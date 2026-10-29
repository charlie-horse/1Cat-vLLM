# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission of the single-token TurboMind-operand NVFP4 GEMV (CPU only)."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn.parameter import Parameter

import vllm
from vllm.model_executor.kernels.linear import sm70_provider
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

FLINT_SHAPES = sorted(sm70_tm.MODELOPT_NVFP4_GEMV_SHAPES)
GEMV_OP = "nvfp4_tm_gemv_dispatch_sm70_out"


class FakeCudaTensor:
    """Shape/device stand-in for a resident TurboMind tensor."""

    is_cuda = True
    device = "cuda:0"

    def __init__(self, *shape):
        self.shape = shape


def _state(k, n, **changes):
    state = SimpleNamespace(
        op_kind="nvfp4",
        weight=FakeCudaTensor(k, n // 8),
        scales=torch.full((k // 16, n), 1e-3, dtype=torch.float16),
        group_size=16,
        k_ld=32 * k,
        q_ld=n,
        output_size=n,
        padded_output_size=n,
        gated_silu=False,
        prescaled_scales=False,
        use_scale_code=False,
        tm_gemv=False,
        tm_gemv_scale_shift=14,
        refreshed=0,
    )
    state.refresh_provider = lambda: setattr(state, "refreshed", 1)
    for name, value in changes.items():
        setattr(state, name, value)
    return state


def _layer(state):
    layer = torch.nn.Module()
    setattr(layer, sm70_tm.STATE_ATTR, state)
    return layer


@pytest.fixture
def sm70(monkeypatch):
    """Exact SM70 device, built operator, kill switch unset."""
    monkeypatch.delenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", raising=False)
    capability = {"value": (7, 0)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability["value"]
    )
    monkeypatch.setattr(sm70_tm, "has_tm_gemv_op", lambda: True)
    return capability


@pytest.mark.parametrize("shape", FLINT_SHAPES)
def test_flint_shapes_are_admitted(sm70, shape):
    k, n = shape
    state = _state(k, n)
    assert sm70_tm.admit_modelopt_nvfp4_gemv(_layer(state))
    assert state.tm_gemv and state.refreshed
    assert state.tm_gemv_scale_shift == 14


def test_kill_switch_keeps_turbomind(sm70, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", "0")
    state = _state(2048, 2560)
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(_layer(state))
    assert not state.tm_gemv and not state.refreshed


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (6, 1)])
def test_other_architectures_are_not_admitted(sm70, capability):
    sm70["value"] = capability
    state = _state(2048, 2048)
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(_layer(state))


def test_sm72_is_admitted(sm70):
    sm70["value"] = (7, 2)
    assert sm70_tm.admit_modelopt_nvfp4_gemv(_layer(_state(2048, 2048)))


def test_missing_operator_is_not_admitted(sm70, monkeypatch):
    monkeypatch.setattr(sm70_tm, "has_tm_gemv_op", lambda: False)
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(_layer(_state(2048, 2048)))


@pytest.mark.parametrize(
    "shape",
    [
        (5120, 8704),  # Qwen3.x TP2 shapes are not Flint shapes
        (4096, 6144),
        (2560, 2048),  # transposed Flint shape
        (2048, 4096),
        (1024, 2048),
    ],
)
def test_other_model_shapes_are_not_admitted(sm70, shape):
    k, n = shape
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(_layer(_state(k, n)))


@pytest.mark.parametrize(
    "changes",
    [
        {"gated_silu": True},
        {"prescaled_scales": True},
        {"use_scale_code": True},
        {"group_size": 32},
        {"padded_output_size": 2080},
        {"k_ld": 0},
        {"q_ld": 0},
        {"op_kind": "nvfp4_qpn2_dense"},
        {"op_kind": "mxfp4"},
        {"scales": torch.zeros(128, 2048, dtype=torch.uint8)},
    ],
)
def test_non_plain_operands_are_not_admitted(sm70, changes):
    state = _state(2048, 2048, **changes)
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(_layer(state))
    assert not state.tm_gemv


def test_unprepared_layer_is_not_admitted(sm70):
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(torch.nn.Module())


def test_cpu_weights_are_not_admitted(sm70):
    state = _state(2048, 2048, weight=torch.empty(2048, 256, dtype=torch.int32))
    assert not sm70_tm.admit_modelopt_nvfp4_gemv(_layer(state))


@pytest.mark.parametrize(
    "peak, shift", [(1e-3, 14), (3.99, 14), (4.0, 13), (10.0, 12), (448.0, 7)]
)
def test_scale_shift_keeps_fp16_scales_finite(peak, shift):
    scales = torch.tensor([[peak, -peak / 2]], dtype=torch.float16)
    assert sm70_tm._tm_gemv_scale_shift(scales) == shift
    assert float(scales.abs().max()) * 2.0**shift <= 65504.0


def test_large_scales_set_the_shift(sm70):
    state = _state(512, 2048, scales=torch.full((32, 2048), 10.0).half())
    assert sm70_tm.admit_modelopt_nvfp4_gemv(_layer(state))
    assert state.tm_gemv_scale_shift == 12


class RecordingOps:
    def __init__(self):
        self.calls: list[tuple] = []

    def nvfp4_tm_gemv_dispatch_sm70_out(self, out, x, *args):
        self.calls.append((GEMV_OP, tuple(x.shape), args))
        out.fill_(1)

    def nvfp4_gemm_sm70_out(self, out, x, *args):
        self.calls.append(("nvfp4_gemm_sm70_out", tuple(x.shape), args))
        out.fill_(2)


def _bound_state(tm_gemv):
    state = _state(2048, 2048, tm_gemv=tm_gemv, tm_gemv_scale_shift=12)
    state.native_ops = RecordingOps()
    state.provider = sm70_provider.bind_prepared_provider(state)
    return state


@pytest.mark.parametrize("rows", [1, 2, 3, 64])
def test_admitted_provider_passes_every_m_to_one_operator(rows):
    """The M <= 2 / TurboMind branch lives in the operator, not in Python,
    so graph capture and torch.compile see one call for every batch size."""
    state = _bound_state(True)
    assert state.provider.name == GEMV_OP
    x = torch.zeros(rows, 2048, dtype=torch.float16)
    out = sm70_provider.apply_prepared(state, x, None, "")
    assert out.shape == (rows, 2048) and bool((out == 1).all())
    ((name, shape, args),) = state.native_ops.calls
    assert name == GEMV_OP and shape == (rows, 2048)
    assert args[2:] == (16, 32 * 2048, 2048, 12)


def test_unadmitted_provider_keeps_turbomind():
    state = _bound_state(False)
    assert state.provider.name == "nvfp4_gemm_sm70_out"
    x = torch.zeros(1, 2048, dtype=torch.float16)
    sm70_provider.apply_prepared(state, x, None, "")
    assert state.native_ops.calls[0][0] == "nvfp4_gemm_sm70_out"


def _modelopt_layer():
    layer = torch.nn.Module()
    layer.weight_scale_2 = Parameter(torch.tensor([0.5, 0.5]), requires_grad=False)
    layer.input_scale = Parameter(torch.ones(2), requires_grad=False)
    return layer


def _record_sm70_preparation(monkeypatch, prepared):
    calls: list[bool] = []

    def fake_prepare(layer, single_token_gemv=False):
        calls.append(single_token_gemv)
        return prepared

    monkeypatch.setattr(modelopt, "_try_prepare_sm70_modelopt_nvfp4", fake_prepare)
    return calls


@pytest.mark.parametrize("prepared", [True, False])
def test_w4a16_method_requests_the_single_token_gemv(monkeypatch, prepared):
    calls = _record_sm70_preparation(monkeypatch, prepared)
    method = object.__new__(modelopt.ModelOptNvFp4W4A16LinearMethod)
    fallback: list[torch.nn.Module] = []
    method.kernel = SimpleNamespace(process_weights_after_loading=fallback.append)
    method.process_weights_after_loading(_modelopt_layer())
    assert calls == [True]
    assert len(fallback) == (0 if prepared else 1)


def test_w4a4_method_keeps_turbomind_only(monkeypatch):
    calls = _record_sm70_preparation(monkeypatch, True)
    method = object.__new__(modelopt.ModelOptNvFp4LinearMethod)
    method.process_weights_after_loading(_modelopt_layer())
    assert calls == [False]


@pytest.mark.parametrize("single_token_gemv", [True, False])
def test_sm70_preparation_forwards_the_request(monkeypatch, single_token_gemv):
    requests: list[bool] = []
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda", lambda tensor, enabled: True)
    monkeypatch.setattr(sm70_tm, "format_enabled", lambda family: True)
    monkeypatch.setattr(
        sm70_tm,
        "prepare_nvfp4_linear",
        lambda layer, single_token_gemv=False: requests.append(single_token_gemv),
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.zeros(4, 8, dtype=torch.uint8), False)
    layer.weight_scale = Parameter(torch.zeros(4, 1).to(torch.float8_e4m3fn), False)
    assert modelopt._try_prepare_sm70_modelopt_nvfp4(layer, single_token_gemv)
    assert requests == [single_token_gemv]


def test_prepare_admits_only_on_request(monkeypatch):
    admitted: list[torch.nn.Module] = []
    monkeypatch.setattr(
        sm70_tm, "admit_modelopt_nvfp4_gemv", lambda layer: admitted.append(layer)
    )
    monkeypatch.setattr(sm70_tm, "_store_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.ops._C, "nvfp4_sm70_prepare", object(), raising=False)

    class FakeOps:
        @staticmethod
        def nvfp4_sm70_prepare(qweight, scales, group_size, interleave):
            return qweight, scales, torch.tensor([0, 0])

    monkeypatch.setattr(vllm, "_sm70_ops", FakeOps, raising=False)
    monkeypatch.setitem(sys.modules, "vllm._sm70_ops", FakeOps)
    for request in (False, True):
        layer = torch.nn.Module()
        layer.weight = Parameter(torch.zeros(64, 32, dtype=torch.uint8), False)
        layer.weight_scale = Parameter(torch.ones(64, 4).to(torch.float8_e4m3fn), False)
        layer.weight_global_scale = Parameter(torch.tensor(1.0), False)
        sm70_tm.prepare_nvfp4_linear(layer, single_token_gemv=request)
        assert len(admitted) == int(request)
