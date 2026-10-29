# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token NVFP4 GEMV on the TurboMind operands of ModelOpt W4A16 layers.

Weights are prepared exactly as the ModelOpt W4A16 linear method does on
SM70 (prepare_nvfp4_linear with single_token_gemv=True). The
reference dequantizes the checkpoint codes with the effective FP16 scales
TurboMind uses: fp16(e4m3 * global).
"""

import pytest
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FLINT_SHAPES = {
    "qkv": (2048, 2560),
    "o": (2048, 2048),
    "shared_gate_up": (2048, 1024),
    "shared_down": (512, 2048),
    "dense_gate_up": (2048, 6144),
    "dense_down": (3072, 2048),
}
GLOBAL_SCALE = 0.05 / (6.0 * 448.0)


def _requires_gemv():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    import vllm._sm70_ops  # noqa: F401 - registers the fake implementations

    if not sm70_tm.has_tm_gemv_op():
        pytest.skip("extension lacks nvfp4_tm_gemv_dispatch_sm70_out")


class ModelOptLinear(torch.nn.Module):
    """The attributes a ModelOpt W4A16 linear holds after loading."""

    def __init__(self, codes, scales, global_scale):
        super().__init__()
        self.weight = Parameter(codes, requires_grad=False)
        self.weight_scale = Parameter(scales, requires_grad=False)
        self.weight_global_scale = Parameter(
            torch.tensor(global_scale, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )


def _modelopt_layer(k, n, seed, global_scale=GLOBAL_SCALE, gemv=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(
        0, 256, (n, k // 2), generator=gen, device="cuda", dtype=torch.uint8
    )
    scales = torch.rand((n, k // 16), generator=gen, device="cuda")
    scales = (scales * 384.0 + 64.0).to(torch.float8_e4m3fn)
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device="cuda")
    values = torch.stack(
        (lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1
    ).flatten(-2)
    effective = (scales.float() * global_scale).half().float()
    reference = values * effective.repeat_interleave(16, dim=-1)  # [N, K]
    layer = ModelOptLinear(codes, scales, global_scale)
    sm70_tm.prepare_nvfp4_linear(layer, single_token_gemv=gemv)
    return layer, reference


def _turbomind(layer, x):
    """The call the unadmitted provider makes, with the layer's captured
    native policy (which can change TurboMind's kernel choice)."""
    state = getattr(layer, sm70_tm.STATE_ATTR)
    out = torch.empty(x.shape[0], state.output_size, device="cuda", dtype=x.dtype)
    state.native_ops.nvfp4_gemm_sm70_out(
        out, x, state.weight, state.scales, 16, state.k_ld, state.q_ld
    )
    return out


def _max_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


@pytest.fixture(scope="module")
def layers():
    _requires_gemv()
    built = {}
    for seed, (name, (k, n)) in enumerate(FLINT_SHAPES.items()):
        layer, reference = _modelopt_layer(k, n, seed)
        assert getattr(layer, sm70_tm.STATE_ATTR).tm_gemv, name
        built[name] = (layer, reference)
    return built


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("name", list(FLINT_SHAPES))
@torch.inference_mode()
def test_gemv_matches_fp32_reference_and_turbomind(layers, name, m):
    layer, weight = layers[name]
    state = getattr(layer, sm70_tm.STATE_ATTR)
    assert state.provider.name == "nvfp4_tm_gemv_dispatch_sm70_out"
    assert state.tm_gemv_scale_shift == 14
    x = torch.randn(m, weight.shape[1], device="cuda").half()
    out = sm70_tm.apply_prepared_linear(layer, x, None)
    reference = x.float() @ weight.t()
    turbomind = _turbomind(layer, x)
    magnitude = reference.abs().max().item()
    # FP32 accumulation of exact FP16 products: well inside FP16 output
    # rounding. TurboMind accumulates in a different order.
    assert _max_error(out, reference) <= 2e-3 * magnitude
    assert _max_error(out, turbomind) <= 2e-3 * magnitude
    # A different kernel ran (summation order differs from TurboMind's).
    assert not torch.equal(out, turbomind)


@pytest.mark.parametrize("name", list(FLINT_SHAPES))
@torch.inference_mode()
def test_larger_batches_keep_turbomind(layers, name):
    layer, weight = layers[name]
    for m in (3, 8):
        x = torch.randn(m, weight.shape[1], device="cuda").half()
        out = sm70_tm.apply_prepared_linear(layer, x, None)
        assert torch.equal(out, _turbomind(layer, x))


@pytest.mark.parametrize("name", ["qkv", "shared_down", "dense_gate_up"])
@torch.inference_mode()
def test_cuda_graph_replay_is_deterministic(layers, name):
    layer, weight = layers[name]
    k = weight.shape[1]
    inputs = {m: torch.zeros(m, k, device="cuda", dtype=torch.half) for m in (1, 2)}
    outputs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for m, x in inputs.items():  # warm up outside capture
            sm70_tm.apply_prepared_linear(layer, x, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for m, x in inputs.items():
            outputs[m] = sm70_tm.apply_prepared_linear(layer, x, None)
    for _ in range(3):
        for x in inputs.values():
            x.normal_()
        graph.replay()
        for m, x in inputs.items():
            eager = sm70_tm.apply_prepared_linear(layer, x, None)
            assert torch.equal(outputs[m], eager)
            reference = x.float() @ weight.t()
            magnitude = reference.abs().max().item()
            assert _max_error(outputs[m], reference) <= 2e-3 * magnitude


@torch.inference_mode()
def test_strided_activation_rows(layers):
    layer, weight = layers["o"]
    k = weight.shape[1]
    wide = torch.randn(2, k + 64, device="cuda").half()
    strided = wide[:, :k]
    assert not strided.is_contiguous()
    out = sm70_tm.apply_prepared_linear(layer, strided, None)
    assert torch.equal(out, sm70_tm.apply_prepared_linear(layer, strided.clone(), None))


@torch.inference_mode()
def test_scales_above_four_use_a_smaller_shift():
    _requires_gemv()
    # global 0.05 * e4m3 up to 448 -> effective scales up to ~22.
    layer, weight = _modelopt_layer(512, 2048, seed=11, global_scale=0.05)
    state = getattr(layer, sm70_tm.STATE_ATTR)
    assert state.tm_gemv
    assert state.tm_gemv_scale_shift < 14
    for m in (1, 2):
        x = torch.randn(m, 512, device="cuda").half() * 1e-2
        out = sm70_tm.apply_prepared_linear(layer, x, None)
        reference = x.float() @ weight.t()
        assert torch.isfinite(out).all()
        magnitude = reference.abs().max().item()
        assert _max_error(out, reference) <= 2e-3 * magnitude


@torch.inference_mode()
def test_non_flint_shape_and_unrequested_layers_keep_turbomind():
    _requires_gemv()
    for k, n, gemv in ((1024, 2048, True), (2048, 2048, False)):
        layer, _ = _modelopt_layer(k, n, seed=3, gemv=gemv)
        state = getattr(layer, sm70_tm.STATE_ATTR)
        assert not state.tm_gemv
        assert state.provider.name == "nvfp4_gemm_sm70_out"
