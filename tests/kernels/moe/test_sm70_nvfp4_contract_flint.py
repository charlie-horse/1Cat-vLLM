# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's NVFP4 MoE contract on the SM70 TurboMind and skinny routes.

Flint MoE layers: hidden 2048, moe_intermediate 512 (TP2 local 256), 256
experts, sigmoid top-6 with renormalize. They share every expert shape with
the validated (2048, 512, 256, 8) contract and differ only in top-k.

The CPU tests cover admission, host-side route selection and buffer sizing
with every native operator intercepted. The GPU tests skip unless an exact
SM70 device and the native extension are present.
"""

import dataclasses
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import vllm.config
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe import sm70_small_routing
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.experts import skinny_sm70_moe
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    Nvfp4SkinnySm70Experts,
    grouped_splitk,
    nvfp4_skinny_scale_reason,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.router import fused_topk_router
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.sm70.fp4_workspace import (
    NvFp4MoEWorkspace,
)
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute as Stage,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

# (hidden size, global expert intermediate size, experts, top-k)
FLINT = (2048, 512, 256, 6)
QWEN36 = (2048, 512, 256, 8)
QWEN38 = (2560, 640, 512, 10)
GLM53 = (4096, 2048, 288, 8)

_POLICY_FLAGS = (
    "qpn_m1",
    "qpn_batch",
    "qpn_dynamic",
    "qpn_mtp5",
    "indexed_prefill",
    "grouped_prefill",
    "grouped_decode",
    "grouped_mtp5",
    "grouped_expert_rows",
    "fused_batch_w13",
    "fused_batch_w2",
    "fused_swiglu_prefill",
    "fast_prefill",
    "raw_scale",
    "w2_direct_reduce",
    "glm53_fused_permute",
    "glm53_qpn_w13",
)

requires_sm70 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)


def _moe_config(
    contract=FLINT,
    tp_size: int = 2,
    *,
    scoring_func: str = "sigmoid",
    **overrides,
) -> FusedMoEConfig:
    hidden, global_intermediate, experts, top_k = contract
    parallel = dataclasses.replace(
        FusedMoEParallelConfig.make_no_parallel(),
        tp_size=tp_size,
        **overrides.pop("parallel", {}),
    )
    values = dict(
        num_experts=experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size_per_partition=global_intermediate // tp_size,
        num_local_experts=experts,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=get_routing_method_type(
            scoring_func=scoring_func,
            top_k=top_k,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=False,
        ),
        moe_parallel_config=parallel,
        in_dtype=torch.float16,
    )
    values.update(overrides)
    return FusedMoEConfig(**values)


def _vllm_config(device: str = "cpu") -> VllmConfig:
    # An explicit device keeps config construction working without a GPU.
    return VllmConfig(device_config=DeviceConfig(device=device))


def _w4a16_config() -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )


def _all_on_policy() -> NS:
    return NS(
        **dict.fromkeys(_POLICY_FLAGS, True),
        route_debug=False,
        explicit_fields=frozenset(),
    )


def _loaded_layer(contract=FLINT, tp_size: int = 2, **attrs) -> NS:
    """The attributes process_weights_after_loading leaves on a layer."""
    hidden, global_intermediate, experts, top_k = contract
    values = dict(
        moe_config=_moe_config(contract, tp_size),
        sm70_moe_policy=_all_on_policy(),
        sm70_nvfp4_batch_capabilities={"w13": True, "w2": True},
        sm70_nvfp4_num_experts=experts,
        sm70_nvfp4_hidden_size=hidden,
        sm70_nvfp4_intermediate_size=global_intermediate // tp_size,
        sm70_nvfp4_top_k=top_k,
        swiglu_limit=None,
        global_num_experts=experts,
        local_num_experts=experts,
        expert_map=None,
        w13_tm_weight=torch.empty(0, dtype=torch.uint8),
        sm70_fp4_codec=object(),
        # Load leaves these policy-level switches on without a shape gate
        # when the native operators exist; apply() must exclude by shape.
        sm70_nvfp4_qwen38_indexed_prefill=True,
        sm70_nvfp4_qwen38_w2_direct_reduce=True,
        # Load gates these on the exact Qwen3.8 / GLM-5.3 shapes.
        sm70_nvfp4_qwen38_fused_swiglu_prefill=False,
        sm70_nvfp4_qwen38_fused_swiglu_decode=False,
        sm70_nvfp4_qwen38_fast_prefill=False,
        sm70_nvfp4_qwen38_raw_scale=False,
        sm70_glm53_fused_permute_q8=False,
        sm70_glm53_qpn_w13_q8=False,
        sm70_nvfp4_grouped_decode=False,
        sm70_nvfp4_grouped_mtp5=False,
        # Load admits the two-kernel M<=2 decode on Flint's TP2 shapes only.
        sm70_nvfp4_decode_qpn=_admits_decode_qpn(contract, tp_size),
    )
    values.update(attrs)
    return NS(**values)


def _admits_decode_qpn(contract, tp_size: int) -> bool:
    hidden, global_intermediate, experts, top_k = contract
    local = (experts, hidden, global_intermediate // tp_size, top_k)
    return local in moe._DECODE_QPN_CONTRACTS


def _route_inputs(tokens: int, contract=FLINT):
    hidden, _, experts, top_k = contract
    x = torch.zeros(tokens, hidden, dtype=torch.float16)
    ids = torch.rand(tokens, experts).topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32)
    return x, ids.contiguous(), weights


def _patch_sort_workspace(monkeypatch) -> Mock:
    size = Mock(return_value=64)
    monkeypatch.setattr(
        torch.ops._moe_C, "moe_permute_sort_workspace_size", size, raising=False
    )
    return size


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def test_flint_contract_is_registered_beside_existing_contracts():
    assert FLINT in moe._SUPPORTED_CONTRACTS
    for contract in (QWEN36, QWEN38, GLM53):
        assert contract in moe._SUPPORTED_CONTRACTS
    assert moe._SUPPORTED_TP_SIZES == (1, 2, 4)


@pytest.mark.parametrize("tp_size", (1, 2, 4))
def test_contract_accepts_flint(tp_size):
    moe.validate_nvfp4_sm70_moe_contract(_moe_config(FLINT, tp_size))


@pytest.mark.parametrize(
    ("contract", "tp_size"),
    [
        (QWEN36, 1),
        (QWEN36, 2),
        (QWEN36, 4),
        (QWEN38, 4),
        (GLM53, 4),
        (GLM53, 8),
    ],
)
def test_contract_still_accepts_existing_contracts(contract, tp_size):
    moe.validate_nvfp4_sm70_moe_contract(
        _moe_config(contract, tp_size, scoring_func="softmax")
    )


@pytest.mark.parametrize(
    "contract",
    [
        # Top-k next to Flint's 6 and Qwen3.6's 8.
        (2048, 512, 256, 4),
        (2048, 512, 256, 5),
        (2048, 512, 256, 7),
        (2048, 512, 256, 10),
        # One field off Flint.
        (2048, 512, 128, 6),
        (2048, 512, 288, 6),
        (2048, 512, 512, 6),
        (2560, 512, 256, 6),
        (4096, 512, 256, 6),
        (2048, 256, 256, 6),
        (2048, 1024, 256, 6),
        (2048, 640, 256, 6),
    ],
)
@pytest.mark.parametrize("tp_size", (1, 2))
def test_contract_rejects_nearby_shapes(contract, tp_size):
    with pytest.raises(NotImplementedError, match="not validated"):
        moe.validate_nvfp4_sm70_moe_contract(_moe_config(contract, tp_size))


@pytest.mark.parametrize("tp_size", (3, 8))
def test_contract_rejects_flint_at_unsupported_tp(tp_size):
    # TP8 is reserved for the GLM-5.3 contract.
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        moe.validate_nvfp4_sm70_moe_contract(_moe_config(FLINT, tp_size))


def test_contract_rejects_flint_with_all2all():
    config = _moe_config(FLINT, 2, parallel={"dp_size": 2, "use_ep": True})
    assert config.moe_parallel_config.use_all2all_kernels
    with pytest.raises(NotImplementedError, match="all-to-all"):
        moe.validate_nvfp4_sm70_moe_contract(config)


def test_flint_admission_comes_from_the_contract_entry(monkeypatch):
    monkeypatch.setattr(moe, "_SUPPORTED_CONTRACTS", moe._SUPPORTED_CONTRACTS - {FLINT})
    with pytest.raises(NotImplementedError, match="not validated"):
        moe.validate_nvfp4_sm70_moe_contract(_moe_config(FLINT, 2))


# ---------------------------------------------------------------------------
# Method selection and the skinny -> TurboMind fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", (1, 2))
def test_turbomind_method_admits_flint_without_skinny(monkeypatch, tp_size):
    class FakeRoutedExperts:
        moe_config = _moe_config(FLINT, tp_size)

    monkeypatch.setattr(modelopt, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(skinny_sm70_moe, "skinny_backend_admitted", lambda *a: False)
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(sm70_tm, "should_use_nvfp4_moe_turbomind", lambda: True)
    # The native policy ABI needs the built extension; admission does not.
    monkeypatch.setattr(moe, "NativeBindings", lambda values=(): NS(values=values))
    with set_current_vllm_config(_vllm_config()):
        method = _w4a16_config().get_quant_method(
            FakeRoutedExperts(), "model.layers.2.mlp.experts"
        )
    assert isinstance(method, moe.ModelOptNvFp4SM70MoEMethod)
    assert method.use_a16
    assert method.maybe_make_prepare_finalize() is None

    FakeRoutedExperts.moe_config = _moe_config(FLINT, tp_size, experts_per_token=7)
    with (
        set_current_vllm_config(_vllm_config()),
        pytest.raises(NotImplementedError, match="not validated"),
    ):
        _w4a16_config().get_quant_method(
            FakeRoutedExperts(), "model.layers.2.mlp.experts"
        )


def _scale_layer(experts: int, *, split: bool) -> NS:
    w13 = torch.full((experts, 2), 0.01, dtype=torch.float32)
    if split:
        w13[:, 1] *= 2
    return NS(
        w13_weight_scale_2=w13,
        w2_weight_scale_2=torch.full((experts,), 0.02, dtype=torch.float32),
        _replace_quant_method=Mock(),
    )


def _skinny_method(config: FusedMoEConfig) -> ModelOptNvFp4FusedMoE:
    method = object.__new__(ModelOptNvFp4FusedMoE)
    method.moe = config
    method.quant_config = _w4a16_config()
    method.nvfp4_backend = NvFp4MoeBackend.SM70_SKINNY
    return method


@pytest.mark.parametrize("tp_size", (1, 2))
def test_split_gate_up_scales_fall_back_to_admitted_turbomind(monkeypatch, tp_size):
    prepared = []
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(moe, "NativeBindings", lambda values=(): NS(values=values))
    monkeypatch.setattr(
        moe.ModelOptNvFp4SM70MoEMethod,
        "process_weights_after_loading",
        lambda self, layer: prepared.append((self, layer)),
    )
    method = _skinny_method(_moe_config(FLINT, tp_size))
    layer = _scale_layer(256, split=True)
    vllm_config = _vllm_config()
    with set_current_vllm_config(vllm_config):
        method.process_weights_after_loading(layer)

    fallback = layer._replace_quant_method.call_args.args[0]
    assert isinstance(fallback, moe.ModelOptNvFp4SM70MoEMethod)
    assert layer.base_quant_method is fallback
    assert prepared == [(fallback, layer)]
    selection = vllm_config.kernel_config.moe_kernel_selections[
        f"Nvfp4SkinnySm70Experts:2048:{512 // tp_size}"
    ]
    assert selection["enabled"] is False
    assert "shared factor" in selection["reason"]

    # Before the Flint contract existed, this fallback raised mid-load.
    monkeypatch.setattr(moe, "_SUPPORTED_CONTRACTS", moe._SUPPORTED_CONTRACTS - {FLINT})
    with (
        set_current_vllm_config(_vllm_config()),
        pytest.raises(NotImplementedError, match="not validated"),
    ):
        _skinny_method(_moe_config(FLINT, tp_size)).process_weights_after_loading(
            _scale_layer(256, split=True)
        )


def test_split_scale_fallback_still_rejects_unvalidated_top_k(monkeypatch):
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(moe, "NativeBindings", lambda values=(): NS(values=values))
    method = _skinny_method(_moe_config(FLINT, 2, experts_per_token=7))
    with (
        set_current_vllm_config(_vllm_config()),
        pytest.raises(NotImplementedError, match="not validated"),
    ):
        method.process_weights_after_loading(_scale_layer(256, split=True))


def test_flint_global_scales_decide_skinny_eligibility():
    shared = _scale_layer(256, split=False)
    assert (
        nvfp4_skinny_scale_reason(shared.w13_weight_scale_2, shared.w2_weight_scale_2)
        is None
    )
    split = _scale_layer(256, split=True)
    reason = nvfp4_skinny_scale_reason(
        split.w13_weight_scale_2, split.w2_weight_scale_2
    )
    assert reason is not None and "shared factor" in reason


# ---------------------------------------------------------------------------
# Skinny route admission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", (1, 2))
def test_skinny_split_k_specializations_cover_flint(tp_size):
    local_intermediate = 512 // tp_size
    # W13 reduces over hidden, W2 over the local intermediate size.
    assert grouped_splitk(2048, 16) == 16
    assert grouped_splitk(local_intermediate, 16) == 16
    assert grouped_splitk(local_intermediate, 8) == 8


@pytest.mark.parametrize("tp_size", (1, 2))
def test_skinny_route_admits_flint(monkeypatch, tp_size):
    kernel = KernelConfig()
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: NS(kernel_config=kernel),
    )
    monkeypatch.setattr(
        Nvfp4SkinnySm70Experts, "_supports_current_device", lambda: True
    )
    monkeypatch.setattr(torch.ops._C, "skinny_moe_qpn_sm70", object(), raising=False)
    config = _moe_config(FLINT, tp_size)

    # The full oracle contract, including the generic modular-kernel checks.
    assert Nvfp4SkinnySm70Experts.is_supported_config(
        Nvfp4SkinnySm70Experts,
        config,
        kNvfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    ) == (True, None)
    assert skinny_sm70_moe.skinny_backend_admitted(config)
    assert modelopt._sm70_moe_backend_requested_explicitly(NS(moe_config=config))
    selection = kernel.moe_kernel_selections[
        f"Nvfp4SkinnySm70Experts:2048:{512 // tp_size}"
    ]
    assert selection["enabled"] is True and selection["reason"] is None

    # Without the native operator the automatic route keeps TurboMind. Hide the
    # whole namespace: torch.ops._C resolves a built extension's real operator
    # again once the stub attribute is deleted.
    monkeypatch.setattr(torch.ops, "_C", NS())
    assert not skinny_sm70_moe.skinny_backend_admitted(config)


# ---------------------------------------------------------------------------
# TurboMind host-side route selection and buffers
# ---------------------------------------------------------------------------


def test_flint_compact_grouped_bound_is_routed_rows():
    top_k = FLINT[3]
    widths = [n for n in range(1, 64) if moe._use_compact_grouped(n, top_k)]
    assert widths == list(range(1, 14))
    assert 13 * top_k <= moe._COMPACT_GROUPED_MAX_SLOTS < 14 * top_k
    # Compact widths always fit the persistent graph-safe buffers.
    assert widths[-1] <= moe._GRAPH_SAFE_MAX_TOKENS


@pytest.mark.parametrize("tp_size", (1, 2))
def test_flint_workspace_sizes_follow_top_k(monkeypatch, tp_size):
    size = _patch_sort_workspace(monkeypatch)
    hidden, global_intermediate, experts, top_k = FLINT
    intermediate = global_intermediate // tp_size
    layer = _loaded_layer(
        FLINT, tp_size, w13_tm_weight=torch.empty(0, dtype=torch.uint8, device="meta")
    )
    NvFp4MoEWorkspace.allocate(layer)

    max_tokens = moe._GRAPH_SAFE_MAX_TOKENS
    max_slots = max_tokens * top_k
    size.assert_called_once_with(max_slots, experts)
    assert layer._nvfp4_sm70_output.shape == (max_tokens, hidden)
    assert layer._nvfp4_sm70_permuted_input.shape == (max_slots, hidden)
    assert layer._nvfp4_sm70_gate_up.shape == (max_slots, 2 * intermediate)
    assert layer._nvfp4_sm70_intermediate.shape == (max_slots, intermediate)
    assert layer._nvfp4_sm70_sorted_output.shape == (max_slots, hidden)
    assert layer._nvfp4_sm70_token_expert_indices.shape == (max_tokens, top_k)
    assert layer._nvfp4_sm70_inv_permuted_idx.shape == (max_tokens, top_k)
    assert layer._nvfp4_sm70_expert_offsets64.shape == (experts + 1,)
    assert layer._nvfp4_sm70_dense_expert_ids.shape == (experts,)
    assert layer._nvfp4_sm70_compact_offsets.shape == (max_slots + 1,)
    assert max_slots >= moe._COMPACT_GROUPED_MAX_SLOTS

    for tokens in (1, 2, 13, 14, max_tokens, max_tokens + 1, 128):
        slots = tokens * top_k
        buffers = NvFp4MoEWorkspace.get(layer, tokens, False)
        assert buffers["output"].shape == (tokens, hidden)
        assert buffers["permuted_input"].shape == (slots, hidden)
        assert buffers["gate_up"].shape == (slots, 2 * intermediate)
        assert buffers["intermediate"].shape == (slots, intermediate)
        assert buffers["sorted_output"].shape == (slots, hidden)
        assert buffers["topk_ids"].shape == (tokens, top_k)
        assert buffers["token_expert_indices"].shape == (tokens, top_k)
        assert buffers["inv_permuted_idx"].shape == (tokens, top_k)
        assert buffers["compact_offsets"].shape == (slots + 1,)
        assert buffers["active_expert_ids"].shape == (slots,)
        assert buffers["expert_offsets64"].shape == (experts + 1,)
        assert buffers["dense_expert_ids"] is layer._nvfp4_sm70_dense_expert_ids
        persistent = buffers["output"]._base is layer._nvfp4_sm70_output
        assert persistent == (tokens <= max_tokens)


def _engaged_specialized_routes(layer, tokens: int, contract) -> list[str]:
    x, ids, weights = _route_inputs(tokens, contract)
    checks = {
        "decode_qpn": moe._use_decode_qpn(layer, x, weights, ids),
        "qpn_m1": moe._use_qwen38_qpn_m1_decode(layer, x, ids),
        "qpn_batch": moe._use_qwen38_qpn_batch_decode(layer, x, ids),
        "qpn_batch_w13": moe._use_qwen38_qpn_batch_fused_w13(layer, x, ids),
        "qpn_batch_w2": moe._use_qwen38_qpn_batch_fused_w2(layer, x, ids),
        "qpn_mtp5": moe._use_qwen38_qpn_mtp5_decode(layer, x, ids),
        "grouped_mtp5": moe._use_grouped_mtp5(layer, x, ids),
        "grouped_decode": moe._use_grouped_decode(layer, x, ids),
        "indexed_prefill": moe._use_qwen38_indexed_prefill(layer, x, ids),
        "glm_rows": moe._use_glm53_grouped_expert_rows(layer, tokens),
        "glm_permute": moe._use_glm53_fused_permute_q8(layer, x, ids),
        "glm_qpn": moe._use_glm53_qpn_w13_q8(layer, x, ids, fused_permute=True),
    }
    return [name for name, engaged in checks.items() if engaged]


@pytest.mark.parametrize("tp_size", (1, 2, 4))
@pytest.mark.parametrize("tokens", (1, 2, 4, 5, 8, 10, 16, 128, 512))
def test_flint_never_engages_model_specific_routes(tp_size, tokens):
    # Every policy and load-time flag on: only the shape gates remain.
    layer = _loaded_layer(
        FLINT,
        tp_size,
        sm70_glm53_fused_permute_q8=True,
        sm70_glm53_qpn_w13_q8=True,
        sm70_nvfp4_grouped_decode=True,
        sm70_nvfp4_grouped_mtp5=True,
    )
    # Only Flint's own M<=2 decode, and only where load admitted it (TP2).
    expected = ["decode_qpn"] if tp_size == 2 and tokens <= 2 else []
    assert _engaged_specialized_routes(layer, tokens, FLINT) == expected


def test_specialized_route_probe_detects_its_own_contract():
    layer = _loaded_layer(QWEN38, 4)
    assert "qpn_m1" in _engaged_specialized_routes(layer, 1, QWEN38)
    assert "qpn_mtp5" in _engaged_specialized_routes(layer, 5, QWEN38)
    assert "indexed_prefill" in _engaged_specialized_routes(layer, 128, QWEN38)
    assert "decode_qpn" not in _engaged_specialized_routes(layer, 1, QWEN38)


@pytest.mark.parametrize(
    ("contract", "tp_size", "admitted"),
    [
        (FLINT, 2, True),
        (FLINT, 1, False),
        (FLINT, 4, False),
        (QWEN36, 2, False),
        (QWEN38, 4, False),
        (GLM53, 8, False),
    ],
)
def test_decode_qpn_contract_is_flint_tp2_only(contract, tp_size, admitted):
    assert _admits_decode_qpn(contract, tp_size) is admitted


@pytest.mark.parametrize(
    ("peak", "shift"),
    [
        (1.0, 0),
        (1.1826171875, 0),  # largest prepared W2 scale in the real checkpoint
        (3.998046875, 0),  # 3.998 * 2^14 = 65504, the FP16 max
        (4.0, 1),
        (14.0, 2),
        (448.0 * 16, 11),
        (65504.0, 14),
    ],
)
def test_decode_qpn_scale_shift_keeps_fp16_headroom(peak, shift):
    scales = torch.full((2, 3), peak / 4, dtype=torch.float16)
    scales[1, 2] = peak
    assert moe._decode_qpn_scale_shift(scales) == shift
    assert peak * 2.0 ** (14 - shift) <= 65504.0


@pytest.mark.parametrize("bad", (float("inf"), float("nan")))
def test_decode_qpn_scale_shift_rejects_non_finite_scales(bad):
    scales = torch.ones(4, dtype=torch.float16)
    scales[2] = bad
    assert moe._decode_qpn_scale_shift(scales) is None


def _decode_load_layer(scale_peak: float = 1.0) -> NS:
    w13_scales = torch.full((4, 8), 0.01, dtype=torch.float16)
    w2_scales = torch.full((4, 8), 0.01, dtype=torch.float16)
    w2_scales[3, 1] = scale_peak
    return NS(
        sm70_moe_policy=NS(flint_qpn_decode=True, explicit_fields=()),
        w13_tm_weight=torch.empty(0),
        w13_tm_scales=w13_scales,
        w2_tm_scales=w2_scales,
    )


FLINT_TP2_LOCAL = (256, 2048, 256, 6)


def _prepare_decode(layer, contract=FLINT_TP2_LOCAL, *, available=True, **kwargs):
    native = NS(has_nvfp4_moe_decode_dispatch=lambda: available)
    moe._prepare_decode_qpn(layer, native, contract, **kwargs)


def test_decode_qpn_load_admits_flint_tp2_with_workspace():
    layer = _decode_load_layer()
    _prepare_decode(layer, supported=True)
    assert layer.sm70_nvfp4_decode_qpn
    assert layer.sm70_nvfp4_decode_w13_scale_shift == 0
    assert layer.sm70_nvfp4_decode_w2_scale_shift == 0
    assert layer._nvfp4_decode_partials.shape == (moe._DECODE_QPN_PARTIALS,)
    assert layer._nvfp4_decode_partials.dtype == torch.float32
    assert layer._nvfp4_decode_counters.shape == (moe._DECODE_QPN_COUNTERS,)
    assert not layer._nvfp4_decode_counters.any()


def test_decode_qpn_load_pre_shifts_large_scales():
    layer = _decode_load_layer(scale_peak=14.0)
    _prepare_decode(layer, supported=True)
    assert layer.sm70_nvfp4_decode_qpn
    assert layer.sm70_nvfp4_decode_w13_scale_shift == 0
    assert layer.sm70_nvfp4_decode_w2_scale_shift == 2


@pytest.mark.parametrize(
    ("contract", "supported", "enabled", "available", "peak"),
    [
        ((256, 2048, 512, 6), True, True, True, 1.0),  # Flint TP1
        ((256, 2048, 256, 8), True, True, True, 1.0),  # Qwen3.6 TP2
        (FLINT_TP2_LOCAL, False, True, True, 1.0),  # raw scale / clamp / ...
        (FLINT_TP2_LOCAL, True, False, True, 1.0),  # kill switch
        (FLINT_TP2_LOCAL, True, True, False, 1.0),  # ops absent (default)
        (FLINT_TP2_LOCAL, True, True, True, float("inf")),  # no FP16 range
    ],
)
def test_decode_qpn_load_falls_back(contract, supported, enabled, available, peak):
    layer = _decode_load_layer(scale_peak=peak)
    layer.sm70_moe_policy.flint_qpn_decode = enabled
    _prepare_decode(layer, contract, available=available, supported=supported)
    assert not layer.sm70_nvfp4_decode_qpn
    assert not hasattr(layer, "_nvfp4_decode_partials")


def test_decode_qpn_explicit_request_fails_closed_without_ops():
    layer = _decode_load_layer()
    layer.sm70_moe_policy.explicit_fields = ("flint_qpn_decode",)
    with pytest.raises(RuntimeError, match="VLLM_SM70_FLINT_MOE_QPN_DECODE"):
        _prepare_decode(layer, available=False, supported=True)


def test_decode_qpn_kill_switch_env(monkeypatch):
    from vllm.config.sm70_moe import Sm70NvFp4MoEConfig

    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", raising=False)
    default = Sm70NvFp4MoEConfig()
    default.resolve()
    assert default.flint_qpn_decode is True
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    disabled = Sm70NvFp4MoEConfig()
    disabled.resolve()
    assert disabled.flint_qpn_decode is False


@pytest.mark.parametrize("tp_size", (1, 2))
@pytest.mark.parametrize("tokens", (1, 2, 8, 13, 14, 18, 19, 128))
def test_flint_apply_takes_generic_turbomind_routes(monkeypatch, tp_size, tokens):
    _patch_sort_workspace(monkeypatch)
    hidden, _, experts, top_k = FLINT
    slots = tokens * top_k
    layer = _loaded_layer(FLINT, tp_size)
    NvFp4MoEWorkspace.allocate(layer)
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    method.native_ops = Mock()
    execute = Mock()
    single = Mock()
    slot_groups = Mock()
    expert_groups = Mock()
    monkeypatch.setattr(moe, "execute_fp4", execute)
    monkeypatch.setattr(moe, "_prepare_single_token_slots", single)
    monkeypatch.setattr(moe, "_prepare_compact_slot_groups", slot_groups)
    monkeypatch.setattr(moe, "_prepare_compact_expert_groups", expert_groups)
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **k: True)
    x, ids, weights = _route_inputs(tokens)
    # Execute host dispatch on CPU tensors; every GPU operator is intercepted.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    output = method.apply(layer, x, weights, ids, None, None)

    execute.assert_called_once()
    plan = execute.call_args.args[1]
    native = method.native_ops
    if tp_size == 2 and tokens <= 2:
        # Two native launches route on device: no permute, no slot prepare.
        # The fused W2 stage writes the layer's persistent output buffer.
        assert output is execute.return_value
        assert execute.call_args.args[2]["output"].shape == (tokens, hidden)
        assert (plan.w13, plan.w2) == (Stage.DECODE_QPN, Stage.DECODE_REDUCE)
        assert execute.call_args.args[4] is ids
        assert execute.call_args.args[5] is weights
        assert execute.call_args.kwargs == {}
        single.assert_not_called()
        slot_groups.assert_not_called()
        native.moe_permute_with_scratch.assert_not_called()
        return
    assert output.shape == (tokens, hidden)
    assert (plan.w13, plan.w2) == (Stage.DENSE, Stage.DENSE)
    assert not plan.interleaved and not plan.qpn_mtp
    assert plan.w13_split_k == 1
    kwargs = execute.call_args.kwargs
    native = method.native_ops
    native.moe_permute_metadata_with_scratch.assert_not_called()
    native.sm70_glm53_moe_permute_q8_out.assert_not_called()
    expert_groups.assert_not_called()
    if tokens == 1:
        single.assert_called_once()
        native.moe_permute_with_scratch.assert_not_called()
        slot_groups.assert_not_called()
        assert plan.reduction == "triton_single"
        assert kwargs["expert_count"] == top_k
        assert kwargs["offsets"].shape == (top_k + 1,)
        return
    single.assert_not_called()
    native.moe_permute_with_scratch.assert_called_once()
    # (input, ids, indices, expert_map, global, local, top_k, ...)
    permute_args = native.moe_permute_with_scratch.call_args.args
    assert permute_args[4:7] == (experts, experts, top_k)
    assert plan.reduction == "unpermute"
    if tokens <= 13:
        slot_groups.assert_called_once()
        assert kwargs["expert_count"] == slots
        assert kwargs["offsets"].shape == (slots + 1,)
    else:
        slot_groups.assert_not_called()
        assert kwargs["expert_count"] == experts
        assert kwargs["expert_ids"] is layer._nvfp4_sm70_dense_expert_ids


# ---------------------------------------------------------------------------
# Router: sigmoid + renormalize top-6 over 256 experts
# ---------------------------------------------------------------------------


def test_flint_router_is_the_generic_fused_topk_router():
    router = create_fused_moe_router(
        top_k=6, global_num_experts=256, renormalize=True, scoring_func="sigmoid"
    )
    assert type(router) is FusedTopKRouter
    assert router.scoring_func == "sigmoid" and router.renormalize
    assert router.routing_method_type == RoutingMethodType.SigmoidRenorm


def _reference_topk(scores_fn):
    def topk(weights, ids, token_expert_indices, gating, renormalize=False, *_):
        scores = scores_fn(gating.float())
        values, indices = scores.topk(weights.shape[1], dim=-1)
        if renormalize:
            values = values / values.sum(dim=-1, keepdim=True)
        weights.copy_(values)
        ids.copy_(indices)
        token_expert_indices.zero_()

    return Mock(side_effect=topk)


@pytest.fixture
def router_ops(monkeypatch):
    ops = NS(
        sigmoid=_reference_topk(torch.sigmoid),
        softmax=_reference_topk(lambda x: torch.softmax(x, dim=-1)),
        qwen38=Mock(),
    )
    monkeypatch.setattr(fused_topk_router.ops, "topk_sigmoid", ops.sigmoid)
    monkeypatch.setattr(fused_topk_router.ops, "topk_softmax", ops.softmax)
    monkeypatch.setattr(fused_topk_router, "_sm70_qwen38_router_topk", ops.qwen38)
    # Enable the E512/K10 fast path everywhere it could apply.
    monkeypatch.setattr(
        fused_topk_router,
        "moe_routing_policy",
        lambda *a: NS(value=lambda name: True),
    )
    monkeypatch.setattr(
        fused_topk_router,
        "current_platform",
        NS(is_device_capability=lambda *a, **k: True),
    )
    return ops


@pytest.mark.parametrize("tokens", (1, 5, 16, 64))
def test_flint_sigmoid_topk_uses_generic_kernel(router_ops, tokens):
    hidden = torch.zeros(tokens, 2048, dtype=torch.float16)
    gating = torch.randn(tokens, 256).to(torch.float16)
    weights, ids, _ = fused_topk_router.fused_topk(
        hidden, gating, 6, True, scoring_func="sigmoid"
    )
    router_ops.qwen38.assert_not_called()
    router_ops.sigmoid.assert_called_once()
    assert weights.shape == ids.shape == (tokens, 6)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    torch.testing.assert_close(weights.sum(-1), torch.ones(tokens))
    reference = torch.sigmoid(gating.float()).topk(6, dim=-1).indices
    assert torch.equal(ids.long(), reference)


def test_qwen38_router_fast_path_needs_its_exact_contract(router_ops):
    hidden = torch.zeros(4, 2048, dtype=torch.float16)
    fused_topk_router.fused_topk(
        hidden, torch.randn(4, 256).half(), 6, True, scoring_func="softmax"
    )
    router_ops.qwen38.assert_not_called()
    router_ops.softmax.assert_called_once()

    # Positive control: the probe does reach the fast path on E512/K10.
    fused_topk_router.fused_topk(
        torch.zeros(4, 2560, dtype=torch.float16),
        torch.randn(4, 512).half(),
        10,
        True,
        scoring_func="softmax",
    )
    router_ops.qwen38.assert_called_once()


def _fake_route_tensors(tokens: int, hidden: int, top_k: int):
    def contiguous():
        return True

    x = NS(
        is_cuda=True,
        dtype=torch.float16,
        ndim=2,
        shape=(tokens, hidden),
        is_contiguous=contiguous,
        device="cuda:0",
    )
    ids = NS(
        dtype=torch.int32,
        ndim=2,
        shape=(tokens, top_k),
        is_contiguous=contiguous,
        device="cuda:0",
    )
    return x, ids


@pytest.mark.parametrize("tokens", (1, 8, 32))
def test_small_routing_fast_path_excludes_flint_geometry(monkeypatch, tokens):
    monkeypatch.setattr(
        sm70_small_routing,
        "current_platform",
        NS(is_device_capability=lambda *a, **k: True),
    )
    capability = sm70_small_routing.SM70_SMALL_ROUTING
    x, ids = _fake_route_tensors(tokens, 2048, 6)
    assert capability.reason(x, ids, 256) == "unmeasured_expert_geometry"
    x, ids = _fake_route_tensors(tokens, 2560, 10)
    assert capability.reason(x, ids, 512) is None


# ---------------------------------------------------------------------------
# GPU: native kernels on Flint shapes (exact SM70 only)
# ---------------------------------------------------------------------------


@requires_sm70
@pytest.mark.parametrize("tokens", (1, 2, 13))
def test_flint_compact_slot_groups_keep_slots_independent(tokens):
    total_slots = tokens * 6
    sorted_ids = (
        torch.arange(total_slots, dtype=torch.int32, device="cuda") // 2
    ) % 256
    offsets = torch.empty(total_slots + 1, dtype=torch.int32, device="cuda")
    active = torch.empty(total_slots, dtype=torch.int32, device="cuda")
    moe._prepare_compact_slot_groups(sorted_ids, offsets, active)
    torch.accelerator.synchronize()
    assert torch.equal(offsets.cpu(), torch.arange(total_slots + 1, dtype=torch.int32))
    assert torch.equal(active, sorted_ids)


@requires_sm70
def test_flint_single_token_route_matches_unpermute():
    top_k, hidden = 6, 2048
    x = torch.randn(1, hidden, dtype=torch.float16, device="cuda")
    ids = torch.tensor([[201, 7, 130, 99, 255, 3]], dtype=torch.int32, device="cuda")
    weights = torch.sigmoid(torch.randn(1, top_k, device="cuda"))
    weights /= weights.sum(dim=-1, keepdim=True)
    expert_output = torch.randn(top_k, hidden, dtype=torch.float16, device="cuda")
    expanded = torch.empty_like(expert_output)
    active = torch.empty(top_k, dtype=torch.int32, device="cuda")
    output = torch.empty(1, hidden, dtype=torch.float16, device="cuda")
    reference = torch.empty_like(output)
    identity = torch.arange(top_k, dtype=torch.int32, device="cuda").view(1, -1)

    moe._prepare_single_token_slots(x, ids, expanded, active)
    moe._single_token_weighted_reduce(expert_output, weights, output)
    torch.ops._moe_C.moe_unpermute(
        expert_output, weights, identity, None, top_k, reference
    )
    torch.accelerator.synchronize()
    assert torch.equal(expanded, x.expand(top_k, -1))
    assert torch.equal(active, ids[0])
    assert torch.equal(output, reference)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize(codes, block_scales, row_scales):
    """ModelOpt NVFP4 [N, K / 2] (low nibble first) -> FP32 [N, K]."""
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=codes.device)
    values = torch.stack(
        (lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1
    ).flatten(1)
    scales = block_scales.float().repeat_interleave(16, dim=1)
    return values * scales * row_scales


def _flint_expert_weights(intermediate: int, generator: torch.Generator):
    hidden, experts = 2048, 256

    def block_scales(rows, groups):
        values = torch.rand(experts, rows, groups, generator=generator) * 1.5 + 0.5
        return values.to(torch.float8_e4m3fn)

    w13_global = torch.full((experts, 2), 0.0075)
    # Distinct gate/up factors: the case the skinny route hands to TurboMind.
    w13_global[:, 1] *= 1.25
    tensors = dict(
        w13_weight=torch.randint(
            0, 256, (experts, 2 * intermediate, hidden // 2), generator=generator
        ).to(torch.uint8),
        w13_weight_scale=block_scales(2 * intermediate, hidden // 16),
        w13_weight_scale_2=w13_global,
        w2_weight=torch.randint(
            0, 256, (experts, hidden, intermediate // 2), generator=generator
        ).to(torch.uint8),
        w2_weight_scale=block_scales(hidden, intermediate // 16),
        w2_weight_scale_2=torch.full((experts,), 0.04),
    )
    return {name: t.cuda().contiguous() for name, t in tensors.items()}


def _reference_moe(x, weights, ids, raw, intermediate):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        g = raw["w13_weight_scale_2"][expert]
        row_scales = g.repeat_interleave(intermediate)[:, None]
        w13 = _dequantize(
            raw["w13_weight"][expert], raw["w13_weight_scale"][expert], row_scales
        )
        gate_up = x[rows].float() @ w13.t()
        act = F.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
        w2 = _dequantize(
            raw["w2_weight"][expert],
            raw["w2_weight_scale"][expert],
            raw["w2_weight_scale_2"][expert],
        )
        out.index_add_(0, rows, (act @ w2.t()) * weights[rows, slots, None])
    return out


@requires_sm70
@pytest.mark.parametrize("tp_size", (1, 2))
def test_flint_turbomind_moe_matches_dequantized_reference(tp_size):
    if not hasattr(torch.ops._C, "nvfp4_moe_dense_stage_sm70_out"):
        pytest.skip("requires the SM70 TurboMind NVFP4 extension")
    intermediate = 512 // tp_size
    generator = torch.Generator().manual_seed(20261028 + tp_size)
    raw = _flint_expert_weights(intermediate, generator)
    config = _moe_config(FLINT, tp_size, device="cuda")
    with set_current_vllm_config(_vllm_config("cuda")):
        method = moe.ModelOptNvFp4SM70MoEMethod(_w4a16_config(), config)
        layer = NS(
            moe_config=config,
            local_num_experts=256,
            global_num_experts=256,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            swiglu_limit=None,
            w13_input_scale=None,
            w2_input_scale=None,
            **{name: t.clone() for name, t in raw.items()},
        )
        method.process_weights_after_loading(layer)
        assert layer.sm70_nvfp4_top_k == 6
        # Direct M1, compact slot groups, persistent dense, eager and prefill.
        for tokens in (1, 2, 8, 13, 14, 18, 19, 64, 200):
            x = torch.randn(tokens, 2048, generator=generator).half().cuda()
            logits = torch.randn(tokens, 256, generator=generator).cuda()
            scores, ids = torch.sigmoid(logits).topk(6, dim=-1)
            weights = (scores / scores.sum(dim=-1, keepdim=True)).contiguous()
            ids = ids.to(torch.int32).contiguous()
            actual = method.apply(layer, x, weights, ids, None, None).clone()
            expected = _reference_moe(x, weights, ids, raw, intermediate)
            torch.accelerator.synchronize()
            assert torch.isfinite(actual).all(), tokens
            error = (actual.float() - expected).norm() / expected.norm()
            assert error < 1e-2, (tokens, error.item())


# ---------------------------------------------------------------------------
# GPU: two-kernel M<=2 decode (W13+SwiGLU, W2+reduce) on Flint TP2 shapes
# ---------------------------------------------------------------------------


def _prepared_flint_layer(raw, tp_size: int = 2):
    config = _moe_config(FLINT, tp_size, device="cuda")
    method = moe.ModelOptNvFp4SM70MoEMethod(_w4a16_config(), config)
    layer = NS(
        moe_config=config,
        local_num_experts=256,
        global_num_experts=256,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
        swiglu_limit=None,
        w13_input_scale=None,
        w2_input_scale=None,
        **{name: t.clone() for name, t in raw.items()},
    )
    method.process_weights_after_loading(layer)
    return method, layer


def _flint_routing(tokens: int, generator: torch.Generator):
    x = torch.randn(tokens, 2048, generator=generator).half().cuda()
    logits = torch.randn(tokens, 256, generator=generator).cuda()
    scores, ids = torch.sigmoid(logits).topk(6, dim=-1)
    weights = (scores / scores.sum(dim=-1, keepdim=True)).contiguous()
    return x, weights, ids.to(torch.int32).contiguous()


def _turbomind_apply(method, layer, x, weights, ids) -> torch.Tensor:
    layer.sm70_nvfp4_decode_qpn = False
    try:
        return method.apply(layer, x, weights, ids, None, None).clone()
    finally:
        layer.sm70_nvfp4_decode_qpn = True


def _relative_error(actual, expected) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _requires_decode_ops():
    if not hasattr(torch.ops._C, "nvfp4_moe_decode_w13_swiglu_sm70_out"):
        pytest.skip("requires the SM70 NVFP4 M<=2 decode extension")


@requires_sm70
@pytest.mark.parametrize("w2_global", (0.04, 7.0), ids=("checkpoint", "x16_w2"))
def test_flint_decode_qpn_matches_reference_and_turbomind(monkeypatch, w2_global):
    """M=1/2 against the FP32 dequantized oracle and the TurboMind route.

    The second case makes the prepared W2 scales reach 28 (E4M3 up to 2 times
    a global of 7), beyond the FP16 limit of prepared_scale * 2^14, so the
    load-time power-of-two pre-shift must engage.
    """
    _requires_decode_ops()
    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", raising=False)
    generator = torch.Generator().manual_seed(20261029)
    raw = _flint_expert_weights(256, generator)
    raw["w2_weight_scale_2"].fill_(w2_global)
    with set_current_vllm_config(_vllm_config("cuda")):
        method, layer = _prepared_flint_layer(raw)
        assert layer.sm70_nvfp4_decode_qpn
        assert layer.sm70_nvfp4_decode_w13_scale_shift == 0
        peak = layer.w2_tm_scales.float().max().item()
        shift = layer.sm70_nvfp4_decode_w2_scale_shift
        assert peak * 2.0 ** (14 - shift) <= 65504.0
        assert (shift > 0) == (peak * 2.0**14 > 65504.0)
        if w2_global > 1:
            assert shift > 0
        for tokens in (1, 2, 1, 2):
            x, weights, ids = _flint_routing(tokens, generator)
            if tokens == 2 and torch.rand(1, generator=generator) < 0.5:
                ids[1] = ids[0]  # both tokens on the same six experts
            actual = method.apply(layer, x, weights, ids, None, None).clone()
            again = method.apply(layer, x, weights, ids, None, None).clone()
            turbomind = _turbomind_apply(method, layer, x, weights, ids)
            expected = _reference_moe(x, weights, ids, raw, 256)
            torch.accelerator.synchronize()
            assert torch.isfinite(actual).all(), tokens
            assert torch.equal(actual, again), "decode must be deterministic"
            assert _relative_error(actual, expected) < 1e-2, tokens
            # TurboMind rounds the same FP16 boundaries in another K order.
            assert _relative_error(actual, turbomind) < 2e-3, tokens


@requires_sm70
def test_flint_decode_qpn_skips_invalid_routes():
    _requires_decode_ops()
    generator = torch.Generator().manual_seed(7)
    raw = _flint_expert_weights(256, generator)
    with set_current_vllm_config(_vllm_config("cuda")):
        method, layer = _prepared_flint_layer(raw)
        assert layer.sm70_nvfp4_decode_qpn
        x, weights, ids = _flint_routing(2, generator)
        ids[0, 3] = -1
        ids[1, 5] = 256
        actual = method.apply(layer, x, weights, ids, None, None).clone()
        valid = (ids >= 0) & (ids < 256)
        expected = _reference_moe(
            x, weights * valid, torch.where(valid, ids, 0), raw, 256
        )
        assert _relative_error(actual, expected) < 1e-2


@requires_sm70
def test_flint_decode_qpn_cuda_graph_replay():
    """FULL-graph style capture at M=1 and M=2: replays track new inputs.

    Replaying many times also checks that the split-K arrival counters re-arm
    inside the kernel, with no reset launch and no host work.
    """
    _requires_decode_ops()
    generator = torch.Generator().manual_seed(11)
    raw = _flint_expert_weights(256, generator)
    with set_current_vllm_config(_vllm_config("cuda")):
        method, layer = _prepared_flint_layer(raw)
        for tokens in (1, 2):
            static = [t.clone() for t in _flint_routing(tokens, generator)]
            method.apply(layer, *static, None, None)  # warm up outside capture
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = method.apply(layer, *static, None, None)
            for _ in range(4):
                case = _flint_routing(tokens, generator)
                for buffer, value in zip(static, case):
                    buffer.copy_(value)
                for _ in range(3):
                    graph.replay()
                replayed = captured.clone()
                eager = method.apply(layer, *case, None, None).clone()
                expected = _reference_moe(case[0], case[1], case[2], raw, 256)
                torch.accelerator.synchronize()
                assert torch.equal(replayed, eager), tokens
                assert _relative_error(replayed, expected) < 1e-2, tokens
            assert not layer._nvfp4_decode_counters.any()


@requires_sm70
def test_flint_decode_qpn_kill_switch_keeps_turbomind(monkeypatch):
    _requires_decode_ops()
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    generator = torch.Generator().manual_seed(3)
    raw = _flint_expert_weights(256, generator)
    with set_current_vllm_config(_vllm_config("cuda")):
        method, layer = _prepared_flint_layer(raw)
        assert not layer.sm70_nvfp4_decode_qpn
        assert not hasattr(layer, "_nvfp4_decode_partials")
        x, weights, ids = _flint_routing(1, generator)
        actual = method.apply(layer, x, weights, ids, None, None)
        expected = _reference_moe(x, weights, ids, raw, 256)
        assert _relative_error(actual, expected) < 1e-2
