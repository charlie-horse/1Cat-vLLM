# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SM70 decode router for the E256/K6 sigmoid MoE with an FP32 gate.

The generic route casts the FP16 hidden states to FP32, runs an FP32 GEMM
against the FP32 gate (cuBLAS split-K), and selects with ``topk_sigmoid``:
four kernels and an FP32 copy of the activations per MoE layer. For one or
two tokens, ``torch.ops.vllm_sm70_router.flint_route`` reads the FP16 hidden
states and the FP32 gate as loaded, accumulates in FP32 and selects the
renormalized sigmoid top six in the same launch, with ``topk_sigmoid``'s
scoring, tie-breaking and normalization. Only the FP32 summation order of the
logits differs.

The model adapter binds ``admit_flint_fused_router`` as the layer's
``MoERunner.router_provider``; the runner calls it on the first eager forward.
Admission is decided once per layer from its configuration (gate, router,
parallel layout, kill switch ``VLLM_SM70_FLINT_FUSED_ROUTER``); each call then
checks only tensor metadata (one or two FP16 rows), which is fixed for a
captured CUDA graph.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

logger = init_logger(__name__)

HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 6
MAX_TOKENS = 2
# FP32 logits for MAX_TOKENS rows, then one INT32 arrival counter.
_LOGITS = MAX_TOKENS * NUM_EXPERTS


def _flint_route_op() -> Any | None:
    """The fused operator, or None when the extension lacks it."""
    try:
        import vllm._sm70_router_C  # noqa: F401
    except ImportError:
        return None
    namespace = getattr(torch.ops, "vllm_sm70_router", None)
    if namespace is None or not hasattr(namespace, "flint_route"):
        return None
    return namespace.flint_route


def _policy_enabled() -> bool:
    from vllm.config.sm70_moe import moe_routing_policy

    return bool(moe_routing_policy().value("flint_fused"))


def _is_exact_sm70(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in (
        (7, 0),
        (7, 2),
    )


class FlintFusedRouter:
    """Per-layer state: the gate weight and a private workspace.

    The workspace holds the FP32 logits and the arrival counter that the
    kernel's last CTA re-arms; it is per layer so no two launches share it.
    """

    def __init__(self, op: Any, weight: torch.Tensor, router: BaseRouter) -> None:
        self.op = op
        self.weight = weight
        self.router = router
        self.logits = torch.empty(_LOGITS, dtype=torch.float32, device=weight.device)
        # The kernel requires the counter to be zero before its first launch.
        self.counter = torch.zeros(1, dtype=torch.int32, device=weight.device)

    @staticmethod
    def accepts(hidden_states: torch.Tensor) -> bool:
        """Tensor-metadata gate; fixed for a captured graph."""
        return (
            hidden_states.dim() == 2
            and 1 <= hidden_states.shape[0] <= MAX_TOKENS
            and hidden_states.shape[1] == HIDDEN
            and hidden_states.dtype == torch.float16
            and hidden_states.stride(1) == 1
            and hidden_states.stride(0) % 4 == 0
            and hidden_states.data_ptr() % 8 == 0
        )

    def try_route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Top-k weights and ids, or None to take the generic route.

        The fused route returns what ``select_experts`` would: INT32 ids
        (logical ids, no EPLB), unless a routed-id capture hook is set or the
        experts want another id dtype.
        """
        if (
            not self.accepts(hidden_states)
            or self.router.capture_fn is not None
            or self.router._get_indices_type() not in (None, torch.int32)
        ):
            return None
        return self.route(hidden_states)

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = hidden_states.shape[0]
        topk_weights = torch.empty(
            (tokens, TOP_K), dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            (tokens, TOP_K), dtype=torch.int32, device=hidden_states.device
        )
        self.op(
            topk_weights,
            topk_ids,
            hidden_states,
            self.weight,
            self.logits,
            self.counter,
        )
        return topk_weights, topk_ids


def _rejection(runner: "MoERunner") -> str | None:
    """Why the layer cannot take the fused router, or None if it can."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = runner.gate
    router = runner.router
    if not isinstance(gate, GateLinear):
        return "the gate is not a GateLinear"
    weight = gate.weight
    if (
        weight.dtype != torch.float32
        or tuple(weight.shape) != (NUM_EXPERTS, HIDDEN)
        or not weight.is_contiguous()
        or weight.data_ptr() % 16 != 0
        or getattr(gate, "bias", None) is not None
        or gate.out_dtype != torch.float32
        or gate.allow_dsv3_router_gemm
        or gate.allow_cublas_router_gemm
    ):
        return "the gate is not an unbiased FP32 [256, 2048] GEMM"
    if not _is_exact_sm70(weight.device):
        return "the device is not SM70"
    if type(router) is not FusedTopKRouter:
        return "the router is not the plain fused top-k router"
    if (
        router.scoring_func != "sigmoid"
        or not router.renormalize
        or router.top_k != TOP_K
        or router.global_num_experts != NUM_EXPERTS
        or router.eplb_state is not None
    ):
        return "the router is not a renormalized sigmoid top-6 over 256 experts"
    if (
        runner._fse_fuse_gate
        or runner.routed_input_transform is not None
        or runner.enable_dbo
        or runner._quant_method.is_monolithic
    ):
        return "the layer fuses gates, transforms inputs, overlaps or is monolithic"
    moe = runner.moe_config
    if moe.dp_size != 1 or moe.pcp_size != 1 or moe.use_ep:
        return "the layer dispatches tokens (DP, PCP or EP)"
    return None


def admit_flint_fused_router(runner: "MoERunner") -> FlintFusedRouter | None:
    """Decide once whether this layer's router runs fused at M <= 2."""
    if runner.gate is None or not _policy_enabled():
        return None
    reason = _rejection(runner)
    if reason is not None:
        logger.debug(
            "SM70 fused E256/K6 router not admitted for %s: %s.",
            runner.layer_name,
            reason,
        )
        return None
    router = runner.router
    assert isinstance(router, BaseRouter)
    op = _flint_route_op()
    if op is None:
        logger.warning_once(
            "vllm._sm70_router_C lacks flint_route; the E256/K6 sigmoid router "
            "keeps the generic FP32 GEMM and topk_sigmoid."
        )
        return None
    logger.info_once(
        "SM70 fused E256/K6 sigmoid router enabled for one or two tokens "
        "(FP16 input, FP32 gate and accumulation; "
        "VLLM_SM70_FLINT_FUSED_ROUTER=0 disables it)."
    )
    return FlintFusedRouter(op, runner.gate.weight, router)
