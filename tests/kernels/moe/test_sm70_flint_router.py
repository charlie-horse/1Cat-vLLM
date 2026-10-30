# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SM70 E256/K6 sigmoid router against the generic FP32 route.

The generic route is what the MoE runner runs today for this layer: cast the
FP16 hidden states to FP32, F.linear with the FP32 gate, then topk_sigmoid
(renormalized). The fused kernel only sums the logits in another FP32 order,
so ids must match except at genuine near-ties, and weights must agree to FP32
rounding.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.fused_moe.router.sm70_flint_router import (
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
    FlintFusedRouter,
    _flint_route_op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((7, 0), (7, 2))
    or _flint_route_op() is None,
    reason="SM70 and vllm._sm70_router_C with flint_route required",
)

FP32_EPS = 2.0**-24
# A flip is a near-tie when the two experts' exact logits differ by at most
# this many FP32 rounding units of the dot product's magnitude, or when their
# exact sigmoid scores are within a few FP32 ulps of 1 (saturation ties).
LOGIT_TIE_UNITS = 64
SCORE_TIE = 4 * 2.0**-24
WEIGHT_ATOL = 2e-6


def reference(x: torch.Tensor, w: torch.Tensor):
    logits = F.linear(x.to(w.dtype), w)
    weights, ids, _ = fused_topk(
        x, logits, TOP_K, renormalize=True, scoring_func="sigmoid"
    )
    return weights, ids


def fused(x: torch.Tensor, w: torch.Tensor, state: FlintFusedRouter | None = None):
    state = state or FlintFusedRouter(_flint_route_op(), w, router=None)
    return state.route(x)


def compare(x, w, weights, ids, ref_weights, ref_ids):
    """Tie-aware comparison; returns the number of rows with flipped ids."""
    exact = x.double() @ w.double().T
    magnitude = x.double().abs() @ w.double().abs().T
    tol = LOGIT_TIE_UNITS * FP32_EPS * magnitude
    score = torch.sigmoid(exact)
    flips = 0
    for row in range(x.shape[0]):
        got = dict(zip(ids[row].tolist(), weights[row].tolist()))
        want = dict(zip(ref_ids[row].tolist(), ref_weights[row].tolist()))
        assert len(got) == TOP_K, f"row {row}: duplicate ids {ids[row]}"
        extra = sorted(set(got) - set(want))
        missing = sorted(set(want) - set(got))
        if extra:
            flips += 1
            for e_in, e_out in zip(extra, missing):
                logit_gap = (exact[row, e_in] - exact[row, e_out]).abs()
                score_gap = (score[row, e_in] - score[row, e_out]).abs()
                assert (
                    logit_gap <= tol[row, e_in] + tol[row, e_out]
                    or score_gap <= SCORE_TIE
                ), (
                    f"row {row}: expert {e_in} replaced {e_out} with exact "
                    f"logit gap {float(logit_gap):.3e} and score gap "
                    f"{float(score_gap):.3e}"
                )
        for expert in set(got) & set(want):
            assert abs(got[expert] - want[expert]) <= WEIGHT_ATOL, (
                f"row {row} expert {expert}: {got[expert]} vs {want[expert]}"
            )
        total = sum(got.values())
        assert total == 0.0 or abs(total - 1.0) <= 1e-5
    return flips


def gate(seed: int, scale: float = 0.02) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=g) * scale
    # Real router weights are FP16 values held in FP32.
    return w.half().float()


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("x_scale", [0.05, 1.0, 8.0])
def test_random_matches_generic_route(tokens, x_scale):
    torch.manual_seed(tokens * 7 + int(x_scale * 10))
    w = gate(1234)
    state = FlintFusedRouter(_flint_route_op(), w, router=None)
    flips = rows = 0
    for _ in range(150):
        x = (torch.randn(tokens, HIDDEN, device="cuda") * x_scale).half()
        # Massive-activation channels, as in real residual-normed inputs.
        x[:, torch.randint(0, HIDDEN, (4,))] *= 30
        weights, ids = fused(x, w, state)
        ref_weights, ref_ids = reference(x, w)
        assert ids.dtype == ref_ids.dtype == torch.int32
        assert weights.dtype == ref_weights.dtype == torch.float32
        flips += compare(x, w, weights, ids, ref_weights, ref_ids)
        rows += tokens
    # Near-ties in FP32 are rare on continuous data.
    assert flips <= max(2, rows // 100)


def test_selection_order_matches_generic_route():
    """Ids come out in descending score order, like topk_sigmoid."""
    torch.manual_seed(3)
    w = gate(77)
    x = torch.randn(2, HIDDEN, device="cuda").half()
    weights, ids = fused(x, w)
    ref_weights, ref_ids = reference(x, w)
    if torch.equal(ids, ref_ids):
        torch.testing.assert_close(weights, ref_weights, atol=WEIGHT_ATOL, rtol=0)
    scores = torch.sigmoid(F.linear(x.float(), w)).gather(1, ids.long())
    assert torch.all(scores[:, :-1] >= scores[:, 1:])


def test_saturated_ties_take_lowest_ids():
    w = torch.zeros(NUM_EXPERTS, HIDDEN, device="cuda")
    hot = [200, 5, 77, 3, 150, 9, 31, 64]
    w[hot] = 0.01
    x = torch.ones(2, HIDDEN, device="cuda", dtype=torch.float16)
    weights, ids = fused(x, w)
    ref_weights, ref_ids = reference(x, w)
    assert ids.tolist() == [sorted(hot)[:TOP_K]] * 2
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_weights)


def test_zero_and_nan_rows_match_exactly():
    w = gate(5)
    x = torch.zeros(2, HIDDEN, device="cuda", dtype=torch.float16)
    x[1, 17] = float("nan")
    weights, ids = fused(x, w)
    ref_weights, ref_ids = reference(x, w)
    assert ids.tolist() == [list(range(TOP_K))] * 2
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_weights)
    assert torch.all(weights[1] == 0)


def test_strided_rows():
    torch.manual_seed(11)
    w = gate(9)
    base = torch.randn(2, HIDDEN + 64, device="cuda").half()
    x = base[:, 32 : 32 + HIDDEN]
    assert not x.is_contiguous()
    assert FlintFusedRouter.accepts(x)
    weights, ids = fused(x, w)
    ref_weights, ref_ids = reference(x.contiguous(), w)
    compare(x, w, weights, ids, ref_weights, ref_ids)


def test_repeated_launches_and_graph_replay():
    """The arrival counter re-arms: every launch and replay sees new input."""
    torch.manual_seed(21)
    w = gate(42)
    state = FlintFusedRouter(_flint_route_op(), w, router=None)
    for tokens in (1, 2):
        static_x = torch.randn(tokens, HIDDEN, device="cuda").half()
        for _ in range(3):
            state.route(static_x)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_weights, out_ids = state.route(static_x)
        for replay in range(40):
            static_x.copy_(
                torch.randn(tokens, HIDDEN, device="cuda") * (0.1, 1.0, 4.0)[replay % 3]
            )
            graph.replay()
            eager_weights, eager_ids = state.route(static_x)
            # Fixed-order reductions: replay is bitwise equal to eager.
            assert torch.equal(out_ids, eager_ids)
            assert torch.equal(out_weights, eager_weights)
            ref_weights, ref_ids = reference(static_x, w)
            compare(static_x, w, out_weights, out_ids, ref_weights, ref_ids)
        assert int(state.counter.item()) == 0
