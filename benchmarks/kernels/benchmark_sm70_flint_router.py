# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SM70 E256/K6 router on real gate weights: agreement and timing.

Loads ``model.layers.<i>.mlp.gate.weight`` (FP16 in the checkpoint, held in
FP32 as vLLM loads it) and ``post_attention_layernorm.weight`` for several MoE
layers from a safetensors checkpoint. For each layer it routes many hidden
vectors one or two at a time through

  generic: x.float() -> F.linear(FP32 gate) -> topk_sigmoid (renormalized),
           the route the MoE runner takes today;
  fused:   torch.ops.vllm_sm70_router.flint_route,

and reports id agreement (set and order), id flips split into near-ties
(exact FP64 logits within 64 FP32 rounding units of the dot product, or exact
sigmoid scores within 4 FP32 ulps) and hard flips, the max weight error on
common ids and the max logit difference. Hidden vectors are RMS-normalized
like the router input, times the layer's norm weight:

  gauss:    Gaussian residual;
  massive:  Gaussian residual with a few channels 30-300x larger;
  heavy:    Student-t (3 dof) residual;
  dump:     rows of --hidden FILE (a [N, 2048] tensor saved with torch.save),
            used as-is, when given.

With --bench it also times both routes per call in CUDA graphs that cycle
through every loaded layer's gate (so the gate is not L2 resident), median of
--iters replays. Pass: zero hard flips and max weight error <= 2e-6.

Example (c4130, one GPU):
  python benchmarks/kernels/benchmark_sm70_flint_router.py --bench
"""

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.fused_moe.router.sm70_flint_router import (
    HIDDEN,
    TOP_K,
    FlintFusedRouter,
    _flint_route_op,
)

FP32_EPS = 2.0**-24
LOGIT_TIE_UNITS = 64
SCORE_TIE = 4 * 2.0**-24


def load_layers(model: Path, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        names = {
            "gate": f"model.layers.{layer}.mlp.gate.weight",
            "norm": f"model.layers.{layer}.post_attention_layernorm.weight",
        }
        tensors = {}
        for key, name in names.items():
            with safe_open(model / weight_map[name], framework="pt") as f:
                tensors[key] = f.get_tensor(name)
        print(
            f"layer {layer}: gate {tuple(tensors['gate'].shape)} "
            f"{tensors['gate'].dtype}, norm {tensors['norm'].dtype}"
        )
        out[layer] = {
            # GateLinear keeps the router in FP32; loading copies (upcasts).
            "gate": tensors["gate"].to("cuda", torch.float32).contiguous(),
            "norm": tensors["norm"].to("cuda", torch.float32),
        }
    return out


def hidden_vectors(
    kind: str, count: int, norm: torch.Tensor, gen: torch.Generator, dump=None
) -> torch.Tensor:
    if kind == "dump":
        return dump[:count].to("cuda", torch.float16)
    shape = (count, HIDDEN)
    if kind == "heavy":
        normal = torch.randn(shape, device="cuda", generator=gen)
        chi2 = torch.randn((3, *shape), device="cuda", generator=gen).square().sum(0)
        residual = normal / (chi2 / 3).sqrt()
    else:
        residual = torch.randn(shape, device="cuda", generator=gen)
        if kind == "massive":
            channels = torch.randint(
                0, HIDDEN, (count, 4), device="cuda", generator=gen
            )
            gains = 30 + 270 * torch.rand((count, 4), device="cuda", generator=gen)
            residual.scatter_(1, channels, residual.gather(1, channels) * gains)
    rms = residual.square().mean(-1, keepdim=True).add(1e-6).rsqrt()
    return (residual * rms * norm).to(torch.float16)


def generic(x: torch.Tensor, gate: torch.Tensor):
    logits = F.linear(x.float(), gate)
    weights, ids, _ = fused_topk(
        x, logits, TOP_K, renormalize=True, scoring_func="sigmoid"
    )
    return weights, ids, logits


def agreement(
    x: torch.Tensor,
    gate: torch.Tensor,
    state: FlintFusedRouter,
    tokens: int,
) -> dict:
    rows = x.shape[0] - x.shape[0] % tokens
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device="cuda")
    ref_ids = torch.empty_like(ids)
    weights = torch.empty((rows, TOP_K), device="cuda")
    ref_weights = torch.empty_like(weights)
    max_logit_diff = torch.zeros((), device="cuda")
    for start in range(0, rows, tokens):
        chunk = x[start : start + tokens]
        w, i = state.route(chunk)
        rw, ri, logits = generic(chunk, gate)
        fused_logits = state.logits[: tokens * logits.shape[1]].view_as(logits)
        max_logit_diff = torch.maximum(
            max_logit_diff, (fused_logits - logits).abs().max()
        )
        ids[start : start + tokens] = i
        weights[start : start + tokens] = w
        ref_ids[start : start + tokens] = ri
        ref_weights[start : start + tokens] = rw
    x = x[:rows]

    exact = x.double() @ gate.double().T
    tol = LOGIT_TIE_UNITS * FP32_EPS * (x.double().abs() @ gate.double().abs().T)
    score = torch.sigmoid(exact)
    order_agree = (ids == ref_ids).all(1)
    sorted_ids = ids.sort(1).values
    sorted_ref = ref_ids.sort(1).values
    set_agree = (sorted_ids == sorted_ref).all(1)
    near = hard = 0
    worst_hard_gap = 0.0
    for row in (~set_agree).nonzero().flatten().tolist():
        got, want = set(ids[row].tolist()), set(ref_ids[row].tolist())
        for e_in, e_out in zip(sorted(got - want), sorted(want - got)):
            logit_gap = float((exact[row, e_in] - exact[row, e_out]).abs())
            score_gap = float((score[row, e_in] - score[row, e_out]).abs())
            if (
                logit_gap <= float(tol[row, e_in] + tol[row, e_out])
                or score_gap <= SCORE_TIE
            ):
                near += 1
            else:
                hard += 1
                worst_hard_gap = max(worst_hard_gap, logit_gap)
    # Weight error on the experts both routes chose.
    match = ids[:, :, None] == ref_ids[:, None, :]
    diff = (weights[:, :, None] - ref_weights[:, None, :]).abs()
    weight_err = float(diff.masked_fill(~match, 0).max())
    return {
        "rows": rows,
        "order_agree": float(order_agree.float().mean()),
        "set_agree": float(set_agree.float().mean()),
        "near_tie_flips": near,
        "hard_flips": hard,
        "worst_hard_logit_gap": worst_hard_gap,
        "max_weight_err": weight_err,
        "max_logit_diff": float(max_logit_diff),
    }


def time_calls(calls: list[Callable[[], object]], iters: int) -> float:
    """Median device microseconds per call, CUDA graph replay."""
    for _ in range(3):
        for call in calls:
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
    torch.cuda._sleep(100_000_000)
    events[0].record()
    for i in range(iters):
        graph.replay()
        events[i + 1].record()
    torch.cuda.synchronize()
    return statistics.median(
        events[i].elapsed_time(events[i + 1]) * 1000.0 / len(calls)
        for i in range(iters)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.path.expanduser("~/flint/models/flint-nvfp4-modelopt")),
    )
    parser.add_argument("--layers", default="2,3,10,20,30,40,46,48")
    parser.add_argument("--vectors", type=int, default=2000)
    parser.add_argument("--hidden", type=Path, help="optional [N, 2048] tensor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    capability = torch.cuda.get_device_capability()
    assert capability in ((7, 0), (7, 2)), f"SM70 required, got {capability}"
    op = _flint_route_op()
    assert op is not None, "vllm._sm70_router_C lacks flint_route; rebuild it"

    layers = load_layers(args.model, [int(v) for v in args.layers.split(",")])
    dump = torch.load(args.hidden, map_location="cpu") if args.hidden else None
    kinds = ["gauss", "massive", "heavy"] + (["dump"] if dump is not None else [])
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    print(
        f"{'layer':>5} {'kind':>8} {'M':>2} {'rows':>6} {'order':>7} {'set':>7} "
        f"{'near':>5} {'hard':>5} {'max dw':>9} {'max dlogit':>10}"
    )
    for layer, tensors in layers.items():
        state = FlintFusedRouter(op, tensors["gate"], router=None)
        for kind in kinds:
            x = hidden_vectors(kind, args.vectors, tensors["norm"], gen, dump)
            for tokens in (1, 2):
                r = agreement(x, tensors["gate"], state, tokens)
                r.update(layer=layer, kind=kind, tokens=tokens)
                records.append(r)
                print(
                    f"{layer:>5} {kind:>8} {tokens:>2} {r['rows']:>6} "
                    f"{r['order_agree']:>7.4f} {r['set_agree']:>7.4f} "
                    f"{r['near_tie_flips']:>5} {r['hard_flips']:>5} "
                    f"{r['max_weight_err']:>9.2e} {r['max_logit_diff']:>10.2e}"
                )
    hard = sum(r["hard_flips"] for r in records)
    weight_err = max(r["max_weight_err"] for r in records)
    verdict = "PASS" if hard == 0 and weight_err <= 2e-6 else "FAIL"
    print(f"\nhard flips {hard}, max weight error {weight_err:.2e}: {verdict}")

    timings = {}
    if args.bench:
        states = [FlintFusedRouter(op, t["gate"], router=None) for t in layers.values()]
        gates = [t["gate"] for t in layers.values()]
        for tokens in (1, 2):
            x = torch.randn(tokens, HIDDEN, device="cuda").half()
            fused_us = time_calls(
                [lambda s=s, x=x: s.route(x) for s in states], args.iters
            )
            generic_us = time_calls(
                [lambda g=g, x=x: generic(x, g) for g in gates], args.iters
            )
            timings[tokens] = {"generic_us": generic_us, "fused_us": fused_us}
            print(
                f"M={tokens}: generic {generic_us:.2f} us, fused {fused_us:.2f} us "
                f"per layer ({len(gates)} gates cycled; x47 layers = "
                f"{(generic_us - fused_us) * 47 / 1000:.3f} ms/step saved)"
            )
    if args.json:
        args.json.write_text(
            json.dumps({"records": records, "timings": timings}, indent=2)
        )


if __name__ == "__main__":
    main()
