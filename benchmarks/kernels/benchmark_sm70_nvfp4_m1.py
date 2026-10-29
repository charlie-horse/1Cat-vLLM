# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token (M=1/2) NVFP4 GEMM microbenchmark for Flint TP2 on SM70.

Times the per-rank Flint projection shapes and the routed-expert stage of one
MoE layer with random ModelOpt NVFP4 weights (packed e2m1 ``[N, K/2]``, e4m3
block scales ``[N, K/16]``, fp32 global scale).

Backends:
  turbomind    What Flint runs today: ``sm70_turbomind.prepare_nvfp4_linear``
               plus ``apply_prepared_linear`` (ModelOpt W4A16 on SM70). For
               MoE, ``ModelOptNvFp4SM70MoEMethod.apply`` on fixed top-k ids.
  qpn2_shared  ``nvfp4_qpn2_tm_dispatch_sm70_out`` over the same TurboMind
               codes with compact E4M3 scales, called through
               ``Qpn2NvFp4LinearKernel.apply_qpn2`` (dense only).
  new          Optional. Dense: ``torch.ops._C.<--new-op>`` if registered.
               MoE: re-prepares the layer with ``<--moe-new-env>=1`` if this
               tree reads that variable. Skipped with a message otherwise.

Every call is captured in a CUDA graph. Each graph replays several weight
copies (dense) or routings (MoE) so consecutive calls do not reuse L2 (6 MB
on V100); the reported time is per call. Replays are queued behind a GPU
sleep so the per-replay CUDA events measure device time, not launch gaps.
GB/s = touched weight bytes / time, with 0.5625 bytes per weight (4-bit code
plus one e4m3 scale per 16).

Example (c4130):
  python benchmarks/kernels/benchmark_sm70_nvfp4_m1.py \\
      --backends turbomind,qpn2_shared,new --json out.json --markdown out.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch.nn import Parameter

BYTES_PER_WEIGHT = 0.5625
NVFP4_GROUP = 16

# Per-rank Flint TP2 projections: name -> (K, N).
DENSE_SHAPES: dict[str, tuple[int, int]] = {
    "qkv": (2048, 2560),
    "o": (2048, 2048),
    "shared_gate_up": (2048, 1024),
    "shared_down": (512, 2048),
    "dense_gate_up": (2048, 6144),
    "dense_down": (3072, 2048),
}

# Flint MoE: hidden, global intermediate, experts, top-k (TP2 local I=256).
MOE_HIDDEN, MOE_INTERMEDIATE, MOE_EXPERTS, MOE_TOP_K = 2048, 512, 256, 6

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _random_nvfp4(
    n: int, k: int, gen: torch.Generator, leading: tuple[int, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """ModelOpt-like codes ``[..., N, K/2]`` and e4m3 block scales."""
    codes = torch.randint(
        0, 256, (*leading, n, k // 2), generator=gen, device="cuda", dtype=torch.uint8
    )
    # ModelOpt block scales are block_amax / (6 * global): spread up to 448.
    scales = torch.rand((*leading, n, k // NVFP4_GROUP), generator=gen, device="cuda")
    scales = (scales * 384.0 + 64.0).to(torch.float8_e4m3fn)
    return codes, scales


def _dequantize(codes, block_scales, global_scale) -> torch.Tensor:
    """ModelOpt NVFP4 ``[N, K/2]`` (low nibble first) -> fp32 ``[N, K]``."""
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=codes.device)
    values = torch.stack(
        (lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1
    ).flatten(-2)
    scales = block_scales.float().repeat_interleave(NVFP4_GROUP, dim=-1)
    return values * scales * global_scale


class _NvFp4Linear(torch.nn.Module):
    """The attributes a ModelOpt W4A16 linear holds after loading."""

    def __init__(self, k, n, codes, scales, global_scale):
        super().__init__()
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.prefix = ""
        self.weight = Parameter(codes, requires_grad=False)
        self.weight_scale = Parameter(scales, requires_grad=False)
        self.weight_global_scale = Parameter(
            torch.tensor(global_scale, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )


def _prepare_dense_layer(k: int, n: int, gen: torch.Generator) -> _NvFp4Linear:
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.kernels.linear.qpn.nvfp4 import _qpn2_config
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    codes, scales = _random_nvfp4(n, k, gen)
    layer = _NvFp4Linear(k, n, codes, scales, 0.05 / (6.0 * 448.0))
    # Same preparation as modelopt._try_prepare_sm70_modelopt_nvfp4.
    sm70_tm.prepare_nvfp4_linear(layer)
    # Shared-weight QPN2 attributes, as Qpn2NvFp4LinearKernel attaches them.
    layer.sm70_nvfp4_qpn2_scales = None
    if hasattr(torch.ops._C, "nvfp4_qpn2_prepare_scales_sm70"):
        layer.sm70_nvfp4_qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_scales_sm70(
            layer.weight_scale.data
        )
    layer.sm70_nvfp4_qpn2_shared_weight = True
    layer.sm70_nvfp4_qpn2_global_scale = float(layer.weight_global_scale.item())
    layer.sm70_nvfp4_qpn2_output_size = (n + 31) // 32 * 32
    split_k, nacc = _qpn2_config(k, layer.sm70_nvfp4_qpn2_output_size, False)
    layer.sm70_nvfp4_qpn2_split_k = split_k
    layer.sm70_nvfp4_qpn2_nacc = nacc
    layer.sm70_nvfp4_qpn2_gated_silu = False
    layer.sm70_nvfp4_qpn2_prefill_enabled = True
    layer.sm70_nvfp4_qpn2_prefill_min_m = 1024
    return layer


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_calls(
    calls: list[Callable[[], object]], iters: int, warmup: int
) -> tuple[float, str]:
    """Median device microseconds per call; returns (us, mode)."""
    for _ in range(3):
        for call in calls:
            call()
    torch.cuda.synchronize()
    mode = "graph"
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for call in calls:
                call()
        replay: Callable[[], object] = graph.replay
    except Exception as exc:  # noqa: BLE001 - fall back and report it
        print(f"  [warn] CUDA graph capture failed ({exc!r}); timing eagerly")
        torch.cuda.synchronize()
        mode = "eager"

        def replay() -> None:
            for call in calls:
                call()

    for _ in range(warmup):
        replay()
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
    # Keep the GPU busy while the host queues every replay and event, so the
    # event gaps hold device time only.
    torch.cuda._sleep(100_000_000)
    events[0].record()
    for i in range(iters):
        replay()
        events[i + 1].record()
    torch.cuda.synchronize()
    per_call = [
        events[i].elapsed_time(events[i + 1]) * 1000.0 / len(calls)
        for i in range(iters)
    ]
    return statistics.median(per_call), mode


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Max abs error and max abs error relative to max |expected|."""
    delta = (actual.float() - expected.float()).abs().max().item()
    scale = expected.float().abs().max().item()
    return delta, delta / scale if scale else math.inf


# ---------------------------------------------------------------------------
# Dense projections
# ---------------------------------------------------------------------------


def _dense_runner(backend: str, layer, x: torch.Tensor, args):
    """Returns a call computing ``x @ W^T`` for ``layer``, or a skip reason."""
    from vllm.model_executor.kernels.linear.qpn.nvfp4 import Qpn2NvFp4LinearKernel
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    if backend == "turbomind":
        return lambda: sm70_tm.apply_prepared_linear(layer, x, None)
    if backend == "qpn2_shared":
        if layer.sm70_nvfp4_qpn2_scales is None or not hasattr(
            torch.ops._C, "nvfp4_qpn2_tm_dispatch_sm70_out"
        ):
            return "nvfp4_qpn2_tm_dispatch_sm70_out not built"
        return lambda: Qpn2NvFp4LinearKernel.apply_qpn2(
            layer, x, None, gated_silu=False
        )
    if backend == "new":
        op = getattr(torch.ops._C, args.new_op, None)
        if op is None:
            return f"torch.ops._C.{args.new_op} not registered"
        state = getattr(layer, sm70_tm.STATE_ATTR)
        n = state.padded_output_size or state.output_size
        out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
        if args.new_signature == "tm":
            extra = (state.scales, state.group_size, state.k_ld, state.q_ld)
        else:
            extra = (
                layer.sm70_nvfp4_qpn2_scales,
                layer.sm70_nvfp4_qpn2_global_scale,
                state.k_ld,
                state.q_ld,
            )

        def call() -> torch.Tensor:
            op(out, x, state.weight, *extra)
            return out[:, : state.output_size]

        return call
    return f"unknown backend {backend}"


def bench_dense(args, records: list[dict]) -> None:
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for name in args.shapes:
        k, n = DENSE_SHAPES[name]
        weight_bytes = k * n * BYTES_PER_WEIGHT
        copies = max(
            1, min(args.max_copies, math.ceil(args.rotate_mb * 2**20 / weight_bytes))
        )
        layers = [_prepare_dense_layer(k, n, gen) for _ in range(copies)]
        ref_w = _dequantize(
            layers[0].weight.data,
            layers[0].weight_scale.data,
            layers[0].sm70_nvfp4_qpn2_global_scale,
        )
        for m in args.m:
            x = torch.randn((m, k), generator=gen, device="cuda").half()
            reference = x.float() @ ref_w.t()
            baseline = None
            for backend in args.backends:
                record = dict(
                    kind="dense",
                    name=name,
                    m=m,
                    k=k,
                    n=n,
                    backend=backend,
                    weight_bytes=weight_bytes,
                    copies=copies,
                )
                if backend == "qpn2_shared":
                    record["qpn2_split_k"] = layers[0].sm70_nvfp4_qpn2_split_k
                    record["qpn2_nacc"] = layers[0].sm70_nvfp4_qpn2_nacc
                runners = [_dense_runner(backend, layer, x, args) for layer in layers]
                if isinstance(runners[0], str):
                    print(f"  [skip] {name} M={m} {backend}: {runners[0]}")
                    record["skipped"] = runners[0]
                    records.append(record)
                    continue
                try:
                    out = runners[0]().clone()
                    torch.cuda.synchronize()
                    us, mode = _time_calls(runners, args.iters, args.warmup)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"  [skip] {name} M={m} {backend}: {exc!r}")
                    record["skipped"] = repr(exc)
                    records.append(record)
                    continue
                if baseline is None and backend == "turbomind":
                    baseline = out
                record.update(us=us, gbps=weight_bytes / us / 1e3, mode=mode)
                record["ref_max_abs"], record["ref_rel"] = _errors(out, reference)
                if baseline is not None and backend != "turbomind":
                    record["tm_max_abs"], record["tm_rel"] = _errors(out, baseline)
                records.append(record)
                print(
                    f"  {name:15s} M={m} {backend:12s} {us:8.2f} us "
                    f"{record['gbps']:7.1f} GB/s ({mode}, copies={copies})"
                )
        del layers, ref_w
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Routed experts (one MoE layer)
# ---------------------------------------------------------------------------


def _moe_config(tp_size: int):
    import dataclasses

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        get_routing_method_type,
    )

    parallel = dataclasses.replace(
        FusedMoEParallelConfig.make_no_parallel(), tp_size=tp_size
    )
    return FusedMoEConfig(
        num_experts=MOE_EXPERTS,
        experts_per_token=MOE_TOP_K,
        hidden_dim=MOE_HIDDEN,
        intermediate_size_per_partition=MOE_INTERMEDIATE // tp_size,
        num_local_experts=MOE_EXPERTS,
        num_logical_experts=MOE_EXPERTS,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=get_routing_method_type(
            scoring_func="sigmoid",
            top_k=MOE_TOP_K,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=False,
        ),
        moe_parallel_config=parallel,
        in_dtype=torch.float16,
    )


def _moe_raw_weights(intermediate: int, gen: torch.Generator) -> dict:
    w13, w13_scale = _random_nvfp4(2 * intermediate, MOE_HIDDEN, gen, (MOE_EXPERTS,))
    w2, w2_scale = _random_nvfp4(MOE_HIDDEN, intermediate, gen, (MOE_EXPERTS,))
    # Equal gate/up global scales, as ModelOpt exports fused W13.
    w13_global = torch.full((MOE_EXPERTS, 2), 0.05 / 2688.0, device="cuda")
    w2_global = torch.full((MOE_EXPERTS,), 0.05 / 2688.0, device="cuda")
    return dict(
        w13_weight=w13,
        w13_weight_scale=w13_scale,
        w13_weight_scale_2=w13_global,
        w2_weight=w2,
        w2_weight_scale=w2_scale,
        w2_weight_scale_2=w2_global,
    )


def _moe_reference(x, weights, ids, raw, intermediate) -> torch.Tensor:
    import torch.nn.functional as F

    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        g = raw["w13_weight_scale_2"][expert].repeat_interleave(intermediate)
        w13 = _dequantize(
            raw["w13_weight"][expert], raw["w13_weight_scale"][expert], g[:, None]
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


def _env_read_by_tree(name: str) -> bool:
    import vllm

    root = Path(vllm.__file__).parent
    for path in root.rglob("*.py"):
        try:
            if name in path.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _build_moe(raw: dict, tp_size: int):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    config = _moe_config(tp_size)
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    with set_current_vllm_config(vllm_config):
        quant = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
        )
        method = moe.ModelOptNvFp4SM70MoEMethod(quant, config)
        layer = NS(
            moe_config=config,
            local_num_experts=MOE_EXPERTS,
            global_num_experts=MOE_EXPERTS,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            swiglu_limit=None,
            w13_input_scale=None,
            w2_input_scale=None,
            **{name: t.clone() for name, t in raw.items()},
        )
        method.process_weights_after_loading(layer)
    return method, layer, vllm_config


def bench_moe(args, records: list[dict]) -> None:
    from vllm.config import set_current_vllm_config

    tp_size = 2
    intermediate = MOE_INTERMEDIATE // tp_size
    gen = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    raw = _moe_raw_weights(intermediate, gen)
    per_expert = (
        MOE_HIDDEN * 2 * intermediate + intermediate * MOE_HIDDEN
    ) * BYTES_PER_WEIGHT

    routings: dict[int, list] = {}
    for m in args.m:
        routings[m] = []
        for _ in range(args.moe_routings):
            x = torch.randn((m, MOE_HIDDEN), generator=gen, device="cuda").half()
            logits = torch.randn((m, MOE_EXPERTS), generator=gen, device="cuda")
            scores, ids = torch.sigmoid(logits).topk(MOE_TOP_K, dim=-1)
            weights = (scores / scores.sum(dim=-1, keepdim=True)).contiguous()
            routings[m].append((x, weights, ids.to(torch.int32).contiguous()))

    builders = []
    for backend in args.backends:
        if backend == "turbomind":
            builders.append((backend, None))
        elif backend == "new":
            env = args.moe_new_env
            if not _env_read_by_tree(env):
                print(f"  [skip] moe new: {env} is not read by this vllm tree")
                records.append(
                    dict(kind="moe", backend="new", skipped=f"{env} unknown")
                )
                continue
            builders.append((backend, env))
        else:
            print(f"  [skip] moe {backend}: no routed-expert path for this backend")

    baselines: dict[int, torch.Tensor] = {}
    for backend, env in builders:
        saved = os.environ.get(env) if env else None
        if env:
            os.environ[env] = "1"
        try:
            method, layer, vllm_config = _build_moe(raw, tp_size)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [skip] moe {backend}: {exc!r}")
            records.append(dict(kind="moe", backend=backend, skipped=repr(exc)))
            continue
        finally:
            if env:
                if saved is None:
                    os.environ.pop(env, None)
                else:
                    os.environ[env] = saved
        with set_current_vllm_config(vllm_config):
            for m in args.m:
                cases = routings[m]
                touched = statistics.mean(ids.unique().numel() for _, _, ids in cases)
                weight_bytes = touched * per_expert
                calls = [
                    (
                        lambda c=c, apply=method.apply, lay=layer: apply(
                            lay, c[0], c[1], c[2], None, None
                        )
                    )
                    for c in cases
                ]
                record = dict(
                    kind="moe",
                    name="moe_routed",
                    m=m,
                    k=MOE_HIDDEN,
                    n=2 * intermediate,
                    backend=backend,
                    weight_bytes=weight_bytes,
                    experts_touched=touched,
                    copies=len(cases),
                )
                try:
                    out = calls[0]().clone()
                    torch.cuda.synchronize()
                    us, mode = _time_calls(calls, args.iters, args.warmup)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"  [skip] moe M={m} {backend}: {exc!r}")
                    record["skipped"] = repr(exc)
                    records.append(record)
                    continue
                x, weights, ids = cases[0]
                reference = _moe_reference(x, weights, ids, raw, intermediate)
                record.update(us=us, gbps=weight_bytes / us / 1e3, mode=mode)
                record["ref_max_abs"], record["ref_rel"] = _errors(out, reference)
                if backend == "turbomind":
                    baselines[m] = out
                elif m in baselines:
                    record["tm_max_abs"], record["tm_rel"] = _errors(out, baselines[m])
                records.append(record)
                print(
                    f"  moe_routed      M={m} {backend:12s} {us:8.2f} us "
                    f"{record['gbps']:7.1f} GB/s ({mode}, "
                    f"{touched:.1f} experts touched)"
                )
        del method, layer
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def markdown(records: list[dict], header: dict) -> str:
    lines = [
        "# SM70 NVFP4 M=1/2 microbenchmark (Flint TP2 per-rank shapes)",
        "",
    ]
    lines += [f"- {key}: `{value}`" for key, value in header.items()]
    lines += [
        "",
        (
            "| shape | M | K | N | weight MB | backend | us | GB/s | "
            "err vs fp32 ref (rel) | err vs turbomind (abs / rel) | note |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if "name" not in r:
            continue
        if r.get("skipped"):
            lines.append(
                f"| {r['name']} | {r['m']} | {r['k']} | {r['n']} | "
                f"{r['weight_bytes'] / 2**20:.2f} | {r['backend']} | - | - | - | - | "
                f"skipped: {r['skipped'][:60]} |"
            )
            continue
        note = [r.get("mode", "")]
        if "qpn2_split_k" in r:
            note.append(f"split_k={r['qpn2_split_k']} nacc={r['qpn2_nacc']}")
        if "experts_touched" in r:
            note.append(f"{r['experts_touched']:.1f} experts")
        vs_tm = (
            f"{r['tm_max_abs']:.2e} / {r['tm_rel']:.2e}" if "tm_max_abs" in r else "-"
        )
        lines.append(
            f"| {r['name']} | {r['m']} | {r['k']} | {r['n']} | "
            f"{r['weight_bytes'] / 2**20:.2f} | {r['backend']} | {r['us']:.2f} | "
            f"{r['gbps']:.1f} | {_fmt(r.get('ref_rel'), '.2e')} | {vs_tm} | "
            f"{'; '.join(n for n in note if n)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backends", default="turbomind,qpn2_shared")
    parser.add_argument("--m", default="1,2", help="comma-separated token counts")
    parser.add_argument("--shapes", default=",".join(DENSE_SHAPES))
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument(
        "--rotate-mb",
        type=float,
        default=24.0,
        help="dense weight copies replayed per graph so the working set "
        "exceeds L2 (MiB)",
    )
    parser.add_argument("--max-copies", type=int, default=16)
    parser.add_argument(
        "--moe-routings",
        type=int,
        default=8,
        help="distinct top-k routings replayed per MoE graph",
    )
    parser.add_argument("--new-op", default="nvfp4_tm_gemv_sm70_out")
    parser.add_argument(
        "--new-signature",
        choices=("tm", "compact"),
        default="tm",
        help="tm: (out, x, tm_weight, tm_scales, group, k_ld, q_ld); "
        "compact: (out, x, tm_weight, e4m3_scales, global, k_ld, q_ld)",
    )
    parser.add_argument("--moe-new-env", default="VLLM_SM70_FLINT_MOE_QPN_DECODE")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    args.backends = [b for b in args.backends.split(",") if b]
    args.m = [int(v) for v in args.m.split(",") if v]
    args.shapes = [s for s in args.shapes.split(",") if s]
    unknown = set(args.shapes) - set(DENSE_SHAPES)
    if unknown:
        parser.error(f"unknown shapes {sorted(unknown)}")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (7, 0),
        (7, 2),
    ):
        sys.exit("requires an SM70 CUDA device")
    import vllm
    import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

    torch.manual_seed(args.seed)
    records: list[dict] = []
    if not args.no_dense:
        bench_dense(args, records)
    if not args.no_moe:
        bench_moe(args, records)

    header = dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        vllm_tree=str(Path(vllm.__file__).parent.parent),
        argv=" ".join(sys.argv[1:]),
        timing=f"CUDA graph, median of {args.iters} replays, per-call device time",
    )
    report = markdown(records, header)
    print()
    print(report)
    if args.markdown:
        args.markdown.write_text(report)
    if args.json:
        args.json.write_text(json.dumps(dict(header=header, records=records), indent=2))


if __name__ == "__main__":
    main()
