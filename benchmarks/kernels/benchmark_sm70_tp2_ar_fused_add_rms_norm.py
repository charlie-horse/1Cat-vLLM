# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 TP2 all-reduce + fused_add_rms_norm microbenchmark (two GPUs).

Times one decode step's worth of layer boundaries (``--calls``, default 98)
captured in one CUDA graph, at the Flint TP2 shape [rows, 2048] FP16:

  ar          custom all-reduce only (cross_device_reduce_1stage)
  unfused     custom all-reduce, then the C fused_add_rms_norm kernel
  fused       tp2_all_reduce_fused_add_rms_norm
  unfused_sum2  shared + routed add, all-reduce, fused_add_rms_norm (MoE)
  fused_sum2  the fused collective with both partial outputs

Each replay is queued behind a GPU sleep so the CUDA events measure device
time, not launch gaps. Reported: median us per call over the replays.

Example (c4130):
  python benchmarks/kernels/benchmark_sm70_tp2_ar_fused_add_rms_norm.py
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 2048
EPS = 1e-6
VARIANTS = ("ar", "unfused", "fused", "unfused_sum2", "fused_sum2")


def _call(ca, variant, x, x_b, residual, weight) -> None:
    if variant == "ar":
        ca.custom_all_reduce(x)
    elif variant == "unfused":
        out = ca.custom_all_reduce(x)
        torch.ops._C.fused_add_rms_norm(out, residual, weight, EPS)
    elif variant == "unfused_sum2":
        out = ca.custom_all_reduce(x + x_b)
        torch.ops._C.fused_add_rms_norm(out, residual, weight, EPS)
    elif variant == "fused":
        ca.tp2_all_reduce_fused_add_rms_norm(x, None, residual, weight, EPS, 1.0)
    else:
        ca.tp2_all_reduce_fused_add_rms_norm(x, x_b, residual, weight, EPS, 1.0)


def _worker(rank: int, rendezvous: str, args: argparse.Namespace) -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=300),
    )
    ca = CustomAllreduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not ca.disabled
    results: dict[tuple[int, str], float] = {}
    try:
        weight = torch.ones(H, device="cuda", dtype=torch.float16)
        for rows in args.rows:
            x = torch.randn(rows, H, device="cuda", dtype=torch.float16)
            x_b = torch.randn_like(x)
            residual = torch.randn_like(x)

            for variant in args.variants:
                _call(ca, variant, x, x_b, residual, weight)
                torch.accelerator.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with ca.capture(), torch.cuda.graph(graph):
                    for _ in range(args.calls):
                        _call(ca, variant, x, x_b, residual, weight)
                for _ in range(args.warmup):
                    graph.replay()
                torch.accelerator.synchronize()
                dist.barrier()
                times = []
                for _ in range(args.replays):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda._sleep(200_000)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 1e3 / args.calls)
                results[(rows, variant)] = statistics.median(times)
                del graph
                torch.accelerator.synchronize()
                dist.barrier()
    finally:
        ca.close()
        dist.destroy_process_group()

    if rank == 0:
        print(f"us per call, median of {args.replays} replays of {args.calls} calls")
        print(f"{'rows':>4} " + " ".join(f"{v:>13}" for v in args.variants))
        for rows in args.rows:
            print(
                f"{rows:>4} "
                + " ".join(f"{results[(rows, v)]:13.2f}" for v in args.variants)
            )
        for rows in args.rows:
            if {"unfused", "fused"} <= set(args.variants):
                saved = results[(rows, "unfused")] - results[(rows, "fused")]
                print(f"rows={rows}: fused saves {saved:.2f} us per boundary")
            if {"unfused_sum2", "fused_sum2"} <= set(args.variants):
                saved = results[(rows, "unfused_sum2")] - results[(rows, "fused_sum2")]
                print(f"rows={rows}: fused_sum2 saves {saved:.2f} us per boundary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--calls", type=int, default=98)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    assert set(args.variants) <= set(VARIANTS), args.variants
    if torch.accelerator.device_count() < 2:
        raise SystemExit("needs two GPUs")
    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = (Path(tmp) / "rendezvous").as_uri()
        mp.spawn(_worker, args=(rendezvous, args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
