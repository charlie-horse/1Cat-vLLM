# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank eager GPU time of the decode sampler on one SM70 GPU.

reference: the regular path after the full-logits gather (FP32 cast,
temperature, apply_top_k_top_p, softmax, exponential race) on [rows, vocab].
candidates: this rank's local top-32 on its [rows, vocab / tp] shard plus the
candidate sampler on the gathered rows (the gather itself is not timed; it
moves rows * tp * 256 bytes instead of rows * vocab * 2).
lm_head: the rank's FP16 projection, for the bandwidth floor.

    .venv/bin/python benchmarks/kernels/benchmark_sm70_topk_candidates.py
"""

import argparse
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm.v1.sample.ops import sm70_topk_candidates as candidates
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample


def _time(fn, iters):
    """Return (GPU us, host us) per call.

    GPU time is measured with the stream held behind a sleep kernel while
    the host enqueues all iterations, so host overhead does not count.
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    host_start = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - host_start) * 1e6 / iters
    torch.cuda.synchronize()
    # Sleep about twice the host enqueue time (V100 at ~1.4 GHz).
    torch.cuda._sleep(int(2 * host_us * iters * 1400))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters, host_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=150144)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0)
    assert candidates.ops_available(), "build _sm70_sampler_C"
    shard = args.vocab // args.tp
    weight = torch.randn(shard, args.hidden, device="cuda").half() * 0.02
    for rows in (1, 2):
        _bench(rows, args, shard, weight)


def _bench(rows, args, shard, weight):
    meta = SimpleNamespace(
        all_random=True,
        all_greedy=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=torch.ones(rows, device="cuda"),
        temperature_cpu=(1.0,) * rows,
        top_k=torch.full((rows,), 20, dtype=torch.int32, device="cuda"),
        top_k_cpu=(20,) * rows,
        top_p=torch.full((rows,), 0.95, device="cuda"),
        top_p_cpu=(0.95,) * rows,
        generators={},
    )
    logits = (torch.randn(rows, args.vocab, device="cuda") * 2.5).half()
    local = logits[:, :shard].contiguous()
    hidden = torch.randn(rows, args.hidden, device="cuda").half()

    def reference():
        x = logits.float().div_(meta.temperature.unsqueeze(1))
        x = apply_top_k_top_p(x, meta.top_k, meta.top_p)
        random_sample(x.softmax(dim=-1, dtype=torch.float32), {})

    wire = torch.cat(
        [candidates.local_topk_candidates(local, vocab_start=0, num_valid=shard)]
        * args.tp,
        dim=0,
    )

    def local_select():
        candidates.local_topk_candidates(local, vocab_start=0, num_valid=shard)

    def sample():
        candidates.sample_topk_candidates(
            wire, meta, tp_size=args.tp, vocab_size=args.vocab
        )

    def lm_head():
        F.linear(hidden, weight)

    ref_gpu, ref_host = _time(reference, args.iters)
    sel_gpu, sel_host = _time(local_select, args.iters)
    smp_gpu, smp_host = _time(sample, args.iters)
    gemm_gpu, _ = _time(lm_head, args.iters)
    gbps = weight.numel() * 2 / (gemm_gpu * 1e-6) / 1e9
    print(
        f"rows={rows} GPU us: reference={ref_gpu:.1f} "
        f"candidates={sel_gpu + smp_gpu:.1f} (local top-32 {sel_gpu:.1f} + "
        f"sample {smp_gpu:.1f}) lm_head={gemm_gpu:.1f} ({gbps:.0f} GB/s) | "
        f"host us/call: reference={ref_host:.1f} local top-32={sel_host:.1f} "
        f"sample={smp_host:.1f}"
    )


if __name__ == "__main__":
    main()
