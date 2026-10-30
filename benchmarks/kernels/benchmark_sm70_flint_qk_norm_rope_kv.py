# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-L2 CUDA-graph timing of Flint attention input preparation on SM70.

Compares, per layer at TP2 shapes (16 q heads, 2 KV heads, D128):
  unfused: rms_norm(q) + rms_norm(k) + compiled NeoX RoPE + KV-cache write
  fused:   one sm70_flint qk_norm_rope_kv kernel
for sliding (RoPE) and full (no RoPE) layers. Each sample replays a graph of
``--layers`` copies, so launch gaps are included, and reports us per layer.

  python benchmarks/kernels/benchmark_sm70_flint_qk_norm_rope_kv.py
"""

import argparse
import json
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.models.flint.qk_norm_rope_kv import qk_norm_rope_kv
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

D = 128
EPS = 1e-6


def _rope_native(positions, q, k, cos_sin):
    cos, sin = cos_sin.index_select(0, positions).chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)

    def rot(x):
        x = x.view(x.shape[0], -1, D)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)

    return rot(q).flatten(1), rot(k).flatten(1)


def _routes(tokens, hq, hkv, qw, kw, cos_sin, kc, vc, scale, rope_compiled):
    blocks, page = kc.shape[:2]
    qkv = torch.randn(tokens, (hq + 2 * hkv) * D, device="cuda").half()
    positions = torch.randint(0, 4096, (tokens,), device="cuda")
    slots = torch.randint(0, blocks * page, (tokens,), device="cuda")
    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    qn = torch.empty(tokens, hq, D, device="cuda", dtype=torch.float16)
    kn = torch.empty(tokens, hkv, D, device="cuda", dtype=torch.float16)
    q_raw, k_raw, v = qkv.split([hq * D, hkv * D, hkv * D], dim=-1)

    def unfused(rope):
        ops.rms_norm(qn, q_raw.view(tokens, hq, D), qw, EPS)
        ops.rms_norm(kn, k_raw.view(tokens, hkv, D), kw, EPS)
        qq, kk = qn.flatten(1), kn.flatten(1)
        if rope:
            qq, kk = rope_compiled(positions, qq, kk, cos_sin)
        triton_reshape_and_cache_flash(
            kk.view(tokens, hkv, D),
            v.view(tokens, hkv, D),
            kc,
            vc,
            slots,
            "auto",
            scale,
            scale,
        )

    def fused(rope):
        qk_norm_rope_kv(
            qkv,
            q,
            k,
            qw,
            kw,
            positions,
            cos_sin if rope else None,
            EPS,
            kc,
            vc,
            slots,
        )

    return (("unfused", unfused), ("fused", fused))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    hq, hkv = args.q_heads, args.kv_heads
    rope_compiled = torch.compile(_rope_native, dynamic=False)
    max_pos, page, blocks = 262144, 16, 4096
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    freqs = torch.outer(torch.arange(max_pos).float(), inv)
    cos_sin = torch.cat((freqs.cos(), freqs.sin()), -1).half().cuda()
    qw = torch.rand(D, device="cuda").half() + 0.5
    kw = torch.rand(D, device="cuda").half() + 0.5
    kv = torch.zeros(blocks, 2, page, hkv, D, device="cuda", dtype=torch.float16)
    kc, vc = kv.unbind(1)
    scale = torch.ones(1, device="cuda")
    eviction = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for tokens in args.tokens:
        routes = _routes(tokens, hq, hkv, qw, kw, cos_sin, kc, vc, scale, rope_compiled)
        for route, fn in routes:
            for rope in (True, False):
                for _ in range(3):
                    fn(rope)
                torch.accelerator.synchronize()
                # Capture only the work; timing events are recorded on the
                # stream around each replay (events recorded inside a capture
                # are not usable for elapsed_time).
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.layers):
                        fn(rope)
                torch.accelerator.synchronize()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                samples = []
                for i in range(args.replays + 5):
                    eviction.fill_(1)  # Cold L2, outside the timed span.
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    if i >= 5:
                        samples.append(begin.elapsed_time(end) * 1000 / args.layers)
                graph.reset()
                records.append(
                    {
                        "tokens": tokens,
                        "route": route,
                        "rope": rope,
                        "median_us_per_layer": round(statistics.median(samples), 2),
                    }
                )
                print(json.dumps(records[-1]))


if __name__ == "__main__":
    main()
