# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused q/k RMSNorm + RoPE + FP16 paged KV write against the unfused ops."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.models.flint.qk_norm_rope_kv import qk_norm_rope_kv
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="Requires SM70",
)

D = 128
EPS = 1e-6
K_FILL = 13.0
V_FILL = 17.0


def _reference_norm(x, w):
    # vllm rms_norm_kernel: FP32 variance, fp16(x * rsqrt(var + eps) * w).
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * inv * w.float()).half()


def _reference_rope(x, cos_sin, positions):
    # The compiled NeoX RoPE: FP32 arithmetic on FP16 inputs, one rounding.
    cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).half()


def _make(tokens, hq, hkv, blocks, page, seed):
    torch.manual_seed(seed)
    qkv = torch.randn(tokens, (hq + 2 * hkv) * D, device="cuda").half() * 3
    qw = (torch.randn(D, device="cuda") * 0.5 + 1).half()
    kw = (torch.randn(D, device="cuda") * 0.5 + 1).half()
    kv = torch.empty(blocks, 2, page, hkv, D, device="cuda", dtype=torch.float16)
    kv[:, 0].fill_(K_FILL)
    kv[:, 1].fill_(V_FILL)
    perm = torch.randperm(blocks * page, device="cuda")[:tokens]
    slots = perm.to(torch.int64)
    if tokens > 1:
        slots[-1] = -1  # CUDA-graph padding row: must not touch the cache.
    return qkv, qw, kw, kv, slots


def _unfused(qkv, qw, kw, positions, rope, kv, slots, hq, hkv):
    tokens = qkv.shape[0]
    q_raw, k_raw, v = qkv.split([hq * D, hkv * D, hkv * D], dim=-1)
    q = torch.empty(tokens, hq, D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv, D, device="cuda", dtype=torch.float16)
    ops.rms_norm(q, q_raw.view(tokens, hq, D), qw, EPS)
    ops.rms_norm(k, k_raw.view(tokens, hkv, D), kw, EPS)
    if rope is not None:
        q = _reference_rope(q, rope.cos_sin_cache, positions)
        k = _reference_rope(k, rope.cos_sin_cache, positions)
    kc, vc = kv.unbind(1)
    scale = torch.ones(1, device="cuda")
    triton_reshape_and_cache_flash(
        k, v.view(tokens, hkv, D), kc, vc, slots, "auto", scale, scale
    )
    return q.reshape(tokens, hq * D), k.reshape(tokens, hkv * D)


def _assert_fp16_close(actual, expected):
    # Same formula; only the FP32 reduction order and FMA contraction differ.
    # Allow one FP16 ulp, and require nearly every element to match exactly.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)
    exact = (actual == expected).float().mean().item()
    assert exact > 0.97, exact


@pytest.mark.parametrize("tokens", [1, 2, 7, 300])
@pytest.mark.parametrize("hq,hkv", [(16, 2), (32, 4)])
@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize("page", [16, 32])
@torch.inference_mode()
def test_fused_matches_unfused(tokens, hq, hkv, use_rope, page):
    blocks = 64
    qkv, qw, kw, kv_ref, slots = _make(tokens, hq, hkv, blocks, page, tokens)
    kv = kv_ref.clone()
    positions = torch.randint(0, 262144, (tokens,), device="cuda")
    rope = None
    if use_rope:
        with set_current_vllm_config(VllmConfig()):
            rope = RotaryEmbedding(D, D, 262144, 10000.0, True, torch.float16).cuda()
    exp_q, exp_k = _unfused(qkv, qw, kw, positions, rope, kv_ref, slots, hq, hkv)

    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    kc, vc = kv.unbind(1)
    qk_norm_rope_kv(
        qkv,
        q,
        k,
        qw,
        kw,
        positions,
        None if rope is None else rope.cos_sin_cache,
        EPS,
        kc,
        vc,
        slots,
    )
    _assert_fp16_close(q, exp_q)
    _assert_fp16_close(k, exp_k)
    # V is copied bit for bit; untouched slots keep their fill.
    assert torch.equal(kv[:, 1], kv_ref[:, 1])
    written = torch.zeros(blocks * page, dtype=torch.bool, device="cuda")
    written[slots[slots >= 0]] = True
    k_flat = kv[:, 0].reshape(blocks * page, hkv * D)
    assert torch.all(k_flat[~written] == K_FILL)
    _assert_fp16_close(k_flat, kv_ref[:, 0].reshape(blocks * page, hkv * D))

    # Layout is bit-identical: the stock writer fed the fused k/v produces
    # exactly the fused cache.
    layout = torch.empty_like(kv)
    layout[:, 0].fill_(K_FILL)
    layout[:, 1].fill_(V_FILL)
    lk, lv = layout.unbind(1)
    scale = torch.ones(1, device="cuda")
    v = qkv[:, (hq + hkv) * D :]
    triton_reshape_and_cache_flash(
        k.view(tokens, hkv, D),
        v.view(tokens, hkv, D),
        lk,
        lv,
        slots,
        "auto",
        scale,
        scale,
    )
    assert torch.equal(layout, kv)


@torch.inference_mode()
def test_fused_matches_eager_modules():
    """Looser check against the eager PyTorch modules (which round each op)."""
    tokens, hq, hkv = 2, 16, 2
    qkv, qw, kw, kv, slots = _make(tokens, hq, hkv, 8, 16, 7)
    positions = torch.tensor([5, 4097], device="cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(D, D, 262144, 10000.0, True, torch.float16).cuda()
    q_raw, k_raw, _ = qkv.split([hq * D, hkv * D, hkv * D], dim=-1)
    q_ref, k_ref = rope.forward_native(
        positions,
        _reference_norm(q_raw.reshape(tokens, hq, D), qw).reshape(tokens, -1),
        _reference_norm(k_raw.reshape(tokens, hkv, D), kw).reshape(tokens, -1),
    )
    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    qk_norm_rope_kv(qkv, q, k, qw, kw, positions, rope.cos_sin_cache, EPS)
    torch.testing.assert_close(q, q_ref, rtol=4e-3, atol=8e-3)
    torch.testing.assert_close(k, k_ref, rtol=4e-3, atol=8e-3)


@torch.inference_mode()
def test_graph_replay_reads_live_inputs():
    tokens, hq, hkv = 2, 16, 2
    qkv, qw, kw, kv, slots = _make(tokens, hq, hkv, 8, 16, 3)
    slots[-1] = 9
    positions = torch.tensor([0, 1], device="cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(D, D, 4096, 10000.0, True, torch.float16).cuda()
    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    kc, vc = kv.unbind(1)

    def run():
        qk_norm_rope_kv(
            qkv, q, k, qw, kw, positions, rope.cos_sin_cache, EPS, kc, vc, slots
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    qkv.zero_()
    slots.fill_(-1)
    kv[:, 0].fill_(K_FILL)
    kv[:, 1].fill_(V_FILL)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.count_nonzero(q) == torch.count_nonzero(k) == 0
    assert torch.all(kv[:, 0] == K_FILL) and torch.all(kv[:, 1] == V_FILL)
