# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Learned attention sinks on the Flash-V100 D128 paged prefill kernel.

A sink is one fp32 logit per query head (natural-log units, gpt-oss
semantics): it adds ``exp(sink)`` to the softmax denominator and has no
value row. The GPU tests compare ``flash_attn_prefill_paged(..., sinks=...)``
with a float32 torch reference for the shapes the backend sends there:
causal full attention and a 513-token sliding window, GQA 8:1 at 32q/4kv and
16q/2kv, FP16 paged KV with shuffled block tables, fresh prefill, chunked
prefill over an existing prefix, and both a single KV tile and many KV tiles.
D128 paged prefill has no split-KV kernel (the split-KV entry is D256 only),
so the multi-tile cases cover the online-softmax merge across KV tiles.

The CPU tests check the threading in ``prefill.py`` with a torch stand-in
for the kernel and the interface contract with a stub extension.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import prefill, workspace

SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
requires_sm70 = pytest.mark.skipif(
    not SM70, reason="FlashAttention-V100 is SM70/V100 only"
)
SWA = (512, 0)  # sliding_window=513: the query token and the 512 before it
FULL = (-1, -1)
ATOL = 2e-2
RTOL = 2e-2


def _reference_attention(q, k, v, *, scale, causal, window, sinks):
    """fp32 attention of ``q [M, H, D]`` over all ``k, v [N, Hkv, D]``."""
    m, num_heads, _ = q.shape
    n, num_kv_heads, _ = k.shape
    group = num_heads // num_kv_heads
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k) * scale
    # The queries are the last M positions of the sequence.
    q_pos = torch.arange(m, device=q.device)[:, None] + max(n - m, 0)
    kv_pos = torch.arange(n, device=q.device)[None, :]
    keep = torch.ones(m, n, dtype=torch.bool, device=q.device)
    if causal:
        keep &= kv_pos <= q_pos
    left, right = window
    if left >= 0:
        keep &= kv_pos >= q_pos - left
    if right >= 0:
        keep &= kv_pos <= q_pos + right
    scores = scores.masked_fill(~keep, float("-inf"))
    if sinks is None:
        probs = torch.softmax(scores, dim=-1)
    else:
        sink = sinks.float().to(q.device).view(num_heads, 1, 1)
        logits = torch.cat([scores, sink.expand(num_heads, m, 1)], dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., :n]
    return torch.einsum("hmn,nhd->mhd", probs, v)


def _reference_paged_prefill(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    *,
    softmax_scale,
    causal=True,
    window_size=FULL,
    sinks=None,
    **_unused,
):
    """Torch stand-in with the ``flash_attn_prefill_paged`` contract."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    page = k_cache.shape[1]
    for b in range(q.shape[0]):
        n = int(seq_lens[b])
        blocks = block_table[b, : -(-n // page)].long()
        out[b] = _reference_attention(
            q[b],
            k_cache[blocks].flatten(0, 1)[:n],
            v_cache[blocks].flatten(0, 1)[:n],
            scale=softmax_scale,
            causal=causal,
            window=window_size,
            sinks=sinks,
        )
    return out.to(q.dtype)


def _sinks(num_heads, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    sinks = torch.randn(num_heads, generator=gen) * 2.0
    sinks[0] = 10.0  # dominates the denominator
    sinks[1] = -30.0  # negligible
    return sinks.to(device)


def _paged_cache(k_lin, v_lin, page, seed):
    """Scatter linear ``[N, Hkv, D]`` K/V into shuffled pages with spares."""
    n, num_kv_heads, head_dim = k_lin.shape
    used = -(-n // page)
    num_blocks = used + 3
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=gen).to(k_lin.device)
    # Unused slots hold finite garbage that must never reach the output.
    shape = (num_blocks, page, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, dtype=k_lin.dtype, device=k_lin.device)
    v_cache = torch.randn(shape, dtype=k_lin.dtype, device=k_lin.device)
    for j in range(used):
        rows = slice(j * page, min((j + 1) * page, n))
        k_cache[int(perm[j]), : rows.stop - rows.start] = k_lin[rows]
        v_cache[int(perm[j]), : rows.stop - rows.start] = v_lin[rows]
    block_table = perm[:used].to(torch.int32)[None]
    return k_cache, v_cache, block_table


@pytest.fixture
def flash_interface():
    fi = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    if not hasattr(fi.flash_attn_v100_cuda, "prefill_paged_sinks_fwd"):
        pytest.skip("flash_attn_v100 extension predates the sink prefill entry")
    return fi


def _random_qkv(q_len, kv_len, num_heads, num_kv_heads, head_dim, seed):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to("cuda", torch.float16)

    return (
        randn(q_len, num_heads, head_dim),
        randn(kv_len, num_kv_heads, head_dim),
        randn(kv_len, num_kv_heads, head_dim),
    )


@requires_sm70
@torch.inference_mode()
@pytest.mark.parametrize("page", [16, 256])
@pytest.mark.parametrize(
    "q_len,kv_len",
    [
        (40, 150),  # fresh prefill inside one 176-token KV tile
        (300, 300),  # fresh prefill over two KV tiles
        (96, 1400),  # chunk over a prefix, eight KV tiles
        (1, 777),  # one-row chunk over a prefix
    ],
)
@pytest.mark.parametrize("window", [FULL, SWA], ids=["full", "swa513"])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(32, 4), (16, 2)])
def test_paged_prefill_sinks_match_reference(
    flash_interface, num_heads, num_kv_heads, window, q_len, kv_len, page
):
    head_dim = 128
    scale = head_dim**-0.5
    q, k_lin, v_lin = _random_qkv(
        q_len, kv_len, num_heads, num_kv_heads, head_dim, seed=q_len + kv_len
    )
    k_cache, v_cache, block_table = _paged_cache(k_lin, v_lin, page, seed=page)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    sinks = _sinks(num_heads, "cuda")

    def run(**kwargs):
        return flash_interface.flash_attn_prefill_paged(
            q[None],
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            **kwargs,
        )[0]

    out = run(sinks=sinks)
    out_default = run()
    out_none = run(sinks=None)
    out_neg_inf = run(sinks=torch.full_like(sinks, float("-inf")))
    torch.accelerator.synchronize()

    kwargs = dict(scale=scale, causal=True, window=window)
    ref = _reference_attention(q, k_lin, v_lin, sinks=sinks, **kwargs)
    ref_none = _reference_attention(q, k_lin, v_lin, sinks=None, **kwargs)
    assert not torch.allclose(ref, ref_none, atol=ATOL, rtol=RTOL), "vacuous sinks"

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
    # Without sinks the kernel is unchanged; a -inf sink adds exactly zero.
    assert torch.equal(out_none, out_default)
    assert torch.equal(out_neg_inf, out_default)
    torch.testing.assert_close(out_default.float(), ref_none, atol=ATOL, rtol=RTOL)


@requires_sm70
@torch.inference_mode()
@pytest.mark.parametrize("window", [FULL, SWA], ids=["full", "swa513"])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(32, 4), (16, 2)])
def test_chunked_prefill_with_sinks_matches_single_pass(
    flash_interface, num_heads, num_kv_heads, window
):
    """Chunks over a growing cached prefix reproduce one full prefill."""
    head_dim, page, total = 128, 16, 1500
    scale = head_dim**-0.5
    q, k_lin, v_lin = _random_qkv(
        total, total, num_heads, num_kv_heads, head_dim, seed=7
    )
    k_cache, v_cache, block_table = _paged_cache(k_lin, v_lin, page, seed=3)
    sinks = _sinks(num_heads, "cuda", seed=1)
    chunks = []
    for start, end in ((0, 700), (700, 1300), (1300, 1333), (1333, total)):
        seq_lens = torch.tensor([end], dtype=torch.int32, device="cuda")
        chunks.append(
            flash_interface.flash_attn_prefill_paged(
                q[None, start:end],
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                softmax_scale=scale,
                causal=True,
                window_size=window,
                sinks=sinks,
            )[0]
        )
    torch.accelerator.synchronize()
    ref = _reference_attention(
        q, k_lin, v_lin, scale=scale, causal=True, window=window, sinks=sinks
    )
    torch.testing.assert_close(torch.cat(chunks).float(), ref, atol=ATOL, rtol=RTOL)


@requires_sm70
@torch.inference_mode()
@pytest.mark.parametrize("window", [FULL, SWA], ids=["full", "swa513"])
def test_batched_requests_index_sinks_per_head(flash_interface, window):
    """Two requests with different prefixes and page tables in one launch."""
    num_heads, num_kv_heads, head_dim, page, q_len = 16, 2, 128, 16, 64
    scale = head_dim**-0.5
    lengths = (900, 200)
    torch.manual_seed(13)
    sinks = _sinks(num_heads, "cuda", seed=2)
    width = -(-max(lengths) // page)
    shape = (2 * width + 1, page, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, dtype=torch.float16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    queries, tables, refs = [], [], []
    for b, kv_len in enumerate(lengths):
        count = -(-kv_len // page)
        # The requests' pages interleave in one pool.
        pages = (torch.arange(count) * 2 + b + 1).to(torch.int32)
        table = torch.zeros(width, dtype=torch.int32)
        table[:count] = pages
        tables.append(table)
        index = pages.long().cuda()
        q = torch.randn(q_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
        queries.append(q)
        refs.append(
            _reference_attention(
                q,
                k_cache[index].flatten(0, 1)[:kv_len],
                v_cache[index].flatten(0, 1)[:kv_len],
                scale=scale,
                causal=True,
                window=window,
                sinks=sinks,
            )
        )
    out = flash_interface.flash_attn_prefill_paged(
        torch.stack(queries),
        k_cache,
        v_cache,
        torch.stack(tables).cuda(),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        softmax_scale=scale,
        causal=True,
        window_size=window,
        sinks=sinks,
    )
    torch.accelerator.synchronize()
    for b, ref in enumerate(refs):
        torch.testing.assert_close(out[b].float(), ref, atol=ATOL, rtol=RTOL)


@requires_sm70
@torch.inference_mode()
@pytest.mark.parametrize("head_dim,num_heads,num_kv_heads", [(64, 8, 2), (256, 8, 2)])
def test_other_head_sizes_use_the_generic_kernel(
    flash_interface, head_dim, num_heads, num_kv_heads
):
    """Sinks keep every head size on the generic kernel's epilogue."""
    scale = head_dim**-0.5
    q, k_lin, v_lin = _random_qkv(
        100, 300, num_heads, num_kv_heads, head_dim, seed=head_dim
    )
    k_cache, v_cache, block_table = _paged_cache(k_lin, v_lin, 16, seed=5)
    sinks = _sinks(num_heads, "cuda", seed=3)
    out = flash_interface.flash_attn_prefill_paged(
        q[None],
        k_cache,
        v_cache,
        block_table,
        torch.tensor([300], dtype=torch.int32, device="cuda"),
        softmax_scale=scale,
        causal=True,
        sinks=sinks,
    )[0]
    torch.accelerator.synchronize()
    ref = _reference_attention(
        q, k_lin, v_lin, scale=scale, causal=True, window=FULL, sinks=sinks
    )
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------
# Backend threading in prefill.py (the GPU variants use the real kernel).
# --------------------------------------------------------------------------


def _executor(paged, *, window=FULL, use_paged=True, anchored=None, tree=None):
    policy = SimpleNamespace(
        use_flash_v100_prefill_paged=use_paged,
        prefix_anchored_decode_window=anchored,
        use_triton_prefill=False,
        use_prefill_paged_cache=False,
    )
    ops = prefill.PrefillDriverOps(
        paged=paged,
        window=lambda causal: window,
        tree_requires_branch=tree,
        compare_triton=lambda *args: None,
        small_query_enabled=lambda metadata: False,
        prefix_dump_enabled=lambda: False,
    )
    settings = prefill.PrefillConfig(policy, 128**-0.5, "auto")
    return prefill.PrefillExecutor(settings, ops, workspace.V100Workspace())


Calls = list[tuple[tuple, dict]]


def _recording(op, calls: Calls):
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return op(*args, **kwargs)

    return run


def _prefix_case(device, dtype, *, q_lens, kv_lens, heads, page, seed):
    """Two requests in a ``[blocks, 2, page, Hkv, D]`` cache, one with a prefix."""
    num_heads, num_kv_heads, head_dim = heads
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device, dtype)

    used = [-(-n // page) for n in kv_lens]
    kv_cache = randn(sum(used) + 1, 2, page, num_kv_heads, head_dim)
    table = torch.zeros(len(kv_lens), max(used), dtype=torch.int32)
    linear = []
    first = 1  # block 0 stays a spare
    for i, (n, count) in enumerate(zip(kv_lens, used)):
        table[i, :count] = torch.arange(first, first + count)
        pages = kv_cache[first : first + count]
        linear.append((pages[:, 0].flatten(0, 1)[:n], pages[:, 1].flatten(0, 1)[:n]))
        first += count
    query = randn(sum(q_lens), num_heads, head_dim)
    offsets = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()])
    lengths = torch.tensor(kv_lens, dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=True,
        num_actual_tokens=int(offsets[-1]),
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.to(device, torch.int32),
        seq_lens_cpu=lengths,
        seq_lens=lengths.to(device),
        block_table=table.to(device),
    )
    return query, kv_cache, metadata, linear


LAYER = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)


def _check_prefix_route(paged, device, dtype, window, atol, rtol):
    heads = (16, 2, 128)
    q_lens, kv_lens = (37, 300), (529, 300)
    query, kv_cache, metadata, linear = _prefix_case(
        device, dtype, q_lens=q_lens, kv_lens=kv_lens, heads=heads, page=16, seed=0
    )
    sinks = _sinks(heads[0], device)
    calls: Calls = []
    executor = _executor(_recording(paged, calls), window=window)
    output = torch.zeros_like(query)
    result = executor._flash_v100_prefill_with_prefix(
        LAYER, query, None, None, kv_cache, metadata, output, sinks=sinks
    )
    assert result is output
    assert len(calls) == 2
    assert all(kwargs["sinks"] is sinks for _, kwargs in calls)
    start = 0
    for q_len, (k_lin, v_lin) in zip(q_lens, linear):
        ref = _reference_attention(
            query[start : start + q_len],
            k_lin,
            v_lin,
            scale=executor.scale,
            causal=True,
            window=window,
            sinks=sinks,
        )
        torch.testing.assert_close(
            output[start : start + q_len].float(), ref, atol=atol, rtol=rtol
        )
        start += q_len


def _check_dense_route(paged, device, dtype, window, atol, rtol):
    num_heads, num_kv_heads, head_dim = 16, 2, 128
    lengths = (300, 300, 129)
    total = sum(lengths)
    gen = torch.Generator().manual_seed(4)
    # Packed projection output: K and V are strided views, not copies.
    qkv = torch.randn(total, (num_heads + 2 * num_kv_heads) * head_dim, generator=gen)
    qkv = qkv.to(device, dtype)
    q_end = num_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim
    query = qkv[:, :q_end].view(total, num_heads, head_dim)
    key = qkv[:, q_end:k_end].view(total, num_kv_heads, head_dim)
    value = qkv[:, k_end:].view(total, num_kv_heads, head_dim)
    offsets = torch.tensor([0, 300, 600, total])
    metadata = SimpleNamespace(
        causal=True,
        num_actual_tokens=total,
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.to(device, torch.int32),
    )
    sinks = _sinks(num_heads, device, seed=5)
    calls: Calls = []
    executor = _executor(_recording(paged, calls), window=window)
    output = torch.zeros_like(query)
    executor._flash_v100_prefill(query, key, value, metadata, output, sinks=sinks)
    # Equal-length requests share one launch with one page per request.
    assert [args[1].shape[:2] for args, _ in calls] == [(2, 300), (1, 129)]
    first_args, first_kwargs = calls[0]
    assert first_args[1].data_ptr() == key.data_ptr()
    assert first_args[2].data_ptr() == value.data_ptr()
    assert first_args[3].tolist() == [[0], [1]]
    assert first_args[4].tolist() == [300, 300]
    assert first_kwargs["sinks"] is sinks
    assert first_kwargs["kv_cache_dtype"] == "auto"
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        ref = _reference_attention(
            query[start:end],
            key[start:end],
            value[start:end],
            scale=executor.scale,
            causal=True,
            window=window,
            sinks=sinks,
        )
        torch.testing.assert_close(output[start:end].float(), ref, atol=atol, rtol=rtol)


@pytest.mark.cpu_test
@pytest.mark.parametrize("window", [FULL, SWA], ids=["full", "swa513"])
def test_prefix_route_runs_every_request_on_the_paged_kernel(monkeypatch, window):
    def no_planner(*args, **kwargs):
        raise AssertionError("sinks must bypass the sink-unaware batch routes")

    monkeypatch.setattr(prefill, "execute_prefill_batch", no_planner)
    monkeypatch.setattr(prefill, "execute_prefill_sequence", no_planner)
    _check_prefix_route(
        _reference_paged_prefill, "cpu", torch.float32, window, 1e-5, 1e-5
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("window", [FULL, SWA], ids=["full", "swa513"])
def test_dense_route_reads_packed_kv_as_one_page_per_request(monkeypatch, window):
    def no_dense(*args, **kwargs):
        raise AssertionError("the dense kernels have no sink term")

    monkeypatch.setattr(prefill._dense_prefill, "flash_v100_dense_prefill", no_dense)
    _check_dense_route(
        _reference_paged_prefill, "cpu", torch.float32, window, 1e-5, 1e-5
    )


@requires_sm70
@torch.inference_mode()
@pytest.mark.parametrize("window", [FULL, SWA], ids=["full", "swa513"])
def test_backend_routes_with_sinks_on_the_kernel(flash_interface, window):
    paged = flash_interface.flash_attn_prefill_paged
    _check_prefix_route(paged, "cuda", torch.float16, window, ATOL, RTOL)
    _check_dense_route(paged, "cuda", torch.float16, window, ATOL, RTOL)


@pytest.mark.cpu_test
def test_sink_free_routes_are_unchanged(monkeypatch):
    def no_paged(*args, **kwargs):
        raise AssertionError("sink-free prefill keeps its routes")

    executor = _executor(no_paged)
    seen = []

    def dense(**kwargs):
        seen.append("dense")
        return kwargs["output"]

    def batch(*args):
        seen.append("batch")
        return True, "batch-output", set()

    monkeypatch.setattr(prefill._dense_prefill, "flash_v100_dense_prefill", dense)
    monkeypatch.setattr(prefill, "execute_prefill_batch", batch)
    query, kv_cache, metadata, _ = _prefix_case(
        "cpu", torch.float32, q_lens=(3,), kv_lens=(9,), heads=(4, 2, 8), page=4, seed=1
    )
    output = torch.zeros_like(query)
    executor._flash_v100_prefill(query, query, query, metadata, output)
    result = executor._flash_v100_prefill_with_prefix(
        LAYER, query, None, None, kv_cache, metadata, output
    )
    assert result == "batch-output"
    assert seen == ["dense", "batch"]


@pytest.mark.cpu_test
@pytest.mark.parametrize("prefix", [False, True])
def test_forward_passes_sinks_only_when_given(prefix):
    query, kv_cache, metadata, _ = _prefix_case(
        "cpu",
        torch.float32,
        q_lens=(3,),
        kv_lens=(9 if prefix else 3,),
        heads=(4, 2, 8),
        page=4,
        seed=2,
    )
    executor = _executor(_reference_paged_prefill)
    calls: list[dict] = []
    target = "_flash_v100_prefill_with_prefix" if prefix else "_flash_v100_prefill"

    def record(*args, **kwargs):
        calls.append(kwargs)
        return args[-1]

    # Instance overrides replace the executor methods, as legacy hooks do.
    vars(executor)[target] = record
    sinks = _sinks(4, "cpu")
    output = torch.zeros_like(query)
    for kwargs in ({}, {"sinks": None}, {"sinks": sinks}):
        prefill.forward(
            executor,
            LAYER,
            query,
            query,
            query,
            kv_cache,
            metadata,
            output,
            None,
            None,
            False,
            "layer",
            **kwargs,
        )
    assert calls == [{}, {}, {"sinks": sinks}]


@pytest.mark.cpu_test
def test_sink_routes_fail_closed():
    query, kv_cache, metadata, _ = _prefix_case(
        "cpu", torch.float32, q_lens=(3,), kv_lens=(9,), heads=(4, 2, 8), page=4, seed=3
    )
    sinks = _sinks(4, "cpu")
    output = torch.zeros_like(query)
    cases = {
        "disabled or unavailable": _executor(_reference_paged_prefill, use_paged=False),
        "prefix-anchored": _executor(_reference_paged_prefill, anchored=64),
        "tree attention": _executor(
            _reference_paged_prefill, tree=lambda metadata, loc: True
        ),
    }
    for message, executor in cases.items():
        with pytest.raises(RuntimeError, match=message):
            executor._flash_v100_prefill_with_prefix(
                LAYER, query, None, None, kv_cache, metadata, output, sinks=sinks
            )
    with pytest.raises(RuntimeError, match="unavailable"):
        _executor(None)._flash_v100_prefill(
            query, query, query, metadata, output, sinks=sinks
        )


# --------------------------------------------------------------------------
# Interface contract with a stub extension (no GPU, no build required).
# --------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parents[3] / "flash-attention-v100"


@pytest.fixture
def stub_interface(monkeypatch):
    """Load ``flash_attn_interface`` under a private name with a stub native."""
    name = "_sinks_prefill_flash_attn_v100"
    package = types.ModuleType(name)
    package.__path__ = [str(_PACKAGE_DIR / "flash_attn_v100")]
    stub = types.ModuleType("flash_attn_v100_cuda")
    monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, f"{name}.flash_attn_v100_cuda", stub)
    spec = importlib.util.spec_from_file_location(
        f"{name}.flash_attn_interface",
        _PACKAGE_DIR / "flash_attn_v100" / "flash_attn_interface.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(f"{name}.legacy_policy", None)


class _OldNative:
    """An extension build that predates the sink entry."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def prefill_paged_fwd(self, *args):
        self.calls.append(("plain", args))
        return args[0].clone()


class _Native(_OldNative):
    def prefill_paged_sinks_fwd(self, *args):
        self.calls.append(("sinks", args))
        return args[0].clone()


def _interface_inputs():
    q = torch.randn(1, 5, 4, 8)
    k_cache = torch.randn(3, 4, 2, 8)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([7], dtype=torch.int32)
    return q, k_cache, block_table, seq_lens


@pytest.mark.cpu_test
def test_interface_selects_the_sink_entry(stub_interface):
    q, k_cache, block_table, seq_lens = _interface_inputs()
    native = _Native()
    runtime = SimpleNamespace(native=native)
    sinks = torch.randn(8)[::2]  # non-contiguous view of four heads
    for kwargs in ({}, {"sinks": None}, {"sinks": sinks}):
        out = stub_interface.flash_attn_prefill_paged(
            q,
            k_cache,
            k_cache,
            block_table,
            seq_lens,
            softmax_scale=0.5,
            causal=True,
            window_size=SWA,
            _runtime=runtime,
            **kwargs,
        )
        assert out.shape == q.shape
    kinds = [kind for kind, _ in native.calls]
    assert kinds == ["plain", "plain", "sinks"]
    plain, with_sinks = native.calls[0][1], native.calls[2][1]
    # The sink-free ABI is unchanged; the sink entry appends one tensor.
    assert len(plain) == 15 and len(with_sinks) == 16
    assert plain[11:] == (512, 0, None, 0) == with_sinks[11:15]
    passed = with_sinks[15]
    assert passed.is_contiguous() and torch.equal(passed, sinks)
    assert torch.equal(with_sinks[0], q.permute(0, 2, 1, 3))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "sinks,error,message",
    [
        (torch.zeros(4, dtype=torch.float16), TypeError, "float32"),
        (torch.zeros(5), ValueError, "shape"),
        (torch.zeros(1, 4), ValueError, "shape"),
        (torch.zeros(4, device="meta"), ValueError, "device"),
        ([0.0] * 4, TypeError, "torch.Tensor"),
    ],
)
def test_interface_validates_sinks(stub_interface, sinks, error, message):
    q, k_cache, block_table, seq_lens = _interface_inputs()
    native = _Native()
    with pytest.raises(error, match=message):
        stub_interface.flash_attn_prefill_paged(
            q,
            k_cache,
            k_cache,
            block_table,
            seq_lens,
            sinks=sinks,
            _runtime=SimpleNamespace(native=native),
        )
    assert native.calls == []


@pytest.mark.cpu_test
def test_interface_rejects_an_extension_without_the_sink_entry(stub_interface):
    q, k_cache, block_table, seq_lens = _interface_inputs()
    native = _OldNative()
    with pytest.raises(RuntimeError, match="rebuild flash_attn_v100"):
        stub_interface.flash_attn_prefill_paged(
            q,
            k_cache,
            k_cache,
            block_table,
            seq_lens,
            sinks=torch.zeros(4),
            _runtime=SimpleNamespace(native=native),
        )
    assert native.calls == []
