# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Learned attention sinks in the SM70 Flash-V100 paged decoder.

The GPU cases compare ``flash_attn_decode_paged(..., sinks=...)`` with an FP32
torch reference for the D128 GQA shapes of a 32q/4kv model (TP1 and TP2
local heads), sliding-window 513 and full attention, FP16 paged KV, single
and split-KV partitions, decode batches 1-8, expanded verifier rows and CUDA
graph replay. They skip without an SM70 device and a built extension.

The CPU cases check the Python dispatch of the interface against an injected
native module and the decode-executor wiring with injected operators.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

FLASH_V100_ROOT = Path(__file__).resolve().parents[3] / "flash-attention-v100"
HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
# A 513-token sliding window: the query token plus 512 earlier keys.
SWA_WINDOW = (512, 0)
FULL_WINDOW = (-1, -1)
WINDOWS = pytest.mark.parametrize(
    "window", [SWA_WINDOW, FULL_WINDOW], ids=["swa513", "full"]
)
# Query/KV heads per rank: TP1 32q/4kv and TP2 16q/2kv, both GQA 8:1.
HEADS = pytest.mark.parametrize("heads", [(32, 4), (16, 2)], ids=["tp1", "tp2"])
ATOL = 5e-4
RTOL = 3e-2


# --------------------------------------------------------------------------
# GPU helpers
# --------------------------------------------------------------------------


def _sm70_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    for index in range(torch.accelerator.device_count()):
        if torch.cuda.get_device_capability(index) == (7, 0):
            return torch.device(f"cuda:{index}")
    pytest.skip("SM70/V100 CUDA device is required")


@pytest.fixture
def sm70():
    device = _sm70_device()
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    if not interface.flash_attn_decode_paged_sinks_available():
        pytest.skip("flash_attn_v100 was built without decode attention sinks")
    return device, interface


def _paged_kv(
    seq_lens: list[int],
    num_kv_heads: int,
    block_size: int,
    device: torch.device,
    seed: int,
):
    """FP16 K/V in shuffled physical blocks with a per-sequence block table."""
    generator = torch.Generator().manual_seed(seed)
    blocks_per_seq = [max(1, math.ceil(length / block_size)) for length in seq_lens]
    max_blocks = max(blocks_per_seq)
    num_blocks = sum(blocks_per_seq) + 3
    physical = torch.randperm(num_blocks, generator=generator)
    block_table = torch.zeros((len(seq_lens), max_blocks), dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(blocks_per_seq):
        block_table[row, :count] = physical[cursor : cursor + count]
        cursor += count
    shape = (num_blocks, block_size, num_kv_heads, HEAD_DIM)
    k_cache = torch.randn(shape, generator=generator).to(device, torch.float16)
    v_cache = torch.randn(shape, generator=generator).to(device, torch.float16)
    return k_cache, v_cache, block_table.to(device)


def _query(batch: int, num_heads: int, device: torch.device, seed: int):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((batch, num_heads, HEAD_DIM), generator=generator)
    return q.to(device, torch.float16)


def _sinks(num_heads: int, device: torch.device, seed: int) -> torch.Tensor:
    """Per-head sinks from negligible to dominant, shuffled across heads."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.linspace(-2.0, 10.0, num_heads, dtype=torch.float32)
    return values[torch.randperm(num_heads, generator=generator)].to(device)


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    window: tuple[int, int],
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """FP32 decode attention: each row attends to its last seq_len keys."""
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_heads // num_kv_heads
    block_size = k_cache.shape[1]
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim)
    out = torch.empty((batch, num_heads, head_dim), device=q.device)
    for row in range(batch):
        length = int(seq_lens[row])
        first = max(0, length - 1 - window[0]) if window[0] >= 0 else 0
        pos = torch.arange(first, length, device=q.device)
        slots = (
            block_table[row, pos // block_size].long() * block_size + pos % block_size
        )
        k = k_flat[slots].float().repeat_interleave(group, dim=1)
        v = v_flat[slots].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,lhd->hl", q[row].float(), k) * SCALE
        if sinks is None:
            probs = torch.softmax(scores, dim=-1)
        else:
            logits = torch.cat([scores, sinks.float()[:, None]], dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, :-1]
        out[row] = torch.einsum("hl,lhd->hd", probs, v)
    return out


def _decode(interface, q, k_cache, v_cache, block_table, seq_lens, window, **kw):
    return interface.flash_attn_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=SCALE,
        window_size=window,
        **kw,
    )


def _check_case(sm70, seq_lens, heads, window, *, block_size=16, seed=0, **kw):
    device, interface = sm70
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table = _paged_kv(
        seq_lens, num_kv_heads, block_size, device, seed
    )
    q = _query(len(seq_lens), num_heads, device, seed + 1)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sinks = _sinks(num_heads, device, seed + 2)
    kw.setdefault("max_seq_len_hint", max(seq_lens))

    out = _decode(interface, q, k_cache, v_cache, block_table, lens, window, **kw)
    expected = _reference(q, k_cache, v_cache, block_table, lens, window, sinks=None)
    torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)

    out_sinks = _decode(
        interface, q, k_cache, v_cache, block_table, lens, window, sinks=sinks, **kw
    )
    expected_sinks = _reference(q, k_cache, v_cache, block_table, lens, window, sinks)
    torch.testing.assert_close(out_sinks.float(), expected_sinks, rtol=RTOL, atol=ATOL)
    # The dominant sinks must visibly shrink the output, or the case is vacuous.
    effect = (expected_sinks - expected).abs().max()
    assert effect > max(4 * ATOL, 0.05 * expected.abs().max())
    return out, out_sinks


# --------------------------------------------------------------------------
# GPU cases
# --------------------------------------------------------------------------


@WINDOWS
@HEADS
@pytest.mark.parametrize(
    "seq_len",
    [1, 100, 256, 257, 513, 514, 600, 8192],
)
@torch.inference_mode()
def test_decode_sinks_single_sequence(sm70, window, heads, seq_len):
    """One partition (<=256) through many partitions at the default P256."""
    _check_case(sm70, [seq_len], heads, window, seed=seq_len)


@WINDOWS
@HEADS
@pytest.mark.parametrize("partition_size", [256, 512, 1024])
@torch.inference_mode()
def test_decode_sinks_every_partition_instantiation(
    sm70, window, heads, partition_size
):
    _check_case(
        sm70,
        [3000, 513, 1025],
        heads,
        window,
        partition_size_hint=partition_size,
        seed=partition_size,
    )


@WINDOWS
@HEADS
@pytest.mark.parametrize(
    "seq_lens",
    [
        [100],
        [600, 100],
        [1, 255, 256, 257],
        [100, 513, 600, 1024, 2049, 4096, 8192, 777],
    ],
    ids=["b1", "b2", "b4", "b8"],
)
@torch.inference_mode()
def test_decode_sinks_batches(sm70, window, heads, seq_lens):
    _check_case(sm70, seq_lens, heads, window, seed=len(seq_lens))


@WINDOWS
@pytest.mark.parametrize("block_size", [32, 64])
@torch.inference_mode()
def test_decode_sinks_other_page_sizes(sm70, window, block_size):
    _check_case(sm70, [700, 2500], (16, 2), window, block_size=block_size)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_64k_context(sm70, window):
    """The >=32K default plan switches to P1024 split-KV partitions."""
    _check_case(sm70, [65536], (16, 2), window, seed=7)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_static_graph_envelope(sm70, window):
    """A capacity hint launches idle partitions, as CUDA-graph decode does."""
    _check_case(
        sm70,
        [600, 70],
        (32, 4),
        window,
        workspace_seq_capacity_hint=16384,
        seed=11,
    )


@WINDOWS
@HEADS
@torch.inference_mode()
def test_decode_sinks_expanded_verifier_rows(sm70, window, heads):
    """Multi-token verification runs one decode row per query token."""
    device, interface = sm70
    num_heads, num_kv_heads = heads
    context, num_tokens = 1500, 4
    k_cache, v_cache, block_table = _paged_kv(
        [context], num_kv_heads, 16, device, seed=21
    )
    rows = block_table.expand(num_tokens, -1).contiguous()
    lens = torch.arange(
        context - num_tokens + 1, context + 1, dtype=torch.int32, device=device
    )
    q = _query(num_tokens, num_heads, device, seed=22)
    sinks = _sinks(num_heads, device, seed=23)
    out = _decode(
        interface,
        q,
        k_cache,
        v_cache,
        rows,
        lens,
        window,
        max_seq_len_hint=context,
        sinks=sinks,
    )
    expected = _reference(q, k_cache, v_cache, rows, lens, window, sinks)
    torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_none_is_bit_identical(sm70, window):
    """``sinks=None`` is the sink-free kernel; -inf sinks add an exact zero.

    With sinks, GQA 8:1 D128 layers take the grouped partition kernel, whose
    fp32 sums run in another order than the per-head kernel's, so -inf sinks
    match the sink-free output to rounding rather than bit for bit.
    """
    device, interface = sm70
    seq_lens = [100, 600, 8192]
    k_cache, v_cache, block_table = _paged_kv(seq_lens, 4, 16, device, seed=31)
    q = _query(len(seq_lens), 32, device, seed=32)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    kw = {"max_seq_len_hint": max(seq_lens)}
    plain = _decode(interface, q, k_cache, v_cache, block_table, lens, window, **kw)
    none = _decode(
        interface, q, k_cache, v_cache, block_table, lens, window, sinks=None, **kw
    )
    neg_inf = torch.full((32,), -math.inf, dtype=torch.float32, device=device)
    masked = _decode(
        interface, q, k_cache, v_cache, block_table, lens, window, sinks=neg_inf, **kw
    )
    assert torch.equal(none, plain)
    torch.testing.assert_close(masked, plain, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)


# The grouped kernel against the per-head kernel: both round partials to
# fp16, so they agree to a couple of fp16 ulps.
GROUPED_ATOL = 2e-3
GROUPED_RTOL = 1e-2


@WINDOWS
@HEADS
@pytest.mark.parametrize("partition_size", [256, 512, 1024])
@pytest.mark.parametrize(
    "seq_lens",
    [[1], [255, 256, 257], [513, 3000, 70], [8192]],
    ids=["one", "edges", "mixed", "long"],
)
@torch.inference_mode()
def test_decode_sinks_grouped_kernel_matches_per_head_kernel(
    sm70, window, heads, partition_size, seq_lens
):
    """GQA 8:1 D128 sink layers run the grouped partition kernel.

    A sink of -1e4 adds an exact zero to every denominator, so the grouped
    output must match the per-head kernel's sink-free output.
    """
    device, interface = sm70
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table = _paged_kv(
        seq_lens, num_kv_heads, 16, device, seed=81 + partition_size
    )
    q = _query(len(seq_lens), num_heads, device, seed=82)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    kw = {"max_seq_len_hint": max(seq_lens), "partition_size_hint": partition_size}
    per_head = _decode(interface, q, k_cache, v_cache, block_table, lens, window, **kw)
    silent = torch.full((num_heads,), -1.0e4, dtype=torch.float32, device=device)
    grouped = _decode(
        interface, q, k_cache, v_cache, block_table, lens, window, sinks=silent, **kw
    )
    torch.testing.assert_close(grouped, per_head, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_cuda_graph_replay(sm70, window):
    """Captured once; replays follow new queries, lengths and sink values."""
    device, interface = sm70
    capacity = 4096
    seq_lens = [capacity, capacity]
    k_cache, v_cache, block_table = _paged_kv(seq_lens, 4, 16, device, seed=41)
    q = _query(2, 32, device, seed=42)
    lens = torch.tensor([600, 50], dtype=torch.int32, device=device)
    sinks = _sinks(32, device, seed=43)
    out = torch.empty_like(q)

    def run():
        _decode(
            interface,
            q,
            k_cache,
            v_cache,
            block_table,
            lens,
            window,
            out=out,
            max_seq_len_hint=capacity,
            workspace_seq_capacity_hint=capacity,
            sinks=sinks,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for step, new_lens in enumerate(([600, 50], [3000, 1], [4096, 2049])):
        q.copy_(_query(2, 32, device, seed=50 + step))
        sinks.copy_(_sinks(32, device, seed=60 + step))
        lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        graph.replay()
        torch.accelerator.synchronize()
        expected = _reference(q, k_cache, v_cache, block_table, lens, window, sinks)
        torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


FUSED_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE"


@pytest.fixture
def sm70_fused(sm70):
    _, interface = sm70
    if not hasattr(interface.flash_attn_v100_cuda, "decode_paged_sinks_fused_fwd"):
        pytest.skip("flash_attn_v100 was built without the fused sink decode")
    return sm70


@WINDOWS
@HEADS
@pytest.mark.parametrize(
    "seq_lens",
    [[1], [63, 64, 65], [255, 256, 257, 513, 514], [3000, 70], [4096], [65536]],
    ids=["one", "min-chunk", "edges", "mixed", "4k", "64k"],
)
@pytest.mark.parametrize("capacity", [None, 262144], ids=["eager", "envelope"])
@torch.inference_mode()
def test_decode_sinks_fused_matches_two_launches(
    sm70_fused, monkeypatch, window, heads, seq_lens, capacity
):
    """One launch per layer against the partition kernel plus reducer.

    ``capacity`` reproduces the server's capture hints (a 262K workspace
    envelope): the fused grid fills the device and the last CTA reduces many
    partials. Both routes round partials to fp16, so they agree to a couple
    of fp16 ulps, and both match the FP32 reference.
    """
    device, interface = sm70_fused
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table = _paged_kv(
        seq_lens, num_kv_heads, 16, device, seed=91 + len(seq_lens)
    )
    q = _query(len(seq_lens), num_heads, device, seed=92)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sinks = _sinks(num_heads, device, seed=93)
    kw = {"max_seq_len_hint": max(seq_lens), "sinks": sinks}
    if capacity is not None:
        kw["workspace_seq_capacity_hint"] = capacity
        kw["partition_size_hint"] = 256
    monkeypatch.setenv(FUSED_ENV, "0")
    two = _decode(interface, q, k_cache, v_cache, block_table, lens, window, **kw)
    monkeypatch.setenv(FUSED_ENV, "1")
    fused = _decode(interface, q, k_cache, v_cache, block_table, lens, window, **kw)
    # Repeat: the arrival counters must be back at zero after each launch.
    again = _decode(interface, q, k_cache, v_cache, block_table, lens, window, **kw)
    torch.testing.assert_close(fused, two, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)
    assert torch.equal(fused, again)
    expected = _reference(q, k_cache, v_cache, block_table, lens, window, sinks)
    torch.testing.assert_close(fused.float(), expected, rtol=RTOL, atol=ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_fused_cuda_graph_server_envelope(sm70_fused, window):
    """The server's capture: hint 1, a 262K envelope, replays at any length.

    Replays must follow new lengths in both directions (the grid never
    depends on them), stay bit-stable on repeat, and keep the empty row zero.
    """
    device, interface = sm70_fused
    capacity = 262144
    k_cache, v_cache, block_table = _paged_kv([40000, 40000], 2, 16, device, seed=95)
    q = _query(2, 16, device, seed=96)
    lens = torch.ones(2, dtype=torch.int32, device=device)
    active = torch.ones(1, dtype=torch.int32, device=device)
    sinks = _sinks(16, device, seed=97)
    out = torch.empty_like(q)

    def run():
        _decode(
            interface,
            q,
            k_cache,
            v_cache,
            block_table,
            lens,
            window,
            out=out,
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=capacity,
            active_num_partitions=active,
            partition_size_hint=256,
            sinks=sinks,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for step, new_lens in enumerate(
        ([4096, 513], [40000, 1], [600, 0], [32768, 4097], [65, 40000])
    ):
        q.copy_(_query(2, 16, device, seed=100 + step))
        lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        active.fill_(max(1, math.ceil(max(new_lens) / 256)))
        graph.replay()
        torch.accelerator.synchronize()
        first = out.clone()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, first)
        expected = _reference(q, k_cache, v_cache, block_table, lens, window, sinks)
        torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


@torch.inference_mode()
def test_decode_sinks_dominant_sink_stays_finite(sm70):
    """A sink far above every score must not overflow the denominator."""
    device, interface = sm70
    k_cache, v_cache, block_table = _paged_kv([2000], 2, 16, device, seed=71)
    q = _query(1, 16, device, seed=72)
    lens = torch.tensor([2000], dtype=torch.int32, device=device)
    sinks = torch.full((16,), 120.0, dtype=torch.float32, device=device)
    out = _decode(
        interface,
        q,
        k_cache,
        v_cache,
        block_table,
        lens,
        FULL_WINDOW,
        max_seq_len_hint=2000,
        sinks=sinks,
    )
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e-3


# --------------------------------------------------------------------------
# CPU: interface dispatch against an injected native module
# --------------------------------------------------------------------------


class _Native(ModuleType):
    def __init__(self, *, with_sinks: bool = True, with_fused: bool = False):
        super().__init__("flash_attn_v100.flash_attn_v100_cuda")
        self.calls: list[tuple[str, tuple]] = []
        self.decode_paged_fwd = self._record("decode_paged_fwd")
        if with_sinks:
            self.decode_paged_sinks_fwd = self._record("decode_paged_sinks_fwd")
        if with_fused:
            self.decode_paged_sinks_fused_fwd = self._record(
                "decode_paged_sinks_fused_fwd"
            )

    def _record(self, name):
        def call(*args):
            self.calls.append((name, args))
            return args[3]

        return call


def _import_interface(monkeypatch, native: _Native):
    for name in list(sys.modules):
        if name == "flash_attn_v100" or name.startswith("flash_attn_v100."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "flash_attn_v100.flash_attn_v100_cuda", native)
    monkeypatch.syspath_prepend(str(FLASH_V100_ROOT))
    return importlib.import_module("flash_attn_v100.flash_attn_interface")


@pytest.fixture
def cpu_interface(monkeypatch):
    before = set(sys.modules)
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv(FUSED_ENV, raising=False)

    def load(with_sinks: bool = True, with_fused: bool = False):
        native = _Native(with_sinks=with_sinks, with_fused=with_fused)
        return _import_interface(monkeypatch, native), native

    yield load
    for name in set(sys.modules) - before:
        if name == "flash_attn_v100" or name.startswith("flash_attn_v100."):
            del sys.modules[name]


def _cpu_inputs(num_heads=16, num_kv_heads=2):
    q = torch.zeros((2, num_heads, HEAD_DIM), dtype=torch.float16)
    k_cache = torch.zeros((4, 16, num_kv_heads, HEAD_DIM), dtype=torch.float16)
    block_table = torch.zeros((2, 2), dtype=torch.int32)
    seq_lens = torch.tensor([20, 7], dtype=torch.int32)
    return q, k_cache, k_cache.clone(), block_table, seq_lens


def test_interface_sink_free_call_keeps_native_abi(cpu_interface):
    interface, native = cpu_interface()
    q, k, v, table, lens = _cpu_inputs()
    out = torch.empty_like(q)
    interface.flash_attn_decode_paged(q, k, v, table, lens, out=out, sinks=None)
    interface.flash_attn_decode_paged(q, k, v, table, lens, out=out)
    assert [name for name, _ in native.calls] == ["decode_paged_fwd"] * 2
    assert all(len(args) == 20 for _, args in native.calls)
    assert native.calls[0][1][3] is out


def test_interface_sinks_take_the_sink_entry_point(cpu_interface):
    interface, native = cpu_interface()
    q, k, v, table, lens = _cpu_inputs()
    sinks = torch.zeros(16, dtype=torch.float32)
    interface.flash_attn_decode_paged(
        q, k, v, table, lens, window_size=SWA_WINDOW, sinks=sinks
    )
    ((name, args),) = native.calls
    assert name == "decode_paged_sinks_fwd"
    assert len(args) == 21 and args[-1] is sinks
    assert args[16:18] == SWA_WINDOW
    assert interface.flash_attn_decode_paged_sinks_available()


def test_interface_grouped_sinks_take_the_fused_entry_point(cpu_interface):
    interface, native = cpu_interface(with_fused=True)
    q, k, v, table, lens = _cpu_inputs()
    sinks = torch.zeros(16, dtype=torch.float32)
    for _ in range(2):
        interface.flash_attn_decode_paged(
            q, k, v, table, lens, window_size=SWA_WINDOW, sinks=sinks
        )
    assert [name for name, _ in native.calls] == ["decode_paged_sinks_fused_fwd"] * 2
    (_, first), (_, second) = native.calls
    assert len(first) == 22 and first[20] is sinks
    counters = first[21]
    # One zeroed int32 counter per (row, head) at least, reused across calls
    # (graph replays keep its address) together with the shared workspace.
    assert counters.dtype == torch.int32 and counters.is_contiguous()
    assert counters.numel() >= q.shape[0] * k.shape[2]
    assert torch.count_nonzero(counters) == 0
    assert second[21] is counters
    assert first[6].data_ptr() == second[6].data_ptr()


def test_interface_fused_decode_kill_switch(cpu_interface, monkeypatch):
    monkeypatch.setenv(FUSED_ENV, "0")
    interface, native = cpu_interface(with_fused=True)
    q, k, v, table, lens = _cpu_inputs()
    sinks = torch.zeros(16, dtype=torch.float32)
    interface.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    ((name, args),) = native.calls
    assert name == "decode_paged_sinks_fwd" and len(args) == 21


@pytest.mark.parametrize(
    ("heads", "kwargs"),
    [
        ((16, 4), {}),
        ((16, 2), {"anchor_lens": "rows", "anchored_window": 8}),
    ],
    ids=["gqa4", "anchored"],
)
def test_interface_fused_decode_only_for_grouped_shapes(cpu_interface, heads, kwargs):
    interface, native = cpu_interface(with_fused=True)
    q, k, v, table, lens = _cpu_inputs(*heads)
    if kwargs.get("anchor_lens") == "rows":
        kwargs = {**kwargs, "anchor_lens": torch.zeros(2, dtype=torch.int32)}
    sinks = torch.zeros(heads[0], dtype=torch.float32)
    interface.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks, **kwargs)
    interface.flash_attn_decode_paged(q, k, v, table, lens)
    assert [name for name, _ in native.calls] == [
        "decode_paged_sinks_fwd",
        "decode_paged_fwd",
    ]


def test_bound_runtime_policy_carries_the_fused_switch(cpu_interface, monkeypatch):
    interface, _ = cpu_interface(with_fused=True)
    runtime = importlib.import_module("flash_attn_v100.runtime")
    fields = runtime.PythonPolicy.__dataclass_fields__
    assert fields["sink_fused_decode"].default is True
    # A configured engine's policy wins over the process environment.
    monkeypatch.setenv(FUSED_ENV, "1")
    off = SimpleNamespace(policy=SimpleNamespace(sink_fused_decode=False))
    assert not interface._sink_fused_decode_enabled(_runtime=off)
    monkeypatch.setenv(FUSED_ENV, "0")
    on = SimpleNamespace(policy=SimpleNamespace(sink_fused_decode=True))
    assert interface._sink_fused_decode_enabled(_runtime=on)
    assert not interface._sink_fused_decode_enabled()


@pytest.mark.parametrize(
    ("sinks", "error"),
    [
        (torch.zeros(16, dtype=torch.float16), TypeError),
        (torch.zeros(8, dtype=torch.float32), ValueError),
        (torch.zeros((16, 1), dtype=torch.float32), ValueError),
        (torch.zeros(32, dtype=torch.float32)[::2], ValueError),
        ([0.0] * 16, TypeError),
    ],
    ids=["fp16", "kv-heads", "2d", "strided", "list"],
)
def test_interface_rejects_malformed_sinks(cpu_interface, sinks, error):
    interface, native = cpu_interface()
    q, k, v, table, lens = _cpu_inputs()
    with pytest.raises(error):
        interface.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert native.calls == []


def test_interface_old_extension_rejects_sinks(cpu_interface):
    interface, native = cpu_interface(with_sinks=False)
    q, k, v, table, lens = _cpu_inputs()
    with pytest.raises(RuntimeError, match="Rebuild Flash-V100"):
        interface.flash_attn_decode_paged(
            q, k, v, table, lens, sinks=torch.zeros(16, dtype=torch.float32)
        )
    assert native.calls == []
    assert not interface.flash_attn_decode_paged_sinks_available()


# --------------------------------------------------------------------------
# CPU: decode executor wiring
# --------------------------------------------------------------------------


def _unused(*args, **kwargs):
    raise AssertionError("unexpected route")


def _executor(*, sinks=None, scalar_tail=None, **policy):
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_v100 import decode as _decode
    from vllm.v1.attention.backends.flash_v100.workspace import V100Workspace

    settings = {
        "prefix_anchored_decode_window": None,
        "use_decode_xqa": False,
        "use_flash_v100": True,
        "use_flash_v100_decode": True,
        "use_flash_v100_prefill_paged": True,
        "use_decode_paged_prefill": False,
        "use_decode_dense_cache": False,
        "use_decode_dense_reference": False,
        "use_decode_scalar_paged": True,
        "allow_triton_fallback": False,
        "decode_strategy": "legacy",
    }
    settings.update(policy)
    calls: list[dict] = []

    def paged(*args, **kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(1)

    config = _decode.DecodeConfig(
        SimpleNamespace(**settings),
        SCALE,
        "auto",
        AttentionType.DECODER,
        SWA_WINDOW,
        sinks=sinks,
    )
    ops = _decode.DecodeOps(
        dense=_unused,
        paged=paged,
        xqa=_unused,
        wmma=_unused,
        prefill=_unused,
        prefill_bhmd=_unused,
        paged_keywords={"window_size", "max_seq_len_hint"},
        scalar_tail=scalar_tail,
        reject_xqa=lambda codec, metadata: False,
        reserve_bhmd_compare=_unused,
        write_bhmd_compare=_unused,
        compare_bhmd=_unused,
        compare_triton=lambda *args, **kwargs: None,
        triton_forward=_unused,
        profile_trace=lambda *args, **kwargs: None,
        draft_debug_enabled=lambda: False,
        draft_debug_log=_unused,
        format_debug=_unused,
    )
    return _decode, _decode.DecodeExecutor(config, ops, V100Workspace()), calls


def _decode_inputs(num_heads=16, num_kv_heads=2):
    query = torch.zeros((2, num_heads, HEAD_DIM), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 3, 16, num_kv_heads, HEAD_DIM), dtype=torch.float16)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        seq_lens=torch.tensor([40, 9], dtype=torch.int32),
        block_table=torch.zeros((2, 3), dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=40,
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    return layer, query, kv_cache, metadata, output


def _request(module, layer, query, kv_cache, metadata, output):
    return module.DecodeRequest(
        layer,
        query,
        query,
        query,
        kv_cache,
        metadata,
        output,
        None,
        None,
        False,
        "model.layers.0.self_attn.attn",
    )


def test_executor_forward_applies_configured_sinks():
    sinks = torch.zeros(16, dtype=torch.float32)
    module, executor, calls = _executor(sinks=sinks)
    layer, query, kv_cache, metadata, output = _decode_inputs()
    request = _request(module, layer, query, kv_cache, metadata, output)
    assert executor.forward(request) is output
    ((kwargs,),) = [calls]
    assert kwargs["sinks"] is sinks
    assert kwargs["window_size"] == SWA_WINDOW
    assert torch.all(output == 1)


def test_executor_sink_free_call_keeps_keyword_set():
    module, executor, calls = _executor()
    layer, query, kv_cache, metadata, output = _decode_inputs()
    executor.forward(_request(module, layer, query, kv_cache, metadata, output))
    ((kwargs,),) = [calls]
    assert "sinks" not in kwargs


def test_executor_decode_helpers_take_explicit_sinks():
    sinks = torch.zeros(16, dtype=torch.float32)
    _, executor, calls = _executor()
    layer, query, kv_cache, metadata, output = _decode_inputs()
    executor._flash_v100_decode(
        layer, query, query, query, kv_cache, metadata, output, sinks=sinks
    )
    key_cache = kv_cache[0]
    executor._call_flash_attn_decode_paged(
        query,
        key_cache,
        key_cache,
        metadata.block_table,
        metadata.seq_lens,
        softmax_scale=SCALE,
        out=output,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
        window_size=SWA_WINDOW,
        sinks=sinks,
    )
    assert [kwargs["sinks"] for kwargs in calls] == [sinks, sinks]


def test_executor_rejects_conflicting_sinks():
    configured = torch.zeros(16, dtype=torch.float32)
    _, executor, calls = _executor(sinks=configured)
    layer, query, kv_cache, metadata, output = _decode_inputs()
    executor._flash_v100_decode(
        layer, query, query, query, kv_cache, metadata, output, sinks=configured
    )
    with pytest.raises(ValueError, match="differ"):
        executor._flash_v100_decode(
            layer,
            query,
            query,
            query,
            kv_cache,
            metadata,
            output,
            sinks=configured.clone(),
        )
    assert len(calls) == 1


def test_executor_sinks_bypass_the_compact_scalar_tail():
    tail_calls = []

    def scalar_tail(*args, **kwargs):
        tail_calls.append(kwargs)
        return True

    sinks = torch.zeros(16, dtype=torch.float32)
    _, executor, calls = _executor(scalar_tail=scalar_tail)
    _, query, kv_cache, metadata, output = _decode_inputs()
    kwargs = {
        "softmax_scale": SCALE,
        "out": output,
        "kv_cache_dtype": "auto",
        "k_scale": 1.0,
        "v_scale": 1.0,
    }
    args = (query, kv_cache[0], kv_cache[1], metadata.block_table, metadata.seq_lens)
    executor._call_flash_attn_decode_paged(*args, **kwargs)
    assert len(tail_calls) == 1 and calls == []
    executor._call_flash_attn_decode_paged(*args, **kwargs, sinks=sinks)
    assert len(tail_calls) == 1 and calls[0]["sinks"] is sinks


@pytest.mark.parametrize(
    ("flag", "candidate"),
    [
        ("use_decode_paged_prefill", "DecodePagedPrefill"),
        ("use_decode_dense_cache", "DecodeDenseCache"),
        ("use_decode_dense_reference", "DecodeDenseReference"),
    ],
)
def test_sink_layers_skip_sink_free_bridges(flag, candidate):
    layer, query, kv_cache, metadata, output = _decode_inputs()
    module, plain, _ = _executor(**{flag: True})
    request = _request(module, layer, query, kv_cache, metadata, output)
    assert getattr(module, candidate)(plain).admit(request)

    sinks = torch.zeros(16, dtype=torch.float32)
    module, executor, calls = _executor(sinks=sinks, **{flag: True})
    assert not getattr(module, candidate)(executor).admit(request)
    executor.forward(request)
    assert calls[0]["sinks"] is sinks


@pytest.mark.parametrize(
    "method",
    [
        "_flash_v100_decode_as_paged_prefill",
        "_flash_v100_decode_dense_cache",
        "_flash_v100_decode_dense_reference",
    ],
)
def test_sink_free_bridges_raise_on_sinks(method):
    _, executor, _ = _executor()
    layer, query, kv_cache, metadata, output = _decode_inputs()
    args = (layer, query, query, query) if method.endswith("cache") else (layer, query)
    with pytest.raises(RuntimeError, match="does not support attention sinks"):
        getattr(executor, method)(
            *args,
            kv_cache,
            metadata,
            output,
            sinks=torch.zeros(16, dtype=torch.float32),
        )


def test_xqa_decode_rejects_sink_layers():
    _, executor, _ = _executor(sinks=torch.zeros(16, dtype=torch.float32))
    layer, query, kv_cache, metadata, output = _decode_inputs()
    with pytest.raises(RuntimeError, match="does not support attention sinks"):
        executor.xqa(
            layer,
            query,
            kv_cache[0],
            kv_cache[1],
            metadata,
            output,
            output,
            SWA_WINDOW,
            None,
        )
