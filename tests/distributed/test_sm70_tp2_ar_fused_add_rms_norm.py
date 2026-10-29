# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 TP2 all-reduce + fused_add_rms_norm against the unfused ops.

Two V100s. The reference is what the model runs without the fusion: the
custom all-reduce (cross_device_reduce_1stage), the elementwise scale or
partial-sum add where present, then the vLLM C fused_add_rms_norm kernel.
Every output must be bitwise equal to it, on both ranks.

Run under the server's conditions as well: the allocator with
expandable_segments (the serve script's PYTORCH_ALLOC_CONF), operands
allocated before graph capture (static buffers the peer cannot open over
IPC), graphs captured inside and outside the communicator's capture context
(FULL and PIECEWISE capture, and capture-time warmup that runs eagerly), and
ranks entering the collective at different times.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 2048
EPS = 1e-6
ROWS = (1, 2, 3, 4, 5, 8, 16)


def _reference(ca, x, x_b, residual, weight, scale):
    """Unfused ops; under capture the collective input is a pool buffer."""
    total = x.clone() if x_b is None else x + x_b
    reduced = ca.custom_all_reduce(total)
    if scale != 1.0:
        reduced = reduced * scale
    out = reduced.clone()
    res = residual.clone()
    torch.ops._C.fused_add_rms_norm(out, res, weight, EPS)
    return out, res


def _fill(rank, seed, x, x_b, residual, weight, magnitude):
    # Peer inputs differ; the residual stream and the weight are replicated.
    gen = torch.Generator(device="cuda").manual_seed(seed * 7 + rank)
    x.copy_(torch.randn(x.shape, generator=gen, device="cuda") * magnitude)
    if x_b is not None:
        x_b.copy_(torch.randn(x.shape, generator=gen, device="cuda") * magnitude)
    gen = torch.Generator(device="cuda").manual_seed(seed * 7 + 5)
    residual.copy_(torch.randn(residual.shape, generator=gen, device="cuda") * 8)
    weight.copy_(1 + 0.1 * torch.randn(weight.shape, generator=gen, device="cuda"))
    if seed % 4 == 1:
        # Exact cancellation across ranks and against the residual.
        x.copy_(-x if rank else x)
        residual[0].zero_()


def _assert_same_on_both_ranks(*tensors):
    for t in tensors:
        gathered = [torch.empty_like(t, device="cpu") for _ in range(2)]
        dist.all_gather(gathered, t.cpu())
        assert torch.equal(gathered[0], gathered[1])


def _assert_equal(actual, expected, context):
    for a, e in zip(actual, expected):
        assert torch.equal(a, e), context


def _check(rank: int, rendezvous: str, expandable: bool) -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    if expandable:
        # Before the first allocation, as PYTORCH_ALLOC_CONF would.
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    ca = CustomAllreduce(
        dist.group.WORLD, torch.device("cuda", rank), max_size=8 * 1024 * 1024
    )
    assert not ca.disabled
    fused = ca.tp2_all_reduce_fused_add_rms_norm
    try:
        weight = torch.empty(H, device="cuda", dtype=torch.float16)

        # Admission: decode-sized FP16 [1..16, 2048] only.
        def admitted(rows, hidden=H, dtype=torch.float16):
            t = torch.zeros(rows, hidden, device="cuda", dtype=dtype)
            w = torch.zeros(hidden, device="cuda", dtype=dtype)
            return ca.can_tp2_all_reduce_fused_add_rms_norm(t, None, t, w)

        assert admitted(1) and admitted(16)
        assert not admitted(17) and not admitted(2, hidden=4096)
        assert not admitted(2, dtype=torch.bfloat16)

        # Eager, and eagerly inside the capture context (graph warmup).
        for rows in ROWS:
            for sum2 in (False, True):
                for scale in (1.0, 0.125):
                    for seed, magnitude in enumerate((1.0, 64.0, 1.0, 2048.0)):
                        x = torch.empty(rows, H, device="cuda", dtype=torch.float16)
                        x_b = torch.empty_like(x) if sum2 else None
                        residual = torch.empty_like(x)
                        _fill(rank, seed, x, x_b, residual, weight, magnitude)
                        assert ca.can_tp2_all_reduce_fused_add_rms_norm(
                            x, x_b, residual, weight
                        )
                        context = (rows, sum2, scale, seed)
                        expected = _reference(ca, x, x_b, residual, weight, scale)
                        actual = fused(x, x_b, residual, weight, EPS, scale)
                        with ca.capture():
                            warmup = fused(x, x_b, residual, weight, EPS, scale)
                        torch.accelerator.synchronize()
                        _assert_equal(actual, expected, context)
                        _assert_equal(warmup, expected, context)
                        _assert_same_on_both_ranks(*actual)

        # CUDA graphs over operands allocated before capture (static
        # buffers; with expandable segments they are not IPC-exportable).
        for rows in (1, 2, 3, 5):
            x = torch.zeros(rows, H, device="cuda", dtype=torch.float16)
            x_b = torch.zeros_like(x)
            residual = torch.zeros_like(x)
            for sum2 in (False, True):
                for scale in (1.0, 0.125):
                    b = x_b if sum2 else None
                    args = (x, b, residual, weight, EPS, scale)
                    fused(*args)
                    _reference(ca, x, b, residual, weight, scale)
                    torch.accelerator.synchronize()
                    dist.barrier()
                    # FULL-style capture: interleaved with ordinary
                    # all-reduces, then the communicator registers the
                    # captured collective buffers.
                    graph = torch.cuda.CUDAGraph()
                    with ca.capture(), torch.cuda.graph(graph):
                        for _ in range(12):
                            out, res = fused(*args)
                            ref = _reference(ca, x, b, residual, weight, scale)
                    # A graph captured without the communicator's context.
                    bare = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(bare):
                        for _ in range(12):
                            bare_out = fused(*args)
                    for cycle in range(6):
                        context = (rows, sum2, scale, cycle)
                        _fill(rank, 100 + cycle, x, b, residual, weight, 4.0)
                        dist.barrier()
                        if rank == cycle % 2:
                            torch.cuda._sleep(20000)
                        for _ in range(3):
                            graph.replay()
                        torch.accelerator.synchronize()
                        _assert_equal((out, res), ref, context)
                        _assert_same_on_both_ranks(out, res)
                        dist.barrier()
                        if rank != cycle % 2:
                            torch.cuda._sleep(20000)
                        for _ in range(3):
                            bare.replay()
                        torch.accelerator.synchronize()
                        _assert_equal(bare_out, ref, context)
                        eager = fused(*args)
                        torch.accelerator.synchronize()
                        _assert_equal(eager, ref, context)
                    del graph, bare
    finally:
        ca.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("expandable", [False, True])
def test_sm70_tp2_all_reduce_fused_add_rms_norm(tmp_path, expandable):
    if torch.accelerator.device_count() < 2 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires two peer-capable V100 GPUs")
    import vllm._custom_ops as ops

    if not hasattr(
        ops._custom_ar_owner_namespace(), "tp2_all_reduce_fused_add_rms_norm"
    ):
        pytest.skip("the custom all-reduce extension lacks the fused collective")
    mp.spawn(
        _check,
        args=((tmp_path / "rendezvous").as_uri(), expandable),
        nprocs=2,
        join=True,
    )
