# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 vocab-parallel top-k candidate sampler against the reference sampler.

The reference is the regular full-vocabulary path: FP32 logits divided by the
temperature, ``apply_top_k_top_p_pytorch`` and ``random_sample``. TP ranks are
simulated by splitting the vocabulary into shards.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.sample.ops import sm70_topk_candidates as candidates
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p_pytorch,
    random_sample,
)

VOCAB = 150144


def _require_sm70():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 only")
    if not candidates.ops_available():
        pytest.fail("Build the SM70 sampler module (_sm70_sampler_C)")


def _metadata(rows, temperature, top_k, top_p, generators=None):
    return SimpleNamespace(
        all_random=True,
        all_greedy=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=torch.full((rows,), temperature, device="cuda"),
        temperature_cpu=(temperature,) * rows,
        top_k=torch.full((rows,), top_k, dtype=torch.int32, device="cuda"),
        top_k_cpu=(top_k,) * rows,
        top_p=(
            None
            if top_p is None
            else torch.full((rows,), top_p, dtype=torch.float32, device="cuda")
        ),
        top_p_cpu=None if top_p is None else (top_p,) * rows,
        generators=generators or {},
    )


def _gathered_wire(logits, tp_size):
    shard = logits.shape[1] // tp_size
    wires = [
        candidates.local_topk_candidates(
            logits[:, rank * shard : (rank + 1) * shard],
            vocab_start=rank * shard,
            num_valid=shard,
        )
        for rank in range(tp_size)
    ]
    assert all(wire is not None for wire in wires)
    return torch.cat(wires, dim=0)


def _reference(logits, metadata):
    scaled = logits.float().div_(metadata.temperature.unsqueeze(1))
    masked = apply_top_k_top_p_pytorch(scaled, metadata.top_k, metadata.top_p)
    probs = masked.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, metadata.generators), masked


def _logits(rows, kind, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, VOCAB, device="cuda", generator=gen) * 2.5
    if kind == "peaked":
        hot = torch.randint(0, VOCAB, (rows, 4), device="cuda", generator=gen)
        x.scatter_add_(1, hot, torch.full((rows, 4), 6.0, device="cuda"))
    elif kind == "ties":
        x = (x * 16).round() / 16
    return x.half()


def _offset():
    return torch.cuda.default_generators[torch.cuda.current_device()].get_offset()


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("cols", [75072, 37536, 1000, 131072])
@pytest.mark.parametrize("vocab_start", [0, 75072])
@torch.inference_mode()
def test_local_top32_matches_topk(rows, cols, vocab_start):
    _require_sm70()
    torch.manual_seed(cols + rows)
    full = (torch.randn(rows, cols + 3, device="cuda") * 3).half()
    # An odd row stride with two rows exercises the unaligned scalar loads.
    logits = full[:, :cols]
    num_valid = cols - 5
    wire = candidates.local_topk_candidates(
        logits, vocab_start=vocab_start, num_valid=num_valid
    )
    assert wire is not None and wire.shape == (rows, 64)
    expected = torch.topk(logits[:, :num_valid].float(), 32, dim=-1).values
    values = wire[:, :32]
    ids = wire[:, 32:].long() - vocab_start
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    assert bool(((ids >= 0) & (ids < num_valid)).all())
    torch.testing.assert_close(logits.float().gather(1, ids), values, rtol=0, atol=0)
    for row in range(rows):
        assert ids[row].unique().numel() == 32


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize(
    "temperature,top_k,top_p,kind",
    [
        (1.0, 20, 0.95, "flat"),
        (1.0, 20, 0.95, "peaked"),
        (0.6, 20, 0.8, "peaked"),
        (1.3, 5, None, "flat"),
        (1.0, 1, 0.95, "flat"),
        (1.0, 20, 0.95, "ties"),
    ],
)
@torch.inference_mode()
def test_same_token_as_reference(tp_size, rows, temperature, top_k, top_p, kind):
    """Same Philox state -> same token and same generator offset."""
    _require_sm70()
    trials = 64
    matches = 0
    for trial in range(trials):
        logits = _logits(rows, kind, seed=1000 * trial + rows)
        metadata = _metadata(rows, temperature, top_k, top_p)
        wire = _gathered_wire(logits, tp_size)

        torch.manual_seed(trial)
        sampled = candidates.sample_topk_candidates(
            wire, metadata, tp_size=tp_size, vocab_size=VOCAB
        )
        offset = _offset()
        torch.manual_seed(trial)
        expected, masked = _reference(logits, metadata)
        assert _offset() == offset
        assert sampled is not None and sampled.dtype == torch.int32
        kept = masked.gather(1, sampled.long().unsqueeze(1))
        assert bool(torch.isfinite(kept).all()), "sampled a masked token"
        matches += int(torch.equal(sampled.long(), expected))
    # Mismatches need an FP32 rounding tie in a cumulative probability or a
    # race score; ties beyond the 12 spare candidates of a rank are rarer.
    assert matches >= trials - (2 if kind == "ties" else 1)


@torch.inference_mode()
def test_request_generator_matches_reference():
    _require_sm70()
    for trial in range(32):
        logits = _logits(1, "peaked", seed=trial)
        wire = _gathered_wire(logits, 2)
        mine = torch.Generator(device="cuda").manual_seed(trial)
        ref = torch.Generator(device="cuda").manual_seed(trial)
        default_offset = _offset()
        sampled = candidates.sample_topk_candidates(
            wire,
            _metadata(1, 1.0, 20, 0.95, {0: mine}),
            tp_size=2,
            vocab_size=VOCAB,
        )
        assert _offset() == default_offset
        expected, _ = _reference(logits, _metadata(1, 1.0, 20, 0.95, {0: ref}))
        assert torch.equal(sampled.long(), expected)
        assert mine.get_offset() == ref.get_offset()


@torch.inference_mode()
def test_distribution_chi_square():
    """Draw frequencies follow the reference's masked softmax."""
    _require_sm70()
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = (torch.randn(1, VOCAB, device="cuda", generator=gen) * 1.2).half()
    metadata = _metadata(1, 1.0, 20, 0.95)
    wire = _gathered_wire(logits, 2)
    _, masked = _reference(logits, metadata)
    keep = torch.isfinite(masked[0]).nonzero().flatten()
    probs = torch.softmax(masked[0, keep].double(), dim=0)
    draws = 6000
    torch.manual_seed(123)
    counts = torch.zeros(VOCAB, dtype=torch.float64, device="cuda")
    for _ in range(draws):
        token = candidates.sample_topk_candidates(
            wire, metadata, tp_size=2, vocab_size=VOCAB
        )
        counts[token.long()] += 1
    assert counts.sum().item() == draws
    assert counts[keep].sum().item() == draws, "sampled outside the kept set"
    expected = probs * draws
    big = expected >= 5
    observed = counts[keep][big]
    chi2 = (((observed - expected[big]) ** 2) / expected[big]).sum().item()
    dof = int(big.sum().item()) - 1
    assert dof >= 5
    assert chi2 < dof + 6 * math.sqrt(2 * dof) + 10, (chi2, dof)
