# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only admission of the SM70 vocab-parallel top-k candidate sampler."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
from vllm.v1.sample.ops.sm70_topk_candidates import (
    MAX_TOP_K,
    admission_reject_reason,
    argmax_invariant_reject_reason,
    metadata_reject_reason,
)


def _metadata(rows=1, **overrides):
    fields = dict(
        all_random=True,
        all_greedy=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=object(),
        temperature_cpu=(1.0,) * rows,
        top_k=object(),
        top_k_cpu=(20,) * rows,
        top_p=object(),
        top_p_cpu=(0.95,) * rows,
        generators={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("rows", [1, 2])
def test_served_defaults_are_admitted(rows):
    assert metadata_reject_reason(_metadata(rows), rows) is None


def test_other_temperatures_top_p_and_small_k_are_admitted():
    meta = _metadata(
        2, temperature_cpu=(0.6, 1.3), top_k_cpu=(1, MAX_TOP_K), top_p=None
    )
    assert metadata_reject_reason(meta, 2) is None


def test_one_seeded_request_is_admitted():
    assert metadata_reject_reason(_metadata(generators={0: object()}), 1) is None


@pytest.mark.parametrize(
    "rows,overrides,reason",
    [
        (3, {}, "batch_size"),
        (0, {}, "batch_size"),
        (1, {"all_random": False}, "sampling_mode"),
        (1, {"all_greedy": True}, "sampling_mode"),
        (1, {"max_num_logprobs": 0}, "logprobs"),
        (1, {"logprob_token_ids": {0: [1]}}, "logprob_token_ids"),
        (1, {"no_penalties": False}, "penalties"),
        (1, {"allowed_token_ids_mask": object()}, "allowed_token_ids"),
        (1, {"bad_words_token_ids": {0: [[1]]}}, "bad_words"),
        (1, {"top_k": None, "top_k_cpu": None}, "top_k"),
        (1, {"top_k_cpu": (MAX_TOP_K + 1,)}, "top_k"),
        (1, {"top_k_cpu": (150144,)}, "top_k"),
        (1, {"top_k_cpu": (0,)}, "top_k"),
        (2, {"top_k_cpu": (20, 150144)}, "top_k"),
        (1, {"temperature": None}, "temperature"),
        (2, {"generators": {0: object()}}, "generators"),
    ],
)
def test_rejections(rows, overrides, reason):
    meta = _metadata(max(rows, 1), **overrides)
    assert metadata_reject_reason(meta, rows) == reason


def _min_p(count):
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = count
    return processor


def _with_argmax_invariant(*processors):
    return SimpleNamespace(
        logitsprocs=SimpleNamespace(argmax_invariant=list(processors))
    )


@pytest.mark.parametrize(
    "processors,reason",
    [
        ((), None),
        ((_min_p(0),), None),
        ((_min_p(1),), "argmax_invariant_logits_processor"),
        ((object(),), "argmax_invariant_logits_processor"),
        ((_min_p(0), object()), "argmax_invariant_logits_processor"),
    ],
)
def test_argmax_invariant_processors(processors, reason):
    meta = _with_argmax_invariant(*processors)
    assert argmax_invariant_reject_reason(meta) == reason


def test_non_cuda_device_is_rejected_first():
    meta = _metadata(3)
    assert admission_reject_reason(meta, 3, torch.device("cpu")) == "non_cuda"
