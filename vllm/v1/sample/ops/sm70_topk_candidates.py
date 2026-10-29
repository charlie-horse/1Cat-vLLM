# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch top-k/top-p sampling from vocab-parallel logit candidates.

Each TP rank selects the 32 largest logits of its FP16 shard; the ranks
exchange only those (value, token) rows, and one kernel applies temperature,
top-k (keeping ties at the k-th value), top-p and the exponential race of
``random_sample`` to the gathered candidates. With k <= 20 this is the
reference sampler (``apply_top_k_top_p_pytorch`` + ``random_sample``)
restricted to the only tokens it can return. The kernel draws the exact
exponentials ``q.exponential_()`` would draw for those tokens and advances
the generator by the same offset, so the same RNG state samples the same
token.

The route replaces the full-vocabulary logits all-gather, two
full-vocabulary softmaxes and the radix sort of the eager sampler.
"""

from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.v1.sample.metadata import SamplingMetadata

# Candidates per TP rank; the kernel's wire row holds 32 values + 32 ids.
LOCAL_K = 32
WIRE_WIDTH = 2 * LOCAL_K
# Largest admitted top-k: leaves 12 spare candidates per rank for logits
# tied with the k-th value, which the reference keeps.
MAX_TOP_K = 20
# Request rows admitted by the runner (decode at batch 1-2).
MAX_ROWS = 2
_MAX_KERNEL_ROWS = 8
_TILE = 2048
_MAX_LOCAL_VOCAB = 64 * _TILE
_MAX_TP = 4


@cache
def ops_available() -> bool:
    import vllm._sm70_ops  # noqa: F401  (loads the sampler sidecar library)

    return hasattr(torch.ops._C, "sm70_local_top32_half_wire") and hasattr(
        torch.ops._C, "sm70_sample_topk_candidates_out"
    )


def metadata_reject_reason(
    sampling_metadata: SamplingMetadata, num_reqs: int
) -> str | None:
    """CPU-only admission of the request batch; None means admitted.

    Penalties, logprobs, masks, logits processors and greedy rows need the
    full-vocabulary path. The runner checks its own state (spec decode,
    structured output, logits processors) separately.
    """
    if not 1 <= num_reqs <= MAX_ROWS:
        return "batch_size"
    if not sampling_metadata.all_random or sampling_metadata.all_greedy:
        return "sampling_mode"
    if sampling_metadata.max_num_logprobs is not None:
        return "logprobs"
    if sampling_metadata.logprob_token_ids:
        return "logprob_token_ids"
    if not sampling_metadata.no_penalties:
        return "penalties"
    if sampling_metadata.allowed_token_ids_mask is not None:
        return "allowed_token_ids"
    if sampling_metadata.bad_words_token_ids:
        return "bad_words"
    top_k_cpu = sampling_metadata.top_k_cpu
    if (
        sampling_metadata.top_k is None
        or top_k_cpu is None
        or len(top_k_cpu) != num_reqs
        or not all(1 <= k <= MAX_TOP_K for k in top_k_cpu)
    ):
        return "top_k"
    temperature_cpu = sampling_metadata.temperature_cpu
    if (
        sampling_metadata.temperature is None
        or temperature_cpu is None
        or len(temperature_cpu) != num_reqs
    ):
        return "temperature"
    generators = sampling_metadata.generators
    if generators and (num_reqs != 1 or len(generators) != 1):
        # Mixed seeded/unseeded rows draw from several generators.
        return "generators"
    return None


def local_topk_candidates(
    logits: torch.Tensor, *, vocab_start: int, num_valid: int
) -> torch.Tensor | None:
    """Return this rank's [rows, 64] FP32 wire rows, or None if unsupported.

    ``logits`` are the rank's raw FP16 logits; columns >= ``num_valid`` are
    vocabulary padding and never selected.
    """
    if not shard_supported(logits, vocab_start=vocab_start, num_valid=num_valid):
        return None
    rows, cols = logits.shape
    tiles = (cols + _TILE - 1) // _TILE
    partial = torch.empty(
        (rows, tiles * LOCAL_K), dtype=torch.int32, device=logits.device
    )
    wire = torch.empty((rows, WIRE_WIDTH), dtype=torch.float32, device=logits.device)
    torch.ops._C.sm70_local_top32_half_wire(
        logits, partial, wire, vocab_start, num_valid
    )
    return wire


def shard_supported(logits: torch.Tensor, *, vocab_start: int, num_valid: int) -> bool:
    return (
        logits.is_cuda
        and logits.dtype == torch.float16
        and logits.ndim == 2
        and logits.stride(1) == 1
        and 1 <= logits.shape[0] <= _MAX_KERNEL_ROWS
        and LOCAL_K <= num_valid <= logits.shape[1] <= _MAX_LOCAL_VOCAB
        and vocab_start >= 0
        and vocab_start + logits.shape[1] <= 1 << 24
        and current_platform.is_device_capability(70, device_id=logits.device.index)
        and ops_available()
    )


def sample_topk_candidates(
    wire: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    tp_size: int,
    vocab_size: int,
) -> torch.Tensor | None:
    """Sample one token per row from rank-major gathered wire rows.

    Returns int32 token ids [rows], or None when the inputs do not qualify
    (the caller then runs the full-vocabulary sampler).
    """
    if (
        not 1 <= tp_size <= _MAX_TP
        or wire.dtype != torch.float32
        or wire.ndim != 2
        or wire.shape[1] != WIRE_WIDTH
        or wire.shape[0] % tp_size != 0
        or not wire.is_contiguous()
        or not LOCAL_K <= vocab_size <= 1 << 24
    ):
        return None
    rows = wire.shape[0] // tp_size
    if metadata_reject_reason(sampling_metadata, rows) is not None:
        return None
    temperature = sampling_metadata.temperature
    top_k = sampling_metadata.top_k
    top_p = sampling_metadata.top_p
    assert temperature is not None and top_k is not None
    if (
        temperature.dtype != torch.float32
        or temperature.shape != (rows,)
        or top_k.dtype != torch.int32
        or top_k.shape != (rows,)
        or (
            top_p is not None
            and (top_p.dtype != torch.float32 or top_p.shape != (rows,))
        )
    ):
        return None
    generator = next(iter(sampling_metadata.generators.values()), None)
    sampled = torch.empty((rows,), dtype=torch.int32, device=wire.device)
    torch.ops._C.sm70_sample_topk_candidates_out(
        sampled,
        wire,
        temperature.contiguous(),
        top_k.contiguous(),
        None if top_p is None else top_p.contiguous(),
        generator,
        vocab_size,
        tp_size,
    )
    return sampled
