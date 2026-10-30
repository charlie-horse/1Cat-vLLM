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

Known limit: a rank sends at most 32 candidates. When a rank's 32nd
candidate still reaches the k-th value, logits tied with it beyond those 32
cannot be sampled, while the reference would keep them (top-k keeps all
ties). With k <= 20 that needs more than 12 extra exact FP16 ties on one
rank. The kernel counts such rows on the device; ``sample_topk_candidates``
reads the count back asynchronously every ``_OVERFLOW_CHECK_INTERVAL``
calls and logs a warning when it grows.
"""

from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.sample.metadata import SamplingMetadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

logger = init_logger(__name__)

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
_MAX_LOCAL_VOCAB = 128 * _TILE
_MAX_TP = 4
_OVERFLOW_CHECK_INTERVAL = 64


@cache
def ops_available() -> bool:
    import vllm._sm70_ops  # noqa: F401  (loads the sampler sidecar library)

    return hasattr(torch.ops._C, "sm70_local_top32_half") and hasattr(
        torch.ops._C, "sm70_sample_topk_candidates"
    )


@cache
def _is_sm70(device_index: int | None) -> bool:
    return current_platform.is_device_capability(70, device_id=device_index)


class _OverflowMonitor:
    """Device counter of rows whose ties may exceed a rank's candidates."""

    def __init__(self, device: torch.device):
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.mirror = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.calls = 0
        self.reported = 0

    def after_call(self) -> None:
        self.calls += 1
        if self.calls % _OVERFLOW_CHECK_INTERVAL:
            return
        # The mirror holds the count copied asynchronously one interval ago.
        seen = int(self.mirror[0])
        if seen > self.reported:
            logger.warning(
                "SM70 top-k candidate sampler: %d sampled rows had more logits "
                "tied with the k-th value than one rank's 32 candidates hold; "
                "those ties were not sampleable.",
                seen,
            )
            self.reported = seen
        self.mirror.copy_(self.counter, non_blocking=True)


_OVERFLOW_MONITORS: dict[torch.device, _OverflowMonitor] = {}


def _overflow_monitor(device: torch.device) -> _OverflowMonitor:
    monitor = _OVERFLOW_MONITORS.get(device)
    if monitor is None:
        monitor = _OVERFLOW_MONITORS[device] = _OverflowMonitor(device)
    return monitor


def overflow_count(device: torch.device) -> int:
    """Synchronously read the device's tie-overflow count (tests, debugging)."""
    return int(_overflow_monitor(device).counter.item())


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


def argmax_invariant_reject_reason(
    sampling_metadata: SamplingMetadata,
) -> str | None:
    """Only an inactive min-p processor may be among the argmax-invariant ones."""
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        if isinstance(processor, MinPLogitsProcessor) and not processor.min_p_count:
            continue
        return "argmax_invariant_logits_processor"
    return None


def admission_reject_reason(
    sampling_metadata: SamplingMetadata, num_reqs: int, device: torch.device
) -> str | None:
    """Device, batch and kernel admission of the sampler; None means admitted.

    The runner checks its own state (spec decode, pooling, structured output,
    thinking budget, non-argmax-invariant logits processors) before this.
    """
    if device.type != "cuda":
        return "non_cuda"
    if torch.cuda.get_device_capability(device) != (7, 0):
        return "non_sm70"
    reason = metadata_reject_reason(sampling_metadata, num_reqs)
    if reason is not None:
        return reason
    reason = argmax_invariant_reject_reason(sampling_metadata)
    if reason is not None:
        return reason
    if not ops_available():
        return "missing_ops"
    return None


def lm_head_topk_candidates(
    logits_processor: "LogitsProcessor",
    lm_head: "VocabParallelEmbedding",
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Rank-local top-32 logits of every TP rank for top-k sampling.

    Returns rank-major FP32 rows ``[tp_size * rows, 64]`` (32 values, then
    32 global token ids) for ``sample_topk_candidates``, or None before any
    collective when the shard does not qualify. The decision depends only on
    shapes and configuration, so every rank takes the same branch.
    """
    shard = lm_head.shard_indices
    if (
        logits_processor.soft_cap is not None
        or logits_processor.scale != 1.0
        or shard.num_added_elements != 0
        or lm_head.weight.dtype != torch.float16
    ):
        return None
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=None)
    wire = local_topk_candidates(
        logits,
        vocab_start=shard.org_vocab_start_index,
        num_valid=shard.num_org_elements,
    )
    if wire is not None and get_tensor_model_parallel_world_size() > 1:
        wire = tensor_model_parallel_all_gather(wire, dim=0)
    return wire


def sample_lm_head_candidates(
    logits_processor: "LogitsProcessor",
    lm_head: "VocabParallelEmbedding",
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    vocab_size: int,
) -> torch.Tensor | None:
    """Sample admitted rows from the LM head's vocab-parallel candidates.

    Returns int32 token ids [rows], or None when the shard or tensors do not
    qualify (the caller then runs the full-vocabulary sampler).
    """
    wire = lm_head_topk_candidates(logits_processor, lm_head, hidden_states)
    if wire is None:
        return None
    sampled = sample_topk_candidates(
        wire,
        sampling_metadata,
        tp_size=get_tensor_model_parallel_world_size(),
        vocab_size=vocab_size,
    )
    if sampled is not None:
        logger.info_once("SM70 top-k candidate sampler enabled.")
    return sampled


def local_topk_candidates(
    logits: torch.Tensor, *, vocab_start: int, num_valid: int
) -> torch.Tensor | None:
    """Return this rank's [rows, 64] FP32 wire rows, or None if unsupported.

    ``logits`` are the rank's raw FP16 logits (row-major, any row stride);
    columns >= ``num_valid`` are vocabulary padding and never selected.
    """
    if not shard_supported(logits, vocab_start=vocab_start, num_valid=num_valid):
        return None
    return torch.ops._C.sm70_local_top32_half(logits, vocab_start, num_valid)


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
        and _is_sm70(logits.device.index)
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

    The caller admits the batch with ``metadata_reject_reason`` first.
    Returns int32 token ids [rows], or None when the tensors do not qualify
    (the caller then runs the full-vocabulary sampler).
    """
    temperature = sampling_metadata.temperature
    top_k = sampling_metadata.top_k
    top_p = sampling_metadata.top_p
    if (
        not 1 <= tp_size <= _MAX_TP
        or wire.dtype != torch.float32
        or wire.ndim != 2
        or wire.shape[1] != WIRE_WIDTH
        or wire.shape[0] % tp_size != 0
        or not wire.is_contiguous()
        or not LOCAL_K <= vocab_size <= 1 << 24
        or temperature is None
        or top_k is None
    ):
        return None
    rows = wire.shape[0] // tp_size
    generators = sampling_metadata.generators
    if (
        not 1 <= rows <= _MAX_KERNEL_ROWS
        or temperature.dtype != torch.float32
        or temperature.shape != (rows,)
        or not temperature.is_contiguous()
        or top_k.dtype != torch.int32
        or top_k.shape != (rows,)
        or not top_k.is_contiguous()
        or (
            top_p is not None
            and (
                top_p.dtype != torch.float32
                or top_p.shape != (rows,)
                or not top_p.is_contiguous()
            )
        )
        or (generators and (rows != 1 or len(generators) != 1))
    ):
        return None
    generator = next(iter(generators.values()), None) if generators else None
    monitor = _overflow_monitor(wire.device)
    sampled = torch.ops._C.sm70_sample_topk_candidates(
        wire,
        temperature,
        top_k,
        top_p,
        generator,
        vocab_size,
        tp_size,
        monitor.counter,
    )
    monitor.after_call()
    return sampled
