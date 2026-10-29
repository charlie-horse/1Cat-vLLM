# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
__version__ = "1.2.0"

from flash_attn_v100.flash_attn_interface import (
    flash_attn_bhmd_func,
    flash_attn_decode_paged,
    flash_attn_decode_paged_sinks_available,
    flash_attn_decode_paged_wmma,
    flash_attn_decode_paged_xqa,
    flash_attn_decode_paged_xqa_available,
    flash_attn_decode_qk_scores,
    flash_attn_func,
    flash_attn_grouped_e4m3_fp32_available,
    flash_attn_grouped_e4m3_fp32_paged,
    flash_attn_grouped_verify_max_query_tokens,
    flash_attn_grouped_verify_paged,
    flash_attn_grouped_verify_request_major_abi_version,
    flash_attn_lse,
    flash_attn_prefill_paged,
    flash_attn_prefill_paged_bfla,
    flash_attn_prefill_paged_bhmd,
    flash_attn_prefill_paged_d256_bm32_allp_pair_scratch,
    flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3,
    flash_attn_prefill_paged_sinks_available,
    flash_attn_prefill_paged_splitkv,
    flash_attn_qk_scores,
    flash_attn_turboquant_decode_paged,
    flash_attn_turboquant_decode_paged_available,
    fp8_e4m3_paged_kv_to_fp16,
    fp8_e5m2_paged_kv_to_fp16,
)

# Learned attention sinks (``sinks=`` on flash_attn_prefill_paged and
# flash_attn_decode_paged) need both native sink entry points. Computed from
# the built extension, so an older build reports False; without an extension
# this package does not import, and readers fall back to False.
SUPPORTS_SINKS: bool = (
    flash_attn_prefill_paged_sinks_available()
    and flash_attn_decode_paged_sinks_available()
)

__all__ = [
    "SUPPORTS_SINKS",
    "fp8_e4m3_paged_kv_to_fp16",
    "fp8_e5m2_paged_kv_to_fp16",
    "flash_attn_decode_qk_scores",
    "flash_attn_decode_paged",
    "flash_attn_decode_paged_sinks_available",
    "flash_attn_decode_paged_xqa",
    "flash_attn_decode_paged_xqa_available",
    "flash_attn_decode_paged_wmma",
    "flash_attn_grouped_verify_max_query_tokens",
    "flash_attn_grouped_verify_paged",
    "flash_attn_grouped_verify_request_major_abi_version",
    "flash_attn_grouped_e4m3_fp32_paged",
    "flash_attn_grouped_e4m3_fp32_available",
    "flash_attn_turboquant_decode_paged",
    "flash_attn_turboquant_decode_paged_available",
    "flash_attn_bhmd_func",
    "flash_attn_func",
    "flash_attn_lse",
    "flash_attn_prefill_paged",
    "flash_attn_prefill_paged_sinks_available",
    "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch",
    "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3",
    "flash_attn_prefill_paged_bfla",
    "flash_attn_prefill_paged_bhmd",
    "flash_attn_prefill_paged_splitkv",
    "flash_attn_qk_scores",
]
