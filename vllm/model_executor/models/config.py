# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import CacheConfig, ModelConfig, VllmConfig


logger = init_logger(__name__)


class VerifyAndUpdateConfig:
    @staticmethod
    def apply_runtime_defaults(vllm_config, defaults, phase: str, *, is_sm70: bool):
        from vllm.model_executor.models.runtime_defaults import apply_runtime_defaults

        return apply_runtime_defaults(vllm_config, defaults, phase, is_sm70=is_sm70)

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        return

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        return


def fla_schedule_family(model_config) -> str | None:
    """Existing FLA consumers; this describes ownership, not new admission."""
    if any(
        getattr(getattr(model_config, name, None), "linear_key_head_dim", None)
        is not None
        for name in ("hf_text_config", "hf_config")
    ):
        return "gdn"
    if getattr(model_config, "architecture", None) in {
        "KimiLinearForCausalLM",
        "BailingMoeV3ForCausalLM",
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "Glm5NextMTPModel",
    }:
        return "kda"
    return None


def effective_layer_policy_hash_fields(model, speculative, fields):
    """Omit policy families with no consumer in the model contract.

    This only selects cache factors; provider admission and dynamic dispatch
    remain unchanged. Unknown/plugin architectures conservatively keep factors.
    """
    selected = set(fields)
    architecture = getattr(model, "architecture", None)
    if architecture is not None and architecture not in {
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "Glm5NextMTPModel",
    }:
        selected.difference_update(
            {
                "glm_exact_kda_gemv",
                "glm_cublaslt",
                "glm_fused_fg_b",
                "glm_pp_mhc_materialize",
            }
        )
    text = getattr(model, "hf_text_config", None)
    if text is not None and getattr(text, "hc_count", 1) <= 1:
        selected.discard("hc_prefill_sequence_parallel")
        selected.discard("hc_mtp_prefill_sequence_parallel")
        selected.discard("hc_prefill_peer_norm")
    if text is not None and not getattr(text, "ple_layer_ids", None):
        selected.discard("ple_spec_conv")
    # A speculative draft can have a different quantization format. Keep the
    # shared loader inputs until its own worker config describes that format.
    if model is not None and speculative is None:
        quantization = getattr(model, "quantization", None)
        for field, formats in (
            ("gptq_turbomind", ("gptq", "gptq_marlin")),
            ("compressed_tensors_turbomind", ("compressed-tensors",)),
            ("mxfp4_turbomind", ("mxfp4",)),
        ):
            if quantization not in formats:
                selected.discard(field)
    return tuple(field for field in fields if field in selected)


def sm70_flash_next_batch_qualified(vllm_config: "VllmConfig") -> bool:
    """Batch quality covers ordinary decode and MTP; other proposers await gates."""
    speculative = vllm_config.speculative_config
    return speculative is None or getattr(speculative, "method", None) == "mtp"


def sm70_fp8_serialized_pipeline_qualified(vllm_config: "VllmConfig") -> bool:
    """Retain the measured PP2/TP4 no-spec single-request validation boundary.

    QPN8 and reversible-scale M1 use the same qualification. They do not
    require this scheduling layout to execute; wider layouts need paired
    output and concurrent workspace tests before lifting the boundary.
    """
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    return bool(
        parallel_config.pipeline_parallel_size == 2
        and parallel_config.tensor_parallel_size == 4
        and scheduler_config.max_num_seqs == 1
        and not getattr(parallel_config, "enable_dbo", False)
        and int(getattr(parallel_config, "ubatch_size", 0)) <= 1
        and getattr(vllm_config, "speculative_config", None) is None
    )


def sm70_awq_prefill_projection_qualified(prefix: str) -> bool:
    """Retain the measured AWQ exact-dense projection roles.

    This is a quality boundary, not a shape or TP requirement. Renamed and
    additional roles need a paired kernel/quality check before admission.
    """
    return prefix.rsplit(".", 1)[-1] in {
        "gate_up_proj",
        "down_proj",
        "in_proj_qkvz",
        "out_proj",
        "o_proj",
    }


def sm70_dflash2_nvfp4_qualified(vllm_config: "VllmConfig") -> bool:
    """Retained QPN2 whole-workload qualification, not operator capability.

    TP, KV dtype and service concurrency are already outside this boundary.
    Keep the draft selector/state contract until a broader quality gate passes.
    """
    parallel = vllm_config.parallel_config
    spec = vllm_config.speculative_config
    draft = getattr(spec, "draft_model_config", None)
    hf = getattr(draft, "hf_config", None)
    dflash = getattr(hf, "dflash_config", None) or {}
    selector = (
        int(dflash.get("selector_top_k", 0) or 0) if isinstance(dflash, Mapping) else 0
    )
    return bool(
        getattr(spec, "method", None) == "dflash"
        and int(getattr(spec, "num_speculative_tokens", 0) or 0) == 7
        and selector == 16
        and parallel.pipeline_parallel_size == 1
        and not parallel.enable_dbo
        and int(parallel.ubatch_size or 0) <= 1
    )


def sm70_nvfp4_projection_qualified(prefix: str) -> bool:
    """Roles covered by the existing real-weight QPN2 quality audit.

    This whitelist is a temporary qualification boundary. Local K/N checks
    belong to Qpn2NvFp4LinearKernel.can_implement, independent of these names.
    """
    return prefix.rsplit(".", 1)[-1] in {
        "in_proj_qkvz",
        "qkv_proj",
        "out_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    }


def sm70_nvfp4_gate_up_qualified(layer) -> bool:
    """Retained dense gated layout audited with the 5120-wide model."""
    return bool(
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "gate_up_proj"
        and getattr(layer, "input_size_per_partition", 0) == 5120
        and getattr(layer, "output_size_per_partition", 0) > 0
        and layer.output_size_per_partition % 64 == 0
        and getattr(layer, "logical_widths", None)
        == [layer.output_size_per_partition // 2] * 2
    )


def sm70_nvfp4_down_qualified(layer) -> bool:
    """Retained QPN4 down projection audited with the 5120-wide model."""
    return bool(
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "down_proj"
        and getattr(layer, "input_size_per_partition", 0) > 0
        and layer.input_size_per_partition % 128 == 0
        and getattr(layer, "output_size_per_partition", 0) == 5120
    )


def sm70_nvfp4_qpn4_qualified(vllm_config: "VllmConfig") -> bool:
    """QPN4's quality audit covers a single sequence without speculation."""
    scheduler = getattr(vllm_config, "scheduler_config", None)
    return (
        int(getattr(scheduler, "max_num_seqs", 1)) == 1
        and vllm_config.speculative_config is None
    )


# The platform treats both Volta capabilities as SM70 (platforms/cuda.py).
_SM70_CAPABILITIES = ((7, 0), (7, 2))

FLASH_V100_SINK_HEAD_SIZES = (128,)
"""Head sizes whose generic Flash-V100 paged prefill and decode apply sinks.

The D256-specialized kernels, XQA, the FP8 prefill bridge and the dense
prefill kernels have no sink term. Widen this only with a paired
kernel/quality check.
"""


def flash_v100_supports_sinks() -> bool:
    """Whether the installed Flash-V100 extension applies learned sinks.

    Reads the ``flash_attn_v100.SUPPORTS_SINKS`` capability flag. The native
    extension is optional (``flash_v100/ops.py`` imports it lazily), so a
    missing or unloadable build reports no support.
    """
    try:
        import flash_attn_v100
    except (ImportError, OSError):
        return False
    return bool(getattr(flash_attn_v100, "SUPPORTS_SINKS", False))


def sm70_deployment(vllm_config: "VllmConfig") -> bool:
    """Whether every participating CUDA device is SM70.

    The attention backend choice is engine-wide, so a stage on another
    architecture keeps the platform's own per-device selection.
    """
    from vllm.platforms import current_platform
    from vllm.platforms import runtime_defaults as _platform_defaults

    if not current_platform.is_cuda():
        return False
    devices = _platform_defaults._participating_cuda_device_ids(vllm_config)
    return bool(devices) and all(
        any(
            current_platform.is_device_capability(capability, device_id=device_id)
            for capability in _SM70_CAPABILITIES
        )
        for device_id in devices
    )


class DeepseekV32ForCausalLM(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config

        # Mirror the check in vllm/model_executor/models/deepseek_v2.py
        is_v32 = hasattr(hf_config, "index_topk")
        assert is_v32

        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
            logger.info("Using bfloat16 kv-cache for DeepSeekV3.2")


class Ernie4_5_VLMoeForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        # Ernie4.5-VL conditionally executes text/vision MoE branches, so
        # fast_moe_cold_start can silently produce incorrect execution order.
        vllm_config.compilation_config.fast_moe_cold_start = False


def _flash_v100_sink_route_rejection(vllm_config: "VllmConfig") -> str | None:
    """Why the Flash-V100 sink route cannot serve this engine, or None.

    Mirrors the per-layer admission in ``flash_v100/impl.py`` so that an
    engine is never routed to a backend whose layers would then fail closed.
    """
    import torch

    from vllm.v1.attention.kv_codecs import FP16, resolve_kv_codec

    flash_v100 = getattr(vllm_config.attention_config, "flash_v100", None)
    if getattr(flash_v100, "enabled", None) is False:
        return "Flash-V100 automatic selection is disabled"
    if not flash_v100_supports_sinks():
        return "the Flash-V100 extension does not report SUPPORTS_SINKS"
    model_config = vllm_config.model_config
    if model_config.dtype != torch.float16:
        return f"Flash-V100 computes in float16, not {model_config.dtype}"
    head_dim = getattr(model_config.hf_text_config, "head_dim", None)
    if head_dim not in FLASH_V100_SINK_HEAD_SIZES:
        return f"Flash-V100 has no sink route for head_dim={head_dim}"
    cache_dtype = vllm_config.cache_config.cache_dtype
    if resolve_kv_codec(cache_dtype) is not FP16:
        return (
            "the Flash-V100 sink route needs an FP16 KV cache, not "
            f"kv_cache_dtype={cache_dtype!r}"
        )
    if vllm_config.speculative_config is not None:
        return "the Flash-V100 speculative verification routes do not apply sinks"
    return None


def _sm70_tp2_allreduce_add_rms_rejection(vllm_config: "VllmConfig") -> str | None:
    """Why Flint does not request the TP2 all-reduce + RMSNorm fusion, or None.

    Called only for SM70 deployments. The route needs TP2 without pipeline
    stages, FP16, hidden size 2048 and the custom all-reduce. Its kill switch
    (VLLM_SM70_FLINT_AR_RMSNORM, parallel_config.communication.ar_add_rms_tp2)
    is resolved here, so =0 leaves the pass configuration untouched.
    """
    import torch

    from vllm.distributed.device_communicators.collective_provider import (
        TP2_AR_ADD_RMS_HIDDEN_SIZE,
    )

    parallel_config = vllm_config.parallel_config
    if parallel_config.tensor_parallel_size != 2:
        return f"tensor_parallel_size={parallel_config.tensor_parallel_size}"
    if parallel_config.pipeline_parallel_size != 1:
        return f"pipeline_parallel_size={parallel_config.pipeline_parallel_size}"
    if parallel_config.disable_custom_all_reduce:
        return "the custom all-reduce is disabled"
    model_config = vllm_config.model_config
    if model_config.dtype != torch.float16:
        return f"dtype={model_config.dtype}"
    if model_config.get_hidden_size() != TP2_AR_ADD_RMS_HIDDEN_SIZE:
        return f"hidden_size={model_config.get_hidden_size()}"
    if vllm_config.compilation_config.pass_config.fuse_allreduce_rms is False:
        return "fuse_allreduce_rms is disabled"
    if not parallel_config.communication.early_value("ar_add_rms_tp2"):
        return "VLLM_SM70_FLINT_AR_RMSNORM=0"
    return None


class FlintModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        if not sm70_deployment(vllm_config):
            return
        FlintModelConfig._configure_sm70_allreduce_add_rms(vllm_config)
        FlintModelConfig._select_sm70_attention_backend(vllm_config)

    @staticmethod
    def _configure_sm70_allreduce_add_rms(vllm_config: "VllmConfig") -> None:
        """Fuse every hidden-state all-reduce with the norm that follows it.

        On SM70 TP2 each layer boundary runs the custom one-stage all-reduce
        and then fused_add_rms_norm. The all-reduce + RMSNorm compile pass
        replaces each pair with one collective whose epilogue does the
        residual add and the RMSNorm, bitwise equal to the pair.
        """
        reason = _sm70_tp2_allreduce_add_rms_rejection(vllm_config)
        if reason is not None:
            logger.debug(
                "Flint on SM70: all-reduce + fused_add_rms_norm fusion is off (%s).",
                reason,
            )
            return
        pass_config = vllm_config.compilation_config.pass_config
        pass_config.fuse_allreduce_rms = True
        pass_config.fuse_allreduce_add_rms_tp2 = True
        logger.info(
            "Flint on SM70 TP2: requesting the fusion of each custom all-reduce "
            "with the residual add and RMSNorm that follow it "
            "(VLLM_SM70_FLINT_AR_RMSNORM=0 disables)."
        )

    @staticmethod
    def _select_sm70_attention_backend(vllm_config: "VllmConfig") -> None:
        """Select an SM70 attention backend that applies the learned sinks.

        Every Flint layer has learned attention sinks. On SM70 the platform
        prefers FLASH_ATTN_V100, which inherits ``supports_sink()`` from the
        Triton backend but admits sinks only on its generic D128 paged
        prefill/decode, and only when the extension reports
        ``flash_attn_v100.SUPPORTS_SINKS``. Otherwise every layer would select
        FLASH_ATTN_V100 and fail at its first forward, so TRITON_ATTN is used
        instead. An explicitly chosen attention backend is left unchanged.
        """
        attention_config = vllm_config.attention_config
        if attention_config.backend is not None:
            return

        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.attention.kv_codecs import FP8_E4M3, resolve_kv_codec

        reason = _flash_v100_sink_route_rejection(vllm_config)
        if reason is None:
            attention_config.backend = AttentionBackendEnum.FLASH_ATTN_V100
            logger.info(
                "Flint on SM70: using FLASH_ATTN_V100, whose generic D128 paged "
                "prefill and decode apply the learned attention sinks."
            )
            return

        attention_config.backend = AttentionBackendEnum.TRITON_ATTN
        logger.info(
            "Flint on SM70: using TRITON_ATTN for the learned attention sinks "
            "because %s.",
            reason,
        )
        cache_dtype = vllm_config.cache_config.cache_dtype
        if resolve_kv_codec(cache_dtype) is FP8_E4M3:
            logger.warning(
                "Flint on SM70 runs TRITON_ATTN with an FP8 E4M3 KV cache "
                "(kv_cache_dtype=%r), which Triton cannot compile on SM70. "
                "Use --kv-cache-dtype auto (FP16 KV cache).",
                cache_dtype,
            )


class Gemma3TextModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        hf_config = model_config.hf_config
        hf_config.is_causal = not hf_config.use_bidirectional_attention


class Gemma4Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Force unified attention backend for models with heterogeneous
        head dimensions.

        Some Gemma4 variants use different head dimensions for
        sliding window (head_dim) vs full attention (global_head_dim) layers.
        When global_head_dim > 256, FlashAttention rejects those layers
        (head_size <= 256 kernel limit), causing vLLM to select a different
        backend for each layer type. This mixed-backend execution produces
        numerical divergence and output corruption.

        The fix detects heterogeneous head dimensions from the model config
        and forces TRITON_ATTN (which has no head_size ceiling) for all
        layers when the user hasn't explicitly chosen a backend.

        TODO: Heterogeneous head_sizes (head_dim != global_head_dim)
        require NixlConnector changes to support per-layer KV transfer
        with different head dimensions for prefill-decode disaggregation.
        """
        hf_text_config = vllm_config.model_config.hf_text_config
        head_dim = getattr(hf_text_config, "head_dim", None)
        global_head_dim = getattr(hf_text_config, "global_head_dim", None)

        # Only force Triton when head dimensions actually differ AND the
        # larger one exceeds FlashAttention's kernel limit (head_size <= 256).
        # This avoids unnecessary backend forcing on smaller models where
        # the config carries global_head_dim but all layers can still use
        # the same FA backend.
        max_head_dim = max(head_dim or 0, global_head_dim or 0)
        if (
            head_dim is not None
            and global_head_dim is not None
            and head_dim != global_head_dim
            and max_head_dim > 256
            and vllm_config.attention_config.backend is None
        ):
            from vllm.v1.attention.backends.registry import (
                AttentionBackendEnum,
            )

            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "Gemma4 model has heterogeneous head dimensions "
                "(head_dim=%d, global_head_dim=%d). Forcing TRITON_ATTN "
                "backend to prevent mixed-backend numerical divergence.",
                head_dim,
                global_head_dim,
            )


class DeepseekV4ForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        if quant_config is not None and quant_config.get("quant_method") == "fp8":
            model_type = getattr(model_config.hf_config, "model_type", None)
            if model_type == "deepseek_v4":
                model_config.hf_config.quantization_config["quant_method"] = (
                    "deepseek_v4_fp8"
                )

        hf_text_quant_config = getattr(
            model_config.hf_text_config, "quantization_config", None
        )
        if (
            hf_text_quant_config is not None
            and hf_text_quant_config.get("quant_method") == "fp8"
        ):
            model_type = getattr(model_config.hf_text_config, "model_type", None)
            if model_type == "deepseek_v4":
                model_config.hf_text_config.quantization_config["quant_method"] = (
                    "deepseek_v4_fp8"
                )


class GptOssForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        if quant_config is not None and quant_config.get("quant_method") == "mxfp4":
            model_config.hf_config.quantization_config["quant_method"] = "gpt_oss_mxfp4"

        hf_text_quant_config = getattr(
            model_config.hf_text_config, "quantization_config", None
        )
        if (
            hf_text_quant_config is not None
            and hf_text_quant_config.get("quant_method") == "mxfp4"
        ):
            model_config.hf_text_config.quantization_config["quant_method"] = (
                "gpt_oss_mxfp4"
            )

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        structured_outputs_config = vllm_config.structured_outputs_config
        if structured_outputs_config.reasoning_parser == "":
            structured_outputs_config.reasoning_parser = "openai_gptoss"

        # Increase the max capture size from 512 to 1024 for performance.
        # NOTE(woosuk): This will increase the number of CUDA graphs
        # from 67 to 83.
        compilation_config = vllm_config.compilation_config
        # Only override when the user has not set either of
        # cudagraph_capture_sizes or max_cudagraph_capture_size.
        if (
            compilation_config.cudagraph_capture_sizes is None
            and compilation_config.max_cudagraph_capture_size is None
        ):
            compilation_config.max_cudagraph_capture_size = 1024
            logger.info(
                "Overriding max cuda graph capture size to %d for performance.", 1024
            )


class GteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        assert config.__class__.__name__ == "NewConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
        config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": config.max_position_embeddings,
            "rope_parameters": config.rope_parameters,
        }


class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Perform early validation and setup for hybrid attention/mamba models.

        Block size alignment with mamba page sizes is handled later by
        Platform.update_block_size_for_backend(), which runs after model
        layers are constructed and the attention backend is known.

        Args:
            vllm_config: vLLM Config
        """
        cache_config = vllm_config.cache_config

        # Disable calculate_kv_scales for hybrid models: uninitialized
        # recurrent state corrupts scales during the calibration pass.
        # See issue: https://github.com/vllm-project/vllm/issues/37554

        if cache_config.calculate_kv_scales:
            logger.warning(
                "Disabling calculate_kv_scales for hybrid model '%s'. "
                "Hybrid models with recurrent layers (GDN, Mamba, SSM) "
                "produce unreliable KV cache scales during the "
                "calibration pass because recurrent state is "
                "uninitialized. Using default scale of 1.0 instead.",
                vllm_config.model_config.model,
            )
            cache_config.calculate_kv_scales = False

        # Enable FULL_AND_PIECEWISE by default
        MambaModelConfig.verify_and_update_config(vllm_config)


class JambaForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        pooler_config = model_config.pooler_config
        if pooler_config.use_activation is None:
            pooler_config.use_activation = False


class JinaForRankingConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        model_config.hf_config.embedding_size = 512


class JinaRobertaModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        if config.position_embedding_type == "rotary":
            assert config.__class__.__name__ == "XLMRobertaFlashConfig"

            head_dim = config.hidden_size // config.num_attention_heads
            max_position = config.max_position_embeddings
            # Jina-embeddings-v3 has max_position_embeddings=8194, which will cause
            # out-of-bound index issue at RoPE for long prompts with torch.compile,
            # because it can't be divided by triton num_warps(default=4 or 8).
            # To deal with this, we increase max_position to multiple of n_warps,
            # so that triton kernel won't hit out-of-bound index in RoPE cache.
            if not model_config.enforce_eager:
                max_position = round_up(max_position, 8)

            rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
            config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim

            config.rotary_kwargs = {
                "head_size": head_dim,
                "max_position": max_position,
                "rope_parameters": config.rope_parameters,
            }


class JinaVLForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config
        config.num_labels = 1
        pooler_config = model_config.pooler_config
        if pooler_config.logit_mean is None:
            pooler_config.logit_mean = 2.65


class LlamaBidirectionalConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        from vllm.config.pooler import SequencePoolingType

        hf_config = model_config.hf_config
        hf_config.is_causal = False

        pooling_type_map: dict[str, SequencePoolingType] = {
            "avg": "MEAN",
            "cls": "CLS",
            "last": "LAST",
        }

        pooling_type = pooling_type_map.get(hf_config.pooling)
        if pooling_type is None:
            raise ValueError(f"pool_type {hf_config.pooling!r} not supported")

        model_config.pooler_config.seq_pooling_type = pooling_type


class LlamaNemotronVLConfig(VerifyAndUpdateConfig):
    """Config handler for LlamaNemotronVL embedding models."""

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        from vllm.config.pooler import SequencePoolingType

        hf_config = model_config.hf_config

        # Set bidirectional attention on the language model config
        hf_config.is_causal = False
        if hasattr(hf_config, "llm_config"):
            hf_config.llm_config.is_causal = False

        if hasattr(hf_config, "vision_config"):
            hf_config.patch_size = hf_config.vision_config.patch_size

        # Set up pooling type
        pooling_type_map: dict[str, SequencePoolingType] = {
            "avg": "MEAN",
            "cls": "CLS",
            "last": "LAST",
        }

        # Get pooling type from config (check both top-level and llm_config)
        pooling = getattr(hf_config, "pooling", None)
        if pooling is None and hasattr(hf_config, "llm_config"):
            pooling = getattr(hf_config.llm_config, "pooling", "avg")

        pooling_type = pooling_type_map.get(pooling)
        if pooling_type is None:
            raise ValueError(f"pool_type {pooling!r} not supported")

        model_config.pooler_config.seq_pooling_type = pooling_type


class MambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Enable FULL_AND_PIECEWISE cuda graph mode by default (required
        to get good performance for mamba layers in V1).

        Args:
            vllm_config: vLLM Config
        """
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if cache_config.enable_prefix_caching:
            if cache_config.mamba_cache_mode == "none":
                if (
                    model_config.supports_mamba_prefix_caching
                    and vllm_config.speculative_config is not None
                ):
                    cache_config.mamba_cache_mode = "align"
                    logger.warning(
                        "Mamba cache mode is set to 'align' for %s by default "
                        "when prefix caching and speculative decoding are enabled",
                        model_config.architecture,
                    )
                else:
                    cache_config.mamba_cache_mode = (
                        "all" if model_config.supports_mamba_prefix_caching else "align"
                    )
                    logger.warning(
                        "Mamba cache mode is set to '%s' for %s by default "
                        "when prefix caching is enabled",
                        cache_config.mamba_cache_mode,
                        model_config.architecture,
                    )
            if (
                cache_config.mamba_cache_mode == "all"
                and not model_config.supports_mamba_prefix_caching
            ):
                cache_config.mamba_cache_mode = "align"
                logger.warning(
                    "Hybrid or mamba-based model detected without support "
                    "for prefix caching with Mamba cache 'all' mode: "
                    "falling back to 'align' mode."
                )
            if cache_config.mamba_cache_mode == "align":
                assert vllm_config.scheduler_config.enable_chunked_prefill, (
                    "Chunked prefill is required for mamba cache mode 'align'."
                )
            logger.info_once(
                "Prefix caching is enabled with Mamba cache '%s' mode.",
                cache_config.mamba_cache_mode,
            )
            # By default, mamba block size will be set to max_model_len (see
            # below). When enabling prefix caching, we align mamba block size
            # to the block size as the basic granularity for prefix caching.
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = cache_config.block_size
        else:
            if cache_config.mamba_cache_mode != "none":
                cache_config.mamba_cache_mode = "none"
                logger.warning(
                    "Mamba cache mode is set to 'none' when prefix caching is disabled"
                )
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = model_config.max_model_len


class NemotronHForCausalLMConfig(VerifyAndUpdateConfig):
    DEFAULT_MAMBA_SSM_CACHE_DTYPE = "float32"
    """Only `float32` is known to have no accuracy issues by default."""

    @classmethod
    def update_mamba_ssm_cache_dtype(
        cls, *, cache_config: "CacheConfig", hf_config: "PretrainedConfig"
    ) -> None:
        """Update mamba_ssm_cache_dtype for NemotronH models when set to 'auto'
        (or not explicitly set), to the value specified in the HF config, or to
        `float32` if not specified.
        """
        if cache_config.mamba_ssm_cache_dtype == "auto":
            mamba_ssm_cache_dtype = getattr(
                hf_config, "mamba_ssm_cache_dtype", cls.DEFAULT_MAMBA_SSM_CACHE_DTYPE
            )
            logger.info(
                "Updating mamba_ssm_cache_dtype to '%s' for NemotronH model",
                mamba_ssm_cache_dtype,
            )
            cache_config.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype

    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        cls.update_mamba_ssm_cache_dtype(
            cache_config=vllm_config.cache_config,
            hf_config=vllm_config.model_config.hf_config,
        )


class NemotronHNanoVLV2Config(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        NemotronHForCausalLMConfig.update_mamba_ssm_cache_dtype(
            cache_config=vllm_config.cache_config,
            hf_config=vllm_config.model_config.hf_config.text_config,
        )

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        mm_config = model_config.multimodal_config
        if mm_config is not None:
            video_kwargs = mm_config.media_io_kwargs.setdefault("video", {})
            video_kwargs.setdefault("video_backend", "nemotron_vl")


class NomicBertModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        assert config.__class__.__name__ == "NomicBertConfig"
        assert config.activation_function in ["swiglu", "gelu"]
        config.position_embedding_type = getattr(
            config, "position_embedding_type", "rope"
        )

        if config.activation_function == "swiglu":
            config.hidden_act = "silu"
        else:
            config.hidden_act = config.activation_function

        assert config.mlp_fc1_bias == config.mlp_fc2_bias == config.qkv_proj_bias
        config.bias = config.qkv_proj_bias

        assert config.rotary_emb_scale_base is None
        assert not config.rotary_emb_interleaved

        config.layer_norm_eps = config.layer_norm_epsilon
        config.intermediate_size = config.n_inner
        config.hidden_size = config.n_embd
        config.num_hidden_layers = config.n_layer
        model_config.model_arch_config.hidden_size = config.hidden_size
        model_config.model_arch_config.total_num_hidden_layers = (
            config.num_hidden_layers
        )

        head_dim = config.hidden_size // config.num_attention_heads
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        max_trained_positions = getattr(
            config, "max_trained_positions", max_position_embeddings
        )

        rope_parameters = {
            "max_trained_positions": max_trained_positions,
            **(config.rope_parameters or {}),
        }

        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": model_config.max_model_len,
            "rope_parameters": rope_parameters,
        }


class Qwen2ForProcessRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        pooler_config = model_config.pooler_config

        if pooler_config.step_tag_id is None:
            pooler_config.step_tag_id = 151651


class Qwen2ForRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        pooler_config = model_config.pooler_config

        if pooler_config.use_activation is None:
            pooler_config.use_activation = False


class Qwen3ForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        is_original_qwen3_reranker = getattr(
            config, "is_original_qwen3_reranker", False
        )

        if not is_original_qwen3_reranker:
            return

        tokens = getattr(config, "classifier_from_token", None)
        assert tokens is not None and len(tokens) == 2, (
            "Try loading the original Qwen3 Reranker?, see: "
            "https://github.com/vllm-project/vllm/tree/main/examples/pooling/score/qwen3_reranker_offline.py"
        )
        text_config = config.get_text_config()
        text_config.method = "from_2_way_softmax"
        text_config.classifier_from_token = tokens


class Qwen3VLForSequenceClassificationConfig(Qwen3ForSequenceClassificationConfig):
    pass


class Qwen3_5ForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Update mamba_ssm_cache_dtype for Qwen3.5 models when set to 'auto'
        (or not explicitly set), to the value specified in the HF config's
        mamba_ssm_dtype field. Warn if the user explicitly overrides it to a
        different value.
        """
        cache_config = vllm_config.cache_config
        hf_text_config = vllm_config.model_config.hf_text_config
        mamba_ssm_dtype = getattr(hf_text_config, "mamba_ssm_dtype", None)
        if cache_config.mamba_ssm_cache_dtype == "auto":
            if mamba_ssm_dtype is not None:
                cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype
        elif (
            mamba_ssm_dtype is not None
            and cache_config.mamba_ssm_cache_dtype != mamba_ssm_dtype
        ):
            logger.warning(
                "Qwen3.5 model specifies mamba_ssm_dtype='%s' in its config, "
                "but --mamba-ssm-cache-dtype='%s' was passed. "
                "Using the user-specified value.",
                mamba_ssm_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )


def _strip_qwen4_exp_mrope(model_config: "ModelConfig") -> None:
    configs = {
        id(config): config
        for config in (
            getattr(model_config, "hf_config", None),
            model_config.hf_text_config,
        )
        if config is not None
    }
    for config in configs.values():
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


class Qwen4ExpForConditionalGenerationConfig(Qwen3_5ForConditionalGenerationConfig):
    """Apply the Qwen3.5 hybrid-cache contract to Qwen4Exp."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        if text_config.hc_count <= 1:
            raise ValueError("Qwen4Exp requires hc_count > 1")

        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size > 1 and getattr(
            text_config, "indexer_n_heads", 0
        ):
            # One all-to-all per QSA layer replaces all-gather(LSE) plus
            # reduce-scatter(output); on TP4/DCP2 V100 decode the combine drops
            # from 31.9 to 19.9 us per layer. An explicit user choice still wins.
            parallel_config.set_dcp_defaults(comm_backend="a2a")
        uses_ple_or_qsa = bool(text_config.ple_layer_ids) or (
            getattr(text_config, "indexer_n_heads", None) is not None
        )
        if uses_ple_or_qsa and (
            parallel_config.enable_dbo or parallel_config.ubatch_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support dual-batch overlap or microbatching"
            )

        model_config = vllm_config.model_config
        multimodal_config = model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:
            _strip_qwen4_exp_mrope(model_config)

        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method not in {
            "mtp",
            "ngram",
            "ngram_gpu",
        }:
            raise NotImplementedError(
                "Qwen4Exp speculative decoding supports only its native MTP "
                "checkpoint and linear n-gram proposers"
            )


class Qwen4ExpForCausalLMConfig(Qwen4ExpForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        _strip_qwen4_exp_mrope(vllm_config.model_config)


class Qwen4ExpMTPConfig(Qwen4ExpForConditionalGenerationConfig):
    """Preserve MRoPE for a VL target and use 1D RoPE for a text target."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        if hasattr(vllm_config.model_config.hf_config, "vision_config"):
            return
        _strip_qwen4_exp_mrope(vllm_config.model_config)


class SnowflakeGteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        assert config.__class__.__name__ == "GteConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
        config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": config.max_position_embeddings,
            "rope_parameters": config.rope_parameters,
        }


class VoyageQwen3BidirectionalEmbedModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        model_config.hf_config.is_causal = False
        model_config.hf_config.embedding_size = model_config.hf_config.num_labels


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "ColBERTJinaRobertaModel": JinaRobertaModelConfig,
    "ColQwen3_5": Qwen3_5ForConditionalGenerationConfig,
    "DeepseekV4ForCausalLM": DeepseekV4ForCausalLMConfig,
    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,
    "Ernie4_5_VLMoeForConditionalGeneration": Ernie4_5_VLMoeForConditionalGenerationConfig,  # noqa: E501
    "FalconMambaForCausalLM": MambaModelConfig,
    "FlintForCausalLM": FlintModelConfig,
    "Gemma3TextModel": Gemma3TextModelConfig,
    "Gemma4ForCausalLM": Gemma4Config,
    "Gemma4ForConditionalGeneration": Gemma4Config,
    "GptOssForCausalLM": GptOssForCausalLMConfig,
    "GteModel": SnowflakeGteNewModelConfig,
    "GteNewForSequenceClassification": GteNewModelConfig,
    "GteNewModel": GteNewModelConfig,
    "JambaForSequenceClassification": JambaForSequenceClassificationConfig,
    "JinaForRanking": JinaForRankingConfig,
    "JinaVLForRanking": JinaVLForSequenceClassificationConfig,
    "LlamaBidirectionalForSequenceClassification": LlamaBidirectionalConfig,
    "LlamaBidirectionalModel": LlamaBidirectionalConfig,
    "LlamaNemotronVLForSequenceClassification": LlamaNemotronVLConfig,
    "LlamaNemotronVLModel": LlamaNemotronVLConfig,
    "Mamba2ForCausalLM": MambaModelConfig,
    "MambaForCausalLM": MambaModelConfig,
    "NemotronHForCausalLM": NemotronHForCausalLMConfig,
    "NemotronHPuzzleForCausalLM": NemotronHForCausalLMConfig,
    "NemotronH_Nano_VL_V2": NemotronHNanoVLV2Config,
    "NomicBertModel": NomicBertModelConfig,
    "Qwen2ForProcessRewardModel": Qwen2ForProcessRewardModelConfig,
    "Qwen2ForRewardModel": Qwen2ForRewardModelConfig,
    "Qwen3ForSequenceClassification": Qwen3ForSequenceClassificationConfig,
    "Qwen3VLForSequenceClassification": Qwen3VLForSequenceClassificationConfig,
    "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,
    "Qwen4ExpForCausalLM": Qwen4ExpForCausalLMConfig,
    "Qwen4ExpForConditionalGeneration": Qwen4ExpForConditionalGenerationConfig,
    "Qwen4ExpMTP": Qwen4ExpMTPConfig,
    "VoyageQwen3BidirectionalEmbedModel": VoyageQwen3BidirectionalEmbedModelConfig,
    "XLMRobertaModel": JinaRobertaModelConfig,
}


def sm70_dflash2_verifier_qualified(
    model_config: Any,
    speculative_config: Any,
    parallel_config: Any,
) -> bool:
    """Admit the quality-audited Qwen3.8 DFlash2 verifier contract.

    Target quantization, KV dtype, TP degree, and service capacity are
    intentionally not part of this admission. Each fast operator capability-
    checks its local weight, cache dtype, and live batch shape, then falls back
    independently when it cannot handle that contract.
    """
    import torch

    if any(
        config is None
        for config in (
            model_config,
            speculative_config,
            parallel_config,
        )
    ):
        return False

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    dflash_config = getattr(draft_hf_config, "dflash_config", None) or {}
    selector_top_k = (
        int(dflash_config.get("selector_top_k", 0) or 0)
        if isinstance(dflash_config, Mapping)
        else 0
    )
    hf_text_config = getattr(model_config, "hf_text_config", None)
    architectures = set(getattr(model_config, "architectures", ()) or ())
    return bool(
        architectures & {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"}
        and getattr(model_config, "dtype", None) == torch.float16
        and getattr(hf_text_config, "hidden_size", None) == 5120
        and getattr(hf_text_config, "num_attention_heads", None) == 24
        and getattr(hf_text_config, "num_key_value_heads", None) == 4
        and getattr(hf_text_config, "head_dim", None) == 256
        and getattr(speculative_config, "method", None) == "dflash"
        and int(getattr(speculative_config, "num_speculative_tokens", 0) or 0) == 7
        and selector_top_k == 16
        and getattr(parallel_config, "pipeline_parallel_size", 0) == 1
        and not getattr(parallel_config, "enable_dbo", False)
        and int(getattr(parallel_config, "ubatch_size", 0) or 0) <= 1
    )
