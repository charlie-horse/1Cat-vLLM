# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Flint model on stock vLLM layers.

Flint is a Qwen3-MoE-style decoder with learned attention sinks, hybrid
sliding-window/full attention (full layers use no RoPE), dense SwiGLU MLPs on
``mlp_only_layers``, sigmoid top-k routing with renormalized weights, and an
ungated shared expert.

Checkpoints are HF-named safetensors, either BF16 (cast to the model dtype at
load) or ModelOpt NVFP4 W4A16. In the NVFP4 checkpoint ``lm_head.weight`` is
stored dequantized without its global scale, which comes separately as the
scalar ``lm_head.weight_scale_2``.
"""

from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import islice

import regex as re
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config_or_none
from vllm.config.execution_policy import layer_policy
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name

from .configs import FlintConfig
from .qk_norm_rope_kv import fused_route_rejection

logger = init_logger(__name__)

_FP16_MAX = torch.finfo(torch.float16).max

# Checkpoint shards that one fused linear loads. A fused NVFP4 linear keeps a
# single global scale, so the shards' ``weight_scale_2`` must agree.
_MERGED_SHARDS = {
    ".q_proj.": ".qkv_proj.",
    ".k_proj.": ".qkv_proj.",
    ".v_proj.": ".qkv_proj.",
    ".gate_proj.": ".gate_up_proj.",
    ".up_proj.": ".gate_up_proj.",
}
_LM_HEAD_SCALE = "lm_head.weight_scale_2"

# FP16 headroom for the MoE intermediate silu(gate) * up. On a few
# massive-activation tokens it exceeds FP16 (about 9.4e4 in layer 46's routed
# experts on V100), while gate, up and the expert outputs stay in range. The
# intermediate is rebalanced by this power of two: up is divided by it and
# down multiplied by it, so the expert output is unchanged and only the FP16
# intermediate shrinks. In NVFP4 checkpoints this shifts the up projection's
# E4M3 block scales, which keeps the gate/up global scales shared, and
# down's global scale.
_MOE_INTERMEDIATE_SHIFT = 16
_MOE_UP = (".mlp.experts.", ".mlp.shared_expert.")

# FP16 headroom for the residual stream. From layer 46 on, a few tokens carry
# about 3.1e4 (2.1x below the FP16 max) in the MoE output and in the residual
# stream; earlier layers stay below 1e2. In FP16 the stream is carried divided
# by this power of two from that layer on: the layer scales its residual after
# the attention add, later layers scale their attention output, and every MoE
# from that layer on is scaled through its down projection's scale at load.
# Every RMSNorm, the final one included, is invariant to the scale.
_FP16_STREAM_SHIFT_LAYER = 46
_FP16_STREAM_SHIFT = 8
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _sink_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Load this rank's heads of the ``[total_num_heads]`` sink tensor."""
    heads_per_rank = param.shape[0]
    start = get_tensor_model_parallel_rank() * heads_per_rank
    param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))


@lru_cache(maxsize=4)
def _expert_params_mapping(num_experts: int) -> list[tuple[str, str, int, str]]:
    # One list shared by every MoE layer; FusedMoE indexes it by identity.
    return FusedMoE.make_expert_params_mapping(
        nn.Module(),
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )


def _layer_rope_parameters(config: FlintConfig, layer_idx: int) -> dict:
    rope_parameters = dict(config.rope_parameters or {})
    if rope_parameters and all(isinstance(v, dict) for v in rope_parameters.values()):
        # Nested per layer type (transformers v5 schema).
        layer_type = config.layer_types[layer_idx]
        rope_parameters = dict(rope_parameters.get(layer_type) or {})
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    return rope_parameters


class FlintMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class FlintSparseMoeBlock(nn.Module):
    """Sigmoid top-k routed experts (renormalized, no score bias) plus an
    ungated shared expert, combined and all-reduced inside FusedMoE."""

    def __init__(
        self,
        config: FlintConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.router_score_function != "sigmoid":
            raise ValueError(
                "Flint routes with sigmoid scores, got "
                f"router_score_function={config.router_score_function!r}"
            )
        if get_tensor_model_parallel_world_size() > config.num_experts:
            raise ValueError(
                "Tensor parallel size is greater than the number of experts "
                f"({config.num_experts})."
            )
        # The router is unquantized and, like the reference implementation,
        # computes its logits in fp32 from an fp32 weight.
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert = FlintMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_expert",
        )
        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            scoring_func="sigmoid",
            use_grouped_topk=False,
            e_score_correction_bias=None,
            router_logits_dtype=torch.float32,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            expert_mapping=_expert_params_mapping(config.num_experts),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The gate runs inside FusedMoE (after the shared-expert stream sync).
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


class FlintAttention(nn.Module):
    def __init__(
        self,
        config: FlintConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = None
        if layer_idx not in config.no_rope_layer_indices:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters=_layer_rope_parameters(config, layer_idx),
            )
            if self.rotary_emb.rotary_dim != self.head_dim:
                raise ValueError("Flint applies RoPE to the full head dimension")

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        # ``sliding_window`` counts the current token, which is also what the
        # attention backends take: a window of W covers W - 1 earlier keys.
        self.sliding_window = config.sliding_window if is_sliding else None
        # One learned sink logit per head, in fp32 whatever the model dtype.
        # The backend keeps a reference to this very tensor, so it is passed
        # as is and loaded in place: each TP rank copies its own heads into it.
        self.softmax_offset = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(self.softmax_offset, {"weight_loader": _sink_weight_loader})
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
            sinks=self.softmax_offset,
        )
        # kernel_config.sm70_flint_qk_rope_kv (VLLM_SM70_FLINT_QK_ROPE_KV),
        # resolved by VllmConfig; on by default.
        vllm_config = get_current_vllm_config_or_none()
        enabled = (
            vllm_config is None
            or vllm_config.kernel_config.sm70_flint_qk_rope_kv is not False
        )
        is_sm70 = current_platform.is_cuda() and current_platform.is_device_capability(
            70
        )
        reason = fused_route_rejection(
            enabled=enabled,
            is_sm70=is_sm70,
            dtype=torch.get_default_dtype(),
            head_dim=self.head_dim,
            q_eps=self.q_norm.variance_epsilon,
            k_eps=self.k_norm.variance_epsilon,
            rotary_emb=self.rotary_emb,
            kv_cache_dtype=self.attn.kv_cache_dtype,
            attn_type=self.attn.attn_type,
            calculate_kv_scales=self.attn.calculate_kv_scales,
            kv_sharing=self.attn.kv_sharing_target_layer_name is not None,
            backend_writes_kv_in_forward=(
                self.attn.attn_backend.forward_includes_kv_cache_update
            ),
        )
        # One kernel for the q/k norms, RoPE and the KV-cache write.
        self.fused_qk_rope_kv = reason is None
        if self.fused_qk_rope_kv:
            logger.info_once("Flint SM70: fused q/k norm, RoPE and KV-cache write.")
        elif is_sm70:
            logger.info_once(
                "Flint SM70: separate q/k norm, RoPE and KV-cache write kernels "
                "because %s.",
                reason,
            )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fused_qk_rope_kv:
            return self._forward_fused(positions, qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_fused(
        self, positions: torch.Tensor, qkv: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = qkv.shape[0]
        q = torch.empty((num_tokens, self.q_size), dtype=qkv.dtype, device=qkv.device)
        k = torch.empty((num_tokens, self.kv_size), dtype=qkv.dtype, device=qkv.device)
        v = qkv[:, self.q_size + self.kv_size :]
        torch.ops.vllm.qk_norm_rope_kv_cache_update(
            qkv,
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            positions,
            None if self.rotary_emb is None else self.rotary_emb.cos_sin_cache,
            self.q_norm.variance_epsilon,
            _encode_layer_name(self.attn.layer_name),
        )
        attn_output = self.attn(q, k, v, kv_cache_updated=True)
        output, _ = self.o_proj(attn_output)
        return output


class FlintDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FlintConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
        fp16_stream_shift: bool = False,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        # FP16 residual-stream rescale (see _FP16_STREAM_SHIFT_LAYER).
        self.residual_scale: float | None = None
        self.attn_out_scale: float | None = None
        if fp16_stream_shift and layer_idx == _FP16_STREAM_SHIFT_LAYER:
            self.residual_scale = 1.0 / _FP16_STREAM_SHIFT
        elif fp16_stream_shift and layer_idx > _FP16_STREAM_SHIFT_LAYER:
            self.attn_out_scale = 1.0 / _FP16_STREAM_SHIFT
        self.self_attn = FlintAttention(
            config,
            layer_idx,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp: FlintMLP | FlintSparseMoeBlock
        if layer_idx in config.mlp_only_layers:
            self.mlp = FlintMLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = FlintSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        if self.attn_out_scale is not None:
            hidden_states = hidden_states * self.attn_out_scale
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.residual_scale is not None:
            residual = residual * self.residual_scale
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class FlintModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        # Sliding windows are set per layer from ``layer_types``. Without this,
        # Attention would fall back to the model-level ``sliding_window`` on
        # the full-attention layers (per_layer_sliding_window=None).
        if cache_config is not None:
            cache_config.sliding_window = None

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        fp16_stream_shift = vllm_config.model_config.dtype == torch.float16
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlintDecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix=prefix,
                fp16_stream_shift=fp16_stream_shift,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class FlintForCausalLM(nn.Module, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.check_fp16_range = vllm_config.model_config.dtype == torch.float16

        self.model = FlintModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            # Flint checkpoints store lm_head unquantized (the ModelOpt config
            # excludes it), so it takes the plain embedding method. With the
            # quant config, ModelOpt would hand this ParallelLMHead an
            # UnquantizedLinearMethod, whose SM70 dense preparation needs
            # LinearBase state that an embedding layer does not have.
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        sampler_on = layer_policy(vllm_config).value("flint_sampler")
        if get_pp_group().is_last_rank and sampler_on:
            # Vocab-parallel sampling: the runner asks for these per step and
            # admits them only for the batches they reproduce exactly (pure
            # greedy; top-k <= 20 random sampling without logprobs).
            self.get_top_tokens = self._get_top_tokens
            self.get_topk_candidates = self._get_topk_candidates
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def _get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Greedy argmax over the regular LM-head logits; the fused SM70
        # top-1 projection forms FP16 products, too coarse for this head's
        # folded output scale.
        return self.logits_processor.get_top_tokens(
            self.lm_head, hidden_states, fused_top1=False
        )

    def _get_topk_candidates(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor.get_topk_candidates(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _expert_params_mapping(self.config.num_experts)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._lm_head_scale: float | None = None
        weights = self._intercept_lm_head_scale(weights)
        weights = _check_merged_global_scales(weights)
        if self.check_fp16_range:
            weights = _rebalance_moe_intermediate(weights, _MOE_INTERMEDIATE_SHIFT)
            weights = _rescale_fp16_stream(
                weights, _FP16_STREAM_SHIFT_LAYER, _FP16_STREAM_SHIFT
            )
            weights = _check_fp16_range(weights)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if self._lm_head_scale is not None:
            self._apply_lm_head_scale(self._lm_head_scale)
        return loaded

    def _intercept_lm_head_scale(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name == _LM_HEAD_SCALE:
                self._lm_head_scale = float(tensor)
            else:
                yield name, tensor

    def _apply_lm_head_scale(self, scale: float) -> None:
        """Apply the lm_head global scale that the checkpoint stores apart.

        With BF16/FP32 weights the scale multiplies the logits, as upstream.
        FP16 cannot hold the unscaled logits (they are the true logits divided
        by a scale of order 1e-4), so in FP16 it is folded into the weight.
        """
        if isinstance(self.lm_head, PPMissingLayer):
            return
        weight = self.lm_head.weight
        if weight.dtype != torch.float16:
            self.logits_processor.scale = scale
            return
        with torch.no_grad():
            for rows in weight.data.split(8192):
                folded = rows.float().mul_(scale)
                if not bool(folded.abs().amax() <= _FP16_MAX):
                    raise ValueError(
                        "lm_head.weight times lm_head.weight_scale_2 exceeds "
                        "the FP16 range"
                    )
                rows.copy_(folded)


def _check_merged_global_scales(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass tensors through, raising if the NVFP4 global scales of the shards
    fused into one linear (q/k/v, gate/up) differ: the fused layer keeps one
    global scale, so shards with other scales would load with wrong values."""
    seen: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        if name.endswith(".weight_scale_2") and ".experts." not in name:
            for shard, merged in _MERGED_SHARDS.items():
                if shard in name:
                    key = name.replace(shard, merged)
                    value = tensor.detach().reshape(()).float().cpu()
                    ref = seen.setdefault(key, value)
                    if not torch.equal(ref, value):
                        raise ValueError(
                            f"{name}: global scale {value.item()} differs from "
                            f"the other shards of {key} ({ref.item()}); the "
                            "checkpoint must share one weight_scale_2 per "
                            "fused linear"
                        )
                    break
        yield name, tensor


def _rebalance_moe_intermediate(
    weights: Iterable[tuple[str, torch.Tensor]], shift: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Divide MoE up projections by ``shift`` and multiply the matching down
    projections by it (see ``_MOE_INTERMEDIATE_SHIFT``).

    A power of two only moves exponents, so this is exact except where an
    E4M3 block scale or an FP16 value leaves the normal range. NVFP4 tensors
    move the up projection's E4M3 block scales (``weight_scale``) and the
    down projection's FP32 global scale (``weight_scale_2``); unquantized
    tensors move the weights themselves.
    """
    for name, tensor in weights:
        if any(block in name for block in _MOE_UP):
            dense = name.endswith(".weight") and tensor.is_floating_point()
            if ".up_proj." in name and (name.endswith(".weight_scale") or dense):
                tensor = (tensor.float() / shift).to(tensor.dtype)
            elif ".down_proj." in name and (name.endswith(".weight_scale_2") or dense):
                tensor = (tensor.float() * shift).to(tensor.dtype)
        yield name, tensor


def _rescale_fp16_stream(
    weights: Iterable[tuple[str, torch.Tensor]], start_layer: int, shift: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Divide the MLP/MoE down projections of layers ``>= start_layer`` by
    ``shift`` (see ``_FP16_STREAM_SHIFT_LAYER``): their outputs then join the
    rescaled residual stream. NVFP4 tensors move the FP32 global scale
    (``weight_scale_2``); unquantized tensors move the weights."""
    for name, tensor in weights:
        match = _LAYER_INDEX.search(name)
        if (
            match is not None
            and int(match.group(1)) >= start_layer
            and ".mlp." in name
            and ".down_proj." in name
            and (
                name.endswith(".weight_scale_2")
                or (name.endswith(".weight") and tensor.is_floating_point())
            )
        ):
            tensor = (tensor.float() / shift).to(tensor.dtype)
        yield name, tensor


def _check_fp16_range(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Reject BF16/FP32 tensors that would overflow when cast to FP16."""
    for name, tensor in weights:
        if tensor.dtype in (torch.bfloat16, torch.float32) and tensor.numel():
            amax = tensor.detach().abs().amax()
            if not bool(amax <= _FP16_MAX):
                raise ValueError(
                    f"{name}: max |x| = {float(amax)} does not fit in FP16 "
                    f"(max {_FP16_MAX})"
                )
        yield name, tensor
