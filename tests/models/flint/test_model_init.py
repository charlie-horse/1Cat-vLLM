# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the Flint model: layer layout, sinks and weight loading.

The model is built on CPU from a tiny config. Two platform lookups that need
a GPU platform are stubbed: the attention backend (the Triton backend is
used, which also fixes how ``sliding_window`` is mapped) and the unquantized
MoE kernel oracle. NVFP4 tests also stub the NVFP4 MoE kernel oracle; they
check that ModelOpt-named tensors land in the right parameters, not kernels.
"""

import contextlib
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.models.flint.configs import FlintConfig

LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]
NO_ROPE = [0, 3]
WINDOW = 9


def tiny_config(**overrides) -> FlintConfig:
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=LAYER_TYPES,
        no_rope_layer_indices=NO_ROPE,
        mlp_only_layers=[0],
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        sliding_window=WINDOW,
        max_position_embeddings=256,
    )
    kwargs.update(overrides)
    return FlintConfig(**kwargs)


@pytest.fixture(scope="module")
def dist():
    if model_parallel_is_initialized():
        yield
        return
    fd, path = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{path}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(OSError):
            os.unlink(path)


@pytest.fixture
def cpu_layers(monkeypatch):
    import vllm.model_executor.layers.attention.attention as attention
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as moe
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    monkeypatch.setattr(
        attention, "get_attn_backend", lambda *args, **kwargs: TritonAttentionBackend
    )
    monkeypatch.setattr(
        moe,
        "select_unquantized_moe_backend",
        lambda moe_config: (UnquantizedMoeBackend.CPU, None),
    )


@pytest.fixture
def nvfp4_moe_stub(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization import modelopt

    monkeypatch.setattr(
        modelopt,
        "select_nvfp4_moe_backend",
        lambda **kwargs: (NvFp4MoeBackend.MARLIN, None),
    )
    monkeypatch.setattr(
        modelopt, "is_global_sf_supported_for_nvfp4_backend", lambda backend: False
    )
    # The W4A16 linear pins a GPU kernel at construction; loading needs none.
    monkeypatch.setattr(
        modelopt, "MarlinNvFp4LinearKernel", lambda config: SimpleNamespace()
    )


def nvfp4_config():
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    return ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=["model.embed_tokens", "*.mlp.gate", "lm_head"],
        group_size=16,
    )


@contextlib.contextmanager
def runtime(cfg, dtype=torch.float16, quant_config=None):
    from vllm.models.flint.model import FlintForCausalLM

    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    vllm_config.model_config = SimpleNamespace(
        hf_config=cfg,
        hf_text_config=cfg,
        dtype=dtype,
        is_mm_prefix_lm=False,
        is_moe=True,
    )
    if quant_config is not None:
        # As the model loader does (configure_quant_config).
        quant_config.packed_modules_mapping = FlintForCausalLM.packed_modules_mapping
        quant_config.apply_vllm_mapper(FlintForCausalLM.hf_to_vllm_mapper)
        vllm_config.quant_config = quant_config
    # As ModelConfig sets it from hf_config.sliding_window.
    vllm_config.cache_config.sliding_window = cfg.sliding_window
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            yield vllm_config
    finally:
        torch.set_default_dtype(default_dtype)


def build(cfg, dtype=torch.float16, quant_config=None):
    from vllm.models.flint.model import FlintForCausalLM

    with runtime(cfg, dtype, quant_config) as vllm_config:
        model = FlintForCausalLM(vllm_config=vllm_config)
    return model, vllm_config


def c3_state_dict(cfg: FlintConfig, nvfp4: bool, seed: int = 0) -> dict:
    """Synthetic checkpoint with the C3 tensor names."""
    gen = torch.Generator().manual_seed(seed)
    h, hd = cfg.hidden_size, cfg.head_dim
    q_out = cfg.num_attention_heads * hd
    kv_out = cfg.num_key_value_heads * hd
    sd: dict[str, torch.Tensor] = {}

    def vector(name, n, dtype=torch.bfloat16):
        sd[name] = (1 + 0.1 * torch.randn(n, generator=gen)).to(dtype)

    def linear(base, out_f, in_f, global_scale):
        if nvfp4:
            sd[f"{base}.weight"] = torch.randint(
                0, 256, (out_f, in_f // 2), dtype=torch.uint8, generator=gen
            )
            sd[f"{base}.weight_scale"] = (
                torch.rand(out_f, in_f // 16, generator=gen) * 4
            ).to(torch.float8_e4m3fn)
            sd[f"{base}.weight_scale_2"] = torch.tensor(global_scale)
        else:
            sd[f"{base}.weight"] = (0.02 * torch.randn(out_f, in_f, generator=gen)).to(
                torch.bfloat16
            )

    sd["model.embed_tokens.weight"] = (
        0.02 * torch.randn(cfg.vocab_size, h, generator=gen)
    ).to(torch.float16 if nvfp4 else torch.bfloat16)
    vector("model.norm.weight", h, torch.float16 if nvfp4 else torch.bfloat16)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        vector(f"{p}.input_layernorm.weight", h)
        vector(f"{p}.post_attention_layernorm.weight", h)
        vector(f"{p}.self_attn.q_norm.weight", hd)
        vector(f"{p}.self_attn.k_norm.weight", hd)
        sd[f"{p}.self_attn.softmax_offset"] = torch.randn(
            cfg.num_attention_heads, generator=gen
        )
        qkv_scale = 1e-3 * (i + 1)
        linear(f"{p}.self_attn.q_proj", q_out, h, qkv_scale)
        linear(f"{p}.self_attn.k_proj", kv_out, h, qkv_scale)
        linear(f"{p}.self_attn.v_proj", kv_out, h, qkv_scale)
        linear(f"{p}.self_attn.o_proj", h, q_out, 2e-3 * (i + 1))
        if i in cfg.mlp_only_layers:
            inter = cfg.intermediate_size
            linear(f"{p}.mlp.gate_proj", inter, h, 3e-3)
            linear(f"{p}.mlp.up_proj", inter, h, 3e-3)
            linear(f"{p}.mlp.down_proj", h, inter, 4e-3)
            continue
        sd[f"{p}.mlp.gate.weight"] = (
            0.02 * torch.randn(cfg.num_experts, h, generator=gen)
        ).to(torch.bfloat16)
        shared = cfg.shared_expert_intermediate_size
        linear(f"{p}.mlp.shared_expert.gate_proj", shared, h, 5e-3)
        linear(f"{p}.mlp.shared_expert.up_proj", shared, h, 5e-3)
        linear(f"{p}.mlp.shared_expert.down_proj", h, shared, 6e-3)
        inter = cfg.moe_intermediate_size
        for e in range(cfg.num_experts):
            base = f"{p}.mlp.experts.{e}"
            # Expert shards may carry their own global scales.
            linear(f"{base}.gate_proj", inter, h, 1e-3 * (e + 1))
            linear(f"{base}.up_proj", inter, h, 1e-3 * (e + 2))
            linear(f"{base}.down_proj", h, inter, 1e-3 * (e + 3))
    if nvfp4:
        # Dequantized without its global scale (C3), plus that scale.
        sd["lm_head.weight"] = (
            torch.randint(-12, 13, (cfg.vocab_size, h), generator=gen).to(torch.float16)
            * 224
        )
        sd["lm_head.weight_scale_2"] = torch.tensor(1.5e-4)
    else:
        sd["lm_head.weight"] = (
            0.02 * torch.randn(cfg.vocab_size, h, generator=gen)
        ).to(torch.bfloat16)
    return sd


def test_config_matches_checkpoint_fields():
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert _CONFIG_REGISTRY["flint"] is FlintConfig
    cfg = tiny_config(rope_theta=10000.0, softmax_type="learnable")
    assert cfg.model_type == "flint"
    assert cfg.rope_parameters == {"rope_type": "default", "rope_theta": 10000.0}
    assert cfg.router_score_function == "sigmoid" and cfg.norm_topk_prob
    assert not cfg.tie_word_embeddings
    restored = FlintConfig(**cfg.to_dict())
    assert restored.layer_types == LAYER_TYPES
    assert restored.no_rope_layer_indices == NO_ROPE
    assert restored.sliding_window == WINDOW
    with pytest.raises(ValueError, match="layer_types"):
        tiny_config(num_hidden_layers=3)


def test_registered_architecture():
    from vllm.model_executor.models.registry import _VLLM_MODELS, ModelRegistry

    # The model module, not the lazy package __init__, so that the registry's
    # model-info cache is keyed on the file that defines the class.
    assert _VLLM_MODELS["FlintForCausalLM"] == (
        "vllm.models.flint.model",
        "FlintForCausalLM",
    )
    model_cls = ModelRegistry._try_load_model_cls("FlintForCausalLM")
    from vllm.models.flint import FlintForCausalLM

    assert model_cls is FlintForCausalLM
    from vllm.model_executor.models.interfaces import supports_pp

    assert supports_pp(model_cls)


def test_layer_layout(dist, cpu_layers):
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.models.flint.model import FlintMLP, FlintSparseMoeBlock

    cfg = tiny_config()
    model, vllm_config = build(cfg)
    # Full layers must not inherit the model-level window.
    assert vllm_config.cache_config.sliding_window is None
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if LAYER_TYPES[i] == "sliding_attention":
            assert attn.attn.sliding_window == WINDOW
            # The current token plus WINDOW - 1 earlier ones.
            assert attn.attn.impl.sliding_window == (WINDOW - 1, 0)
        else:
            assert attn.attn.sliding_window is None
            assert attn.attn.impl.sliding_window == (-1, -1)
        if i in NO_ROPE:
            assert attn.rotary_emb is None
        else:
            assert attn.rotary_emb.is_neox_style
            assert attn.rotary_emb.rotary_dim == cfg.head_dim
        assert attn.softmax_offset.shape == (cfg.num_attention_heads,)
        assert attn.softmax_offset.dtype == torch.float32
        assert attn.attn.impl.sinks is attn.softmax_offset
        assert attn.qkv_proj.weight.dtype == torch.float16

        if i in cfg.mlp_only_layers:
            assert isinstance(layer.mlp, FlintMLP)
            assert layer.mlp.down_proj.input_size == cfg.intermediate_size
            continue
        assert isinstance(layer.mlp, FlintSparseMoeBlock)
        experts = layer.mlp.experts
        assert isinstance(experts, FusedMoE)
        assert experts.scoring_func == "sigmoid"
        assert experts.renormalize is True
        assert experts.top_k == cfg.num_experts_per_tok
        assert experts.e_score_correction_bias is None
        assert experts.is_internal_router
        assert experts.shared_experts is not None
        assert experts.moe_config.router_logits_dtype == torch.float32
        gate = layer.mlp.gate
        assert gate.weight.dtype == torch.float32
        assert gate.out_dtype == torch.float32
        shared = layer.mlp.shared_expert
        assert shared.down_proj.reduce_results is False
        assert shared.down_proj.input_size == cfg.shared_expert_intermediate_size


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_sinks_are_sharded_by_tp_rank(dist, cpu_layers, monkeypatch, tp_rank):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.flint import model as flint_model

    for module in (flint_model, linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    cfg = tiny_config()
    with runtime(cfg) as vllm_config:
        attn = flint_model.FlintAttention(
            cfg, 1, vllm_config.cache_config, None, prefix="model.layers.1.self_attn"
        )
    assert attn.num_heads == 2 and attn.num_kv_heads == 1
    sinks = attn.softmax_offset
    assert sinks.shape == (2,) and sinks.dtype == torch.float32
    assert attn.attn.impl.sinks is sinks
    full = torch.arange(cfg.num_attention_heads, dtype=torch.bfloat16)
    sinks.weight_loader(sinks, full)
    assert attn.softmax_offset is sinks and attn.attn.impl.sinks is sinks
    assert sinks.dtype == torch.float32 and sinks.is_contiguous()
    torch.testing.assert_close(sinks.data, full[2 * tp_rank :][:2].float())


@pytest.mark.parametrize("nvfp4", [False, True], ids=["bf16", "nvfp4"])
def test_sinks_stay_the_backend_fp32_tensor(dist, cpu_layers, nvfp4_moe_stub, nvfp4):
    """The backend keeps a reference to the sink tensor and needs it fp32,
    contiguous and 1-D, so it must be loaded in place and never cast."""
    cfg = tiny_config()
    model, _ = build(cfg, quant_config=nvfp4_config() if nvfp4 else None)
    attentions = [layer.self_attn for layer in model.model.layers]
    before = []
    for attn in attentions:
        sinks = attn.softmax_offset
        # Built under an FP16 default dtype, so fp32 must be explicit.
        assert isinstance(sinks, torch.nn.Parameter)
        assert sinks.dtype == torch.float32 and sinks.dim() == 1
        assert sinks.shape == (cfg.num_attention_heads,) and sinks.is_contiguous()
        assert attn.attn.impl.sinks is sinks
        before.append((sinks, sinks.data_ptr()))

    sd = c3_state_dict(cfg, nvfp4=nvfp4)
    if not nvfp4:
        # HF BF16 checkpoints may store the sinks in BF16.
        for i in range(cfg.num_hidden_layers):
            name = f"model.layers.{i}.self_attn.softmax_offset"
            sd[name] = sd[name].bfloat16()
    model.load_weights(sd.items())
    for attn in attentions:
        attn.attn.process_weights_after_loading(torch.float16)

    for i, (attn, (sinks, ptr)) in enumerate(zip(attentions, before)):
        assert attn.softmax_offset is sinks and attn.attn.impl.sinks is sinks
        assert sinks.data_ptr() == ptr
        assert sinks.dtype == torch.float32 and sinks.is_contiguous()
        assert sinks.shape == (cfg.num_attention_heads,)
        expected = sd[f"model.layers.{i}.self_attn.softmax_offset"].float()
        torch.testing.assert_close(sinks.data, expected, rtol=0, atol=0)


def test_load_bf16_checkpoint(dist, cpu_layers):
    cfg = tiny_config()
    model, _ = build(cfg)
    sd = c3_state_dict(cfg, nvfp4=False)
    loaded = model.load_weights(sd.items())
    params = dict(model.named_parameters())
    assert loaded == set(params)

    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    layer = model.model.layers[1]
    p = "model.layers.1"
    qkv = layer.self_attn.qkv_proj.weight
    for name, rows in (
        ("q_proj", slice(0, q_out)),
        ("k_proj", slice(q_out, q_out + kv_out)),
        ("v_proj", slice(q_out + kv_out, q_out + 2 * kv_out)),
    ):
        torch.testing.assert_close(
            qkv[rows], sd[f"{p}.self_attn.{name}.weight"].half(), rtol=0, atol=0
        )
    torch.testing.assert_close(
        layer.self_attn.softmax_offset.data, sd[f"{p}.self_attn.softmax_offset"]
    )
    gate = layer.mlp.gate.weight
    torch.testing.assert_close(gate, sd[f"{p}.mlp.gate.weight"].float())
    shared = layer.mlp.shared_expert.gate_up_proj.weight
    torch.testing.assert_close(
        shared,
        torch.cat(
            [
                sd[f"{p}.mlp.shared_expert.gate_proj.weight"],
                sd[f"{p}.mlp.shared_expert.up_proj.weight"],
            ]
        ).half(),
    )
    experts = layer.mlp.experts
    for e in (0, cfg.num_experts - 1):
        base = f"{p}.mlp.experts.{e}"
        torch.testing.assert_close(
            experts.w13_weight[e],
            torch.cat(
                [sd[f"{base}.gate_proj.weight"], sd[f"{base}.up_proj.weight"]]
            ).half(),
        )
        torch.testing.assert_close(
            experts.w2_weight[e], sd[f"{base}.down_proj.weight"].half()
        )
    dense = model.model.layers[0].mlp.gate_up_proj.weight
    torch.testing.assert_close(
        dense[: cfg.intermediate_size], sd["model.layers.0.mlp.gate_proj.weight"].half()
    )
    torch.testing.assert_close(model.lm_head.weight, sd["lm_head.weight"].half())
    assert model.logits_processor.scale == 1.0


def test_fp16_load_rejects_out_of_range_values(dist, cpu_layers):
    cfg = tiny_config()
    model, _ = build(cfg)
    sd = c3_state_dict(cfg, nvfp4=False)
    sd["model.layers.2.mlp.experts.3.up_proj.weight"][0, 0] = 70000.0
    with pytest.raises(ValueError, match="does not fit in FP16"):
        model.load_weights(sd.items())


def test_load_nvfp4_checkpoint(dist, cpu_layers, nvfp4_moe_stub):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4W4A16LinearMethod,
    )

    cfg = tiny_config()
    model, _ = build(cfg, quant_config=nvfp4_config())
    layer = model.model.layers[1]
    assert isinstance(
        layer.self_attn.qkv_proj.quant_method, ModelOptNvFp4W4A16LinearMethod
    )
    assert isinstance(layer.mlp.experts.quant_method, ModelOptNvFp4FusedMoE)
    assert isinstance(
        layer.mlp.shared_expert.gate_up_proj.quant_method,
        ModelOptNvFp4W4A16LinearMethod,
    )
    assert isinstance(
        model.model.layers[0].mlp.gate_up_proj.quant_method,
        ModelOptNvFp4W4A16LinearMethod,
    )
    assert isinstance(layer.mlp.gate.quant_method, UnquantizedLinearMethod)
    assert isinstance(model.lm_head.quant_method, UnquantizedLinearMethod)
    assert model.lm_head.weight.dtype == torch.float16
    assert model.model.embed_tokens.weight.dtype == torch.float16

    sd = c3_state_dict(cfg, nvfp4=True)
    loaded = model.load_weights(sd.items())
    params = dict(model.named_parameters())
    assert loaded <= set(params)
    # Only the unused activation-scale placeholders (W4A16) and the KV-cache
    # scales (FP16 KV) stay unloaded.
    unloaded = set(params) - loaded
    assert all(name.endswith("input_scale") or ".attn." in name for name in unloaded), (
        sorted(unloaded)
    )
    assert not any(name.endswith(("weight", "weight_scale_2")) for name in unloaded)

    q_out = cfg.num_attention_heads * cfg.head_dim
    p = "model.layers.1"
    qkv = layer.self_attn.qkv_proj
    assert torch.equal(qkv.weight[:q_out], sd[f"{p}.self_attn.q_proj.weight"])
    assert torch.equal(
        qkv.weight_scale[:q_out].view(torch.uint8),
        sd[f"{p}.self_attn.q_proj.weight_scale"].view(torch.uint8),
    )
    torch.testing.assert_close(qkv.weight_scale_2.data, torch.full((3,), 2e-3))
    experts = layer.mlp.experts
    for e in (0, cfg.num_experts - 1):
        base = f"{p}.mlp.experts.{e}"
        assert torch.equal(
            experts.w13_weight[e],
            torch.cat([sd[f"{base}.gate_proj.weight"], sd[f"{base}.up_proj.weight"]]),
        )
        assert torch.equal(
            experts.w2_weight_scale[e].view(torch.uint8),
            sd[f"{base}.down_proj.weight_scale"].view(torch.uint8),
        )
        torch.testing.assert_close(
            experts.w13_weight_scale_2[e],
            torch.tensor([1e-3 * (e + 1), 1e-3 * (e + 2)]),
        )
        torch.testing.assert_close(
            experts.w2_weight_scale_2[e], torch.tensor(1e-3 * (e + 3))
        )
    # FP16 cannot hold the unscaled logits, so the scale is folded in.
    torch.testing.assert_close(
        model.lm_head.weight,
        (sd["lm_head.weight"].float() * 1.5e-4).half(),
        rtol=0,
        atol=0,
    )
    assert model.logits_processor.scale == 1.0


def test_lm_head_scale_on_logits_outside_fp16(dist, cpu_layers):
    cfg = tiny_config()
    model, _ = build(cfg, dtype=torch.bfloat16)
    sd = c3_state_dict(cfg, nvfp4=False)
    sd["lm_head.weight_scale_2"] = torch.tensor(1.5e-4)
    model.load_weights(sd.items())
    torch.testing.assert_close(model.lm_head.weight, sd["lm_head.weight"])
    assert model.logits_processor.scale == pytest.approx(1.5e-4)


def test_nvfp4_merged_shards_need_one_global_scale(dist, cpu_layers, nvfp4_moe_stub):
    cfg = tiny_config()
    model, _ = build(cfg, quant_config=nvfp4_config())
    sd = c3_state_dict(cfg, nvfp4=True)
    sd["model.layers.2.self_attn.v_proj.weight_scale_2"] = torch.tensor(0.5)
    with pytest.raises(ValueError, match="qkv_proj"):
        model.load_weights(sd.items())


class ReferenceAttention(torch.nn.Module):
    """Eager attention over the arguments the backend impl was given: its
    ``(left, right)`` window, sinks and scale. Masks like the Triton kernel:
    a key is visible when ``0 <= q_pos - k_pos <= left``."""

    def __init__(self, layer):
        super().__init__()
        self.impl = layer.impl
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size

    def forward(self, q, k, v):
        n = q.shape[0]
        groups = self.num_heads // self.num_kv_heads
        q = q.view(n, self.num_heads, self.head_size).transpose(0, 1)
        k = k.view(n, self.num_kv_heads, self.head_size).transpose(0, 1)
        v = v.view(n, self.num_kv_heads, self.head_size).transpose(0, 1)
        k = k.repeat_interleave(groups, dim=0)
        v = v.repeat_interleave(groups, dim=0)
        scores = (q @ k.transpose(1, 2)) * self.impl.scale
        distance = torch.arange(n)[:, None] - torch.arange(n)[None, :]
        visible = distance >= 0
        left = self.impl.sliding_window[0]
        if left >= 0:
            visible &= distance <= left
        scores = scores.masked_fill(~visible, float("-inf"))
        sinks = self.impl.sinks.view(-1, 1, 1).expand(-1, n, 1)
        probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1]
        return (probs @ v).transpose(0, 1).reshape(n, -1)


@pytest.mark.parametrize("layer_idx", [1, 3], ids=["sliding_rope", "full_nope"])
def test_attention_matches_transformers(dist, cpu_layers, layer_idx):
    modeling = pytest.importorskip("transformers.models.flint.modeling_flint")
    hf_configuration = pytest.importorskip(
        "transformers.models.flint.configuration_flint"
    )
    from transformers.masking_utils import (
        causal_mask_function,
        sliding_window_causal_mask_function,
    )

    window, seq_len = 5, 12
    cfg = tiny_config(sliding_window=window)
    model, _ = build(cfg, dtype=torch.float32)
    sd = c3_state_dict(cfg, nvfp4=False, seed=layer_idx)
    model.load_weights(sd.items())
    attn = model.model.layers[layer_idx].self_attn
    attn.attn = ReferenceAttention(attn.attn)

    hf_cfg = hf_configuration.FlintConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_parameters=dict(cfg.rope_parameters),
        layer_types=LAYER_TYPES,
        sliding_window=window,
        mlp_only_layers=[0],
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
    )
    hf_cfg._attn_implementation = "eager"
    hf_attn = modeling.FlintAttention(hf_cfg, layer_idx).float()
    p = f"model.layers.{layer_idx}.self_attn"
    hf_attn.load_state_dict(
        {
            **{
                f"{name}.weight": sd[f"{p}.{name}.weight"].float()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            },
            "sinks": sd[f"{p}.softmax_offset"],
        }
    )
    rotary = modeling.FlintRotaryEmbedding(hf_cfg)

    x = torch.randn(
        1, seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(7)
    )
    positions = torch.arange(seq_len)
    mask_fn = (
        sliding_window_causal_mask_function(window)
        if LAYER_TYPES[layer_idx] == "sliding_attention"
        else causal_mask_function
    )
    visible = mask_fn(0, 0, positions[:, None], positions[None, :])
    additive = torch.zeros(seq_len, seq_len).masked_fill(~visible, float("-inf"))
    with torch.no_grad():
        expected, _ = hf_attn(x, rotary(x, positions[None]), additive[None, None])
        actual = attn(positions, x[0])
    torch.testing.assert_close(actual, expected[0], rtol=1e-4, atol=1e-5)
