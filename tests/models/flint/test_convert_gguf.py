# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flint GGUF -> ModelOpt NVFP4 safetensors converter must be lossless,
must write a quantization config that vLLM's ModelOpt loader path parses, and
must refuse BF16 tensors that do not fit in FP16."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from vllm.models.flint import convert_gguf
from vllm.models.flint.convert_gguf import (
    convert,
    main,
    modelopt_quantization_config,
    parse_size,
    rescale_block_scales,
)
from vllm.models.flint.gguf_nvfp4 import _repack_nvfp4, dequantize_nvfp4

gguf = pytest.importorskip("gguf")
if not hasattr(gguf.GGMLQuantizationType, "NVFP4"):
    pytest.skip("gguf >= 0.19.0 (NVFP4) is required", allow_module_level=True)

NVFP4 = gguf.GGMLQuantizationType.NVFP4
BF16 = gguf.GGMLQuantizationType.BF16

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 128, 2, 1, 64
DENSE, MOE, SHARED, EXPERTS, VOCAB = 128, 64, 64, 4, 256

# The exact quantization_config the converter must emit (contract C3).
EXPECTED_QUANT_CONFIG = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["model.embed_tokens", "*.mlp.gate", "lm_head"],
    },
}

TINY_CONFIG = {
    "architectures": ["FlintForCausalLM"],
    "model_type": "flint",
    "hidden_size": HIDDEN,
    "intermediate_size": DENSE,
    "num_hidden_layers": 2,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "moe_intermediate_size": MOE,
    "shared_expert_intermediate_size": SHARED,
    "num_experts": EXPERTS,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "router_score_function": "sigmoid",
    "mlp_only_layers": [0],
    "layer_types": ["full_attention", "sliding_attention"],
    "no_rope_layer_indices": [0],
    "sliding_window": 513,
    "softmax_type": "learnable",
    "vocab_size": VOCAB,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


def _random_blocks(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (rows, cols // 64, 36), dtype=np.uint8)
    blocks[..., :4] = np.where(blocks[..., :4] == 0xFF, 0x7F, blocks[..., :4])
    return blocks.reshape(rows, -1)


def _bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def _write_gguf(path, embed_scale: float = 1.0, shared: bool = False) -> dict:
    """A two-layer Flint GGUF (dense layer 0, MoE layer 1) with every tensor
    kind of the real checkpoint. Unless ``shared``, the global scales of
    layer 1's q/k/v differ by a power of two and those of layer 0's gate/up by
    a ratio of 0.8. Returns the reference values by GGUF name."""
    writer = gguf.GGUFWriter(str(path), "flint")
    ref: dict = {"nvfp4": {}, "scale": {}, "bf16": {}, "f32": {}}
    seed = iter(range(1000))

    def nvfp4(name, rows, cols, scales, normal=False):
        num = len(scales)
        blocks = _random_blocks(num * rows, cols, next(seed)).reshape(num, rows, -1)
        if normal:
            # Block scales of at least 2^-3, so that halving them stays exact.
            scale_bytes = blocks.reshape(-1, 36)[:, :4]
            blocks.reshape(-1, 36)[:, :4] = 0x20 + scale_bytes % 0x5F
        data = blocks[0] if num == 1 else blocks
        writer.add_tensor(name + ".weight", data, raw_dtype=NVFP4)
        writer.add_tensor(name + ".scale", np.array(scales, dtype=np.float32))
        ref["nvfp4"][name] = data
        ref["scale"][name] = np.array(scales, dtype=np.float32)

    def bf16(name, values):
        writer.add_tensor(name, _bf16_bits(values), raw_dtype=BF16)
        ref["bf16"][name] = values.to(torch.bfloat16)

    def f32(name, n):
        values = np.random.default_rng(next(seed)).standard_normal(n, np.float32)
        writer.add_tensor(name, values)
        ref["f32"][name] = values

    q_rows, kv_rows = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for layer in range(2):
        blk = f"blk.{layer}."
        for norm in ("attn_norm", "ffn_norm"):
            f32(blk + norm + ".weight", HIDDEN)
        f32(blk + "attn_q_norm.weight", HEAD_DIM)
        f32(blk + "attn_k_norm.weight", HEAD_DIM)
        f32(blk + "attn_sinks.weight", HEADS)
        # Layer 0 shares the q/k/v global scale, layer 1 does not.
        k_scale = 0.25 if layer == 0 or shared else 0.5
        nvfp4(blk + "attn_q", q_rows, HIDDEN, [0.25], normal=True)
        nvfp4(blk + "attn_k", kv_rows, HIDDEN, [k_scale])
        nvfp4(blk + "attn_v", kv_rows, HIDDEN, [0.25], normal=True)
        nvfp4(blk + "attn_output", HIDDEN, q_rows, [0.75])
    nvfp4("blk.0.ffn_gate", DENSE, HIDDEN, [0.125])
    nvfp4("blk.0.ffn_up", DENSE, HIDDEN, [0.125 if shared else 0.1])
    nvfp4("blk.0.ffn_down", HIDDEN, DENSE, [0.0625])
    router = torch.linspace(-3, 3, EXPERTS * HIDDEN).reshape(EXPERTS, HIDDEN)
    bf16("blk.1.ffn_gate_inp.weight", router)
    nvfp4("blk.1.ffn_gate_shexp", SHARED, HIDDEN, [0.375])
    nvfp4("blk.1.ffn_up_shexp", SHARED, HIDDEN, [0.375])
    nvfp4("blk.1.ffn_down_shexp", HIDDEN, SHARED, [0.5])
    gate = [0.01 * (e + 1) for e in range(EXPERTS)]
    up = [g * (3 if e == 2 else 1) for e, g in enumerate(gate)]
    nvfp4("blk.1.ffn_gate_exps", MOE, HIDDEN, gate)
    nvfp4("blk.1.ffn_up_exps", MOE, HIDDEN, up)
    nvfp4("blk.1.ffn_down_exps", HIDDEN, MOE, [0.02] * EXPERTS)
    nvfp4("output", VOCAB, HIDDEN, [0.0078125])
    f32("output_norm.weight", HIDDEN)
    embed = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(0))
    embed[3, 4] = embed_scale * 1000.0
    bf16("token_embd.weight", embed)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return ref


def _hf_config_dir(root) -> str:
    os.makedirs(root)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f)
    files = {
        "tokenizer_config.json": "{}",
        "tokenizer.json": "{}",
        "chat_template.jinja": "{{ messages }}",
        "generation_config.json": '{"temperature": 1.0}',
        # Not a config file: must not be copied.
        "model-00001-of-00015.safetensors": "",
    }
    for name, text in files.items():
        with open(os.path.join(root, name), "w") as f:
            f.write(text)
    return str(root)


def _load_checkpoint(out_dir) -> tuple[dict[str, torch.Tensor], dict]:
    with open(os.path.join(out_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(os.path.join(out_dir, shard), "pt") as f:
            for name in f.keys():  # noqa: SIM118
                assert index["weight_map"][name] == shard
                tensors[name] = f.get_tensor(name)
    assert set(tensors) == set(index["weight_map"])
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    assert index["metadata"]["total_size"] == total
    return tensors, index


def _hf_base(gguf_name: str) -> tuple[str, bool]:
    from vllm.models.flint.gguf_nvfp4 import _map_name

    return _map_name(gguf_name)


@pytest.fixture
def converted(tmp_path):
    src = tmp_path / "flint.gguf"
    ref = _write_gguf(src)
    hf_dir = _hf_config_dir(tmp_path / "hf")
    out = tmp_path / "out"
    # Small shards force several files; the CLI parses the size.
    assert (
        main([str(src), str(out), "--hf-config", hf_dir, "--max-shard-size", "100KB"])
        == 0
    )
    return ref, out


def _is_power_of_two(x: float) -> bool:
    return math.frexp(x)[0] == 0.5


def test_round_trip_is_lossless(converted):
    """dequant(GGUF) == dequant(converted), exactly, except for the shards
    whose global scale was unified with a non-power-of-two ratio: their codes
    are unchanged and their block scales are within E4M3 rounding."""
    ref, out = converted
    tensors, index = _load_checkpoint(out)
    assert len(set(index["weight_map"].values())) > 2
    with open(out / "conversion_report.json") as f:
        unified = json.load(f)["unified_weight_scale_2"]
    assert set(unified) == {
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.0.mlp.up_proj",
    }
    expected = set()
    for name, blocks in ref["nvfp4"].items():
        hf_base, stacked = _hf_base(name)
        scales = ref["scale"][name]
        stack = blocks if stacked else blocks[None]
        for e, (b, s) in enumerate(zip(stack, scales)):
            rows = b.shape[0]
            normalized = gguf.quants.dequantize(b, NVFP4).reshape(rows, -1)
            want = torch.from_numpy(normalized * s)
            base = hf_base.format(e) if stacked else hf_base
            scale_2 = tensors[base + ".weight_scale_2"]
            assert scale_2.dtype == torch.float32 and scale_2.shape == ()
            common = unified[base]["weight_scale_2"] if base in unified else s
            assert float(scale_2) == float(common)
            if base == "lm_head":
                weight = tensors["lm_head.weight"]
                assert weight.dtype == torch.float16
                assert torch.equal(weight.float(), torch.from_numpy(normalized))
                got = weight.float() * scale_2
                expected |= {base + ".weight", base + ".weight_scale_2"}
            else:
                weight = tensors[base + ".weight"]
                block_scale = tensors[base + ".weight_scale"]
                assert weight.dtype == torch.uint8
                assert block_scale.dtype == torch.float8_e4m3fn
                assert weight.shape == (rows, normalized.shape[1] // 2)
                assert block_scale.shape == (rows, normalized.shape[1] // 16)
                got = dequantize_nvfp4(weight, block_scale, scale_2)
                expected |= {
                    base + s for s in (".weight", ".weight_scale", ".weight_scale_2")
                }
                if base in unified and not _is_power_of_two(unified[base]["ratio"]):
                    cols = normalized.shape[1]
                    packed, ref_scale = _repack_nvfp4(torch.from_numpy(b), cols)
                    assert torch.equal(weight, packed)
                    want_eff = ref_scale.double() * float(s)
                    got_eff = block_scale.double() * float(common)
                    # Half an E4M3 step: 2^-4 relative (normal), 2^-10 absolute.
                    bound = torch.clamp(want_eff / 16, min=2.0**-10 * float(common))
                    assert torch.all((got_eff - want_eff).abs() <= bound), base
                    continue
            assert torch.equal(got, want), base
    for name, values in ref["bf16"].items():
        hf_name, _ = _hf_base(name)
        assert tensors[hf_name].dtype == torch.float16
        assert torch.equal(tensors[hf_name].float(), values.float())
        expected.add(hf_name)
    for name, values in ref["f32"].items():
        hf_name, _ = _hf_base(name)
        assert tensors[hf_name].dtype == torch.float32
        assert torch.equal(tensors[hf_name], torch.from_numpy(values))
        expected.add(hf_name)
    assert set(tensors) == expected
    assert tensors["model.layers.1.self_attn.softmax_offset"].shape == (HEADS,)
    assert "lm_head.weight_scale" not in tensors
    assert not any(n.endswith("input_scale") for n in tensors)


def test_config_files_and_report(converted):
    ref, out = converted
    with open(out / "config.json") as f:
        config = json.load(f)
    assert config["torch_dtype"] == "float16"
    assert config["quantization_config"] == EXPECTED_QUANT_CONFIG
    assert modelopt_quantization_config() == EXPECTED_QUANT_CONFIG
    for key, value in TINY_CONFIG.items():
        if key != "torch_dtype":
            assert config[key] == value
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        assert (out / name).is_file()
    assert not (out / "model-00001-of-00015.safetensors").exists()
    assert not list(out.glob("*.partial"))

    with open(out / "conversion_report.json") as f:
        report = json.load(f)
    counts = report["tensor_counts"]
    assert counts["nvfp4_modules"] == 7 + 7 + EXPERTS * 3
    assert counts["nvfp4_tensors"] == 3 * counts["nvfp4_modules"]
    assert counts["dequantized_modules"] == 1
    assert counts["passthrough_float32"] == len(ref["f32"])
    assert counts["passthrough_float16"] == len(ref["bf16"])
    assert report["gguf_tensors"] == {
        "F32": len(ref["f32"]) + len(ref["scale"]),
        "NVFP4": len(ref["nvfp4"]),
        "BF16": len(ref["bf16"]),
    }
    merged = report["merged_shards_share_weight_scale_2"]
    assert merged["qkv"]["modules"] == 2 and merged["qkv"]["shared"] == 1
    assert merged["qkv"]["mismatched"] == {
        "model.layers.1.self_attn": {"q": 0.25, "k": 0.5, "v": 0.25}
    }
    assert merged["dense_gate_up"] == {
        "modules": 1,
        "shared": 0,
        "all_shared": False,
        "mismatched": {
            "model.layers.0.mlp": {"gate": 0.125, "up": float(np.float32(0.1))}
        },
    }
    assert merged["shared_expert_gate_up"]["all_shared"]
    assert merged["expert_gate_up"]["modules"] == EXPERTS
    assert list(merged["expert_gate_up"]["mismatched"]) == [
        "model.layers.1.mlp.experts.2"
    ]
    unified = report["unified_weight_scale_2"]
    for shard in ("q_proj", "v_proj"):
        assert unified[f"model.layers.1.self_attn.{shard}"] == {
            "fused_module": "model.layers.1.self_attn",
            "source_weight_scale_2": 0.25,
            "weight_scale_2": 0.5,
            "ratio": 0.5,
            "nonzero_blocks": int(ref["nvfp4"][f"blk.1.attn_{shard[0]}"].size / 36 * 4),
            "max_rel_error": 0.0,
            "underflowed_to_zero": 0,
        }
    up = unified["model.layers.0.mlp.up_proj"]
    assert up["fused_module"] == "model.layers.0.mlp"
    assert up["weight_scale_2"] == 0.125
    assert up["ratio"] == float(np.float32(0.1)) / 0.125
    # 1/16 for normal E4M3 results, up to 1/4 for the smallest subnormals.
    assert 0 < up["max_rel_error"] <= 0.25
    assert up["underflowed_to_zero"] == 0
    # Routed experts keep their own gate/up scales.
    assert not any(".experts." in name for name in unified)
    casts = report["fp16_casts"]
    assert set(casts) == {"model.embed_tokens.weight", "model.layers.1.mlp.gate.weight"}
    assert casts["model.embed_tokens.weight"]["max_abs"] == 1000.0
    assert casts["model.layers.1.mlp.gate.weight"]["max_abs"] == 3.0
    lm_head = report["dequantized"]["lm_head"]
    assert lm_head["dtype"] == "float16" and lm_head["weight_scale_2"] == 0.0078125
    fold = report["fp16_folded_block_scales"]
    nonzero = 0
    for name, blocks in ref["nvfp4"].items():
        if name != "output":
            scales = blocks.reshape(-1, 36)[:, :4] & 0x7F
            nonzero += int(((scales != 0) & (scales != 0x7F)).sum())
    assert fold["nonzero"] == nonzero
    assert 0 < fold["subnormal"] <= nonzero
    assert report["quantization_config"] == EXPECTED_QUANT_CONFIG


def test_quant_config_parses_through_vllm_loader(converted, monkeypatch):
    """Follow vLLM's own path: config.json -> hf_config.quantization_config ->
    ModelArchConfigConvertor -> ModelConfig._verify_quantization override
    detection -> weight_utils.get_quant_config -> ModelOptNvFp4Config."""
    from vllm.config import model as model_mod
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4W4A16LinearMethod,
    )
    from vllm.model_executor.model_loader.weight_utils import get_quant_config
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    _, out = converted
    hf_config = get_config(str(out), trust_remote_code=False)
    assert hf_config.model_type == "flint"
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
        hf_config.model_type, ModelArchConfigConvertorBase
    )
    quant_cfg = convertor_cls(
        hf_config, hf_config.get_text_config()
    ).get_quantization_config()
    assert quant_cfg["quant_method"] == "modelopt"

    # The CPU test platform does not list modelopt_fp4; CUDA allows it.
    monkeypatch.setattr(
        model_mod.current_platform, "verify_quantization", lambda q: None
    )
    model_config = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(quantization_config=quant_cfg),
        hf_config=hf_config,
        hf_overrides={},
        allow_deprecated_quantization=False,
    )
    model_mod.ModelConfig._verify_quantization(model_config)
    assert model_config.quantization == "modelopt_fp4"

    qc = get_quant_config(model_config, SimpleNamespace())
    assert type(qc) is ModelOptNvFp4Config
    assert qc.get_name() == "modelopt_fp4"
    assert qc.quant_method == "W4A16_NVFP4"
    assert qc.is_checkpoint_nvfp4_serialized
    assert qc.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod
    assert qc.group_size == 16
    assert qc.kv_cache_quant_algo is None
    assert qc.exclude_modules == ["model.embed_tokens", "*.mlp.gate", "lm_head"]

    qc.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    for prefix in ("lm_head", "model.embed_tokens", "model.layers.7.mlp.gate"):
        assert qc.is_layer_excluded(prefix), prefix
    for prefix in (
        "model.layers.7.self_attn.qkv_proj",
        "model.layers.7.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.7.mlp.shared_expert.gate_up_proj",
        "model.layers.7.mlp.shared_expert.down_proj",
        "model.layers.7.mlp.experts",
    ):
        assert not qc.is_layer_excluded(prefix), prefix


def test_bf16_overflow_fails_the_conversion(tmp_path):
    src = tmp_path / "flint.gguf"
    _write_gguf(src, embed_scale=100.0)
    hf_dir = _hf_config_dir(tmp_path / "hf")
    out = tmp_path / "out"
    amax = float(torch.tensor(100000.0).to(torch.bfloat16))
    with pytest.raises(ValueError) as err:
        convert(str(src), str(out), hf_dir, verbose=False)
    assert "model.embed_tokens.weight" in str(err.value)
    assert str(amax) in str(err.value)
    assert not (out / "model.safetensors.index.json").exists()
    assert not list(out.glob("*.partial"))


_E4M3_POSITIVE = torch.arange(1, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)


def test_rescale_block_scales_power_of_two_is_exact():
    # Every normal E4M3 scale of at least 2^-4 stays normal at a quarter.
    old = _E4M3_POSITIVE[_E4M3_POSITIVE.float() >= 2**-4].reshape(1, -1)
    new, stats = rescale_block_scales(old, 0.25)
    assert new.dtype == torch.float8_e4m3fn and new.shape == old.shape
    assert torch.equal(new.float(), old.float() * 0.25)
    assert stats == {
        "nonzero_blocks": old.numel(),
        "max_rel_error": 0.0,
        "underflowed_to_zero": 0,
    }
    same, stats = rescale_block_scales(_E4M3_POSITIVE, 1.0)
    assert torch.equal(same.view(torch.uint8), _E4M3_POSITIVE.view(torch.uint8))
    assert stats["max_rel_error"] == 0.0


def test_rescale_block_scales_rounds_to_nearest_e4m3():
    old = torch.cat([_E4M3_POSITIVE, torch.zeros(3, dtype=torch.float8_e4m3fn)])
    table = _E4M3_POSITIVE.double()
    for ratio in (0.8, 0.3, 1 / 3, 2**-4 * 0.9):
        new, stats = rescale_block_scales(old, ratio)
        want = old.double() * ratio
        nearest = (table[None, :] - want[:, None]).abs().min(dim=1).values
        nearest = torch.minimum(nearest, want)  # rounding to zero
        assert torch.equal((new.double() - want).abs(), nearest)
        nonzero = want != 0
        rel = ((new.double() - want).abs()[nonzero] / want[nonzero]).max()
        assert stats["max_rel_error"] == float(rel)
        assert stats["nonzero_blocks"] == len(_E4M3_POSITIVE)
        assert stats["underflowed_to_zero"] == int(
            (nonzero & (new.double() == 0)).sum()
        )
        # Half an E4M3 step: 1/16 of the value for normal results.
        normal = want >= 2**-6
        assert ((new.double() - want).abs()[normal] <= want[normal] / 16).all()
    # The smallest subnormal scale (2^-9) underflows below half of itself.
    _, stats = rescale_block_scales(_E4M3_POSITIVE[:1], 0.4)
    assert stats["underflowed_to_zero"] == 1 and stats["max_rel_error"] == 1.0
    with pytest.raises(ValueError):
        rescale_block_scales(old, 1.5)


def test_strict_merged_scales_raises(tmp_path):
    src = tmp_path / "flint.gguf"
    _write_gguf(src)
    hf_dir = _hf_config_dir(tmp_path / "hf")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="strict-merged-scales") as err:
        main([str(src), str(out), "--hf-config", hf_dir, "--strict-merged-scales"])
    assert "model.layers.1.self_attn" in str(err.value)
    assert "model.layers.0.mlp" in str(err.value)
    assert "experts" not in str(err.value)
    assert not out.exists()

    # Shared dense scales pass in strict mode; expert gate/up may still differ.
    shared_src = tmp_path / "shared.gguf"
    _write_gguf(shared_src, shared=True)
    report = convert(
        str(shared_src), str(out), hf_dir, strict_merged_scales=True, verbose=False
    )
    assert report["unified_weight_scale_2"] == {}
    merged = report["merged_shards_share_weight_scale_2"]
    assert all(merged[k]["all_shared"] for k in ("qkv", "dense_gate_up"))
    assert not merged["expert_gate_up"]["all_shared"]


def test_refuses_mismatched_config_and_existing_output(tmp_path):
    src = tmp_path / "flint.gguf"
    _write_gguf(src)
    hf_dir = _hf_config_dir(tmp_path / "hf")
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump({**TINY_CONFIG, "num_experts": 8}, f)
    with pytest.raises(ValueError, match="num_experts=8"):
        convert(str(src), str(tmp_path / "a"), hf_dir, verbose=False)
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f)
    convert(str(src), str(tmp_path / "b"), hf_dir, verbose=False)
    with pytest.raises(FileExistsError):
        convert(str(src), str(tmp_path / "b"), hf_dir, verbose=False)
    convert(str(src), str(tmp_path / "b"), hf_dir, verbose=False, overwrite=True)


def test_repo_ids_are_downloaded(tmp_path, monkeypatch):
    import huggingface_hub

    src = tmp_path / "flint.gguf"
    _write_gguf(src)
    hf_dir = _hf_config_dir(tmp_path / "hf")
    calls: list[str] = []

    def find_gguf_file(repo, revision, download_dir):
        calls.append(repo)
        return str(src)

    monkeypatch.setattr(convert_gguf, "find_gguf_file", find_gguf_file)

    def snapshot_download(repo, revision, cache_dir, allow_patterns):
        calls.append(repo)
        assert "config.json" in allow_patterns and "*" not in allow_patterns
        return hf_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    report = convert(
        "org/flint-gguf", str(tmp_path / "out"), "org/flint", verbose=False
    )
    assert calls == ["org/flint-gguf", "org/flint"]
    assert report["hf_config"] == "org/flint"


def test_parse_size():
    assert parse_size("5GB") == 5 * 10**9
    assert parse_size("4.5GB") == 4.5 * 10**9
    assert parse_size("4GiB") == 4 * 2**30
    assert parse_size("100KB") == 100_000
    assert parse_size("123") == 123
    with pytest.raises(ValueError):
        parse_size("lots")


_HEADER_BYTES = 8 << 20


def test_real_gguf_header_maps_every_tensor(tmp_path):
    """Map the published GGUF's tensor names from its header only (one range
    request); skipped without network access."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    repo, filename = "RecursiveSI/flint-nvfp4-gguf", "flint-nvfp4.gguf"
    try:
        size = huggingface_hub.HfApi().get_paths_info(repo, [filename])[0].size
        url = huggingface_hub.hf_hub_url(repo, filename)
        response = huggingface_hub.get_session().get(
            url, headers={"Range": f"bytes=0-{_HEADER_BYTES - 1}"}, timeout=60
        )
        response.raise_for_status()
        config_path = huggingface_hub.hf_hub_download(
            "RecursiveSI/flint", "config.json"
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Hugging Face Hub not reachable: {e}")
    header = response.content
    assert response.status_code == 206 and len(header) == _HEADER_BYTES
    # A sparse file of the real size: the reader maps it but only the header
    # is read.
    path = tmp_path / filename
    with open(path, "wb") as f:
        f.truncate(size)
        f.write(header)
    reader = convert_gguf.open_gguf_reader(str(path))
    assert reader.data_offset <= len(header)
    specs = list(convert_gguf.gguf_nvfp4_tensor_specs(reader, convert_gguf.DEQUANTIZED))
    with open(config_path) as f:
        config = json.load(f)
    convert_gguf._check_against_config(config, specs)
    shapes = {name: (dtype, shape) for name, dtype, shape in specs}
    assert len(shapes) == len(specs)
    layers, experts = config["num_hidden_layers"], config["num_experts"]
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    q_rows = config["num_attention_heads"] * config["head_dim"]
    assert shapes["model.embed_tokens.weight"] == (torch.float16, (vocab, hidden))
    assert shapes["lm_head.weight"] == (torch.float16, (vocab, hidden))
    assert shapes["lm_head.weight_scale_2"] == (torch.float32, ())
    assert shapes["model.layers.0.mlp.gate_proj.weight"] == (
        torch.uint8,
        (config["intermediate_size"], hidden // 2),
    )
    last = f"model.layers.{layers - 1}."
    assert shapes[last + "self_attn.q_proj.weight_scale"] == (
        torch.float8_e4m3fn,
        (q_rows, hidden // 16),
    )
    assert shapes[last + "self_attn.softmax_offset"] == (
        torch.float32,
        (config["num_attention_heads"],),
    )
    assert shapes[last + "mlp.gate.weight"] == (torch.float16, (experts, hidden))
    assert shapes[f"{last}mlp.experts.{experts - 1}.down_proj.weight"] == (
        torch.uint8,
        (hidden, config["moe_intermediate_size"] // 2),
    )
    assert shapes[last + "mlp.shared_expert.up_proj.weight"][1] == (
        config["shared_expert_intermediate_size"],
        hidden // 2,
    )
    dense = len(config["mlp_only_layers"])
    nvfp4_modules = dense * 7 + (layers - dense) * (7 + 3 * experts)
    assert sum(n.endswith(".weight_scale") for n in shapes) == nvfp4_modules
    assert not any(dtype == torch.bfloat16 for dtype, _ in shapes.values())
