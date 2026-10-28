# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's NVFP4 GGUF loader must repack llama.cpp NVFP4 blocks into the
ModelOpt layout without changing any weight value, and must emit the 16-bit
tensors in FP16 for SM70 without silently overflowing."""

import numpy as np
import pytest
import torch

from vllm.models.flint import gguf_nvfp4
from vllm.models.flint.gguf_nvfp4 import (
    _repack_nvfp4,
    cast_checked,
    dequantize_nvfp4,
    find_gguf_file,
    gguf_nvfp4_tensor_specs,
    gguf_nvfp4_weights_iterator,
)

# gguf does not define __version__, so importorskip(minversion=...) would
# always skip; NVFP4 support is what gguf >= 0.19.0 adds.
gguf = pytest.importorskip("gguf")
if not hasattr(gguf.GGMLQuantizationType, "NVFP4"):
    pytest.skip("gguf >= 0.19.0 (NVFP4) is required", allow_module_level=True)

NVFP4 = gguf.GGMLQuantizationType.NVFP4
BF16 = gguf.GGMLQuantizationType.BF16


def _random_blocks(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    # Random bytes cover every E2M1 code and every representable scale byte,
    # including the sign bit and the 0x7F NaN pattern that ggml decodes as 0.
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (rows, cols // 64, 36), dtype=np.uint8)
    blocks[..., :4] = np.where(blocks[..., :4] == 0xFF, 0x7F, blocks[..., :4])
    return blocks.reshape(rows, -1)


def _bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def test_repack_matches_gguf_dequantize():
    rows, cols = 32, 512
    blocks = _random_blocks(rows, cols)
    weight, weight_scale = _repack_nvfp4(torch.from_numpy(blocks), cols)
    assert weight.shape == (rows, cols // 2)
    assert weight_scale.shape == (rows, cols // 16)
    ours = dequantize_nvfp4(weight, weight_scale)
    ref = gguf.quants.dequantize(blocks, NVFP4).reshape(rows, cols)
    torch.testing.assert_close(ours, torch.from_numpy(ref), atol=0, rtol=0)


def test_normalized_dequantize_is_exact_in_fp16():
    rows, cols = 64, 1024
    blocks = _random_blocks(rows, cols, seed=7)
    weight, weight_scale = _repack_nvfp4(torch.from_numpy(blocks), cols)
    fp16 = dequantize_nvfp4(weight, weight_scale, out_dtype=torch.float16)
    ref = gguf.quants.dequantize(blocks, NVFP4).reshape(rows, cols)
    torch.testing.assert_close(fp16.float(), torch.from_numpy(ref), atol=0, rtol=0)


def test_unrepresentable_scale_raises():
    blocks = _random_blocks(1, 64)
    blocks[0, 0] = 0xFF
    with pytest.raises(ValueError):
        _repack_nvfp4(torch.from_numpy(blocks), 64)


def _write_tiny(path, embed: torch.Tensor):
    hidden, inter, vocab, experts = 128, 64, 256, 4
    writer = gguf.GGUFWriter(str(path), "flint")

    def add_nvfp4(name, rows, cols, num, seed):
        blocks = _random_blocks(num * rows, cols, seed).reshape(num, rows, -1)
        data = blocks[0] if num == 1 else blocks
        writer.add_tensor(name + ".weight", data, raw_dtype=NVFP4)
        scale = np.full((num,), 0.5, dtype=np.float32) * np.arange(1, num + 1)
        writer.add_tensor(name + ".scale", scale)
        return data

    q = add_nvfp4("blk.2.attn_q", hidden, hidden, 1, 1)
    add_nvfp4("blk.2.ffn_up_exps", inter, hidden, experts, 2)
    head = add_nvfp4("output", vocab, hidden, 1, 3)
    writer.add_tensor("blk.2.attn_sinks.weight", np.zeros(4, dtype=np.float32))
    writer.add_tensor("token_embd.weight", _bf16_bits(embed), raw_dtype=BF16)
    writer.add_tensor(
        "blk.2.ffn_gate_inp.weight",
        _bf16_bits(torch.linspace(-3, 3, experts * hidden).reshape(experts, -1)),
        raw_dtype=BF16,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return q, head, (hidden, inter, vocab, experts)


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_iterator_maps_names_and_scales(tmp_path, out_dtype):
    path = tmp_path / "tiny.gguf"
    embed = torch.randn(256, 128) * 100
    q, head, (hidden, inter, vocab, experts) = _write_tiny(path, embed)

    stats: dict = {}
    tensors = dict(
        gguf_nvfp4_weights_iterator(
            str(path), "cpu", frozenset({"lm_head"}), out_dtype, stats
        )
    )

    q_w = tensors["model.layers.2.self_attn.q_proj.weight"]
    q_s = tensors["model.layers.2.self_attn.q_proj.weight_scale"]
    assert q_w.dtype == torch.uint8 and q_s.dtype == torch.float8_e4m3fn
    ref = gguf.quants.dequantize(q, NVFP4).reshape(hidden, hidden) * 0.5
    torch.testing.assert_close(
        dequantize_nvfp4(
            q_w, q_s, tensors["model.layers.2.self_attn.q_proj.weight_scale_2"]
        ),
        torch.from_numpy(ref),
        atol=0,
        rtol=0,
    )
    # Stacked experts are split per expert with their own global scale.
    for e in range(experts):
        gs = tensors[f"model.layers.2.mlp.experts.{e}.up_proj.weight_scale_2"]
        assert float(gs) == 0.5 * (e + 1)
        assert tensors[f"model.layers.2.mlp.experts.{e}.up_proj.weight"].shape == (
            inter,
            hidden // 2,
        )
    # The LM head is dequantized exactly (global scale kept apart).
    head_w = tensors["lm_head.weight"]
    assert head_w.dtype == out_dtype
    head_ref = gguf.quants.dequantize(head, NVFP4).reshape(vocab, hidden)
    torch.testing.assert_close(
        head_w.float(), torch.from_numpy(head_ref), atol=0, rtol=0
    )
    assert float(tensors["lm_head.weight_scale_2"]) == 0.5
    sinks = tensors["model.layers.2.self_attn.softmax_offset"]
    assert sinks.dtype == torch.float32
    # BF16 pass-through tensors come out in out_dtype with the same values.
    embed_out = tensors["model.embed_tokens.weight"]
    assert embed_out.dtype == out_dtype and embed_out.shape == (vocab, hidden)
    torch.testing.assert_close(
        embed_out.float(), embed.to(torch.bfloat16).float(), atol=0, rtol=0
    )
    assert tensors["model.layers.2.mlp.gate.weight"].dtype == out_dtype
    if out_dtype == torch.float16:
        assert set(stats) == {
            "model.embed_tokens.weight",
            "model.layers.2.mlp.gate.weight",
        }
        amax = float(embed.to(torch.bfloat16).float().abs().max())
        assert stats["model.embed_tokens.weight"]["max_abs"] == amax
        assert stats["model.layers.2.mlp.gate.weight"]["max_abs"] == 3.0
    else:
        assert stats == {}

    # The header-only plan matches what the iterator yields, in order.
    specs = list(gguf_nvfp4_tensor_specs(str(path), frozenset({"lm_head"}), out_dtype))
    assert [(n, d, s) for n, d, s in specs] == [
        (n, t.dtype, tuple(t.shape)) for n, t in tensors.items()
    ]


def test_bf16_overflow_raises(tmp_path):
    path = tmp_path / "tiny.gguf"
    embed = torch.randn(256, 128)
    embed[17, 5] = -70000.0
    _write_tiny(path, embed)
    with pytest.raises(ValueError, match=r"model\.embed_tokens\.weight.*70144"):
        dict(gguf_nvfp4_weights_iterator(str(path), "cpu", frozenset({"lm_head"})))
    # BF16 output keeps the value.
    tensors = dict(
        gguf_nvfp4_weights_iterator(str(path), "cpu", frozenset(), torch.bfloat16)
    )
    assert float(tensors["model.embed_tokens.weight"][17, 5]) == -70144.0


def test_cast_checked_limits():
    # 65280 is the largest bf16 value <= 65504; the next one is 65536.
    ok = torch.tensor([65280.0, -65280.0, -1e-30, 2.0, 0.0], dtype=torch.bfloat16)
    stats: dict = {}
    out = cast_checked("t", ok, torch.float16, stats)
    assert out.dtype == torch.float16
    assert out.tolist() == [65280.0, -65280.0, 0.0, 2.0, 0.0]
    assert stats["t"] == {
        "source_dtype": "bfloat16",
        "max_abs": 65280.0,
        "flushed_to_zero": 1,
    }
    for bad in (65536.0, float("-inf"), float("nan")):
        x = torch.tensor([1.0, bad], dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="t: max"):
            cast_checked("t", x, torch.float16)
    # Same dtype: returned as is, no statistics.
    assert cast_checked("u", out, torch.float16, stats) is out
    assert "u" not in stats


def test_find_gguf_file_resolves_hub_repo_ids(tmp_path, monkeypatch):
    """`vllm serve <org>/<repo>` must detect a GGUF-only repo, not just local paths."""
    (tmp_path / "flint.gguf").touch()
    monkeypatch.setattr(
        gguf_nvfp4,
        "any_pattern_in_repo_files",
        lambda repo, patterns, revision: repo == "org/gguf",
    )
    monkeypatch.setattr(
        gguf_nvfp4, "download_weights_from_hf", lambda *args: str(tmp_path)
    )
    assert find_gguf_file("org/gguf") == str(tmp_path / "flint.gguf")
    assert find_gguf_file("org/safetensors") is None
