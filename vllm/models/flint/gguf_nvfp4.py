# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load Flint's NVFP4 GGUF checkpoint as ModelOpt-layout NVFP4 tensors.

A GGUF NVFP4 super-block holds 64 values as four 16-value sub-blocks, each with
a UE4M3 scale byte and 8 bytes of E2M1 codes (low nibbles = values 0..7, high
nibbles = values 8..15). The value is ``e4m3(scale) * e2m1(code) * tensor_scale``
where ``tensor_scale`` is the companion ``<name>.scale`` F32 tensor (one per
expert for stacked expert tensors). ModelOpt packs consecutive values into one
byte (low nibble first) and stores the same three factors as ``weight``,
``weight_scale`` and ``weight_scale_2``, so conversion is a lossless repack.

SM70 has no BF16 compute, so the 16-bit pass-through tensors (embedding,
routers) are emitted in ``out_dtype`` (FP16 by default). A BF16 -> FP16 cast
refuses any tensor whose largest magnitude does not fit in FP16. F32 tensors
(norms, attention sinks) pass through unchanged. Everything runs on the CPU
unless a ``device`` is given.
"""

import glob
import os
import warnings
from collections.abc import Iterator
from typing import Any

import huggingface_hub
import numpy as np
import regex as re
import torch

from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.transformers_utils.repo_utils import any_pattern_in_repo_files

_GGUF_NVFP4 = 40
_GGUF_F32 = 0
_GGUF_F16 = 1
_GGUF_BF16 = 30
_QK = 64
_BLOCK_BYTES = 36
_FP16_MAX = float(torch.finfo(torch.float16).max)
_CAST_CHUNK = 1 << 24

_LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_sinks.weight": "self_attn.softmax_offset",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}
_EXPERT_NAMES = {
    "ffn_gate_exps": "gate_proj",
    "ffn_up_exps": "up_proj",
    "ffn_down_exps": "down_proj",
}
_GLOBAL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output": "lm_head",
}


def find_gguf_file(
    model_path: str, revision: str | None = None, download_dir: str | None = None
) -> str | None:
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return model_path
    if not os.path.isdir(model_path):
        # Offline, the repo cannot be listed; look in the cached snapshot instead.
        offline = huggingface_hub.constants.HF_HUB_OFFLINE
        if not offline and not any_pattern_in_repo_files(
            model_path, ["*.gguf"], revision=revision
        ):
            return None
        try:
            model_path = download_weights_from_hf(
                model_path, download_dir, ["*.gguf"], revision
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            return None
    files = sorted(glob.glob(os.path.join(model_path, "*.gguf")))
    if len(files) > 1:
        raise ValueError(f"Expected one GGUF file in {model_path}, found {files}")
    return files[0] if files else None


def _repack_nvfp4(
    blocks: torch.Tensor, n_cols: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., n_cols / 64 * 36] GGUF bytes -> (packed [..., n_cols / 2] uint8,
    block scales [..., n_cols / 16] float8_e4m3fn)."""
    lead = blocks.shape[:-1]
    blocks = blocks.reshape(*lead, n_cols // _QK, _BLOCK_BYTES)
    # ggml decodes the scales as unsigned E4M3 and the NaN pattern as zero;
    # 0xFF would decode to 480, which E4M3 cannot represent.
    scales = blocks[..., :4]
    if bool((scales == 0xFF).any()):
        raise ValueError("NVFP4 GGUF block scale 0xFF is not supported")
    scales = torch.where(scales == 0x7F, torch.zeros_like(scales), scales & 0x7F)
    qs = blocks[..., 4:].reshape(*lead, n_cols // _QK, 4, 8)
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return (
        packed.reshape(*lead, n_cols // 2).contiguous(),
        scales.reshape(*lead, n_cols // 16).contiguous().view(torch.float8_e4m3fn),
    )


_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1 = _E2M1_POS + tuple(-v for v in _E2M1_POS)


def dequantize_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """ModelOpt-layout NVFP4 [N, K / 2] -> [N, K]. Without ``global_scale`` the
    result is ``e2m1 * e4m3``, which is exact in fp16 and bf16 (at most 6 * 448
    in magnitude, at least 2^-10 when non-zero, with at most 6 significant
    bits)."""
    chunk_rows = 8192
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    out = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] * 2),
        dtype=out_dtype,
        device=weight.device,
    )
    for start in range(0, weight.shape[0], chunk_rows):
        w = weight[start : start + chunk_rows]
        codes = torch.stack([w & 0x0F, w >> 4], dim=-1).int()
        values = lut[codes].reshape(*w.shape[:-1], -1, 16)
        values = values * weight_scale[start : start + chunk_rows].float().unsqueeze(-1)
        values = values.flatten(-2)
        if global_scale is not None:
            values = values * global_scale.float().to(weight.device)
        out[start : start + chunk_rows] = values
    return out


def cast_checked(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    stats: dict[str, dict[str, Any]] | None = None,
) -> torch.Tensor:
    """Cast a 16-bit pass-through tensor to ``dtype``.

    A cast to FP16 raises ``ValueError`` (naming the tensor and its max |x|)
    when any value is non-finite or larger than 65504 in magnitude, instead of
    silently producing ``inf``. ``stats[name]`` records the max |x| and the
    number of non-zero values that FP16 flushes to zero."""
    if tensor.dtype == dtype:
        return tensor
    if dtype != torch.float16:
        return tensor.to(dtype)
    out = torch.empty(tensor.shape, dtype=dtype, device=tensor.device)
    src = tensor.reshape(-1)
    dst = out.view(-1)
    # A tensor maximum propagates NaN, unlike Python's max().
    amax_t = torch.zeros((), dtype=torch.float32, device=tensor.device)
    flushed = 0
    for start in range(0, src.numel(), _CAST_CHUNK):
        chunk = src[start : start + _CAST_CHUNK].float()
        half = chunk.to(dtype)
        amax_t = torch.maximum(amax_t, chunk.abs().max())
        flushed += int(((chunk != 0) & (half == 0)).sum())
        dst[start : start + _CAST_CHUNK] = half
    amax = float(amax_t)
    if not amax <= _FP16_MAX:
        raise ValueError(
            f"{name}: max |x| = {amax} exceeds the float16 range "
            f"({_FP16_MAX:g}); refusing to cast {tensor.dtype} to float16"
        )
    if stats is not None:
        stats[name] = {
            "source_dtype": str(tensor.dtype).removeprefix("torch."),
            "max_abs": amax,
            "flushed_to_zero": flushed,
        }
    return out


def _as_tensor(data: np.ndarray) -> torch.Tensor:
    # The reader memory-maps the file read-only; tensors are only read.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(np.asarray(data))


def _map_name(name: str) -> tuple[str, bool]:
    """GGUF tensor base name -> (HF base name, is_stacked_experts)."""
    if name in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[name], False
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m is None:
        raise ValueError(f"Unexpected tensor in Flint GGUF: {name}")
    layer, rest = m.groups()
    prefix = f"model.layers.{layer}."
    if rest in _EXPERT_NAMES:
        return prefix + "mlp.experts.{}." + _EXPERT_NAMES[rest], True
    if rest not in _LAYER_NAMES:
        raise ValueError(f"Unexpected tensor in Flint GGUF: {name}")
    return prefix + _LAYER_NAMES[rest], False


def open_gguf_reader(path: Any):
    try:
        import gguf
    except ImportError as err:
        raise ImportError(
            "Loading Flint NVFP4 GGUF checkpoints requires the `gguf` package "
            "(pip install 'gguf>=0.19.0')."
        ) from err

    reader = path if isinstance(path, gguf.GGUFReader) else gguf.GGUFReader(path)
    prec_a4 = reader.fields.get("general.tensor_extra.prec_a4")
    if prec_a4 is not None and any(prec_a4.contents()):
        raise ValueError(
            "GGUF marks tensors with FP4 activations (prec_a4); only weight-only "
            "NVFP4 (W4A16) is supported."
        )
    return reader


def _pass_through_dtype(name: str, ggml_type: int, out_dtype: torch.dtype):
    if ggml_type == _GGUF_F32:
        return torch.float32
    if ggml_type in (_GGUF_BF16, _GGUF_F16):
        return out_dtype
    raise ValueError(f"Unsupported GGUF tensor type for {name}: {ggml_type}")


def _nvfp4_dims(t) -> tuple[int, int, int | None]:
    """(n_cols, rows, experts or None) of a GGUF NVFP4 tensor."""
    dims = [int(d) for d in t.shape]
    if len(dims) not in (2, 3) or dims[0] % _QK != 0:
        raise ValueError(f"Unsupported NVFP4 tensor shape for {t.name}: {dims}")
    return dims[0], dims[1], dims[2] if len(dims) == 3 else None


def gguf_nvfp4_tensor_specs(
    path: Any,
    dequantize: frozenset[str] = frozenset(),
    out_dtype: torch.dtype = torch.float16,
) -> Iterator[tuple[str, torch.dtype, tuple[int, ...]]]:
    """(name, dtype, shape) of every tensor ``gguf_nvfp4_weights_iterator``
    yields, in the same order, read from the GGUF header alone."""
    reader = open_gguf_reader(path)
    tensors = {t.name: t for t in reader.tensors}
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue
        if int(t.tensor_type) != _GGUF_NVFP4:
            hf_name, _ = _map_name(name)
            dtype = _pass_through_dtype(name, int(t.tensor_type), out_dtype)
            yield hf_name, dtype, tuple(int(d) for d in reversed(t.shape))
            continue
        base = name.removesuffix(".weight")
        hf_base, stacked = _map_name(base)
        n_cols, rows, experts = _nvfp4_dims(t)
        if stacked != (experts is not None):
            raise ValueError(f"Unexpected NVFP4 tensor rank for {name}")
        if base + ".scale" not in tensors:
            raise ValueError(f"NVFP4 tensor {name} has no {base}.scale tensor")
        if hf_base in dequantize:
            if stacked:
                raise ValueError(f"Cannot dequantize stacked experts: {name}")
            yield hf_base + ".weight", out_dtype, (rows, n_cols)
            yield hf_base + ".weight_scale_2", torch.float32, ()
            continue
        quantized = (
            (".weight", torch.uint8, (rows, n_cols // 2)),
            (".weight_scale", torch.float8_e4m3fn, (rows, n_cols // 16)),
            (".weight_scale_2", torch.float32, ()),
        )
        prefixes = [hf_base.format(e) for e in range(experts or 0)] or [hf_base]
        for prefix in prefixes:
            for suffix, dtype, shape in quantized:
                yield prefix + suffix, dtype, shape


def gguf_nvfp4_global_scales(path: Any) -> dict[str, float]:
    """HF module base name -> global scale (``weight_scale_2``) of every NVFP4
    module, one per expert for stacked experts. Reads only the small
    ``.scale`` tensors."""
    reader = open_gguf_reader(path)
    tensors = {t.name: t for t in reader.tensors}
    scales: dict[str, float] = {}
    for name, t in tensors.items():
        if int(t.tensor_type) != _GGUF_NVFP4:
            continue
        base = name.removesuffix(".weight")
        hf_base, stacked = _map_name(base)
        if base + ".scale" not in tensors:
            raise ValueError(f"NVFP4 tensor {name} has no {base}.scale tensor")
        values = np.asarray(tensors[base + ".scale"].data, np.float32).reshape(-1)
        if stacked:
            scales.update({hf_base.format(e): float(v) for e, v in enumerate(values)})
        else:
            scales[hf_base] = float(values[0])
    return scales


def gguf_nvfp4_weights_iterator(
    path: Any,
    device: torch.device | str = "cpu",
    dequantize: frozenset[str] = frozenset(),
    out_dtype: torch.dtype = torch.float16,
    cast_stats: dict[str, dict[str, Any]] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield HF-style (name, tensor) pairs; NVFP4 tensors become ``.weight``
    (packed uint8), ``.weight_scale`` (E4M3) and ``.weight_scale_2`` (F32),
    except HF base names in ``dequantize``, which become an ``out_dtype``
    ``.weight`` holding ``e2m1 * e4m3`` (exact) plus the F32
    ``.weight_scale_2`` that the caller applies to the fp32 output.

    BF16/F16 tensors are cast to ``out_dtype`` (see ``cast_checked``; the
    statistics of each FP16 cast go to ``cast_stats``), F32 tensors pass
    through. NVFP4 repacking runs on ``device``; ``path`` may also be an open
    ``gguf.GGUFReader``."""
    reader = open_gguf_reader(path)
    tensors = {t.name: t for t in reader.tensors}
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue
        data = _as_tensor(t.data)
        if int(t.tensor_type) != _GGUF_NVFP4:
            hf_name, _ = _map_name(name)
            dtype = _pass_through_dtype(name, int(t.tensor_type), out_dtype)
            if int(t.tensor_type) == _GGUF_BF16:
                data = data.view(torch.bfloat16)
            if dtype != torch.float32:
                data = cast_checked(hf_name, data, dtype, cast_stats)
            yield hf_name, data
            continue

        base = name.removesuffix(".weight")
        hf_base, stacked = _map_name(base)
        n_cols, _, experts = _nvfp4_dims(t)
        if stacked != (experts is not None):
            raise ValueError(f"Unexpected NVFP4 tensor rank for {name}")
        if base + ".scale" not in tensors:
            raise ValueError(f"NVFP4 tensor {name} has no {base}.scale tensor")
        weight, weight_scale = _repack_nvfp4(data.to(device), n_cols)
        global_scale = _as_tensor(tensors[base + ".scale"].data)
        if global_scale.dtype != torch.float32:
            # ModelOpt global scales are fp32; refuse a lossy narrowing.
            scale32 = global_scale.float()
            if not torch.equal(scale32.to(global_scale.dtype), global_scale):
                raise ValueError(f"{base}.scale is not representable in fp32")
            global_scale = scale32
        if global_scale.numel() != (experts or 1):
            raise ValueError(
                f"{base}.scale has {global_scale.numel()} values, expected "
                f"{experts or 1}"
            )
        if hf_base in dequantize:
            if stacked:
                raise ValueError(f"Cannot dequantize stacked experts: {name}")
            yield (
                hf_base + ".weight",
                dequantize_nvfp4(weight, weight_scale, out_dtype=out_dtype),
            )
            yield hf_base + ".weight_scale_2", global_scale.reshape(())
            continue
        if not stacked:
            yield hf_base + ".weight", weight
            yield hf_base + ".weight_scale", weight_scale
            yield hf_base + ".weight_scale_2", global_scale.reshape(())
            continue
        for e in range(weight.shape[0]):
            expert = hf_base.format(e)
            yield expert + ".weight", weight[e]
            yield expert + ".weight_scale", weight_scale[e]
            yield expert + ".weight_scale_2", global_scale[e]
