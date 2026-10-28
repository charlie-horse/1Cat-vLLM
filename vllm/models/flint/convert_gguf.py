# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert Flint's NVFP4 GGUF checkpoint into a ModelOpt NVFP4 (W4A16)
safetensors checkpoint that SM70 serves in FP16.

    python -m vllm.models.flint.convert_gguf SRC OUT [--hf-config RecursiveSI/flint]

``SRC`` is a local ``.gguf`` file (or a directory holding one) or a Hugging Face
repo id. ``--hf-config`` (a repo id or a local directory) supplies
``config.json``, the tokenizer, ``chat_template.jinja`` and
``generation_config.json``.

The NVFP4 weights are repacked losslessly into the ModelOpt layout
(``weight`` / ``weight_scale`` / ``weight_scale_2``). ``lm_head`` is stored as
the exact FP16 ``e2m1 * e4m3`` product plus a scalar ``lm_head.weight_scale_2``
that the model applies to the logits. BF16 tensors are cast to FP16 and the
conversion fails if one of them does not fit. vLLM's dense W4A16 NVFP4 linear
keeps one global scale per fused layer (q/k/v, gate/up), so fused dense
shards whose global scales differ get the largest one, with their E4M3 block
scales re-encoded (exact for power-of-two ratios); ``--strict-merged-scales``
fails instead. Routed experts keep their own gate/up scales.

Tensors are streamed from the memory-mapped GGUF straight into the shards, so
peak memory stays around the largest single tensor (~1 GB), not the model
size. The output needs about as much disk space as the GGUF (~23 GB), plus the
GGUF itself when ``SRC`` is downloaded.
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import regex as re
import torch

from .gguf_nvfp4 import (
    find_gguf_file,
    gguf_nvfp4_global_scales,
    gguf_nvfp4_tensor_specs,
    gguf_nvfp4_weights_iterator,
    open_gguf_reader,
)

# (name, dtype, shape) of one output tensor.
Spec = tuple[str, torch.dtype, tuple[int, ...]]

DEFAULT_HF_CONFIG = "RecursiveSI/flint"
DEFAULT_MAX_SHARD_SIZE = 5 * 10**9

# Stored dequantized (normalized, so exactly) with a scalar weight_scale_2.
DEQUANTIZED = frozenset({"lm_head"})
EXCLUDE_MODULES = ("model.embed_tokens", "*.mlp.gate", "lm_head")

REPORT_FILE = "conversion_report.json"
INDEX_FILE = "model.safetensors.index.json"
_CONFIG_FILE = "config.json"
_EXTRA_FILES = (
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "LICENSE",
)
_REQUIRED_EXTRA_FILES = ("tokenizer_config.json", "chat_template.jinja")

_ST_DTYPES = {
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
}
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# Shards that vLLM fuses into one linear layer: category -> (regex over the
# HF module name with the fused module as group 1 and the shard as group 2,
# number of shards). The shard module is "<fused module>.<shard>_proj".
_LAYER = r"model\.layers\.\d+"
_MERGED = {
    "qkv": (rf"({_LAYER}\.self_attn)\.([qkv])_proj", 3),
    "dense_gate_up": (rf"({_LAYER}\.mlp)\.(gate|up)_proj", 2),
    "shared_expert_gate_up": (rf"({_LAYER}\.mlp\.shared_expert)\.(gate|up)_proj", 2),
    "expert_gate_up": (rf"({_LAYER}\.mlp\.experts\.\d+)\.(gate|up)_proj", 2),
}
_MERGED_RE = {kind: re.compile(pattern) for kind, (pattern, _) in _MERGED.items()}
# The dense W4A16 NVFP4 linear keeps one global scale per fused layer
# (max(weight_scale_2)), so these shards get a common scale. Routed experts
# keep separate gate/up scales, which the SM70 MoE path honours.
UNIFIED_KINDS = ("qkv", "dense_gate_up", "shared_expert_gate_up")
_FP16_MIN_NORMAL = float(torch.finfo(torch.float16).tiny)


def modelopt_quantization_config() -> dict[str, Any]:
    """The ``quantization_config`` written to ``config.json``.

    This is the nested ModelOpt form (``{"quantization": {...}}``) with an
    explicit ``quant_method``: vLLM detects it as ``modelopt_fp4`` and
    ``ModelOptNvFp4Config.from_config`` requires ``group_size``,
    ``kv_cache_quant_algo`` and ``exclude_modules`` in it. There is no
    ``producer: {"name": "modelopt"}`` entry because vLLM's ModelOpt producer
    normalization only accepts the FP8 and NVFP4 (W4A4) algorithms."""
    return {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": list(EXCLUDE_MODULES),
        },
    }


def parse_size(value: str | int) -> int:
    """'5GB', '4.5GB', '500MB', '4GiB' or a byte count -> bytes."""
    if isinstance(value, int):
        return value
    m = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?)(i?)B?\s*", value, flags=re.I)
    if m is None:
        raise ValueError(f"Cannot parse size: {value!r}")
    number, unit, binary = m.groups()
    base = 1024 if binary else 1000
    return int(float(number) * base ** "_KMGT".index(unit.upper() or "_"))


def resolve_gguf(
    src: str, revision: str | None = None, download_dir: str | None = None
) -> str:
    """Local .gguf file, a directory holding one, or a HF repo id -> path."""
    path = find_gguf_file(src, revision, download_dir)
    if path is None:
        raise FileNotFoundError(f"No .gguf file found for {src!r}")
    return path


def fetch_hf_config(
    hf_config: str, revision: str | None = None, download_dir: str | None = None
) -> str:
    """Local directory or HF repo id -> directory with the config files (only
    the small files are downloaded)."""
    if os.path.isdir(hf_config):
        return hf_config
    import huggingface_hub

    return huggingface_hub.snapshot_download(
        hf_config,
        revision=revision,
        cache_dir=download_dir,
        allow_patterns=[_CONFIG_FILE, *_EXTRA_FILES],
    )


def _nbytes(dtype: torch.dtype, shape: tuple[int, ...]) -> int:
    n = dtype.itemsize
    for d in shape:
        n *= d
    return n


def plan_shards(specs: Iterable[Spec], max_shard_size: int) -> list[list[Spec]]:
    """Split the tensors, in order, into shards of at most ``max_shard_size``
    bytes (a larger tensor gets a shard of its own)."""
    shards: list[list[Spec]] = []
    current: list[Spec] = []
    size = 0
    for spec in specs:
        nbytes = _nbytes(spec[1], spec[2])
        if current and size + nbytes > max_shard_size:
            shards.append(current)
            current, size = [], 0
        current.append(spec)
        size += nbytes
    if current:
        shards.append(current)
    return shards


def _write_shard(
    path: str,
    specs: list[Spec],
    tensors: Iterator[tuple[str, torch.Tensor]],
    on_tensor,
) -> int:
    """Stream ``tensors`` into a safetensors file whose header was planned from
    ``specs``; each yielded tensor must match its spec. Returns the data size."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype, shape in specs:
        end = offset + _nbytes(dtype, shape)
        header[name] = {
            "dtype": _ST_DTYPES[dtype],
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    partial = path + ".partial"
    try:
        with open(partial, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for name, dtype, shape in specs:
                got_name, tensor = next(tensors, (None, None))
                if (
                    got_name != name
                    or tensor is None
                    or tensor.dtype != dtype
                    or tuple(tensor.shape) != shape
                ):
                    got = None if tensor is None else (tensor.dtype, tensor.shape)
                    raise RuntimeError(
                        f"GGUF reader yielded {got_name} {got}; planned {name} "
                        f"{(dtype, shape)}"
                    )
                on_tensor(name, tensor)
                raw = tensor.detach().contiguous().cpu().reshape(-1)
                f.write(memoryview(raw.view(torch.uint8).numpy()))
    except BaseException:
        os.remove(partial)
        raise
    os.replace(partial, path)
    return offset


def merged_groups(scales: dict[str, float]) -> dict[str, dict[str, dict[str, float]]]:
    """category -> fused module -> shard -> global scale."""
    groups: dict[str, dict[str, dict[str, float]]] = {k: {} for k in _MERGED}
    for module, value in scales.items():
        for kind, pattern in _MERGED_RE.items():
            if m := pattern.fullmatch(module):
                groups[kind].setdefault(m.group(1), {})[m.group(2)] = value
                break
    return groups


def merged_summary(
    groups: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Whether the shards of each fused module share one global scale."""
    out: dict[str, Any] = {}
    for kind, modules in groups.items():
        want = _MERGED[kind][1]
        mismatched = {
            module: shards
            for module, shards in modules.items()
            if len(shards) != want or len(set(shards.values())) != 1
        }
        out[kind] = {
            "modules": len(modules),
            "shared": len(modules) - len(mismatched),
            "all_shared": not mismatched,
            "mismatched": mismatched,
        }
    return out


def unification_targets(
    groups: dict[str, dict[str, dict[str, float]]],
) -> dict[str, tuple[str, float, float]]:
    """Shard module -> (fused module, own scale, common scale) for every dense
    shard whose global scale is below the largest one of its fused module."""
    targets: dict[str, tuple[str, float, float]] = {}
    for kind in UNIFIED_KINDS:
        for module, shards in groups[kind].items():
            common = max(shards.values())
            for shard, value in shards.items():
                if value != common:
                    targets[f"{module}.{shard}_proj"] = (module, value, common)
    return targets


def rescale_block_scales(
    block_scale: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Re-encode E4M3 block scales for a ``ratio``-times smaller-or-equal global
    scale: ``round_to_e4m3(block_scale * ratio)``, with the max relative error
    of the effective block scale and the number of non-zero block scales that
    underflow to zero. Exact when ``ratio`` is a power of two and the results
    stay in the normal E4M3 range."""
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    old = block_scale.double()
    want = old * ratio
    new = want.to(torch.float8_e4m3fn)
    nonzero = old != 0
    err = (new.double() - want).abs()[nonzero] / want[nonzero]
    return new, {
        "nonzero_blocks": int(nonzero.sum()),
        "max_rel_error": float(err.max()) if err.numel() else 0.0,
        "underflowed_to_zero": int((nonzero & (new.double() == 0)).sum()),
    }


def _unify_merged_scales(
    tensors: Iterable[tuple[str, torch.Tensor]],
    targets: dict[str, tuple[str, float, float]],
    stats: dict[str, dict[str, Any]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Give the shards in ``targets`` their fused module's common global
    scale, re-encoding their block scales; ``stats`` gets one entry each."""
    for name, tensor in tensors:
        base, _, suffix = name.rpartition(".")
        if base in targets:
            module, own, common = targets[base]
            if suffix == "weight_scale":
                tensor, error = rescale_block_scales(tensor, own / common)
                stats[base] = {
                    "fused_module": module,
                    "source_weight_scale_2": own,
                    "weight_scale_2": common,
                    "ratio": own / common,
                    **error,
                }
            elif suffix == "weight_scale_2":
                if float(tensor) != own:
                    raise RuntimeError(f"{name}: expected {own}, got {float(tensor)}")
                tensor = torch.tensor(common, dtype=torch.float32)
        yield name, tensor


class _ReportBuilder:
    """Collects conversion statistics from the tensors as they are written."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.global_scales: list[float] = []
        self.dequantized: dict[str, dict[str, Any]] = {}
        self.fold: Counter[str] = Counter()
        self.fold_modules: list[tuple[float, str, int, int]] = []
        self._pending_scale: tuple[str, torch.Tensor] | None = None

    def add(self, name: str, tensor: torch.Tensor) -> None:
        base, _, suffix = name.rpartition(".")
        if suffix == "weight_scale":
            self._pending_scale = (base, tensor)
            return
        if suffix == "weight_scale_2":
            value = float(tensor)
            if base in self.dequantized:
                self.dequantized[base]["weight_scale_2"] = value
                return
            self.kinds["nvfp4_modules"] += 1
            self.global_scales.append(value)
            if self._pending_scale is not None and self._pending_scale[0] == base:
                self._add_fold(base, self._pending_scale[1], tensor)
            self._pending_scale = None
            return
        if tensor.dtype == torch.uint8:
            return
        base = name.removesuffix(".weight")
        if base in DEQUANTIZED:
            self.kinds["dequantized_modules"] += 1
            self.dequantized[base] = {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "max_abs": float(tensor.abs().max()),
            }
        else:
            dtype = str(tensor.dtype).removeprefix("torch.")
            self.kinds["passthrough_" + dtype] += 1

    def _add_fold(
        self, base: str, block_scale: torch.Tensor, global_scale: torch.Tensor
    ) -> None:
        # The SM70 NVFP4 kernels fold block_scale * weight_scale_2 into one
        # FP16 scale; count the products that lose precision or vanish there.
        block = block_scale.float()
        nonzero = block != 0
        folded = block * global_scale.float()
        n_nonzero = int(nonzero.sum())
        flushed = int((nonzero & (folded.half() == 0)).sum())
        subnormal = int((nonzero & (folded < _FP16_MIN_NORMAL)).sum())
        self.fold["nonzero"] += n_nonzero
        self.fold["subnormal"] += subnormal
        self.fold["flushed_to_zero"] += flushed
        if subnormal:
            ratio = subnormal / n_nonzero
            self.fold_modules.append((ratio, base, subnormal, flushed))

    def build(self) -> dict[str, Any]:
        gs = self.global_scales
        counts = dict(self.kinds)
        counts["nvfp4_tensors"] = 3 * counts.get("nvfp4_modules", 0)
        return {
            "tensor_counts": counts,
            "dequantized": self.dequantized,
            "weight_scale_2_range": [min(gs), max(gs)] if gs else None,
            "fp16_folded_block_scales": {
                "nonzero": self.fold["nonzero"],
                "subnormal": self.fold["subnormal"],
                "flushed_to_zero": self.fold["flushed_to_zero"],
                "worst_modules": [
                    {"module": b, "subnormal": s, "flushed_to_zero": f}
                    for _, b, s, f in sorted(self.fold_modules, reverse=True)[:20]
                ],
            },
        }


def _check_against_config(config: dict[str, Any], specs: list[Spec]) -> None:
    """Fail early when --hf-config does not describe this GGUF."""
    shapes = {name: shape for name, _, shape in specs}
    layers, experts = set(), set()
    for name in shapes:
        if m := re.match(r"model\.layers\.(\d+)\.(?:mlp\.experts\.(\d+)\.)?", name):
            layers.add(int(m.group(1)))
            if m.group(2) is not None:
                experts.add(int(m.group(2)))
    found = {"num_hidden_layers": max(layers) + 1 if layers else 0}
    if experts:
        found["num_experts"] = max(experts) + 1
    embed = shapes.get("model.embed_tokens.weight")
    if embed is not None:
        found["vocab_size"], found["hidden_size"] = embed
    for key, value in found.items():
        if key in config and config[key] != value:
            raise ValueError(
                f"--hf-config has {key}={config[key]} but the GGUF has {value}"
            )


def _summary(report: dict[str, Any], out_dir: str) -> str:
    def items(d: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in d.items())

    lines = [
        f"Converted {report['source']} -> {out_dir}",
        f"  {report['num_shards']} shard(s), {report['total_size'] / 1e9:.2f} GB, "
        f"{report['seconds']:.0f} s",
        f"  GGUF tensors: {items(report['gguf_tensors'])}",
        f"  Output tensors: {items(report['tensor_counts'])}",
        "  Merged shards share weight_scale_2:",
    ]
    for kind, s in report["merged_shards_share_weight_scale_2"].items():
        status = "yes" if s["all_shared"] else "NO"
        lines.append(f"    {kind:<22} {status:<3} {s['shared']}/{s['modules']}")
        mismatched = list(s["mismatched"].items())
        lines += [f"      {module}: {scales}" for module, scales in mismatched[:5]]
        if len(mismatched) > 5:
            lines.append(f"      ... {len(mismatched) - 5} more in {REPORT_FILE}")
    unified = report["unified_weight_scale_2"]
    if unified:
        worst = max(s["max_rel_error"] for s in unified.values())
        underflow = sum(s["underflowed_to_zero"] for s in unified.values())
        lines.append(
            f"  Unified weight_scale_2 of {len(unified)} dense shard(s): max "
            f"relative block-scale error {worst:.3g}, {underflow} block scale(s) "
            "underflowed to zero"
        )
    lines.append("  FP16-cast tensors: max |x|, non-zeros flushed to zero")
    for name, s in report["fp16_casts"].items():
        lines.append(f"    {name:<36} {s['max_abs']:<12.6g} {s['flushed_to_zero']}")
    for name, s in report["dequantized"].items():
        lines.append(
            f"  {name}: {s['dtype']} normalized weight, max |x| = {s['max_abs']:g}, "
            f"weight_scale_2 = {s['weight_scale_2']:g}"
        )
    fold = report["fp16_folded_block_scales"]
    lines.append(
        f"  FP16 block_scale * weight_scale_2: {fold['subnormal']} of "
        f"{fold['nonzero']} subnormal, {fold['flushed_to_zero']} flushed to zero"
    )
    lines.append(f"  Report: {os.path.join(out_dir, REPORT_FILE)}")
    return "\n".join(lines)


def convert(
    src: str,
    out_dir: str,
    hf_config: str = DEFAULT_HF_CONFIG,
    *,
    revision: str | None = None,
    hf_config_revision: str | None = None,
    download_dir: str | None = None,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    device: str = "cpu",
    out_dtype: torch.dtype = torch.float16,
    overwrite: bool = False,
    strict_merged_scales: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convert ``src`` into ``out_dir``; returns the conversion report.

    Dense shards fused by vLLM (q/k/v, dense and shared-expert gate/up) whose
    global scales differ get the largest one and re-encoded block scales;
    with ``strict_merged_scales`` the conversion fails instead."""
    start = time.monotonic()
    gguf_path = resolve_gguf(src, revision, download_dir)
    config_dir = fetch_hf_config(hf_config, hf_config_revision, download_dir)
    with open(os.path.join(config_dir, _CONFIG_FILE)) as f:
        config = json.load(f)

    reader = open_gguf_reader(gguf_path)
    gguf_types = Counter(t.tensor_type.name for t in reader.tensors)
    specs = list(gguf_nvfp4_tensor_specs(reader, DEQUANTIZED, out_dtype))
    _check_against_config(config, specs)
    groups = merged_groups(gguf_nvfp4_global_scales(reader))
    targets = unification_targets(groups)
    if targets and strict_merged_scales:
        fused = {module for module, _, _ in targets.values()}
        detail = {
            module: shards
            for kind in UNIFIED_KINDS
            for module, shards in groups[kind].items()
            if module in fused
        }
        raise ValueError(
            "Fused dense shards have different weight_scale_2 and "
            f"--strict-merged-scales is set: {detail}"
        )
    shards = plan_shards(specs, max_shard_size)

    os.makedirs(out_dir, exist_ok=True)
    stale = [
        n
        for n in os.listdir(out_dir)
        if n.endswith((".safetensors", ".safetensors.partial")) or n == INDEX_FILE
    ]
    if stale and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds a checkpoint ({stale[0]}); pass --overwrite"
        )
    for name in stale:
        os.remove(os.path.join(out_dir, name))

    copied = []
    for name in _EXTRA_FILES:
        path = os.path.join(config_dir, name)
        if os.path.isfile(path):
            shutil.copyfile(path, os.path.join(out_dir, name))
            copied.append(name)
    missing = [n for n in _REQUIRED_EXTRA_FILES if n not in copied]
    if missing:
        print(f"WARNING: {hf_config} has no {missing}", file=sys.stderr)

    cast_stats: dict[str, dict[str, Any]] = {}
    unified: dict[str, dict[str, Any]] = {}
    builder = _ReportBuilder()
    tensors = _unify_merged_scales(
        gguf_nvfp4_weights_iterator(reader, device, DEQUANTIZED, out_dtype, cast_stats),
        targets,
        unified,
    )
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        size = _write_shard(
            os.path.join(out_dir, shard_name), shard, tensors, builder.add
        )
        total_size += size
        weight_map.update({name: shard_name for name, _, _ in shard})
        if verbose:
            print(
                f"[{i + 1}/{len(shards)}] {shard_name}: {len(shard)} tensors, "
                f"{size / 1e9:.2f} GB, {time.monotonic() - start:.0f} s",
                flush=True,
            )
    leftover = next(tensors, None)
    if leftover is not None:
        raise RuntimeError(f"GGUF reader yielded unplanned tensor {leftover[0]}")

    with open(os.path.join(out_dir, INDEX_FILE), "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
            indent=2,
        )
    dtype_name = str(out_dtype).removeprefix("torch.")
    config["torch_dtype"] = dtype_name
    if "dtype" in config:
        config["dtype"] = dtype_name
    config["quantization_config"] = modelopt_quantization_config()
    with open(os.path.join(out_dir, _CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    report: dict[str, Any] = {
        "source": gguf_path,
        "hf_config": hf_config,
        "out_dtype": dtype_name,
        "num_shards": len(shards),
        "total_size": total_size,
        "seconds": time.monotonic() - start,
        "copied_files": [_CONFIG_FILE, *copied],
        "quantization_config": config["quantization_config"],
        "gguf_tensors": dict(gguf_types),
        **builder.build(),
        "merged_shards_share_weight_scale_2": merged_summary(groups),
        "unified_weight_scale_2": unified,
        "fp16_casts": cast_stats,
    }
    report["tensor_counts"]["total"] = len(specs)
    with open(os.path.join(out_dir, REPORT_FILE), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    if verbose:
        print(_summary(report, out_dir), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.models.flint.convert_gguf",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", help="local .gguf file or Hugging Face repo id")
    parser.add_argument("out", help="output directory")
    parser.add_argument(
        "--hf-config",
        default=DEFAULT_HF_CONFIG,
        help="repo id or directory with config.json and the tokenizer "
        f"(default: {DEFAULT_HF_CONFIG})",
    )
    parser.add_argument("--revision", help="revision of the GGUF repo")
    parser.add_argument("--hf-config-revision", help="revision of --hf-config")
    parser.add_argument("--download-dir", help="Hugging Face cache directory")
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="maximum shard size, e.g. 5GB or 4GiB (default: 5GB)",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="float16",
        help="dtype of the unquantized tensors (default: float16)",
    )
    parser.add_argument(
        "--device", default="cpu", help="device for the NVFP4 repack (default: cpu)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing checkpoint"
    )
    parser.add_argument(
        "--strict-merged-scales",
        action="store_true",
        help="fail instead of unifying differing global scales of fused dense "
        "shards (q/k/v, gate/up)",
    )
    args = parser.parse_args(argv)
    convert(
        args.src,
        args.out,
        args.hf_config,
        revision=args.revision,
        hf_config_revision=args.hf_config_revision,
        download_dir=args.download_dir,
        max_shard_size=parse_size(args.max_shard_size),
        device=args.device,
        out_dtype=_DTYPES[args.dtype],
        overwrite=args.overwrite,
        strict_merged_scales=args.strict_merged_scales,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
