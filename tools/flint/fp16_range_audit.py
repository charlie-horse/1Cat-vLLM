# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure per-layer activation ranges under FP16 serving and flag overflow risk.

A checkpoint trained in BF16 can carry activations that FP16 cannot hold
(FP16 max is 65504). This tool loads the model through vLLM's offline ``LLM``
API with the real serving engine flags, hooks every decoder layer inside each
worker, runs long code/agent-style prompts (prefill and a few decode steps),
and reports max |x| per layer and probe.

Usage, from the repository root on the GPU box:

    python -m tools.flint.fp16_range_audit --model /path/to/checkpoint \\
        --tensor-parallel-size 2 --dtype half --max-model-len 32768 \\
        --output fp16_range_audit.json

Any ``vllm serve`` engine flag passes through unchanged (for example
``--kv-cache-dtype``, ``--max-num-batched-tokens``, ``--quantization``,
``--gpu-memory-utilization``), so paste your serving flags. Tool flags:

    --threshold F         flag finite values with |x| > F (default 30000)
    --quantile Q          also record the Q-quantile of |x| (default 0.9999;
                          0 disables it)
    --prompts-file PATH   JSON list or JSONL. Each entry is a string (raw
                          prompt) or an object {"name", "prompt"} or
                          {"name", "messages", "tools"} (chat template).
                          Replaces the built-in prompts.
    --long-prompt-tokens N  size of the built-in long prompt, built by
                          repetition (default 16384; 0 drops it). It is
                          capped by max-model-len minus --max-tokens.
    --max-tokens N        greedy tokens generated per prompt (default 16)
    --layers-path PATH    dotted path to the decoder ModuleList (default:
                          the largest ModuleList named ``layers``)
    --decode-max-tokens N forward calls with at most N tokens count as
                          decode (default 1; raise it for speculative decode)
    --output PATH         JSON report (default fp16_range_audit.json)

Exit status: 0 when nothing is flagged, 1 when any probe saw |x| above the
threshold or any inf/nan, 2 when the audit itself failed.

Probes, per decoder layer (each only where the submodule exists):

    attn_in_o   attention output, the input of o_proj (per TP rank shard)
    attn_o      o_proj output
    q_norm, k_norm  q/k after their RMSNorm; their max per-head L2 norms give
                ``qk_bound`` = max|q| * max|k| * scaling, an upper bound on
                the attention logits (informational: attention kernels
                accumulate scores in FP32, so it is never flagged)
    mlp_act     input of mlp.down_proj (dense layers: act(gate) * up)
    shared_act  input of the shared expert's down_proj
    mlp_out     MLP / MoE block output
    hidden, residual  the two tensors a vLLM decoder layer returns
    stream      hidden + residual summed in FP32: the residual stream the
                next fused add+RMSNorm writes back in the model dtype. A
                value above 65504 here overflows in the next layer.
    final       norm_out (final norm output) and logits

Per probe and phase (prefill / decode) the report keeps the call count, the
max finite |x|, the largest per-forward Q-quantile of |x|, and counts of nan,
inf and |x| > threshold. Counts are summed over TP ranks (replicated tensors
are counted once per rank); maxima are maxima over ranks.

How it works: the hooks live in the worker processes. The tool passes
``worker_extension_cls`` (this module's ``Fp16RangeAuditExtension``) and
drives it with named ``LLM.collective_rpc`` calls whose arguments and results
are plain JSON-like data, so no ``VLLM_ALLOW_INSECURE_SERIALIZATION`` is
needed (``LLM.apply_model`` would ship a pickled callable). It forces
``enforce_eager=True`` and torch.compile mode NONE: in this fork
``enforce_eager`` alone only disables CUDA graphs, and a compiled forward
would bypass Python hooks. Prefix caching defaults to off so every prompt
token is prefilled and measured. Prompts run one at a time.

Not visible: values inside fused kernels (routed-expert intermediates of
fused MoE, attention scores). ``shared_act`` and ``qk_bound`` are the proxies.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

FP16_MAX = 65504.0
DEFAULT_THRESHOLD = 30000.0
DEFAULT_QUANTILE = 0.9999
DEFAULT_CHUNK_ELEMS = 1 << 22
EXTENSION_QUALNAME = "tools.flint.fp16_range_audit.Fp16RangeAuditExtension"
REPO_ROOT = Path(__file__).resolve().parents[2]

PREFILL = "prefill"
DECODE = "decode"
FINAL = "final"


class AuditError(Exception):
    """A usage or setup problem; reported without a traceback (exit 2)."""


# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

_ATTN = ("self_attn", "attn", "self_attention", "attention")
_O_PROJ = tuple(f"{a}.{o}" for a in _ATTN for o in ("o_proj", "out_proj", "dense"))


@dataclass(frozen=True)
class Probe:
    """One observation point.

    ``paths`` are candidate submodule paths tried in order; the first that
    exists is hooked. ``scope`` says what they are relative to: ``"layer"``
    (a decoder layer), ``"stack"`` (the module that owns the layer list) or
    ``"root"`` (the whole model). ``where`` is ``"input"`` (first tensor
    argument), ``"output"`` (tensor ``index`` of the output) or ``"stream"``
    (FP32 sum of the first two output tensors).
    """

    name: str
    paths: tuple[str, ...]
    where: str = "output"
    index: int = 0
    scope: str = "layer"
    rownorm: bool = False


LAYER_PROBES: tuple[Probe, ...] = (
    Probe("attn_in_o", _O_PROJ, "input"),
    Probe("attn_o", _O_PROJ),
    Probe("q_norm", tuple(f"{a}.q_norm" for a in _ATTN), rownorm=True),
    Probe("k_norm", tuple(f"{a}.k_norm" for a in _ATTN), rownorm=True),
    Probe("mlp_act", ("mlp.down_proj", "feed_forward.down_proj"), "input"),
    Probe(
        "shared_act",
        ("mlp.shared_expert.down_proj", "mlp.shared_experts.down_proj"),
        "input",
    ),
    Probe("mlp_out", ("mlp", "feed_forward", "block_sparse_moe", "moe")),
    Probe("hidden", ("",)),
    Probe("residual", ("",), index=1),
    Probe("stream", ("",), "stream"),
)

FINAL_PROBES: tuple[Probe, ...] = (
    Probe("norm_out", ("norm", "final_layernorm", "final_norm", "ln_f"), scope="stack"),
    Probe("logits", ("logits_processor",), scope="root"),
)

TABLE_COLUMNS = (
    "attn_in_o",
    "attn_o",
    "mlp_act",
    "shared_act",
    "mlp_out",
    "residual",
    "stream",
    "qk_bound",
)

# ---------------------------------------------------------------------------
# Tensor statistics (device side, no host sync)
# ---------------------------------------------------------------------------


def top_count(numel: int, quantile: float) -> int:
    """How many of the largest |x| to keep so the smallest is the quantile.

    Matches ``torch.quantile(..., interpolation="higher")``: the value at
    ascending index ``ceil(q * (n - 1))``. Returns 0 when disabled.
    """
    if numel <= 0 or quantile <= 0:
        return 0
    index = min(numel - 1, max(0, math.ceil(quantile * (numel - 1) - 1e-9)))
    return numel - index


def _max_rownorm(x: torch.Tensor, chunk_elems: int) -> torch.Tensor:
    best = torch.zeros((), dtype=torch.float32, device=x.device)
    dim = x.shape[-1] if x.dim() else 1
    rows = x.reshape(-1, dim)
    step = max(1, chunk_elems // max(dim, 1))
    for start in range(0, rows.shape[0], step):
        norms = torch.linalg.vector_norm(rows[start : start + step].float(), dim=-1)
        norms.masked_fill_(~torch.isfinite(norms), 0)
        best = torch.maximum(best, norms.max())
    return best


def tensor_stats(
    x: torch.Tensor,
    *,
    threshold: float,
    quantile: float,
    chunk_elems: int = DEFAULT_CHUNK_ELEMS,
    other: torch.Tensor | None = None,
    rownorm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Statistics of |x| for one forward call, left on ``x``'s device.

    With ``other``, the statistics are of ``x.float() + other.float()``.
    Works in chunks of ``chunk_elems`` so the temporary memory stays small.

    Returns ``(maxes, counts)``: ``maxes`` is float32
    ``[max finite |x|, quantile of |x|, max L2 norm over the last dim]`` and
    ``counts`` is int64 ``[nan, inf, finite |x| > threshold]``. Non-finite
    values count as 0 in the maxima and the quantile.
    """
    x = x.detach()
    flat = x.reshape(-1)
    other_flat = None if other is None else other.detach().reshape(-1)
    numel = flat.numel()
    device = x.device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    max_abs = zero
    nan, inf, over = torch.zeros(3, dtype=torch.int64, device=device).unbind()
    keep = top_count(numel, quantile)
    candidates = []
    for start in range(0, numel, chunk_elems):
        chunk = flat[start : start + chunk_elems]
        if other_flat is not None:
            chunk = chunk.float() + other_flat[start : start + chunk_elems].float()
            values = chunk.abs_()
        else:
            values = chunk.abs()
        bad = ~torch.isfinite(values)
        num_bad = bad.sum()
        num_nan = torch.isnan(values).sum()
        nan = nan + num_nan
        inf = inf + (num_bad - num_nan)
        values.masked_fill_(bad, 0)
        max_abs = torch.maximum(max_abs, values.max().float())
        over = over + (values > threshold).sum()
        if keep:
            k = min(keep, values.numel())
            candidates.append(torch.topk(values, k, sorted=False).values)
    tail = zero
    if keep:
        merged = torch.cat(candidates)
        tail = torch.topk(merged, keep, sorted=False).values.min().float()
    row = _max_rownorm(x, chunk_elems) if rownorm else zero
    return torch.stack([max_abs, tail, row]), torch.stack([nan, inf, over])


class _Accumulator:
    """Running statistics of one (layer, probe, phase), kept on device."""

    def __init__(self) -> None:
        self.calls = 0
        self.numel = 0
        self.dim = 0
        self.maxes: torch.Tensor | None = None
        self.counts: torch.Tensor | None = None

    def add(
        self, maxes: torch.Tensor, counts: torch.Tensor, numel: int, dim: int
    ) -> None:
        self.calls += 1
        self.numel += numel
        self.dim = dim
        if self.maxes is None or self.counts is None:
            self.maxes, self.counts = maxes, counts
        else:
            torch.maximum(self.maxes, maxes, out=self.maxes)
            self.counts.add_(counts)

    def as_dict(self) -> dict[str, Any]:
        assert self.maxes is not None and self.counts is not None
        max_abs, tail, row = (float(v) for v in self.maxes.cpu().tolist())
        nan, inf, over = (int(v) for v in self.counts.cpu().tolist())
        return dict(
            calls=self.calls,
            numel=self.numel,
            dim=self.dim,
            max_abs=max_abs,
            quantile_abs=tail,
            max_rownorm=row,
            nan=nan,
            inf=inf,
            over=over,
        )


# ---------------------------------------------------------------------------
# Hook installation (pure torch; runs inside each worker)
# ---------------------------------------------------------------------------


def find_layer_stack(
    model: nn.Module, layers_path: str | None = None
) -> tuple[str, nn.ModuleList]:
    """Locate the decoder-layer ModuleList.

    With ``layers_path``, that dotted path. Otherwise the ``ModuleList``
    named ``layers`` with the most entries (``model.layers`` wins ties,
    then the shortest path), so vision towers or draft heads are skipped.
    """
    if layers_path:
        stack = model.get_submodule(layers_path)
        if not isinstance(stack, nn.ModuleList):
            raise TypeError(f"{layers_path} is {type(stack).__name__}, not ModuleList")
        return layers_path, stack
    found = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList)
        and name.rpartition(".")[2] == "layers"
        and len(module)
    ]
    if not found:
        raise RuntimeError("No ModuleList named 'layers' found; pass --layers-path")
    found.sort(key=lambda item: (-len(item[1]), item[0] != "model.layers", item[0]))
    return found[0]


def _get_submodule(base: nn.Module, path: str) -> nn.Module | None:
    if not path:
        return base
    try:
        return base.get_submodule(path)
    except AttributeError:
        return None


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [v for v in value if isinstance(v, torch.Tensor)]
    return []


def _first_tensor(args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value
    return None


def _token_count(args: Sequence[Any], kwargs: dict[str, Any]) -> int | None:
    """Tokens in this forward: rows of the first floating tensor (dim >= 2)."""
    values = (*args, *kwargs.values())
    for value in values:
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.dim() >= 2
        ):
            return value.numel() // max(value.shape[-1], 1)
    for value in values:
        if isinstance(value, torch.Tensor) and value.dim() == 1:
            return value.numel()
    return None


def _synchronize() -> None:
    """Wait for every stream: hooks may run on a side stream (shared experts)."""
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


class RangeAuditor:
    """Forward hooks that accumulate |x| statistics per layer, probe, phase."""

    def __init__(
        self,
        model: nn.Module,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        quantile: float = DEFAULT_QUANTILE,
        layers_path: str | None = None,
        decode_max_tokens: int = 1,
        chunk_elems: int = DEFAULT_CHUNK_ELEMS,
        probes: Sequence[Probe] = LAYER_PROBES,
        final_probes: Sequence[Probe] = FINAL_PROBES,
    ) -> None:
        self.model = model
        self.threshold = float(threshold)
        self.quantile = float(quantile)
        self.layers_path = layers_path
        self.decode_max_tokens = int(decode_max_tokens)
        self.chunk_elems = int(chunk_elems)
        self.probes = tuple(probes)
        self.final_probes = tuple(final_probes)
        self.phase = PREFILL
        self.installed: dict[str, dict[str, str]] = {}
        self.scales: dict[str, float] = {}
        self._handles: list[Any] = []
        self._acc: dict[tuple[str, str, str], _Accumulator] = {}
        self._info: dict[str, Any] = {}

    # -- installation -----------------------------------------------------

    def install(self) -> dict[str, Any]:
        if self._handles:
            raise RuntimeError("RangeAuditor hooks are already installed")
        path, layers = find_layer_stack(self.model, self.layers_path)
        stack = _get_submodule(self.model, path.rpartition(".")[0])
        assert stack is not None
        hooked = []
        for idx, layer in enumerate(layers):
            if isinstance(layer, nn.Identity):  # pipeline-parallel placeholder
                continue
            key = str(idx)
            if not hooked:
                self._handles.append(
                    layer.register_forward_pre_hook(self._set_phase, with_kwargs=True)
                )
            hooked.append(key)
            for probe in self.probes:
                self._attach(key, layer, f"{path}.{idx}", probe)
            for name in _ATTN:
                scaling = getattr(getattr(layer, name, None), "scaling", None)
                if isinstance(scaling, (int, float)):
                    self.scales[key] = float(scaling)
                    break
        if not hooked:
            raise RuntimeError(f"{path} holds no decoder layers on this rank")
        for probe in self.final_probes:
            base, prefix = (
                (stack, path.rpartition(".")[0])
                if probe.scope == "stack"
                else (self.model, "")
            )
            self._attach(FINAL, base, prefix, probe)
        probes: dict[str, list[str]] = {}
        for key, names in self.installed.items():
            for name in names:
                probes.setdefault(name, []).append(key)
        self._info = dict(
            layers_path=path,
            num_layers=len(layers),
            hooked_layers=hooked,
            layer_class=type(layers[int(hooked[0])]).__name__,
            probes=probes,
            paths={
                name: self.installed[keys[0]][name] for name, keys in probes.items()
            },
            scales=dict(self.scales),
            threshold=self.threshold,
            quantile=self.quantile,
        )
        return copy.deepcopy(self._info)

    def _attach(self, key: str, base: nn.Module, prefix: str, probe: Probe) -> None:
        for rel in probe.paths:
            module = _get_submodule(base, rel)
            if module is None:
                continue
            if probe.where == "input":
                handle = module.register_forward_pre_hook(
                    self._input_hook(key, probe), with_kwargs=True
                )
            else:
                handle = module.register_forward_hook(self._output_hook(key, probe))
            self._handles.append(handle)
            full = ".".join(p for p in (prefix, rel) if p) or "<root>"
            self.installed.setdefault(key, {})[probe.name] = full
            return

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # -- hooks --------------------------------------------------------------

    def _set_phase(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        tokens = _token_count(args, kwargs)
        if tokens is not None:
            self.phase = DECODE if tokens <= self.decode_max_tokens else PREFILL

    def _input_hook(self, key: str, probe: Probe):
        def hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            self.record(key, probe, _first_tensor(args, kwargs))

        return hook

    def _output_hook(self, key: str, probe: Probe):
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            tensors = _tensors(output)
            if probe.where == "stream":
                if len(tensors) >= 2 and tensors[0].shape == tensors[1].shape:
                    self.record(key, probe, tensors[0], tensors[1])
                elif tensors:
                    self.record(key, probe, tensors[0])
            elif probe.index < len(tensors):
                self.record(key, probe, tensors[probe.index])

        return hook

    def record(
        self,
        key: str,
        probe: Probe,
        x: Any,
        other: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return
        if x.numel() == 0:
            return
        maxes, counts = tensor_stats(
            x,
            threshold=self.threshold,
            quantile=self.quantile,
            chunk_elems=self.chunk_elems,
            other=other,
            rownorm=probe.rownorm,
        )
        acc = self._acc.setdefault((key, probe.name, self.phase), _Accumulator())
        acc.add(maxes, counts, x.numel(), x.shape[-1] if x.dim() else 1)

    # -- results ------------------------------------------------------------

    def reset(self) -> None:
        self._acc = {}

    def collect(self, reset: bool = True) -> dict[str, Any]:
        """Statistics as ``{"stats": {layer: {probe: {phase: stat}}}}``."""
        _synchronize()
        stats: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for (key, name, phase), acc in self._acc.items():
            stats.setdefault(key, {}).setdefault(name, {})[phase] = acc.as_dict()
        if reset:
            self.reset()
        return {"stats": stats}


class Fp16RangeAuditExtension:
    """Worker extension: named RPCs with plain-data arguments and results."""

    def fp16_audit_install(self, config: dict[str, Any]) -> dict[str, Any]:
        previous = getattr(self, "_fp16_audit", None)
        if previous is not None:
            previous.remove()
        auditor = RangeAuditor(self.get_model(), **config)  # type: ignore[attr-defined]
        info = auditor.install()
        self._fp16_audit = auditor
        info["rank"] = int(getattr(self, "rank", 0))
        return info

    def fp16_audit_collect(self, reset: bool = True) -> dict[str, Any]:
        auditor = getattr(self, "_fp16_audit", None)
        if auditor is None:
            raise RuntimeError("fp16_audit_install has not run on this worker")
        result = auditor.collect(reset=reset)
        result["rank"] = int(getattr(self, "rank", 0))
        return result

    def fp16_audit_remove(self) -> bool:
        auditor = getattr(self, "_fp16_audit", None)
        if auditor is not None:
            auditor.remove()
            self._fp16_audit = None
        return auditor is not None


# ---------------------------------------------------------------------------
# Merging and reporting (host side, plain data)
# ---------------------------------------------------------------------------

_MAX_KEYS = ("max_abs", "quantile_abs", "max_rownorm", "dim")
_SUM_KEYS = ("numel", "nan", "inf", "over")
Stats = dict[str, dict[str, dict[str, dict[str, Any]]]]


def merge_stat(
    a: dict[str, Any], b: dict[str, Any], *, sum_calls: bool = True
) -> dict[str, Any]:
    out = {k: max(a[k], b[k]) for k in _MAX_KEYS}
    out.update({k: a[k] + b[k] for k in _SUM_KEYS})
    out["calls"] = a["calls"] + b["calls"] if sum_calls else max(a["calls"], b["calls"])
    return out


def merge_stats(trees: Iterable[Stats], *, sum_calls: bool = True) -> Stats:
    merged: Stats = {}
    for tree in trees:
        for key, probes in tree.items():
            for name, phases in probes.items():
                slot = merged.setdefault(key, {}).setdefault(name, {})
                for phase, stat in phases.items():
                    prev = slot.get(phase)
                    slot[phase] = (
                        dict(stat)
                        if prev is None
                        else merge_stat(prev, stat, sum_calls=sum_calls)
                    )
    return merged


def merge_ranks(per_rank: Sequence[dict[str, Any]]) -> Stats:
    """Merge one collect() result per worker; every rank runs every call."""
    return merge_stats((r["stats"] for r in per_rank), sum_calls=False)


def _layer_order(key: str) -> tuple[int, int | str]:
    return (0, int(key)) if key.isdigit() else (1, key)


def _flag_reasons(stat: dict[str, Any], threshold: float) -> list[str]:
    reasons = []
    if stat["nan"]:
        reasons.append(f"{stat['nan']} nan")
    if stat["inf"]:
        reasons.append(f"{stat['inf']} inf")
    if stat["max_abs"] > threshold:
        reasons.append(f"max|x| {stat['max_abs']:.1f} > {threshold:g}")
    return reasons


def summarize(
    runs: Sequence[dict[str, Any]], threshold: float, scales: dict[str, float]
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Merge prompts and phases per (layer, probe); list flagged entries."""
    layers: dict[str, dict[str, dict[str, Any]]] = {}
    flags: list[dict[str, Any]] = []
    for run in runs:
        for key, probes in run["stats"].items():
            for name, phases in probes.items():
                entry = layers.setdefault(key, {}).setdefault(
                    name, {"phases": {}, "worst_prompt": None, "_worst": -1.0}
                )
                for phase, stat in phases.items():
                    prev = entry["phases"].get(phase)
                    entry["phases"][phase] = (
                        dict(stat) if prev is None else merge_stat(prev, stat)
                    )
                    if stat["max_abs"] > entry["_worst"]:
                        entry["_worst"] = stat["max_abs"]
                        entry["worst_prompt"] = run["name"]
                    reasons = _flag_reasons(stat, threshold)
                    if reasons:
                        flags.append(
                            dict(
                                prompt=run["name"],
                                layer=key,
                                probe=name,
                                phase=phase,
                                max_abs=stat["max_abs"],
                                quantile_abs=stat["quantile_abs"],
                                nan=stat["nan"],
                                inf=stat["inf"],
                                over=stat["over"],
                                reasons=reasons,
                            )
                        )
    for key, probes in layers.items():
        for entry in probes.values():
            entry.pop("_worst")
            phases = list(entry["phases"].values())
            total = phases[0]
            for stat in phases[1:]:
                total = merge_stat(total, stat)
            entry.update(
                {k: total[k] for k in (*_MAX_KEYS, *_SUM_KEYS)},
                calls={phase: s["calls"] for phase, s in entry["phases"].items()},
                headroom=FP16_MAX / total["max_abs"] if total["max_abs"] else None,
            )
        q, k = probes.get("q_norm"), probes.get("k_norm")
        if q and k and q["max_rownorm"] and k["max_rownorm"]:
            scale = scales.get(key) or q["dim"] ** -0.5
            probes["qk_bound"] = dict(
                max_abs=q["max_rownorm"] * k["max_rownorm"] * scale,
                scaling=scale,
                bound=True,
            )
    flags.sort(key=lambda f: (_layer_order(f["layer"]), f["probe"], f["prompt"]))
    return layers, flags


def build_report(
    runs: Sequence[dict[str, Any]],
    workers: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(config["threshold"])
    scales = dict(workers[0].get("scales", {})) if workers else {}
    layers, flags = summarize(runs, threshold, scales)
    never_fired: dict[str, list[str]] = {}
    for info in workers:
        for name, keys in info.get("probes", {}).items():
            for key in keys:
                if name not in layers.get(key, {}):
                    never_fired.setdefault(name, [])
                    if key not in never_fired[name]:
                        never_fired[name].append(key)
    return dict(
        tool="fp16_range_audit",
        fp16_max=FP16_MAX,
        config=config,
        workers=list(workers),
        prompts=[{k: v for k, v in run.items() if k != "stats"} for run in runs],
        layers={k: layers[k] for k in sorted(layers, key=_layer_order)},
        flags=flags,
        never_fired=never_fired,
        runs=list(runs),
        ok=not flags,
    )


def exit_code(report: dict[str, Any]) -> int:
    return 0 if report["ok"] else 1


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if value >= 1e5 or value < 1e-2:
        return f"{value:.3g}"
    return f"{value:.1f}"


def _cell(entry: dict[str, Any] | None, threshold: float) -> str:
    if entry is None:
        return "-"
    text = _fmt(entry["max_abs"])
    if entry.get("bound"):
        return text
    nan, inf = entry.get("nan", 0), entry.get("inf", 0)
    if nan + inf and nan + inf >= entry.get("numel", 0):
        text = "nan" if nan else "inf"  # no finite value at all
    if nan or inf:
        text += "!"
    if entry["max_abs"] > threshold:
        text += "*"
    return text


def format_report(report: dict[str, Any], max_flags: int = 40) -> str:
    config = report["config"]
    threshold = float(config["threshold"])
    quantile = float(config.get("quantile", 0))
    layers = report["layers"]
    lines = [
        f"FP16 range audit: max |x| per layer (threshold {threshold:g}, "
        f"FP16 max {FP16_MAX:g})"
    ]
    if config.get("model"):
        lines.append(
            f"model {config['model']}  dtype {config.get('dtype')}  "
            f"tp {config.get('tensor_parallel_size')}"
        )
    for run in report["prompts"]:
        lines.append(
            f"  prompt {run['name']}: {run['prompt_tokens']} prompt tokens, "
            f"{run['generated_tokens']} generated ({run.get('template', '?')})"
        )
    numeric = [k for k in layers if k.isdigit()]
    if numeric:
        first = layers[numeric[0]]
        probe = first.get("stream") or next(iter(first.values()))
        calls = probe.get("calls", {})
        lines.append(
            f"forward calls seen at layer {numeric[0]}: "
            f"prefill={calls.get(PREFILL, 0)} decode={calls.get(DECODE, 0)}"
        )
        if not calls.get(DECODE):
            lines.append("WARNING: no decode step observed (see --decode-max-tokens)")
    columns = [c for c in TABLE_COLUMNS if any(c in layers[k] for k in numeric)]
    widths = [max(len(c), 9) for c in columns]
    lines.append("")
    lines.append(
        "layer  " + "  ".join(c.rjust(w) for c, w in zip(columns, widths, strict=True))
    )
    for key in numeric:
        cells = [_cell(layers[key].get(c), threshold) for c in columns]
        lines.append(
            f"{key:>5}  "
            + "  ".join(c.rjust(w) for c, w in zip(cells, widths, strict=True))
        )
    if FINAL in layers:
        final = "  ".join(
            f"{name}={_cell(entry, threshold)}" for name, entry in layers[FINAL].items()
        )
        lines.append(f"final  {final}")
    lines.append(
        "  * above threshold   ! inf/nan seen   qk_bound = max|q|*max|k|*scaling "
        "(upper bound on attention logits, not flagged)"
    )
    entries = [
        (key, name, entry)
        for key, probes in layers.items()
        for name, entry in probes.items()
        if not entry.get("bound")
    ]
    entries.sort(key=lambda item: -item[2]["max_abs"])
    if entries:
        label = f"p{quantile * 100:g}" if quantile else "quantile"
        lines.append("")
        lines.append(f"largest values (max |x|, largest per-forward {label}):")
        for key, name, entry in entries[:10]:
            headroom = entry.get("headroom")
            lines.append(
                f"  layer {key:>5} {name:<11} max {_fmt(entry['max_abs']):>9}  "
                f"{label} {_fmt(entry['quantile_abs']):>9}  "
                f"headroom {_fmt(headroom)}x  prompt {entry['worst_prompt']}"
            )
    if report["never_fired"]:
        lines.append("")
        lines.append("probes installed but never called (fused or bypassed path):")
        for name, keys in report["never_fired"].items():
            lines.append(f"  {name}: layers {', '.join(keys)}")
    flags = report["flags"]
    lines.append("")
    if flags:
        lines.append(f"FLAGGED: {len(flags)} entries")
        for flag in flags[:max_flags]:
            lines.append(
                f"  layer {flag['layer']:>5} {flag['probe']:<11} {flag['phase']:<7} "
                f"prompt {flag['prompt']}: {'; '.join(flag['reasons'])}"
            )
        if len(flags) > max_flags:
            lines.append(f"  ... and {len(flags) - max_flags} more (see the JSON)")
    else:
        lines.append(f"OK: no finite |x| above {threshold:g} and no inf/nan")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REPEAT_MARK = "<<REPEAT>>"


@dataclass
class PromptSpec:
    """A raw ``text`` prompt or chat ``messages`` (with optional ``tools``).

    With ``repeat``, ``REPEAT_MARK`` in the last message is replaced by copies
    of ``repeat`` (``{i}`` becomes the copy index) up to ``target_tokens``.
    """

    name: str
    text: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    repeat: str | None = None
    target_tokens: int = 0


_AGENT_SYSTEM = """\
You are an autonomous software engineering agent working in a git checkout.
You can call tools to read files, search the code, run shell commands and
apply patches. Work in small verified steps: read the relevant code first,
state a hypothesis, make the smallest change that fixes the root cause, then
run the narrowest test that proves it. Never invent file contents or command
output. When you call a tool, emit exactly one JSON tool call and wait for the
result. Report what you changed, which commands you ran and their results."""

_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }
    for name, description, properties in (
        (
            "read_file",
            "Read a UTF-8 file, optionally a line range.",
            {"path": {"type": "string"}, "start_line": {"type": "integer"}},
        ),
        (
            "run_shell",
            "Run a shell command in the repository root and return its output.",
            {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
        ),
        (
            "apply_patch",
            "Apply a unified diff to the working tree.",
            {"patch": {"type": "string"}},
        ),
        (
            "search_code",
            "Regex search over tracked files; returns path:line:text matches.",
            {"pattern": {"type": "string"}, "glob": {"type": "string"}},
        ),
    )
]

_BUGFIX_FILE = """\
# cache/lru.py
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class Entry(Generic[V]):
    value: V
    expires_at: float
    size: int = 1


@dataclass
class LRUCache(Generic[K, V]):
    capacity: int
    ttl_s: float = 300.0
    clock: Callable[[], float] = time.monotonic
    _data: OrderedDict = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _used: int = 0
    hits: int = 0
    misses: int = 0

    def get(self, key: K) -> V | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.expires_at < self.clock():
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return entry.value

    def put(self, key: K, value: V, size: int = 1) -> None:
        with self._lock:
            if key in self._data:
                self._used -= self._data[key].size
            self._data[key] = Entry(value, self.clock() + self.ttl_s, size)
            self._used += size
            while self._used > self.capacity:
                _, evicted = self._data.popitem(last=True)
                self._used -= evicted.size

    def __len__(self) -> int:
        return len(self._data)
"""

_BUGFIX_TEST_OUTPUT = """\
$ python -m pytest tests/test_lru.py -x -q
F
=================================== FAILURES ===================================
___________________________ test_evicts_least_recent ___________________________

    def test_evicts_least_recent():
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        cache.put("c", 3)
>       assert cache.get("b") is None
E       AssertionError: assert 2 is None
E        +  where 2 = <bound method LRUCache.get of LRUCache(capacity=2, ...)>('b')

tests/test_lru.py:14: AssertionError
1 failed, 6 passed in 0.31s"""

_TRANSCRIPT = """\
Previous tool results in this session:

[run_shell] git log --oneline -6
9c1d2e4 scheduler: requeue preempted requests at the front
41b7a90 kv: account sliding-window blocks per group
d03f5aa api: stream tool-call deltas as soon as the name is known
77e21c3 tests: cover 128k prompts in the chunked prefill path
1fe9b02 metrics: export queue time histogram
a84c6d1 bump version to 0.14.2

[search_code] pattern="def _free_blocks" glob="**/*.py"
engine/kv_manager.py:212:    def _free_blocks(self, request_id: str) -> int:
engine/kv_manager.py:388:    def _free_group(self, group: int, ids: list[int]) -> None:

[read_file] engine/kv_manager.py start_line=205
205      def allocate(self, request: Request, num_tokens: int) -> list[int] | None:
206          need = cdiv(request.num_computed + num_tokens, self.block_size)
207          need -= len(self.blocks[request.id])
208          if need > self.free.qsize():
209              return None
210          return [self.free.get_nowait() for _ in range(need)]
211
212      def _free_blocks(self, request_id: str) -> int:
213          blocks = self.blocks.pop(request_id, [])
214          for block in reversed(blocks):
215              if block.ref_count == 1:
216                  self.free.put_nowait(block)
217              block.ref_count -= 1
218          return len(blocks)

[run_shell] python -m pytest tests/engine/test_kv_manager.py -q -k window
....F.
FAILED tests/engine/test_kv_manager.py::test_window_release - assert 37 == 40
{"free_blocks": 37, "expected": 40, "window": 512, "block_size": 16,
 "requests": [{"id": "r0", "tokens": 1311}, {"id": "r1", "tokens": 902}]}

The window test fails after the sliding-window accounting commit. Decide the
next tool call, explain your hypothesis in two sentences, then call it."""

_REVIEW_DIFF = """\
Review this patch for correctness, numerical safety on fp16 and performance.
List concrete problems with file:line references and a suggested fix each.

diff --git a/csrc/layernorm_kernels.cu b/csrc/layernorm_kernels.cu
@@ -41,22 +41,27 @@ template <typename scalar_t, int VEC>
 __global__ void fused_add_rms_norm_kernel(
     scalar_t* __restrict__ input,     // [num_tokens, hidden]
     scalar_t* __restrict__ residual,  // [num_tokens, hidden]
     const scalar_t* __restrict__ weight,
     const float epsilon, const int hidden) {
-  float variance = 0.0f;
+  __shared__ float s_variance;
+  float variance = 0.0f;
   for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
-    float z = (float)input[blockIdx.x * hidden + i];
-    z += (float)residual[blockIdx.x * hidden + i];
-    variance += z * z;
-    residual[blockIdx.x * hidden + i] = (scalar_t)z;
+    const int64_t off = (int64_t)blockIdx.x * hidden + i;
+    scalar_t z = input[off] + residual[off];
+    variance += (float)z * (float)z;
+    residual[off] = z;
   }
-  variance = blockReduceSum<float>(variance);
-  if (threadIdx.x == 0) s_variance = rsqrtf(variance / hidden + epsilon);
+  variance = blockReduceSum<float>(variance);
+  if (threadIdx.x == 0) {
+    s_variance = rsqrtf(variance / hidden + epsilon);
+  }
   __syncthreads();
   for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
-    float x = (float)residual[blockIdx.x * hidden + i];
-    input[blockIdx.x * hidden + i] = (scalar_t)(x * s_variance) * weight[i];
+    const int64_t off = (int64_t)blockIdx.x * hidden + i;
+    float x = (float)residual[off];
+    input[off] = (scalar_t)(x * s_variance * (float)weight[i]);
   }
 }"""

_REPO_HEADER = """\
Below is the source of a service module, concatenated file by file. Read all
of it; afterwards answer the question at the end.

"""

_REPO_BLOCK = """
# ---- services/shard_{i}/handlers.py ----
import json
import logging
from dataclasses import dataclass

log = logging.getLogger("shard_{i}")


@dataclass
class Request{i}:
    user_id: int
    payload: dict
    retries: int = 0


def validate_{i}(req: Request{i}) -> list[str]:
    errors = []
    if req.user_id <= 0:
        errors.append(f"bad user id {req.user_id}")
    if "items" not in req.payload:
        errors.append("missing items")
    elif len(req.payload["items"]) > {i} % 97 + 16:
        errors.append("too many items")
    return errors


def handle_{i}(raw: str) -> dict:
    data = json.loads(raw)
    req = Request{i}(user_id=int(data.get("user", -1)), payload=data)
    errors = validate_{i}(req)
    if errors:
        log.warning("shard %d rejected %s: %s", {i}, req.user_id, errors)
        return {"ok": False, "errors": errors, "shard": {i}}
    total = sum(item.get("qty", 0) * item.get("price", 0.0)
                for item in req.payload["items"])
    return {"ok": True, "total": round(total, 2), "shard": {i}}
"""

_REPO_QUESTION = """
Question: which handlers can return a total for a request whose items list is
longer than 40, and what single change would make validation consistent across
all shards? Answer with the shard numbers, then a patch."""


def default_prompts(long_prompt_tokens: int = 16384) -> list[PromptSpec]:
    """Built-in code/agent prompts; the last one is long, built by repetition."""
    system = {"role": "system", "content": _AGENT_SYSTEM}
    prompts = [
        PromptSpec(
            name="agent_bugfix",
            messages=[
                system,
                {
                    "role": "user",
                    "content": "The test below fails. Find the bug and propose a "
                    "patch.\n\n```python\n"
                    + _BUGFIX_FILE
                    + "```\n\n```\n"
                    + _BUGFIX_TEST_OUTPUT
                    + "\n```",
                },
            ],
            tools=_AGENT_TOOLS,
        ),
        PromptSpec(
            name="agent_transcript",
            messages=[system, {"role": "user", "content": _TRANSCRIPT}],
            tools=_AGENT_TOOLS,
        ),
        PromptSpec(
            name="code_review",
            messages=[system, {"role": "user", "content": _REVIEW_DIFF}],
        ),
    ]
    if long_prompt_tokens > 0:
        prompts.append(
            PromptSpec(
                name="long_repo_context",
                messages=[
                    system,
                    {
                        "role": "user",
                        "content": _REPO_HEADER + REPEAT_MARK + _REPO_QUESTION,
                    },
                ],
                tools=_AGENT_TOOLS,
                repeat=_REPO_BLOCK,
                target_tokens=long_prompt_tokens,
            )
        )
    return prompts


def load_prompts(path: Path) -> list[PromptSpec]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        items = json.loads(text)
        if isinstance(items, dict):
            items = items.get("prompts")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a nonempty list of prompts")
    specs = []
    for i, item in enumerate(items):
        if isinstance(item, str) and item:
            specs.append(PromptSpec(name=f"prompt_{i}", text=item))
        elif isinstance(item, dict) and (item.get("messages") or item.get("prompt")):
            specs.append(
                PromptSpec(
                    name=str(item.get("name", f"prompt_{i}")),
                    text=item.get("prompt") if not item.get("messages") else None,
                    messages=item.get("messages"),
                    tools=item.get("tools"),
                )
            )
        else:
            raise ValueError(f"{path}: entry {i} has no prompt or messages")
    return specs


def _expand(spec: PromptSpec, copies: int) -> list[dict[str, Any]]:
    assert spec.messages is not None and spec.repeat is not None
    body = "".join(spec.repeat.replace("{i}", str(i)) for i in range(copies))
    messages = [dict(m) for m in spec.messages]
    messages[-1]["content"] = messages[-1]["content"].replace(REPEAT_MARK, body)
    return messages


def _plain(messages: Sequence[dict[str, Any]]) -> str:
    turns = [f"{m['role']}: {m.get('content') or ''}" for m in messages]
    return "\n\n".join(turns) + "\n\nassistant:"


def _render(
    tokenizer: Any, spec: PromptSpec, messages: list[dict[str, Any]] | None
) -> tuple[list[int], str]:
    if messages is None:
        assert spec.text is not None
        return list(tokenizer.encode(spec.text)), "raw"
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tools=spec.tools, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:  # no template, or the template rejects the input
        print(f"[{spec.name}] chat template failed ({exc}); using plain text")
        return list(tokenizer.encode(_plain(messages))), "plain"
    if not isinstance(rendered, str):
        return list(rendered), "chat"
    return list(tokenizer.encode(rendered, add_special_tokens=False)), "chat"


def render_prompt(
    spec: PromptSpec, tokenizer: Any, budget: int
) -> tuple[list[int], str]:
    """Token ids and the rendering used (``chat``, ``plain`` or ``raw``)."""
    if spec.repeat:
        target = min(spec.target_tokens, budget)
        sample = spec.repeat.replace("{i}", "0")
        block = max(1, len(tokenizer.encode(sample, add_special_tokens=False)))
        copies = max(1, target // block)
        for _ in range(16):
            ids, how = _render(tokenizer, spec, _expand(spec, copies))
            if len(ids) <= target or copies == 1:
                break
            copies = max(1, min(copies - 1, copies * target // len(ids)))
    else:
        ids, how = _render(tokenizer, spec, spec.messages)
    if len(ids) > budget:
        raise AuditError(
            f"prompt {spec.name!r} has {len(ids)} tokens, more than the "
            f"{budget} that fit (max-model-len minus --max-tokens)"
        )
    return ids, how


# ---------------------------------------------------------------------------
# Engine driver
# ---------------------------------------------------------------------------


def _no_compile(compilation_config: Any) -> Any:
    """The same compilation config with torch.compile disabled (mode NONE)."""
    if compilation_config is None:
        return {"mode": 0}
    if isinstance(compilation_config, int):
        return 0
    if isinstance(compilation_config, dict):
        return {**compilation_config, "mode": 0}
    try:
        from vllm.config.compilation import CompilationMode

        mode: Any = CompilationMode.NONE
    except Exception:
        mode = 0
    compilation_config = copy.deepcopy(compilation_config)
    compilation_config.mode = mode
    return compilation_config


def engine_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    """``LLM(**kwargs)`` for the audit: eager, uncompiled, with the extension."""
    kwargs = dict(options)
    extension = kwargs.get("worker_extension_cls") or ""
    if extension and extension != EXTENSION_QUALNAME:
        raise ValueError(
            f"worker_extension_cls={extension!r} conflicts with the audit's "
            f"{EXTENSION_QUALNAME!r}"
        )
    kwargs["worker_extension_cls"] = EXTENSION_QUALNAME
    kwargs["enforce_eager"] = True
    kwargs["compilation_config"] = _no_compile(kwargs.get("compilation_config"))
    if kwargs.get("enable_prefix_caching") is None:
        kwargs["enable_prefix_caching"] = False
    return kwargs


def _ensure_importable_by_workers() -> None:
    """Workers import the extension by name; put the repo root on their path."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root, *parts])
    module = importlib.import_module(EXTENSION_QUALNAME.rpartition(".")[0])
    if Path(module.__file__ or "").resolve() != Path(__file__).resolve():
        raise AuditError(
            f"{EXTENSION_QUALNAME} resolves to {module.__file__}, not this file; "
            "run from the repository root"
        )


def build_parser() -> tuple[argparse.ArgumentParser, set[str]]:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Measure per-layer activation ranges under FP16 serving "
        "(see the module docstring of tools/flint/fp16_range_audit.py)."
    )
    tool = parser.add_argument_group("fp16 range audit")
    tool.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    tool.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    tool.add_argument("--prompts-file", type=Path, default=None)
    tool.add_argument("--long-prompt-tokens", type=int, default=16384)
    tool.add_argument("--max-tokens", type=int, default=16)
    tool.add_argument("--layers-path", default=None)
    tool.add_argument("--decode-max-tokens", type=int, default=1)
    tool.add_argument("--output", type=Path, default=Path("fp16_range_audit.json"))
    tool_dests = {action.dest for action in tool._group_actions}
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model=None,
        tensor_parallel_size=2,
        dtype="half",
        max_model_len=32768,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    return parser, tool_dests


def run(argv: Sequence[str] | None = None) -> int:
    parser, tool_dests = build_parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required")
    options = vars(args)
    tool = argparse.Namespace(**{k: options[k] for k in tool_dests})
    if tool.max_tokens < 1:
        parser.error("--max-tokens must be at least 1 so decode is covered")
    kwargs = engine_kwargs({k: v for k, v in options.items() if k not in tool_dests})
    prompts = (
        load_prompts(tool.prompts_file)
        if tool.prompts_file
        else default_prompts(tool.long_prompt_tokens)
    )
    _ensure_importable_by_workers()
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    from vllm import LLM, SamplingParams

    llm = LLM(**kwargs)
    model_config = llm.llm_engine.model_config
    if model_config.dtype != torch.float16:
        print(f"WARNING: model dtype is {model_config.dtype}, not float16")
    tokenizer = llm.get_tokenizer()
    budget = model_config.max_model_len - tool.max_tokens
    rendered = [render_prompt(spec, tokenizer, budget) for spec in prompts]
    audit_config = dict(
        threshold=tool.threshold,
        quantile=tool.quantile,
        layers_path=tool.layers_path,
        decode_max_tokens=tool.decode_max_tokens,
    )
    workers = llm.collective_rpc("fp16_audit_install", args=(audit_config,))
    info = workers[0]
    print(
        f"hooked {len(info['hooked_layers'])}/{info['num_layers']} "
        f"{info['layer_class']} at {info['layers_path']} on {len(workers)} "
        f"worker(s); probes: {', '.join(info['probes'])}"
    )
    sampling = SamplingParams(
        temperature=0.0, max_tokens=tool.max_tokens, ignore_eos=True
    )
    runs = []
    for spec, (ids, how) in zip(prompts, rendered, strict=True):
        outputs = llm.generate([{"prompt_token_ids": ids}], sampling, use_tqdm=False)
        per_rank = llm.collective_rpc("fp16_audit_collect")
        generated = len(outputs[0].outputs[0].token_ids)
        print(f"ran {spec.name}: {len(ids)} prompt + {generated} generated tokens")
        runs.append(
            dict(
                name=spec.name,
                prompt_tokens=len(ids),
                generated_tokens=generated,
                template=how,
                stats=merge_ranks(per_rank),
            )
        )
    llm.collective_rpc("fp16_audit_remove")
    config = dict(
        argv=list(sys.argv[1:] if argv is None else argv),
        model=args.model,
        dtype=str(model_config.dtype),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=model_config.max_model_len,
        max_tokens=tool.max_tokens,
        **audit_config,
    )
    report = build_report(runs, workers, config)
    tool.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(format_report(report))
    print(f"\nJSON report: {tool.output}")
    return exit_code(report)


def main(argv: Sequence[str] | None = None) -> int:
    """Exit status: 0 clean, 1 flagged, 2 the audit failed."""
    try:
        return run(argv)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.exit(main())
