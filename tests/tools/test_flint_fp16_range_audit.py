# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dry run of the FP16 range audit's hook and report logic.

A tiny pure-torch model with a vLLM-style ``model.layers`` stack stands in
for the engine; nothing here starts vLLM.
"""

import importlib
import json
from types import SimpleNamespace

import msgspec
import pytest
import torch
from torch import nn

from tools.flint import fp16_range_audit as audit

pytestmark = pytest.mark.cpu_test

HIDDEN, HEADS, HEAD_DIM, VOCAB = 16, 2, 8, 32


class TinyRMSNorm(nn.Module):
    """vLLM RMSNorm calling convention: with a residual, returns (out, sum)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
        return out if residual is None else (out, residual)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(HIDDEN, 3 * HEADS * HEAD_DIM, bias=False)
        self.q_norm = TinyRMSNorm(HEAD_DIM)
        self.k_norm = TinyRMSNorm(HEAD_DIM)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.scaling = HEAD_DIM**-0.5

    def forward(self, positions, hidden_states):
        tokens = hidden_states.shape[0]
        q, k, v = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        q = self.q_norm(q.view(tokens, HEADS, HEAD_DIM))
        k = self.k_norm(k.view(tokens, HEADS, HEAD_DIM))
        v = v.view(tokens, HEADS, HEAD_DIM)
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scaling
        mask = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        out = torch.einsum("hqk,khd->qhd", probs, v).reshape(tokens, -1)
        return self.o_proj(out)


class TinyMLP(nn.Module):
    def __init__(self, inner=24):
        super().__init__()
        self.gate_up_proj = nn.Linear(HIDDEN, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, HIDDEN, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(nn.functional.silu(gate) * up)


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, 2, bias=False)
        self.experts = nn.ModuleList([TinyMLP(12), TinyMLP(12)])
        self.shared_expert = TinyMLP(12)
        self.use_shared = True

    def forward(self, x):
        weights = self.gate(x).sigmoid()
        out = sum(
            w.unsqueeze(-1) * e(x) for w, e in zip(weights.unbind(-1), self.experts)
        )
        return out + self.shared_expert(x) if self.use_shared else out


class TinyLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP() if index == 0 else TinyMoE()
        self.input_layernorm = TinyRMSNorm(HIDDEN)
        self.post_attention_layernorm = TinyRMSNorm(HIDDEN)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class TinyModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(TinyLayer(i) for i in range(num_layers))
        self.norm = TinyRMSNorm(HIDDEN)

    def forward(self, input_ids, positions):
        hidden_states, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class TinyLogitsProcessor(nn.Module):
    def forward(self, lm_head, hidden_states):
        return hidden_states @ lm_head.weight.t()


class TinyForCausalLM(nn.Module):
    def __init__(self, num_layers=3, pp_missing=()):
        super().__init__()
        torch.manual_seed(0)
        self.model = TinyModel(num_layers)
        for index in pp_missing:
            self.model.layers[index] = nn.Identity()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.logits_processor = TinyLogitsProcessor()
        # A vision tower with its own (shorter) layer list must be ignored.
        self.vision = nn.Module()
        self.vision.layers = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return self.logits_processor(self.lm_head, hidden_states)


def step(model, tokens):
    with torch.no_grad():
        ids = torch.arange(tokens) % VOCAB
        return model(ids, torch.arange(tokens))


def capture(model):
    """Independent reference hooks: the raw tensors the audit should see."""
    seen: dict[str, list[torch.Tensor]] = {}

    def keep(name, pick):
        def hook(module, args, output):
            seen.setdefault(name, []).append(pick(args, output).detach().clone())

        return hook

    handles = []
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, nn.Identity):
            continue
        handles += [
            layer.self_attn.o_proj.register_forward_hook(
                keep(f"{idx}.attn_in_o", lambda a, o: a[0])
            ),
            layer.mlp.register_forward_hook(keep(f"{idx}.mlp_out", lambda a, o: o)),
            layer.register_forward_hook(
                keep(f"{idx}.stream", lambda a, o: o[0].float() + o[1].float())
            ),
            layer.register_forward_hook(keep(f"{idx}.residual", lambda a, o: o[1])),
            layer.self_attn.q_norm.register_forward_hook(
                keep(f"{idx}.q_norm", lambda a, o: o)
            ),
        ]
    return seen, handles


def reference_quantile(x, q):
    values = x.detach().float().abs().reshape(-1)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return torch.quantile(values, q, interpolation="higher").item()


def as_plain(maxes, counts):
    max_abs, tail, row = maxes.tolist()
    return dict(max_abs=max_abs, tail=tail, row=row, counts=counts.tolist())


# --------------------------------------------------------------------------
# tensor_stats
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_elems", [7, 100, 1 << 22])
def test_tensor_stats_matches_reference(chunk_elems):
    torch.manual_seed(1)
    x = torch.randn(37, 129) * 3
    x[3, 5] = float("inf")
    x[4, 6] = float("-inf")
    x[5, 7] = float("nan")
    x[6, 8] = 1234.5
    stats = as_plain(
        *audit.tensor_stats(
            x, threshold=8.0, quantile=0.999, chunk_elems=chunk_elems, rownorm=True
        )
    )
    finite = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    assert stats["max_abs"] == pytest.approx(1234.5)
    assert stats["counts"] == [1, 2, int((finite.abs() > 8.0).sum())]
    assert stats["tail"] == pytest.approx(reference_quantile(x, 0.999))
    norms = torch.linalg.vector_norm(x, dim=-1)
    norms = torch.where(torch.isfinite(norms), norms, torch.zeros_like(norms))
    assert stats["row"] == pytest.approx(norms.max().item())


@pytest.mark.parametrize("numel", [1, 2, 10, 2048, 10007])
def test_top_count_matches_torch_quantile(numel):
    torch.manual_seed(numel)
    x = torch.rand(numel)
    for q in (0.5, 0.99, 0.9999):
        keep = audit.top_count(numel, q)
        value = torch.topk(x, keep).values.min().item()
        assert value == pytest.approx(reference_quantile(x, q))
    assert audit.top_count(numel, 0.0) == 0


def test_stream_sum_is_computed_in_fp32():
    # Each FP16 operand fits, but their sum (the residual the next fused
    # add+RMSNorm writes back in FP16) does not.
    hidden = torch.full((4, 8), 40000.0, dtype=torch.float16)
    residual = torch.full((4, 8), 40000.0, dtype=torch.float16)
    assert torch.isinf(hidden + residual).all()
    stats = as_plain(
        *audit.tensor_stats(hidden, threshold=30000.0, quantile=0.9999, other=residual)
    )
    assert stats["max_abs"] == pytest.approx(80000.0)
    assert stats["counts"] == [0, 0, 32]
    alone = as_plain(*audit.tensor_stats(hidden, threshold=30000.0, quantile=0.0))
    assert alone["max_abs"] == pytest.approx(40000.0)
    assert alone["tail"] == 0.0


def test_tensor_stats_fp16_inf_and_nan():
    x = torch.tensor([[1.0, -65504.0], [float("inf"), float("nan")]]).half()
    stats = as_plain(*audit.tensor_stats(x, threshold=30000.0, quantile=0.9999))
    assert stats["max_abs"] == 65504.0
    assert stats["counts"] == [1, 1, 1]


# --------------------------------------------------------------------------
# RangeAuditor on a tiny model
# --------------------------------------------------------------------------


def test_find_layer_stack_prefers_decoder_layers():
    model = TinyForCausalLM()
    path, layers = audit.find_layer_stack(model)
    assert path == "model.layers" and len(layers) == 3
    assert audit.find_layer_stack(model, "vision.layers")[0] == "vision.layers"
    with pytest.raises(TypeError):
        audit.find_layer_stack(model, "model.norm")
    with pytest.raises(RuntimeError):
        audit.find_layer_stack(nn.Sequential(nn.Linear(2, 2)))


def test_install_finds_expected_probes():
    model = TinyForCausalLM()
    auditor = audit.RangeAuditor(model)
    info = auditor.install()
    assert info["layers_path"] == "model.layers"
    assert info["hooked_layers"] == ["0", "1", "2"]
    assert info["layer_class"] == "TinyLayer"
    probes = info["probes"]
    for name in ("attn_in_o", "attn_o", "q_norm", "k_norm", "mlp_out", "stream"):
        assert probes[name] == ["0", "1", "2"]
    assert probes["mlp_act"] == ["0"]  # dense layer only
    assert probes["shared_act"] == ["1", "2"]  # MoE layers only
    assert probes["norm_out"] == probes["logits"] == ["final"]
    assert info["paths"]["attn_in_o"] == "model.layers.0.self_attn.o_proj"
    assert info["paths"]["shared_act"] == "model.layers.1.mlp.shared_expert.down_proj"
    assert info["paths"]["norm_out"] == "model.norm"
    assert info["paths"]["logits"] == "logits_processor"
    assert info["scales"] == {k: pytest.approx(HEAD_DIM**-0.5) for k in "012"}
    with pytest.raises(RuntimeError):
        auditor.install()


def test_pipeline_placeholders_are_skipped():
    model = TinyForCausalLM(num_layers=4, pp_missing=(0, 1))

    # A later pipeline stage receives hidden states instead of token ids.
    class Stage(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner.model

        def forward(self, hidden_states, positions):
            residual = None
            for layer in self.model.layers[2:]:
                hidden_states, residual = layer(positions, hidden_states, residual)
            return hidden_states

    stage = Stage(model)
    auditor = audit.RangeAuditor(stage)
    assert auditor.install()["hooked_layers"] == ["2", "3"]
    with torch.no_grad():
        stage(torch.randn(5, HIDDEN), torch.arange(5))
    # An intermediate stage never runs the final norm or the logits.
    assert set(auditor.collect()["stats"]) == {"2", "3"}


def test_collect_matches_reference_per_phase():
    model = TinyForCausalLM()
    auditor = audit.RangeAuditor(model, threshold=1e9, quantile=0.99, chunk_elems=10)
    auditor.install()
    seen, handles = capture(model)
    step(model, 6)  # prefill
    step(model, 1)  # decode
    step(model, 1)
    stats = auditor.collect()["stats"]
    for handle in handles:
        handle.remove()

    for idx in "012":
        for name in ("attn_in_o", "mlp_out", "stream", "residual"):
            prefill = stats[idx][name]["prefill"]
            decode = stats[idx][name]["decode"]
            tensors = seen[f"{idx}.{name}"]
            assert prefill["calls"] == 1 and decode["calls"] == 2
            assert prefill["numel"] == tensors[0].numel()
            assert prefill["max_abs"] == pytest.approx(tensors[0].abs().max().item())
            assert prefill["quantile_abs"] == pytest.approx(
                reference_quantile(tensors[0], 0.99)
            )
            decode_max = max(t.abs().max().item() for t in tensors[1:])
            assert decode["max_abs"] == pytest.approx(decode_max)
            assert prefill["nan"] == prefill["inf"] == prefill["over"] == 0
        q_norm = stats[idx]["q_norm"]["prefill"]
        assert q_norm["dim"] == HEAD_DIM
        assert q_norm["max_rownorm"] == pytest.approx(
            torch.linalg.vector_norm(seen[f"{idx}.q_norm"][0], dim=-1).max().item()
        )
    assert stats["final"]["logits"]["prefill"]["numel"] == 6 * VOCAB
    # The decode/prefill split follows the token count of each forward.
    assert stats["0"]["attn_in_o"]["decode"]["numel"] == 2 * HEADS * HEAD_DIM
    # collect() resets by default.
    assert auditor.collect()["stats"] == {}


def test_remove_detaches_all_hooks():
    model = TinyForCausalLM()
    auditor = audit.RangeAuditor(model)
    auditor.install()
    auditor.remove()
    step(model, 4)
    assert auditor.collect()["stats"] == {}
    auditor.install()  # can be installed again after removal
    step(model, 4)
    assert auditor.collect()["stats"]


def run_prompts(model, auditor, prompts):
    runs = []
    for name, tokens in prompts:
        step(model, tokens)
        step(model, 1)
        collected = auditor.collect()
        runs.append(
            dict(
                name=name,
                prompt_tokens=tokens,
                generated_tokens=1,
                template="raw",
                stats=audit.merge_ranks([collected]),
            )
        )
    return runs


def test_clean_report_exits_zero():
    model = TinyForCausalLM()
    auditor = audit.RangeAuditor(model)
    info = auditor.install()
    runs = run_prompts(model, auditor, [("short", 5), ("long", 9)])
    report = audit.build_report(runs, [info], dict(threshold=30000.0, quantile=0.9999))
    json.dumps(report, allow_nan=False)
    assert report["ok"] and audit.exit_code(report) == 0
    assert report["flags"] == [] and report["never_fired"] == {}
    layer = report["layers"]["1"]
    assert layer["stream"]["calls"] == {"prefill": 2, "decode": 2}
    assert layer["stream"]["worst_prompt"] in {"short", "long"}
    assert layer["qk_bound"]["bound"]
    expected = (
        layer["q_norm"]["max_rownorm"] * layer["k_norm"]["max_rownorm"] * HEAD_DIM**-0.5
    )
    assert layer["qk_bound"]["max_abs"] == pytest.approx(expected)
    text = audit.format_report(report)
    assert "OK: no finite |x| above 30000" in text
    assert "prefill=2 decode=2" in text
    assert "qk_bound" in text
    for key in ("    0", "    1", "    2", "final"):
        assert key in text


def test_overflow_and_inf_are_flagged():
    model = TinyForCausalLM()
    with torch.no_grad():
        # Layer 1's shared expert blows up: its MoE output goes far above the
        # threshold and the residual stream follows it into the next layers.
        model.model.layers[1].mlp.shared_expert.down_proj.weight.mul_(1e5)
        # Layer 2's attention output contains an inf.
        model.model.layers[2].self_attn.o_proj.weight[0, 0] = float("inf")
    auditor = audit.RangeAuditor(model, threshold=30000.0)
    info = auditor.install()
    runs = run_prompts(model, auditor, [("p", 6)])
    report = audit.build_report(runs, [info], dict(threshold=30000.0, quantile=0.9999))
    json.dumps(report, allow_nan=False)  # no inf/nan leaks into the JSON
    assert not report["ok"] and audit.exit_code(report) == 1
    flagged = {(f["layer"], f["probe"]) for f in report["flags"]}
    assert ("1", "mlp_out") in flagged and ("1", "stream") in flagged
    assert ("0", "mlp_out") not in flagged
    attn = report["layers"]["2"]["attn_o"]
    assert attn["inf"] + attn["nan"] > 0
    assert ("2", "attn_o") in flagged
    assert any("inf" in r or "nan" in r for f in report["flags"] for r in f["reasons"])
    text = audit.format_report(report)
    assert "FLAGGED" in text
    assert "*" in text and "!" in text


def test_never_fired_probes_are_reported():
    model = TinyForCausalLM()
    for layer in model.model.layers[1:]:
        layer.mlp.use_shared = False
    auditor = audit.RangeAuditor(model)
    info = auditor.install()
    runs = run_prompts(model, auditor, [("p", 4)])
    report = audit.build_report(runs, [info], dict(threshold=30000.0, quantile=0.9999))
    assert report["never_fired"] == {"shared_act": ["1", "2"]}
    assert report["ok"]
    assert "never called" in audit.format_report(report)


def test_merge_ranks_and_prompts():
    def stat(max_abs, calls, nan=0):
        return dict(
            calls=calls,
            numel=10,
            dim=4,
            max_abs=max_abs,
            quantile_abs=max_abs / 2,
            max_rownorm=0.0,
            nan=nan,
            inf=0,
            over=0,
        )

    rank0 = {"stats": {"0": {"attn_in_o": {"prefill": stat(3.0, 2)}}}}
    rank1 = {"stats": {"0": {"attn_in_o": {"prefill": stat(5.0, 2, nan=1)}}}}
    merged = audit.merge_ranks([rank0, rank1])["0"]["attn_in_o"]["prefill"]
    assert merged["calls"] == 2  # every rank runs every forward
    assert merged["numel"] == 20 and merged["nan"] == 1
    assert merged["max_abs"] == 5.0 and merged["quantile_abs"] == 2.5
    across_prompts = audit.merge_stats([rank0["stats"], rank1["stats"]])
    assert across_prompts["0"]["attn_in_o"]["prefill"]["calls"] == 4


# --------------------------------------------------------------------------
# Worker extension (the RPC surface used through LLM.collective_rpc)
# --------------------------------------------------------------------------


def test_extension_resolves_by_qualname():
    module_name, _, class_name = audit.EXTENSION_QUALNAME.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    assert cls.__name__ == "Fp16RangeAuditExtension"
    public = [a for a in dir(cls) if not a.startswith("__")]
    assert public and all(a.startswith("fp16_audit_") for a in public)


def test_extension_round_trip_is_msgpack_native():
    model = TinyForCausalLM()

    class FakeWorker(audit.Fp16RangeAuditExtension):
        rank = 1

        def get_model(self):
            return model

    worker = FakeWorker()
    config = dict(threshold=30000.0, quantile=0.9999, layers_path=None)
    # Arguments and results must encode without pickle (no insecure
    # serialization between the engine core and the LLM process).
    config = msgspec.msgpack.decode(msgspec.msgpack.encode(config))
    info = worker.fp16_audit_install(config)
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(info)) == info
    assert info["rank"] == 1
    step(model, 5)
    collected = worker.fp16_audit_collect()
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(collected)) == collected
    assert collected["rank"] == 1 and collected["stats"]["0"]["stream"]["prefill"]
    assert worker.fp16_audit_install(config)["hooked_layers"] == ["0", "1", "2"]
    assert worker.fp16_audit_remove() is True
    step(model, 5)
    assert worker.fp16_audit_remove() is False
    with pytest.raises(RuntimeError):
        worker.fp16_audit_collect()


# --------------------------------------------------------------------------
# Engine arguments and prompts
# --------------------------------------------------------------------------


def test_engine_kwargs_force_eager_uncompiled_extension():
    kwargs = audit.engine_kwargs(dict(model="/m", tensor_parallel_size=2))
    assert kwargs["enforce_eager"] is True
    assert kwargs["compilation_config"] == {"mode": 0}
    assert kwargs["worker_extension_cls"] == audit.EXTENSION_QUALNAME
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["model"] == "/m" and kwargs["tensor_parallel_size"] == 2

    kept = audit.engine_kwargs(
        dict(
            enforce_eager=False,
            enable_prefix_caching=True,
            compilation_config={"mode": 3, "cudagraph_mode": "NONE"},
        )
    )
    assert kept["enforce_eager"] is True and kept["enable_prefix_caching"] is True
    assert kept["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
    assert audit.engine_kwargs(dict(compilation_config=3))["compilation_config"] == 0

    original = SimpleNamespace(mode=3, custom_ops=["all"])
    converted = audit.engine_kwargs(dict(compilation_config=original))
    assert converted["compilation_config"].mode == 0
    assert converted["compilation_config"].custom_ops == ["all"]
    assert original.mode == 3  # the caller's object is not mutated

    with pytest.raises(ValueError):
        audit.engine_kwargs(dict(worker_extension_cls="other.Extension"))


class FakeTokenizer:
    """Whitespace tokenizer with a trivial chat template."""

    def __init__(self, template=True):
        self.template = template

    def encode(self, text, add_special_tokens=True):
        ids = [len(word) for word in text.split()]
        return ([0] if add_special_tokens else []) + ids

    def apply_chat_template(self, messages, tools=None, **kwargs):
        if not self.template:
            raise ValueError("tokenizer has no chat template")
        assert kwargs == dict(tokenize=False, add_generation_prompt=True)
        turns = [f"<{m['role']}> {m['content']}" for m in messages]
        if tools:
            turns.insert(0, "<tools> " + " ".join(t["function"]["name"] for t in tools))
        return "<bos> " + " ".join(turns) + " <assistant>"


def test_default_prompts_and_long_prompt_sizing():
    prompts = audit.default_prompts(long_prompt_tokens=4000)
    names = [p.name for p in prompts]
    assert len(set(names)) == len(names) and names[-1] == "long_repo_context"
    assert len(audit.default_prompts(long_prompt_tokens=0)) == len(prompts) - 1

    tokenizer = FakeTokenizer()
    long_spec = prompts[-1]
    ids, how = audit.render_prompt(long_spec, tokenizer, budget=100000)
    assert how == "chat" and 0.9 * 4000 <= len(ids) <= 4000
    text = " ".join(m["content"] for m in audit._expand(long_spec, 3))
    assert "shard_2" in text and audit.REPEAT_MARK not in text
    # The long prompt shrinks to what fits in the context.
    ids, _ = audit.render_prompt(long_spec, tokenizer, budget=1500)
    assert len(ids) <= 1500
    for spec in prompts[:-1]:
        ids, how = audit.render_prompt(spec, tokenizer, budget=100000)
        assert how == "chat" and len(ids) > 100
    with pytest.raises(audit.AuditError):
        audit.render_prompt(prompts[0], tokenizer, budget=50)


def test_render_without_chat_template_falls_back_to_plain():
    spec = audit.default_prompts(0)[0]
    ids, how = audit.render_prompt(spec, FakeTokenizer(template=False), budget=10**6)
    assert how == "plain" and ids[0] == 0
    raw = audit.PromptSpec(name="raw", text="a bb ccc")
    assert audit.render_prompt(raw, FakeTokenizer(), budget=10) == ([0, 1, 2, 3], "raw")


def test_load_prompts_json_and_jsonl(tmp_path):
    messages = [{"role": "user", "content": "hi"}]
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(["plain text", {"name": "chat", "messages": messages}]))
    specs = audit.load_prompts(path)
    assert [s.name for s in specs] == ["prompt_0", "chat"]
    assert specs[0].text == "plain text" and specs[1].messages == messages

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"name": "a", "prompt": "x y"}) + "\n\n" + json.dumps("z") + "\n"
    )
    specs = audit.load_prompts(path)
    assert [(s.name, s.text) for s in specs] == [("a", "x y"), ("prompt_1", "z")]

    path.write_text(json.dumps({"name": "empty"}) + "\n")
    with pytest.raises(ValueError):
        audit.load_prompts(path)
