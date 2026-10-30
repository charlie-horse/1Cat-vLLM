# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 TP2 all-reduce + fused_add_rms_norm: pattern matching and admission.

CPU only. The patterns are matched on fake-tensor graphs built like the
post-grad graph of a decoder: the custom all-reduce, an optional elementwise
scale or two summed partial outputs, then the fused_add_rms_norm IR op.
"""

from types import SimpleNamespace

import pytest
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.ir.ops
from vllm.compilation.passes.fusion import allreduce_rms_fusion as fusion
from vllm.model_executor.layers import layernorm  # noqa: F401
from vllm.model_executor.models import config as model_config_hooks

pytestmark = pytest.mark.cpu_test

EPS = 1e-6
H = 2048
FUSED = torch.ops.vllm.tp2_all_reduce_fused_add_rms_norm.default
ALL_REDUCE = torch.ops.vllm.all_reduce.default


@pytest.fixture
def tp_group(monkeypatch):
    monkeypatch.setattr(
        fusion, "get_tp_group", lambda: SimpleNamespace(unique_name="tp:0")
    )
    monkeypatch.setattr(fusion, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        fusion,
        "tensor_model_parallel_all_reduce",
        lambda x: torch.ops.vllm.all_reduce(x, group_name="tp:0"),
    )


def _apply(model, *, hidden=H, dtype=torch.float16, sum2=False, eps=EPS):
    with FakeTensorMode():
        inputs = [torch.empty(2, hidden, dtype=dtype)]
        if sum2:
            inputs.append(torch.empty(2, hidden, dtype=dtype))
        inputs += [
            torch.empty(2, hidden, dtype=dtype),
            torch.empty(hidden, dtype=dtype),
        ]
        graph = make_fx(model)(*inputs)
        patterns = PatternMatcherPass()
        fusion.Tp2AllReduceFusedAddRMSNormPattern(eps, dtype, "cpu").register(patterns)
        matches = patterns.apply(graph.graph)
    return matches, [n for n in graph.graph.nodes if n.target == FUSED], graph


def _norm(reduced, residual, weight, eps=EPS):
    return vllm.ir.ops.fused_add_rms_norm(reduced, residual, weight, eps)


@pytest.mark.parametrize("keep_residual", [False, True])
def test_plain_boundary_is_fused(tp_group, keep_residual):
    def model(x, residual, weight):
        out = _norm(
            torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0"), residual, weight
        )
        return out if keep_residual else out[0]

    matches, fused, graph = _apply(model)
    assert matches == 1 and len(fused) == 1
    node = fused[0]
    # (input, input_b, residual, weight, epsilon, scale, group_name)
    assert node.args[1] is None
    assert node.args[4] == EPS and node.args[5] == 1.0
    assert "tp:0" in (*node.args[6:], *node.kwargs.values())
    assert not any(n.target == ALL_REDUCE for n in graph.graph.nodes)


@pytest.mark.parametrize("keep_residual", [False, True])
def test_partial_sum_boundary_is_fused(tp_group, keep_residual):
    def model(a, b, residual, weight):
        reduced = torch.ops.vllm.all_reduce(a.relu() + b.relu(), group_name="tp:0")
        out = _norm(reduced, residual, weight)
        return out if keep_residual else out[0]

    matches, fused, graph = _apply(model, sum2=True)
    assert matches == 1 and len(fused) == 1
    assert fused[0].args[1] is not None
    assert not any(n.target == torch.ops.aten.add.Tensor for n in graph.graph.nodes)


@pytest.mark.parametrize("scale, expected", [(0.125, 1), (0.5, 1), (0.3, 0)])
def test_scaled_boundary_needs_a_power_of_two(tp_group, scale, expected):
    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced * scale, residual, weight)

    matches, fused, _ = _apply(model)
    assert matches == expected and len(fused) == expected
    if fused:
        assert fused[0].args[5] == scale


def test_added_sum_with_another_user_is_not_folded(tp_group):
    def model(a, b, residual, weight):
        total = a + b
        reduced = torch.ops.vllm.all_reduce(total, group_name="tp:0")
        out, res = _norm(reduced, residual, weight)
        return out, res, total

    matches, fused, _ = _apply(model, sum2=True)
    # The add stays (it has another user); the collective + norm still fuse.
    assert matches == 1 and fused[0].args[1] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 4096},  # another hidden size
        {"dtype": torch.bfloat16},
        {"eps": 1e-5},  # the model's epsilon differs from the pattern's
    ],
)
def test_other_shapes_are_left_alone(tp_group, kwargs):
    hidden = kwargs.get("hidden", H)
    eps = 1e-6

    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced, residual, weight, eps)

    matches, fused, _ = _apply(
        model,
        hidden=hidden,
        dtype=kwargs.get("dtype", torch.float16),
        eps=kwargs.get("eps", EPS),
    )
    assert matches == 0 and not fused


def test_unfused_boundary_counter(tp_group):
    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced * 0.3, residual, weight)

    with FakeTensorMode():
        graph = make_fx(model)(
            torch.empty(2, H, dtype=torch.float16),
            torch.empty(2, H, dtype=torch.float16),
            torch.empty(H, dtype=torch.float16),
        )
    assert fusion._tp2_unfused_add_rms_boundaries(graph.graph) == 1


# ---------------------------------------------------------------------------
# Admission: the model config hook requests, the pass checks the engine
# ---------------------------------------------------------------------------


def _config(**overrides):
    values = dict(
        tp=2, pp=1, dtype=torch.float16, hidden=H, disable_ca=False, switch=True
    )
    values.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=values["tp"],
            pipeline_parallel_size=values["pp"],
            disable_custom_all_reduce=values["disable_ca"],
            communication=SimpleNamespace(
                value=lambda field: {"ar_add_rms_tp2": values["switch"]}[field],
                early_value=lambda field: {"ar_add_rms_tp2": values["switch"]}[field],
            ),
        ),
        model_config=SimpleNamespace(
            dtype=values["dtype"], get_hidden_size=lambda: values["hidden"]
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(
                fuse_allreduce_rms=values.get("fuse"),
                fuse_allreduce_add_rms_tp2=False,
            )
        ),
    )


def test_hook_requests_the_pass():
    cfg = _config()
    model_config_hooks.FlintModelConfig._configure_sm70_allreduce_add_rms(cfg)
    assert cfg.compilation_config.pass_config.fuse_allreduce_rms is True
    assert cfg.compilation_config.pass_config.fuse_allreduce_add_rms_tp2 is True


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tp": 1}, "tensor_parallel_size=1"),
        ({"tp": 4}, "tensor_parallel_size=4"),
        ({"pp": 2}, "pipeline_parallel_size=2"),
        ({"disable_ca": True}, "custom all-reduce"),
        ({"dtype": torch.bfloat16}, "dtype"),
        ({"hidden": 4096}, "hidden_size=4096"),
        ({"fuse": False}, "fuse_allreduce_rms"),
        ({"switch": False}, "VLLM_SM70_FLINT_AR_RMSNORM=0"),
    ],
)
def test_hook_rejections(overrides, reason):
    cfg = _config(**overrides)
    assert reason in model_config_hooks._sm70_tp2_allreduce_add_rms_rejection(cfg)
    model_config_hooks.FlintModelConfig._configure_sm70_allreduce_add_rms(cfg)
    assert cfg.compilation_config.pass_config.fuse_allreduce_add_rms_tp2 is False


@pytest.mark.parametrize("raw, requested", [(None, True), ("1", True), ("0", False)])
def test_hook_reads_the_kill_switch(monkeypatch, raw, requested):
    """=0 leaves the pass configuration (and its compile hash) untouched."""
    from vllm.config.execution_policy import CommunicationPolicy

    if raw is None:
        monkeypatch.delenv("VLLM_SM70_FLINT_AR_RMSNORM", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLINT_AR_RMSNORM", raw)
    cfg = _config()
    communication = CommunicationPolicy()
    cfg.parallel_config.communication = communication
    model_config_hooks.FlintModelConfig._configure_sm70_allreduce_add_rms(cfg)
    pass_config = cfg.compilation_config.pass_config
    assert pass_config.fuse_allreduce_add_rms_tp2 is requested
    assert pass_config.fuse_allreduce_rms is (True if requested else None)
    # The engine-wide resolution later keeps the value read by the hook.
    monkeypatch.setenv("VLLM_SM70_FLINT_AR_RMSNORM", "0" if requested else "1")
    communication.resolve()
    assert communication.value("ar_add_rms_tp2") is requested


@pytest.mark.parametrize(
    "overrides, tp, capabilities, has_op, reason",
    [
        ({}, 2, SimpleNamespace(world_size=2), True, None),
        ({"switch": False}, 2, SimpleNamespace(world_size=2), True, "disabled"),
        ({}, 4, SimpleNamespace(world_size=4), True, "TP2"),
        ({"pp": 2}, 2, SimpleNamespace(world_size=2), True, "TP2"),
        ({"dtype": torch.bfloat16}, 2, SimpleNamespace(world_size=2), True, "FP16"),
        ({"hidden": 4096}, 2, SimpleNamespace(world_size=2), True, "4096"),
        ({}, 2, None, True, "custom all-reduce"),
        ({}, 2, SimpleNamespace(world_size=2), False, "extension"),
    ],
)
def test_pass_admission(monkeypatch, overrides, tp, capabilities, has_op, reason):
    monkeypatch.setattr(fusion, "ops_have_tp2_add_rms", lambda: has_op)
    result = fusion.tp2_add_rms_rejection(_config(**overrides), tp, capabilities)
    if reason is None:
        assert result is None
    else:
        assert reason in result


@pytest.mark.parametrize("raw, expected", [(None, True), ("1", True), ("0", False)])
def test_kill_switch_resolves_into_the_communication_policy(monkeypatch, raw, expected):
    from vllm.config.execution_policy import CommunicationPolicy

    if raw is None:
        monkeypatch.delenv("VLLM_SM70_FLINT_AR_RMSNORM", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLINT_AR_RMSNORM", raw)
    policy = CommunicationPolicy()
    policy.resolve()
    assert policy.value("ar_add_rms_tp2") is expected
