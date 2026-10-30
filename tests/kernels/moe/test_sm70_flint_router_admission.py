# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission checks for the fused SM70 E256/K6 router (no GPU needed)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import sm70_flint_router as mod
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


def make_gate(
    shape=(256, 2048), dtype=torch.float32, out_dtype=torch.float32
) -> GateLinear:
    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    gate.bias = None
    gate.out_dtype = out_dtype
    gate.allow_dsv3_router_gemm = False
    gate.allow_cublas_router_gemm = False
    return gate


def make_runner(**overrides):
    fields = dict(
        layer_name="model.layers.2.mlp.experts",
        gate=make_gate(),
        router=FusedTopKRouter(
            top_k=6, global_num_experts=256, scoring_func="sigmoid", renormalize=True
        ),
        _fse_fuse_gate=False,
        routed_input_transform=None,
        enable_dbo=False,
        _quant_method=SimpleNamespace(is_monolithic=False),
        moe_config=SimpleNamespace(dp_size=1, pcp_size=1, use_ep=False),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def admitted(monkeypatch):
    calls = []

    def op(*args):
        calls.append(args)

    monkeypatch.setattr(mod, "_policy_enabled", lambda: True)
    monkeypatch.setattr(mod, "_is_exact_sm70", lambda device: True)
    monkeypatch.setattr(mod, "_flint_route_op", lambda: op)
    return calls


def test_admits_flint_router(admitted):
    state = mod.admit_flint_fused_router(make_runner())
    assert isinstance(state, mod.FlintFusedRouter)
    assert state.logits.numel() == 2 * 256 and state.logits.dtype == torch.float32
    assert state.counter.tolist() == [0] and state.counter.dtype == torch.int32


def test_kill_switch(admitted, monkeypatch):
    monkeypatch.setattr(mod, "_policy_enabled", lambda: False)
    assert mod.admit_flint_fused_router(make_runner()) is None


def test_kill_switch_reads_routing_policy(monkeypatch):
    from vllm.config.sm70_moe import Sm70MoERoutingPolicy

    assert Sm70MoERoutingPolicy.aliases["flint_fused"] == "VLLM_SM70_FLINT_FUSED_ROUTER"
    for raw, expected in (("0", False), ("1", True)):
        monkeypatch.setenv("VLLM_SM70_FLINT_FUSED_ROUTER", raw)
        policy = Sm70MoERoutingPolicy()
        policy.resolve()
        assert policy.value("flint_fused") is expected
    monkeypatch.delenv("VLLM_SM70_FLINT_FUSED_ROUTER")
    policy = Sm70MoERoutingPolicy()
    policy.resolve()
    assert policy.value("flint_fused") is True


def test_missing_operator_keeps_generic_route(admitted, monkeypatch):
    monkeypatch.setattr(mod, "_flint_route_op", lambda: None)
    assert mod.admit_flint_fused_router(make_runner()) is None


def test_non_sm70_keeps_generic_route(admitted, monkeypatch):
    monkeypatch.setattr(mod, "_is_exact_sm70", lambda device: False)
    assert mod.admit_flint_fused_router(make_runner()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(gate=None),
        dict(gate=torch.nn.Linear(2048, 256, bias=False)),
        dict(gate=make_gate(shape=(512, 2048))),
        dict(gate=make_gate(shape=(256, 2560))),
        dict(gate=make_gate(dtype=torch.float16)),
        dict(gate=make_gate(out_dtype=torch.float16)),
        dict(
            router=FusedTopKRouter(
                top_k=6, global_num_experts=256, scoring_func="softmax"
            )
        ),
        dict(
            router=FusedTopKRouter(
                top_k=8,
                global_num_experts=256,
                scoring_func="sigmoid",
                renormalize=True,
            )
        ),
        dict(
            router=FusedTopKRouter(
                top_k=6,
                global_num_experts=256,
                scoring_func="sigmoid",
                renormalize=False,
            )
        ),
        dict(
            router=FusedTopKRouter(
                top_k=6,
                global_num_experts=256,
                scoring_func="sigmoid",
                renormalize=True,
                eplb_state=object(),
            )
        ),
        dict(_fse_fuse_gate=True),
        dict(routed_input_transform=torch.nn.Identity()),
        dict(enable_dbo=True),
        dict(_quant_method=SimpleNamespace(is_monolithic=True)),
        dict(moe_config=SimpleNamespace(dp_size=2, pcp_size=1, use_ep=False)),
        dict(moe_config=SimpleNamespace(dp_size=1, pcp_size=1, use_ep=True)),
    ],
)
def test_other_layers_keep_generic_route(admitted, overrides):
    assert mod.admit_flint_fused_router(make_runner(**overrides)) is None


def test_biased_gate_is_rejected(admitted):
    gate = make_gate()
    gate.bias = torch.nn.Parameter(torch.zeros(256))
    assert mod.admit_flint_fused_router(make_runner(gate=gate)) is None


@pytest.mark.parametrize(
    ("shape", "dtype", "accepted"),
    [
        ((1, 2048), torch.float16, True),
        ((2, 2048), torch.float16, True),
        ((3, 2048), torch.float16, False),
        ((0, 2048), torch.float16, False),
        ((1, 2048), torch.bfloat16, False),
        ((1, 2048), torch.float32, False),
        ((2, 2560), torch.float16, False),
    ],
)
def test_metadata_gate(shape, dtype, accepted):
    x = torch.zeros(shape, dtype=dtype)
    assert mod.FlintFusedRouter.accepts(x) is accepted


def test_metadata_gate_rejects_odd_strides():
    base = torch.zeros(2, 2050, dtype=torch.float16)
    assert not mod.FlintFusedRouter.accepts(base[:, :2048])
    assert not mod.FlintFusedRouter.accepts(base[:, 1:2049])
    assert not mod.FlintFusedRouter.accepts(torch.zeros(2048, 2, dtype=torch.float16).T)


def test_try_route_defers_to_generic_route(admitted):
    runner = make_runner()
    state = mod.admit_flint_fused_router(runner)
    x = torch.zeros(2, 2048, dtype=torch.float16)

    weights, ids = state.try_route(x)
    assert weights.shape == ids.shape == (2, 6)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    (op_args,) = admitted
    assert op_args[2] is x and op_args[3] is runner.gate.weight
    assert op_args[4] is state.logits and op_args[5] is state.counter

    assert state.try_route(torch.zeros(3, 2048, dtype=torch.float16)) is None
    runner.router.capture_fn = lambda ids: None
    assert state.try_route(x) is None
    runner.router.capture_fn = None
    runner.router.indices_type_getter = lambda: torch.int64
    assert state.try_route(x) is None
    runner.router.indices_type_getter = lambda: torch.int32
    assert state.try_route(x) is not None
    assert len(admitted) == 2


def test_runner_hook_admits_once_outside_capture(admitted, monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    hook = MoERunner._maybe_precomputed_route
    x = torch.zeros(1, 2048, dtype=torch.float16)
    runner = make_runner(
        router_provider=None,
        _precomputed_route=None,
        _precomputed_route_decided=False,
    )
    # Unbound layers (every other model) keep the gate route untouched.
    assert hook(runner, x) is None and not runner._precomputed_route_decided

    calls = []

    def provider(layer):
        calls.append(layer)
        return mod.admit_flint_fused_router(layer)

    runner.router_provider = provider
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert hook(runner, x) is None and not calls
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    weights, ids = hook(runner, x)
    assert weights.shape == ids.shape == (1, 6)
    assert hook(runner, x) is not None
    assert calls == [runner] and len(admitted) == 2
