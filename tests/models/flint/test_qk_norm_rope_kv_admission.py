# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the SM70 fused q/k norm + RoPE + KV-write admission."""

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.models.flint.qk_norm_rope_kv import SPLITTING_OP, fused_route_rejection


def _rope(**overrides):
    kwargs = dict(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=64,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float16,
    )
    kwargs.update(overrides)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        return RotaryEmbedding(**kwargs)


def _args(**overrides):
    args = dict(
        enabled=True,
        is_sm70=True,
        dtype=torch.float16,
        head_dim=128,
        q_eps=1e-6,
        k_eps=1e-6,
        rotary_emb=None,
        kv_cache_dtype="auto",
        attn_type="decoder",
        calculate_kv_scales=False,
        kv_sharing=False,
        backend_writes_kv_in_forward=False,
    )
    args.update(overrides)
    return args


def test_admits_flint_shapes_with_and_without_rope():
    assert fused_route_rejection(**_args()) is None
    assert fused_route_rejection(**_args(rotary_emb=_rope())) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(enabled=False),
        dict(is_sm70=False),
        dict(dtype=torch.bfloat16),
        dict(head_dim=64),
        dict(k_eps=1e-5),
        dict(kv_cache_dtype="fp8_e4m3"),
        dict(attn_type="encoder_only"),
        dict(calculate_kv_scales=True),
        dict(kv_sharing=True),
        dict(backend_writes_kv_in_forward=True),
    ],
)
def test_rejects_other_configs(overrides):
    assert fused_route_rejection(**_args(**overrides)) is not None


@pytest.mark.parametrize(
    "rope_overrides",
    [
        dict(is_neox_style=False),
        dict(rotary_dim=64),
        dict(dtype=torch.float32),
    ],
)
def test_rejects_other_rotary_embeddings(rope_overrides):
    rope = _rope(**rope_overrides)
    assert fused_route_rejection(**_args(rotary_emb=rope)) is not None


def test_rejects_rotary_subclasses():
    class Scaled(RotaryEmbedding):
        pass

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rope = Scaled(128, 128, 64, 10000.0, True, torch.float16)
    assert fused_route_rejection(**_args(rotary_emb=rope)) is not None


@pytest.mark.parametrize("value,expected", [(None, True), ("1", True), ("0", False)])
def test_kill_switch_resolves_into_kernel_config(monkeypatch, value, expected):
    from vllm.config import KernelConfig

    if value is None:
        monkeypatch.delenv("VLLM_SM70_FLINT_QK_ROPE_KV", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", value)
    config = KernelConfig()
    config.resolve_flint_qk_rope_kv()
    assert config.sm70_flint_qk_rope_kv is expected
    # An explicit setting wins over the environment.
    config = KernelConfig(sm70_flint_qk_rope_kv=not expected)
    config.resolve_flint_qk_rope_kv()
    assert config.sm70_flint_qk_rope_kv is (not expected)


def test_op_is_a_default_splitting_op():
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode

    config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, use_inductor_graph_partition=False
    )
    config.set_splitting_ops_for_v1(all2all_backend="", data_parallel_size=1)
    assert SPLITTING_OP in config.splitting_ops
