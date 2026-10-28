# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 attention route for learned sinks: backend choice and admission.

CPU only: the platform capability, the ``flash_attn_v100.SUPPORTS_SINKS``
flag and the native operators are faked. The executors in ``decode.py`` and
``prefill.py`` run for real, so the end-to-end cases check that the sinks
reach the generic native kernels through the routes that exist today.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.platforms
from vllm.model_executor.models import config as qualification
from vllm.platforms import runtime_defaults
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_v100 import decode as v100_decode
from vllm.v1.attention.backends.flash_v100 import impl as v100_impl
from vllm.v1.attention.backends.flash_v100 import ops as v100_ops
from vllm.v1.attention.backends.flash_v100.backend import FlashAttnV100Backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

pytestmark = pytest.mark.cpu_test

NUM_HEADS = 32
NUM_KV_HEADS = 4
HEAD_SIZE = 128
SLIDING_WINDOW = 513
BLOCK_SIZE = 16


def _set_native_flag(monkeypatch, supports_sinks: bool | None) -> None:
    """Install a fake ``flash_attn_v100`` module; None removes the flag."""
    module = SimpleNamespace()
    if supports_sinks is not None:
        module.SUPPORTS_SINKS = supports_sinks
    monkeypatch.setitem(sys.modules, "flash_attn_v100", module)


# ---------------------------------------------------------------------------
# Capability flag and platform detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, expected",
    [
        (None, False),  # extension missing: the import raises ImportError
        (SimpleNamespace(), False),  # build without the flag
        (SimpleNamespace(SUPPORTS_SINKS=False), False),
        (SimpleNamespace(SUPPORTS_SINKS=True), True),
    ],
)
def test_capability_flag_reader(monkeypatch, module, expected):
    monkeypatch.setitem(sys.modules, "flash_attn_v100", module)
    assert qualification.flash_v100_supports_sinks() is expected


class _Platform:
    def __init__(self, *capabilities: tuple[int, int], cuda: bool = True):
        self.capabilities = capabilities
        self.cuda = cuda

    def is_cuda(self) -> bool:
        return self.cuda

    def is_device_capability(self, capability, device_id=None) -> bool:
        return tuple(capability) == self.capabilities[device_id]


def _set_platform(monkeypatch, *capabilities, cuda=True):
    monkeypatch.setattr(
        vllm.platforms, "current_platform", _Platform(*capabilities, cuda=cuda)
    )
    monkeypatch.setattr(
        runtime_defaults,
        "_participating_cuda_device_ids",
        lambda cfg: tuple(range(len(capabilities))),
    )


@pytest.mark.parametrize(
    "capabilities, cuda, expected",
    [
        (((7, 0), (7, 0)), True, True),
        (((7, 2),), True, True),
        (((7, 0), (8, 0)), True, False),  # heterogeneous stages
        (((7, 5),), True, False),
        (((8, 0),), True, False),
        (((7, 0),), False, False),
        ((), True, False),  # no participating CUDA device
    ],
)
def test_sm70_deployment(monkeypatch, capabilities, cuda, expected):
    _set_platform(monkeypatch, *capabilities, cuda=cuda)
    assert qualification.sm70_deployment(SimpleNamespace()) is expected


# ---------------------------------------------------------------------------
# FlintModelConfig backend choice
# ---------------------------------------------------------------------------


def _vllm_config(
    *,
    backend=None,
    dtype=torch.float16,
    head_dim=HEAD_SIZE,
    cache_dtype="auto",
    speculative_config=None,
    enabled=True,
):
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=backend, flash_v100=SimpleNamespace(enabled=enabled)
        ),
        model_config=SimpleNamespace(
            dtype=dtype, hf_text_config=SimpleNamespace(head_dim=head_dim)
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        speculative_config=speculative_config,
    )


@pytest.fixture
def config_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(qualification, "logger", logger)
    return logger


def _choose(monkeypatch, *, flag, capabilities=((7, 0), (7, 0)), **kwargs):
    _set_platform(monkeypatch, *capabilities)
    _set_native_flag(monkeypatch, flag)
    vllm_config = _vllm_config(**kwargs)
    qualification.FlintModelConfig.verify_and_update_config(vllm_config)
    return vllm_config.attention_config.backend


def test_flint_architecture_is_registered():
    assert (
        qualification.MODELS_CONFIG_MAP["FlintForCausalLM"]
        is qualification.FlintModelConfig
    )


@pytest.mark.parametrize("capability", [(7, 0), (7, 2)])
def test_sm70_with_sink_kernels_selects_flash_v100(
    monkeypatch, config_logger, capability
):
    backend = _choose(monkeypatch, flag=True, capabilities=(capability,))
    assert backend == AttentionBackendEnum.FLASH_ATTN_V100
    assert "FLASH_ATTN_V100" in config_logger.info.call_args.args[0]
    config_logger.warning.assert_not_called()


@pytest.mark.parametrize("flag", [False, None])
def test_sm70_without_sink_kernels_selects_triton(monkeypatch, config_logger, flag):
    backend = _choose(monkeypatch, flag=flag)
    assert backend == AttentionBackendEnum.TRITON_ATTN
    args = config_logger.info.call_args.args
    assert "TRITON_ATTN" in args[0] and "SUPPORTS_SINKS" in args[1]
    config_logger.warning.assert_not_called()


def test_missing_extension_selects_triton(monkeypatch, config_logger):
    _set_platform(monkeypatch, (7, 0))
    monkeypatch.setitem(sys.modules, "flash_attn_v100", None)
    vllm_config = _vllm_config()
    qualification.FlintModelConfig.verify_and_update_config(vllm_config)
    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"cache_dtype": "fp8_e5m2"}, "FP16 KV cache"),
        ({"dtype": torch.float32}, "float16"),
        ({"head_dim": 256}, "head_dim=256"),
        ({"speculative_config": object()}, "speculative"),
        ({"enabled": False}, "disabled"),
    ],
)
def test_flash_v100_sink_route_limits_select_triton(
    monkeypatch, config_logger, overrides, reason
):
    backend = _choose(monkeypatch, flag=True, **overrides)
    assert backend == AttentionBackendEnum.TRITON_ATTN
    assert reason in config_logger.info.call_args.args[1]
    config_logger.warning.assert_not_called()


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_triton_with_e4m3_kv_cache_warns(monkeypatch, config_logger, flag, cache_dtype):
    backend = _choose(monkeypatch, flag=flag, cache_dtype=cache_dtype)
    assert backend == AttentionBackendEnum.TRITON_ATTN
    message = config_logger.warning.call_args.args[0]
    assert "E4M3" in message and "--kv-cache-dtype auto" in message


@pytest.mark.parametrize(
    "backend",
    [AttentionBackendEnum.FLASH_ATTN_V100, AttentionBackendEnum.TRITON_ATTN],
)
def test_explicit_backend_is_kept(monkeypatch, config_logger, backend):
    for flag in (True, False):
        assert _choose(monkeypatch, flag=flag, backend=backend) is backend
    config_logger.info.assert_not_called()


@pytest.mark.parametrize("capabilities", [((7, 5),), ((8, 0),), ((7, 0), (8, 0))])
def test_non_sm70_deployment_is_left_to_the_platform(
    monkeypatch, config_logger, capabilities
):
    assert _choose(monkeypatch, flag=True, capabilities=capabilities) is None
    config_logger.info.assert_not_called()


def test_flash_v100_backend_keeps_inherited_sink_capability():
    # Not overridden: sink-carrying draft layers with non-causal attention
    # have no other valid SM70 backend (Triton lacks non-causal attention,
    # FlexAttention lacks sinks). Flint relies on its config override.
    assert FlashAttnV100Backend.supports_sink() is True
    assert FlashAttnV100Backend.supports_non_causal() is True
    assert TritonAttentionBackend.supports_non_causal() is False


# ---------------------------------------------------------------------------
# Flash-V100 sink admission
# ---------------------------------------------------------------------------


class _NativeOps:
    """Fake native operators recording each call's keyword arguments."""

    NAMES = (
        "dense",
        "bhmd",
        "decode",
        "xqa",
        "wmma",
        "prefill",
        "prefill_bhmd",
        "bfla",
        "splitkv",
    )

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.ops = tuple(self._op(name) for name in self.NAMES)

    def _op(self, name):
        def op(query, *args, **kwargs):
            self.calls.append((name, kwargs))
            out = kwargs.get("out")
            if out is not None:
                out.fill_(1)
                return out
            return torch.ones_like(query)

        op.__name__ = name
        return op

    def __getitem__(self, name):
        return self.ops[self.NAMES.index(name)]

    def kwargs_of(self, name):
        return [kwargs for op, kwargs in self.calls if op == name]


@pytest.fixture
def native(monkeypatch):
    ops = _NativeOps()
    monkeypatch.setattr(v100_ops, "get_flash_ops", lambda: ops.ops)
    for name in (
        "get_flash_grouped_verify_op",
        "get_fp8_e5m2_paged_kv_bridge_op",
        "get_sm70_v37_e4m3_bridge_op",
    ):
        monkeypatch.setattr(v100_ops, name, lambda: None)
    monkeypatch.delenv("VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_USE_TRITON", raising=False)
    return ops


def _impl(
    monkeypatch,
    *,
    flag=True,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    head_size=HEAD_SIZE,
    kv_cache_dtype="auto",
    sinks="fp32",
    sliding_window=SLIDING_WINDOW,
):
    _set_native_flag(monkeypatch, flag)
    if sinks == "fp32":
        sinks = torch.nn.Parameter(
            torch.linspace(-1.0, 1.0, num_heads), requires_grad=False
        )
    return v100_impl.FlashAttnV100Impl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype=kv_cache_dtype,
        attn_type=AttentionType.DECODER,
        sinks=sinks,
    )


def _assert_not_admitted(impl, reason):
    assert impl._flash_v100_sinks is None
    assert reason in impl._flash_v100_sink_rejection
    assert impl._supports_flash_v100_path() is False


def test_d128_generic_routes_admit_sinks(monkeypatch, native):
    impl = _impl(monkeypatch)
    sinks = impl.sinks

    assert impl._flash_v100_sinks is sinks
    assert sinks.dtype == torch.float32 and sinks.is_contiguous()
    assert isinstance(sinks, torch.nn.Parameter)  # loaded weights stay visible
    assert impl._flash_v100_sink_rejection is None
    assert impl._supports_flash_v100_path() is True

    # Only the generic paged prefill and decode routes stay enabled.
    assert impl.use_flash_v100_prefill_paged is True
    assert impl.use_flash_v100_decode is True
    assert impl.use_decode_scalar_paged is True
    assert impl.use_prefill_paged_cache is True
    assert impl.smallq_decode_max_query_len == 0
    for disabled in (
        "use_decode_paged_prefill",
        "use_decode_dense_cache",
        "use_decode_dense_reference",
        "use_decode_xqa",
        "use_smallq_decode_xqa",
        "use_flash_v100_prefill_splitkv",
        "use_flash_v100_prefill_bfla",
        "use_flash_v100_prefill_contig_dense",
        "use_flash_v100_prefill_gather_dense",
        "use_fp8_prefill_bridge",
    ):
        assert getattr(impl, disabled) is False, disabled

    # Sinks are bound to exactly the two generic native kernels.
    assert impl.flash_attn_prefill_paged is not native["prefill"]
    assert impl.flash_attn_decode_paged is not native["decode"]
    for name, attribute in (
        ("dense", "flash_attn_func"),
        ("bhmd", "flash_attn_bhmd_func"),
        ("xqa", "flash_attn_decode_paged_xqa"),
        ("wmma", "flash_attn_decode_paged_wmma"),
        ("prefill_bhmd", "flash_attn_prefill_paged_bhmd"),
        ("bfla", "flash_attn_prefill_paged_bfla"),
        ("splitkv", "flash_attn_prefill_paged_splitkv"),
    ):
        assert getattr(impl, attribute) is native[name], attribute


def test_bound_operator_passes_sinks_and_keeps_signature(monkeypatch, native):
    impl = _impl(monkeypatch)
    query = torch.zeros(1, NUM_HEADS, HEAD_SIZE)
    impl.flash_attn_decode_paged(query, out=torch.empty_like(query))
    impl.flash_attn_decode_paged(query, out=torch.empty_like(query), sinks=None)
    other = torch.zeros(NUM_HEADS)
    impl.flash_attn_decode_paged(query, out=torch.empty_like(query), sinks=other)
    passed = [kwargs["sinks"] for kwargs in native.kwargs_of("decode")]
    assert passed[0] is impl.sinks and passed[1] is impl.sinks
    assert passed[2] is other
    assert v100_ops.callable_accepts_keyword(
        impl.flash_attn_decode_paged, "window_size"
    )


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"flag": False}, "SUPPORTS_SINKS"),
        ({"flag": None}, "SUPPORTS_SINKS"),
        # D256 owns the specialized prefill kernels and, at GQA 6, XQA decode.
        ({"head_size": 256, "num_heads": 24}, "head size 256"),
        ({"head_size": 64}, "head size 64"),
        ({"kv_cache_dtype": "fp8_e5m2"}, "not fp16"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "not fp16"),
        ({"sinks": torch.zeros(NUM_HEADS, dtype=torch.float16)}, "fp32"),
        ({"sinks": torch.zeros(NUM_HEADS, 2)[:, 0]}, "contiguous"),
    ],
)
def test_other_routes_reject_sinks(monkeypatch, native, kwargs, reason):
    impl = _impl(monkeypatch, **kwargs)
    _assert_not_admitted(impl, reason)
    # Rejected layers keep every route and the unbound native operators.
    assert impl.flash_attn_decode_paged is native["decode"]
    query = torch.zeros(1, 1, impl.num_heads, impl.head_size)
    impl.flash_attn_prefill_paged(query)
    assert "sinks" not in native.kwargs_of("prefill")[-1]
    assert impl.use_prefill_paged_cache is False


def test_d256_xqa_layer_rejects_sinks_with_flag(monkeypatch, native):
    impl = _impl(monkeypatch, head_size=256, num_heads=24, sliding_window=None)
    _assert_not_admitted(impl, "head size 256")


def test_layer_without_sinks_is_unchanged(monkeypatch, native):
    impl = _impl(monkeypatch, sinks=None)
    assert impl._flash_v100_sinks is None
    assert impl._flash_v100_sink_rejection is None
    assert impl._supports_flash_v100_path() is True
    assert impl.flash_attn_decode_paged is native["decode"]
    assert impl.use_prefill_paged_cache is False
    assert impl._decode_helper_sink_kwargs() == {}


def test_unavailable_generic_routes_or_speculation_reject_sinks(monkeypatch, native):
    impl = _impl(monkeypatch)
    spec = SimpleNamespace(speculative_config=object())
    assert "speculative" in impl._flash_v100_sink_rejection_reason(spec)
    impl.use_flash_v100_prefill_paged = False
    assert "unavailable" in impl._flash_v100_sink_rejection_reason(None)


# ---------------------------------------------------------------------------
# Forward through the existing executors
# ---------------------------------------------------------------------------


def _layer():
    return SimpleNamespace(layer_name="layer.0", _k_scale_float=1.0, _v_scale_float=1.0)


def _kv_cache(num_blocks=4):
    return torch.zeros(num_blocks, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)


def _metadata(query_lens, seq_lens):
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    blocks = torch.arange(len(query_lens) * 2, dtype=torch.int32).reshape(-1, 2)
    return SimpleNamespace(
        max_query_len=max(query_lens),
        max_seq_len=int(seq_lens.max()),
        num_actual_tokens=int(query_start_loc[-1]),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        block_table=blocks,
        causal=True,
    )


def _forward(impl, metadata):
    tokens = metadata.num_actual_tokens
    query = torch.zeros(tokens, NUM_HEADS, HEAD_SIZE)
    key = torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE)
    output = torch.zeros_like(query)
    result = impl.forward(
        _layer(), query, key, key.clone(), _kv_cache(), metadata, output
    )
    assert result is output and bool(output.eq(1).all())


def test_decode_reaches_generic_kernel_with_sinks(monkeypatch, native):
    impl = _impl(monkeypatch)
    _forward(impl, _metadata([1, 1], [9, 17]))
    (kwargs,) = native.kwargs_of("decode")
    assert kwargs["sinks"] is impl.sinks
    assert kwargs["window_size"] == (SLIDING_WINDOW - 1, 0)
    assert [name for name, _ in native.calls] == ["decode"]


@pytest.mark.parametrize(
    "query_lens, seq_lens",
    [([12], [12]), ([4], [20])],  # no-prefix and chunked prefill
)
def test_prefill_reaches_generic_kernel_with_sinks(
    monkeypatch, native, query_lens, seq_lens
):
    impl = _impl(monkeypatch)
    _forward(impl, _metadata(query_lens, seq_lens))
    assert [name for name, _ in native.calls] == ["prefill"]
    (kwargs,) = native.kwargs_of("prefill")
    assert kwargs["sinks"] is impl.sinks
    assert kwargs["window_size"] == (SLIDING_WINDOW - 1, 0)


def test_decode_helper_receives_sinks_once_it_accepts_them(monkeypatch, native):
    impl = _impl(monkeypatch)
    received = []

    def helper(self, *args, **kwargs):
        received.append(kwargs.get("sinks"))

    query = torch.zeros(1, NUM_HEADS, HEAD_SIZE)
    call = dict(
        softmax_scale=1.0,
        out=torch.empty_like(query),
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
    )
    caches = (torch.zeros(1), torch.zeros(1), torch.zeros(1, 1), torch.ones(1))
    # Without the helper keyword, the bound operator still applies the sinks.
    impl._call_flash_attn_decode_paged(query, *caches, **call)
    assert native.kwargs_of("decode")[-1]["sinks"] is impl.sinks

    monkeypatch.setattr(
        v100_decode.DecodeExecutor, "_call_flash_attn_decode_paged", helper
    )
    monkeypatch.setattr(v100_impl, "_DECODE_HELPER_TAKES_SINKS", True)
    impl._call_flash_attn_decode_paged(query, *caches, **call)
    assert received == [impl.sinks]


def test_rejected_sink_layer_keeps_the_runtime_error(monkeypatch, native):
    impl = _impl(monkeypatch, flag=False)
    with pytest.raises(RuntimeError, match="cannot run this layer"):
        _forward(impl, _metadata([1], [9]))
    assert native.calls == []
