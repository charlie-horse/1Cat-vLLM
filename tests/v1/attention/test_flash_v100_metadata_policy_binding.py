# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flash-V100 metadata builder binds its per-step policy reads once.

Metadata is built outside a forward context, where the policy adapters fall
back to resolving the standalone Flash-V100 policy from the environment on
every call. Hybrid models build metadata once per KV-cache group and decode
step, so the builder resolves these flags at construction instead.
"""

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.flash_v100 import routing
from vllm.v1.attention.backends.flash_v100.metadata import (
    FlashAttnV100MetadataBuilder,
)


def _builder(**bound):
    builder = object.__new__(FlashAttnV100MetadataBuilder)
    builder._graph_policy = SimpleNamespace(decode_partition_hint=lambda: None)
    builder._decode_active_num_partitions = torch.zeros(1, dtype=torch.int32)
    builder.device = torch.device("cpu")
    for name, value in bound.items():
        setattr(builder, name, value)
    return builder


def _metadata(max_seq_len_hint=4096):
    return SimpleNamespace(
        flash_v100_decode_max_seq_len_hint=max_seq_len_hint,
        flash_v100_decode_workspace_seq_capacity_hint=262144,
        flash_v100_static_decode_seq_hint=None,
        flash_v100_decode_active_num_partitions=None,
    )


def test_bound_flags_skip_the_per_call_policy_reads(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("policy read on the per-step path")

    monkeypatch.setattr(routing, "decode_dynamic_partitions_enabled", unexpected)
    monkeypatch.setattr(routing, "trace_decode_active_metadata", unexpected)
    builder = _builder(_decode_dynamic_partitions=True, _trace_decode_active=False)
    metadata = _metadata()
    builder._update_decode_active_num_partitions(metadata, stage="build")
    active = metadata.flash_v100_decode_active_num_partitions
    assert active is builder._decode_active_num_partitions
    assert int(active[0]) == 4096 // 256


def test_disabled_dynamic_partitions_leave_no_active_count(monkeypatch):
    monkeypatch.setattr(
        routing,
        "decode_dynamic_partitions_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("policy read")),
    )
    builder = _builder(_decode_dynamic_partitions=False, _trace_decode_active=False)
    metadata = _metadata()
    builder._update_decode_active_num_partitions(metadata, stage="build")
    assert metadata.flash_v100_decode_active_num_partitions is None


def test_unbound_builder_resolves_the_flags_per_call(monkeypatch):
    calls: list[str] = []

    def dynamic_enabled() -> bool:
        calls.append("p")
        return True

    def trace(**kwargs) -> None:
        calls.append("trace")

    monkeypatch.setattr(routing, "decode_dynamic_partitions_enabled", dynamic_enabled)
    monkeypatch.setattr(routing, "trace_decode_active_metadata", trace)
    builder = _builder()
    builder._update_decode_active_num_partitions(_metadata(), stage="build")
    assert calls == ["p", "trace"]
