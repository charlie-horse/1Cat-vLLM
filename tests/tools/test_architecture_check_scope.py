# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep local hook filters covering the offline reader without widening scope."""

import regex as re
import yaml

from tools import generate_architecture_reference as reference

HOOKS = {"check-architecture-reference", "test-architecture-reference"}


def test_local_filters_cover_inputs_and_skip_unrelated_files():
    root = reference.ROOT
    hooks = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    selected = [
        hook for repo in hooks["repos"] for hook in repo["hooks"] if hook["id"] in HOOKS
    ]
    assert {hook["id"] for hook in selected} == HOOKS
    related = [
        *reference.MAINTAINED_DOCS,
        reference.CODECS,
        reference.ROUTES,
        reference.BINDINGS,
        "vllm/config/kernel.py",
        "vllm/config/quantization/base.py",
        "tools/generate_architecture_reference.py",
        "tools/config_ownership.py",
        "tools/pre_commit/environment_readers.py",
        "tools/sm70/path_inventory.py",
        "tests/tools/test_architecture_reference.py",
        "tests/tools/test_architecture_doc_urls.py",
        "tests/tools/test_architecture_check_scope.py",
        "docs/mkdocs/hooks/url_schemes.py",
        "docs/mkdocs/hooks/generate_argparse.py",
        ".pre-commit-config.yaml",
        "mkdocs.yaml",
    ]
    unrelated = [
        "vllm/v1/core/sched/scheduler.py",
        "csrc/attention/paged_attention_v1.cu",
        "docs/design/architecture/sm70_phase_d.md",
        "docs/design/sm70_v100_migration_control.md",
    ]
    for name, expected in [(p, True) for p in related] + [
        (p, False) for p in unrelated
    ]:
        for hook in selected:
            assert not hook.get("always_run", False)
            assert bool(re.search(hook["files"], name)) == expected, name
