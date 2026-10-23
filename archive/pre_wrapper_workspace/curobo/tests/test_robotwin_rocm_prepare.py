# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reproducible RoboTwin ROCm integration overlay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_prepare_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "robotwin_rocm_prepare.py"
    spec = importlib.util.spec_from_file_location("robotwin_rocm_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_source_applies_once_and_is_idempotent() -> None:
    """Apply an exact replacement once and accept its migrated form thereafter."""
    module = _load_prepare_module()
    replacement = module.Replacement(Path("example.py"), "old\n", "new\n")

    migrated, changed = module.migrate_source("before\nold\nafter\n", replacement)
    repeated, repeated_changed = module.migrate_source(migrated, replacement)

    assert changed
    assert migrated == "before\nnew\nafter\n"
    assert repeated == migrated
    assert not repeated_changed


def test_migrate_source_accepts_migrated_text_containing_original() -> None:
    """Recognize an idempotent replacement that wraps rather than removes its source."""
    module = _load_prepare_module()
    replacement = module.Replacement(
        Path("example.py"),
        "    operation()\n",
        "    if enabled:\n        operation()\n",
    )

    migrated, changed = module.migrate_source(replacement.migrated, replacement)

    assert migrated == replacement.migrated
    assert not changed


def test_migrate_source_upgrades_legacy_migration_and_remains_idempotent() -> None:
    """Upgrade an older overlay shape directly to the latest exact source."""
    module = _load_prepare_module()
    replacement = module.Replacement(
        Path("example.py"),
        "old\n",
        "new\n",
        legacy_originals=("intermediate\n",),
    )

    migrated, changed = module.migrate_source("intermediate\n", replacement)
    repeated, repeated_changed = module.migrate_source(migrated, replacement)

    assert changed
    assert migrated == "new\n"
    assert repeated == migrated
    assert not repeated_changed


def test_stack_blocks_three_overlay_removes_only_final_third_block_retreat() -> None:
    """Keep ordinary retreats while avoiding the harmful post-success third retreat."""
    module = _load_prepare_module()
    replacement = next(
        item
        for item in module.REPLACEMENTS
        if item.relative_path == Path("envs/stack_blocks_three.py")
    )

    migrated, changed = module.migrate_source(replacement.original, replacement)

    assert changed
    assert "if block is not self.block3:" in migrated
    assert migrated.count("move_by_displacement") == 1


@pytest.mark.parametrize("source", ["missing\n", "old\nold\n", "old\nnew\n"])
def test_migrate_source_rejects_unknown_or_ambiguous_source(source: str) -> None:
    """Refuse to patch an unrecognized upstream source shape."""
    module = _load_prepare_module()
    replacement = module.Replacement(Path("example.py"), "old\n", "new\n")

    with pytest.raises(RuntimeError, match="unexpected source shape"):
        module.migrate_source(source, replacement)
