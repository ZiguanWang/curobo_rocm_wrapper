# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Test the frozen seven-test ROCm migration contract."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_each_profile_contains_exactly_seven_migration_tests() -> None:
    """Each version keeps exactly the requested seven core migration tests."""
    root = Path(__file__).parents[1]
    for version in ("v0.7.8", "v0.8.0"):
        manifest = tomllib.loads(
            (root / f"profiles/{version}/tests.toml").read_text(encoding="utf-8")
        )["migration"]

        assert manifest["count"] == 7
        assert len(manifest["tests"]) == 7
        assert len(set(manifest["tests"])) == 7
