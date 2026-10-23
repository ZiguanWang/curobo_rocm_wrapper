# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for aggregating strict RoboTwin planner request comparisons."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "robotwin_planner_request_compare_summary.py"
    )
    spec = importlib.util.spec_from_file_location(
        "robotwin_planner_request_compare_summary", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(
    seed: int,
    rocm_seconds: list[float],
    mplib_seconds: list[float],
) -> dict[str, Any]:
    matches = []
    for rocm_value, mplib_value in zip(rocm_seconds, mplib_seconds):
        matches.append(
            {
                "rocm_success": True,
                "mplib_success": True,
                "both_success": True,
                "rocm_seconds": rocm_value,
                "mplib_seconds": mplib_value,
            }
        )
    return {
        "task": "stack_blocks_two",
        "seed": seed,
        "robotwin_revision": "robotwin",
        "xpolicy_revision": "xpolicy",
        "matches": matches,
    }


def test_summary_enforces_sample_count_and_speedup() -> None:
    """Require enough samples in addition to a two-times aggregate p50 speedup."""
    module = _load_module()
    comparisons = [
        _comparison(0, [1.0, 1.5], [3.0, 3.5]),
        _comparison(1, [2.0], [4.0]),
    ]

    insufficient = module.summarize(comparisons, minimum_both_success=4)
    passed = module.summarize(comparisons, minimum_both_success=3)

    assert not insufficient["passed"]
    assert not insufficient["enough_samples"]
    assert passed["passed"]
    assert passed["both_success_count"] == 3
    assert passed["latency_seconds"]["rocm_p50_seconds"] == 1.5
    assert passed["latency_seconds"]["mplib_p50_seconds"] == 3.5


def test_summary_keeps_failed_matches_in_quality_rates() -> None:
    """Retain one-sided failures in success-rate denominators."""
    module = _load_module()
    comparison = _comparison(0, [1.0], [3.0])
    comparison["matches"].append(
        {
            "rocm_success": False,
            "mplib_success": True,
            "both_success": False,
            "rocm_seconds": 2.0,
            "mplib_seconds": 4.0,
        }
    )

    result = module.summarize([comparison], minimum_both_success=1)

    assert result["matched_request_count"] == 2
    assert result["rocm_success_rate"] == 0.5
    assert result["mplib_success_rate"] == 1.0
