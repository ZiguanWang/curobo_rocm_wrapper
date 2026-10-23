# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for matching RoboTwin planner requests across backends."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "robotwin_planner_request_compare.py"
    spec = importlib.util.spec_from_file_location("robotwin_planner_request_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence(backend: str, elapsed_seconds: float, success: bool = True) -> dict[str, Any]:
    return {
        "task": "stack_blocks_two",
        "seed": 3,
        "robotwin_revision": "robotwin",
        "xpolicy_revision": "xpolicy",
        "backend": backend,
        "planner_records": [
            {
                "arm": "left",
                "kind": "single",
                "constraint_pose": None,
                "constraint_ignored": False,
                "current_qpos": [0.0, 0.1],
                "target_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "status": "Success" if success else "Fail",
                "elapsed_seconds": elapsed_seconds,
            }
        ],
    }


def test_compare_accepts_two_x_rocm_speedup() -> None:
    """Pass when matched successful ROCm requests are at least twice as fast."""
    module = _load_module()
    result = module.compare(_evidence("hip", 1.0), _evidence("cpu_mplib", 2.5))

    assert result["passed"]
    assert result["matched_request_count"] == 1
    assert result["latency_seconds"]["rocm_speedup_over_mplib"] == 2.5
    assert result["latency_seconds"]["rocm_p90_seconds"] == 1.0
    assert result["latency_seconds"]["mplib_p99_seconds"] == 2.5


def test_compare_rejects_failure_and_ignores_constrained_requests() -> None:
    """Report quality failures and exclude requests with unequal constraint semantics."""
    module = _load_module()
    rocm = _evidence("hip", 1.0, success=False)
    mplib = _evidence("cpu_mplib", 2.0)
    rocm["planner_records"].append(
        {
            "arm": "left",
            "kind": "single",
            "constraint_pose": [1, 1, 1, 0, 0, 0],
            "target_pose": [0.0] * 7,
            "status": "Success",
            "elapsed_seconds": 1.0,
        }
    )
    mplib["planner_records"].append(
        {
            "arm": "left",
            "kind": "single",
            "constraint_ignored": True,
            "target_pose": [0.0] * 7,
            "status": "Success",
            "elapsed_seconds": 0.1,
        }
    )

    result = module.compare(rocm, mplib)

    assert not result["passed"]
    assert result["matched_request_count"] == 1
    assert result["rocm_success_count"] == 0
    assert result["both_success_count"] == 0
    assert result["latency_seconds"] is None


def test_compare_rejects_mismatched_identity() -> None:
    """Do not compare requests from different task seeds or upstream revisions."""
    module = _load_module()
    rocm = _evidence("hip", 1.0)
    mplib = _evidence("cpu_mplib", 2.0)
    mplib["seed"] = 4

    with pytest.raises(ValueError, match="seed"):
        module.compare(rocm, mplib)


def test_compare_requires_matching_start_state() -> None:
    """Exclude target matches whose planner requests start from different joint states."""
    module = _load_module()
    rocm = _evidence("hip", 1.0)
    mplib = _evidence("cpu_mplib", 3.0)
    mplib["planner_records"][0]["current_qpos"] = [0.0, 0.2]

    result = module.compare(rocm, mplib)

    assert result["matched_request_count"] == 0
    assert not result["passed"]


def test_compare_expands_matching_repeated_measurements() -> None:
    """Treat each aligned repetition of the same frozen request as a timing sample."""
    module = _load_module()
    rocm = _evidence("hip", 9.0)
    mplib = _evidence("cpu_mplib", 9.0)
    rocm["planner_records"][0]["repeat_results"] = [
        {"status": "Success", "elapsed_seconds": value} for value in (1.0, 1.5, 2.0)
    ]
    mplib["planner_records"][0]["repeat_results"] = [
        {"status": "Success", "elapsed_seconds": value} for value in (3.0, 3.5, 4.0)
    ]

    result = module.compare(rocm, mplib)

    assert result["matched_request_count"] == 3
    assert result["both_success_count"] == 3
    assert result["latency_seconds"]["rocm_p50_seconds"] == 1.5
    assert result["latency_seconds"]["mplib_p50_seconds"] == 3.5
