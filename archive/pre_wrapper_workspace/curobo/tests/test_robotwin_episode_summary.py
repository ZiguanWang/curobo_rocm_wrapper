# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for RoboTwin episode acceptance aggregation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_summary_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "robotwin_rocm_episode_summary.py"
    spec = importlib.util.spec_from_file_location("robotwin_rocm_episode_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(seed: int, success: bool = True) -> dict[str, Any]:
    return {
        "task": "stack_blocks_two",
        "robotwin_revision": "robotwin-revision",
        "xpolicy_revision": "xpolicy-revision",
        "accelerator": "rocm",
        "backend": "hip",
        "left_planner": "CuroboPlanner",
        "right_planner": "CuroboPlanner",
        "xpolicy_fail_closed": True,
        "seed": seed,
        "task_success": success,
        "total_seconds": 10.0 + seed,
        "play_seconds": 5.0 + seed,
    }


def test_episode_summary_passes_complete_success_set(tmp_path: Path) -> None:
    """Accept a frozen set whose episodes all use HIP and succeed."""
    module = _load_summary_module()
    paths = []
    for seed in range(2):
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_episode(seed)), encoding="utf-8")
        paths.append(path)

    result = module.summarize(paths, minimum_success_rate=1.0)

    assert result["passed"]
    assert result["accelerator"] == "rocm"
    assert result["success_rate"] == 1.0
    assert result["seeds"] == [0, 1]
    assert result["latency_seconds"]["successful_episodes"]["total_p50"] == 10.5
    assert result["latency_seconds"]["all_episode_exits"]["total_p50"] == 10.5
    assert result["input_sha256"][str(paths[0].resolve())] == hashlib.sha256(
        paths[0].read_bytes()
    ).hexdigest()


def test_episode_summary_rejects_failure_or_runtime_fallback(tmp_path: Path) -> None:
    """Reject failed tasks and evidence that did not use the required planner."""
    module = _load_summary_module()
    failed = _episode(0, success=False)
    fallback = _episode(1)
    fallback["right_planner"] = "MplibPlanner"
    paths = []
    for name, evidence in (("failed", failed), ("fallback", fallback)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        paths.append(path)

    result = module.summarize(paths, minimum_success_rate=0.5)

    assert not result["passed"]
    assert result["failed_seeds"] == [0]
    assert result["invalid_runtime_seeds"] == [1]
    assert result["latency_seconds"]["successful_episodes"]["total_p50"] == 11.0
    assert result["latency_seconds"]["all_episode_exits"]["total_p50"] == 10.5


def test_episode_summary_rejects_non_rocm_accelerator(tmp_path: Path) -> None:
    """Reject HIP-labelled evidence that did not run on the ROCm accelerator."""
    module = _load_summary_module()
    evidence = _episode(0)
    evidence["accelerator"] = "cpu"
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = module.summarize([path], minimum_success_rate=1.0)

    assert not result["passed"]
    assert result["invalid_runtime_seeds"] == [0]


def test_episode_summary_supports_explicit_cuda_baseline(tmp_path: Path) -> None:
    """Use the same frozen success-rate contract for the CUDA baseline."""
    module = _load_summary_module()
    evidence = _episode(0)
    evidence["accelerator"] = "cuda"
    evidence["backend"] = "cuda"
    path = tmp_path / "cuda.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = module.summarize(
        [path], minimum_success_rate=1.0, accelerator="cuda", backend="cuda"
    )

    assert result["passed"]
    assert result["accelerator"] == "cuda"
    assert result["backend"] == "cuda"


def test_episode_summary_reports_no_success_latency_for_all_failures(tmp_path: Path) -> None:
    """Do not present failed episode exit time as successful planning latency."""
    module = _load_summary_module()
    paths = []
    for seed in range(2):
        path = tmp_path / f"failed_{seed}.json"
        path.write_text(json.dumps(_episode(seed, success=False)), encoding="utf-8")
        paths.append(path)

    result = module.summarize(paths, minimum_success_rate=1.0)

    assert result["latency_seconds"]["successful_episodes"] is None
    assert result["latency_seconds"]["all_episode_exits"]["total_p50"] == 10.5


def test_episode_summary_rejects_mixed_revisions(tmp_path: Path) -> None:
    """Do not combine results from different upstream revisions."""
    module = _load_summary_module()
    first = _episode(0)
    second = _episode(1)
    second["robotwin_revision"] = "different-revision"
    paths = []
    for index, evidence in enumerate((first, second)):
        path = tmp_path / f"seed_{index}.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="frozen upstream revisions"):
        module.summarize(paths, minimum_success_rate=1.0)


def test_episode_summary_rejects_duplicate_seeds(tmp_path: Path) -> None:
    """Keep repeated evidence from changing the frozen success-rate denominator."""
    module = _load_summary_module()
    paths = []
    for index in range(2):
        path = tmp_path / f"duplicate_{index}.json"
        path.write_text(json.dumps(_episode(0)), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="unique seeds"):
        module.summarize(paths, minimum_success_rate=1.0)
