# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for frozen multi-task RoboTwin ROCm soak aggregation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "robotwin_mixed_task_soak_summary.py"
    spec = importlib.util.spec_from_file_location("robotwin_mixed_task_soak_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(module: ModuleType, task: str, seed: int, success: bool = True) -> dict[str, Any]:
    return {
        "robotwin_revision": module.ROBOTWIN_REVISION,
        "xpolicy_revision": module.XPOLICY_REVISION,
        "accelerator": "rocm",
        "backend": "hip",
        "left_planner": "CuroboPlanner",
        "right_planner": "CuroboPlanner",
        "xpolicy_fail_closed": True,
        "task": task,
        "seed": seed,
        "task_success": success,
    }


def _write(tmp_path: Path, episodes: list[dict[str, Any]]) -> list[Path]:
    paths = []
    for index, episode in enumerate(episodes):
        path = tmp_path / f"episode_{index}.json"
        path.write_text(json.dumps(episode), encoding="utf-8")
        paths.append(path)
    return paths


def test_mixed_soak_accepts_two_complete_four_seed_tasks(tmp_path: Path) -> None:
    """Accept eight successful HIP episodes spanning two frozen tasks."""
    module = _load_module()
    episodes = [
        _episode(module, task, seed)
        for task in ("stack_blocks_two", "stack_blocks_three")
        for seed in range(4)
    ]

    result = module.summarize(_write(tmp_path, episodes))

    assert result["passed"]
    assert result["task_count"] == 2
    assert result["episode_count"] == 8
    assert result["success_rate"] == 1.0


def test_mixed_soak_keeps_failed_seed_in_denominator(tmp_path: Path) -> None:
    """Keep a failed extended-task seed in the release success-rate denominator."""
    module = _load_module()
    episodes = [
        _episode(module, task, seed, success=not (task == "stack_blocks_three" and seed == 2))
        for task in ("stack_blocks_two", "stack_blocks_three")
        for seed in range(4)
    ]

    result = module.summarize(_write(tmp_path, episodes))

    assert not result["passed"]
    assert result["success_rate"] == 0.875
    assert result["failed_episodes"] == [{"task": "stack_blocks_three", "seed": 2}]


def test_mixed_soak_rejects_duplicate_task_seed(tmp_path: Path) -> None:
    """Reject duplicate evidence that could bias the aggregate result."""
    module = _load_module()
    episode = _episode(module, "stack_blocks_two", 0)

    with pytest.raises(ValueError, match="unique task and seed"):
        module.summarize(_write(tmp_path, [episode, episode]), minimum_episodes=2)


def test_mixed_soak_rejects_runtime_fallback(tmp_path: Path) -> None:
    """Reject an otherwise successful set containing a non-cuRobo planner."""
    module = _load_module()
    episodes = [
        _episode(module, task, seed)
        for task in ("stack_blocks_two", "stack_blocks_three")
        for seed in range(4)
    ]
    episodes[-1]["right_planner"] = "MplibPlanner"

    result = module.summarize(_write(tmp_path, episodes))

    assert not result["passed"]
    assert result["fallback_count"] == 1
