# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for machine-readable RoboTwin ROCm episode evidence."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load_episode_module() -> ModuleType:
    benchmark_dir = Path(__file__).parents[2] / "benchmark"
    path = benchmark_dir / "robotwin_rocm_episode.py"
    spec = importlib.util.spec_from_file_location("robotwin_rocm_episode", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(benchmark_dir))
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(benchmark_dir))
    return module


def test_successful_batch_trajectories_only_serializes_successes() -> None:
    """Retain full trajectories only for slots reported as successful."""
    module = _load_episode_module()
    result = {
        "status": ["Failure", "Success", "Failure"],
        "position": np.arange(18, dtype=np.float32).reshape(3, 3, 2),
        "velocity": np.ones((3, 3, 2), dtype=np.float32),
        "last_tstep": np.array([2, 1, 2]),
    }

    trajectories = module._successful_batch_trajectories(result)

    assert len(trajectories) == 1
    assert trajectories[0]["slot"] == 1
    assert trajectories[0]["last_tstep"] == 1
    assert trajectories[0]["position"] == result["position"][1, :1].tolist()
    assert trajectories[0]["velocity"] == result["velocity"][1, :1].tolist()


def test_successful_batch_trajectories_accepts_all_failure_without_plan() -> None:
    """An all-failure request may omit trajectory arrays."""
    module = _load_episode_module()

    assert module._successful_batch_trajectories(
        {"status": ["Failure", "Failure"]}
    ) == []


def test_task_final_state_captures_blocks_and_grippers() -> None:
    """Record enough final geometry to diagnose a failed stack predicate."""
    module = _load_episode_module()
    pose = SimpleNamespace(
        p=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    actor = SimpleNamespace(get_pose=lambda: pose)
    task = SimpleNamespace(
        block1=actor,
        block2=actor,
        block3=actor,
        is_left_gripper_open=lambda: True,
        is_right_gripper_open=lambda: False,
    )

    state = module._task_final_state(task)

    assert state["actor_poses"]["block3"] == pytest.approx(
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    )
    assert state["left_gripper_open"] is True
    assert state["right_gripper_open"] is False


def test_prepare_imports_allows_cuda_without_rocm_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep the shared episode runner from forcing ROCm on CUDA baselines."""
    module = _load_episode_module()
    robotwin_root = tmp_path / "RoboTwin"
    (robotwin_root / "envs" / "robot").mkdir(parents=True)
    (robotwin_root / "envs" / "robot" / "robot.py").touch()
    monkeypatch.setenv("CUROBO_REQUIRE_ROCM", "1")

    module._prepare_imports(robotwin_root, required_accelerator="cuda")

    assert "CUROBO_REQUIRE_ROCM" not in os.environ
    assert os.environ["ASSETS_PATH"] == str(robotwin_root)
    sys.path.remove(str(robotwin_root))


@pytest.mark.parametrize(
    "result",
    [
        {
            "status": ["Success", "Failure"],
            "position": np.zeros((1, 3, 2), dtype=np.float32),
        },
        {
            "status": ["Success"],
            "position": np.zeros((1, 3, 2), dtype=np.float32),
            "velocity": np.zeros((1, 2, 2), dtype=np.float32),
        },
        {
            "status": ["Success"],
            "position": np.zeros((1, 3, 2), dtype=np.float32),
            "last_tstep": np.array([4]),
        },
    ],
)
def test_successful_batch_trajectories_rejects_inconsistent_shapes(result: dict) -> None:
    """Reject evidence that cannot be mapped back to its batch slot."""
    module = _load_episode_module()

    with pytest.raises(ValueError, match="batch"):
        module._successful_batch_trajectories(result)
