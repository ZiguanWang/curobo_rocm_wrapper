# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for equivalent revolute IK seed normalization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from curobo._src.motion.motion_planner import (
    _LOCAL_IK_CHUNK_ORDER,
    MotionPlanner,
    _make_collision_escape_ik_seed_config,
    _make_local_ik_seed_config,
    _normalize_revolute_seed_config,
    _replace_failed_seed_config,
)
from curobo._src.robot.types.joint_types import JointType
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo.types import JointState


def _kinematics(position_limits: torch.Tensor) -> Any:
    config = SimpleNamespace(
        joint_map=torch.tensor([0, 1, 2], dtype=torch.long),
        joint_map_type=torch.tensor(
            [JointType.Z_ROT.value, JointType.X_PRISM.value, JointType.Y_ROT.value],
            dtype=torch.long,
        ),
        num_dof=3,
        joint_limits=SimpleNamespace(position=position_limits),
    )
    return SimpleNamespace(kinematics_config=config)


def test_normalize_revolute_seed_nearest_to_start() -> None:
    """Wrap revolute seeds while leaving a prismatic joint unchanged."""
    start = JointState.from_position(
        torch.tensor([[0.2, 0.4, -0.3]], dtype=torch.float32),
        joint_names=["rot_a", "linear", "rot_b"],
    )
    seed = torch.tensor(
        [[[0.2 + 2.0 * torch.pi, 2.4, -0.3 - 2.0 * torch.pi]]],
        dtype=torch.float32,
    )
    limits = torch.tensor(
        [[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]],
        dtype=torch.float32,
    )

    normalized = _normalize_revolute_seed_config(seed, start, _kinematics(limits))

    torch.testing.assert_close(normalized[..., 0], start.position[..., 0:1])
    torch.testing.assert_close(normalized[..., 1], seed[..., 1])
    torch.testing.assert_close(normalized[..., 2], start.position[..., 2:3])


def test_normalize_revolute_seed_respects_position_limits() -> None:
    """Keep the original equivalent value when the nearest wrap is outside limits."""
    start = JointState.from_position(
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        joint_names=["rot_a", "linear", "rot_b"],
    )
    seed = torch.tensor([[[2.0 * torch.pi, 0.0, 2.0 * torch.pi]]], dtype=torch.float32)
    limits = torch.tensor(
        [[0.5, -1.0, -10.0], [7.0, 1.0, 10.0]],
        dtype=torch.float32,
    )

    normalized = _normalize_revolute_seed_config(seed, start, _kinematics(limits))

    torch.testing.assert_close(normalized[..., 0], seed[..., 0])
    torch.testing.assert_close(normalized[..., 2], torch.zeros_like(normalized[..., 2]))


def test_replace_failed_seed_config_updates_failed_slots() -> None:
    """Replace failed seeds without assigning through a temporary indexed tensor."""
    seeds = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float32,
    )
    success = torch.tensor([[False, True, False, True]])

    result = _replace_failed_seed_config(seeds, success)

    expected = torch.tensor(
        [[[3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [7.0, 8.0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result, expected)


def test_local_ik_seed_config_is_deterministic_and_bounded() -> None:
    """Generate repeatable local seeds without consuming the global RNG."""
    limits = torch.tensor(
        [[-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    current = JointState.from_position(torch.zeros(1, 3))
    state_before = torch.random.get_rng_state()

    first = _make_local_ik_seed_config(current, _kinematics(limits), 16, attempt=3)
    second = _make_local_ik_seed_config(current, _kinematics(limits), 16, attempt=3)
    different_chunk = _make_local_ik_seed_config(current, _kinematics(limits), 16, attempt=4)

    assert first.shape == (1, 16, 3)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different_chunk)
    assert torch.all(first >= limits[0])
    assert torch.all(first <= limits[1])
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_local_ik_chunk_order_spreads_early_search_without_losing_coverage() -> None:
    """Visit distant deterministic chunks early while retaining all sixteen chunks."""
    assert _LOCAL_IK_CHUNK_ORDER[:4] == (0, 8, 4, 12)
    assert sorted(_LOCAL_IK_CHUNK_ORDER) == list(range(16))


def test_collision_escape_seed_config_is_deterministic_and_bounded() -> None:
    """Perturb a failed IK center narrowly without consuming the global RNG."""
    limits = torch.tensor(
        [[-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    center = torch.tensor([[[0.09, -0.19, 0.29]]], dtype=torch.float32)
    state_before = torch.random.get_rng_state()

    first = _make_collision_escape_ik_seed_config(center, _kinematics(limits), 16, attempt=0)
    second = _make_collision_escape_ik_seed_config(center, _kinematics(limits), 16, attempt=0)
    next_chunk = _make_collision_escape_ik_seed_config(center, _kinematics(limits), 16, attempt=1)

    assert first.shape == (1, 16, 3)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, next_chunk)
    assert torch.all(first >= limits[0])
    assert torch.all(first <= limits[1])
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_plan_pose_refines_initially_failed_ik_seeds() -> None:
    """Use the first IK output as a refinement seed before discarding an attempt."""
    planner = MotionPlanner.__new__(MotionPlanner)
    limits = torch.tensor(
        [[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]],
        dtype=torch.float32,
    )
    planner._destroyed = True
    planner.graph_planner = object()
    planner._get_graph_seed_trajectories = lambda *args, **kwargs: None
    first_solution = torch.tensor([[[0.3, 0.4, 0.5]]], dtype=torch.float32)
    refined_solution = torch.tensor([[[0.6, 0.7, 0.8]]], dtype=torch.float32)
    calls: list[torch.Tensor | None] = []

    def solve_ik(*args: Any, seed_config: Any = None, **kwargs: Any) -> Any:
        calls.append(seed_config)
        if seed_config is None:
            return SimpleNamespace(
                success=torch.tensor([[False]]),
                solution=first_solution,
                total_time=1.0,
                solve_time=0.5,
            )
        return SimpleNamespace(
            success=torch.tensor([[True]]),
            solution=refined_solution,
            total_time=2.0,
            solve_time=1.0,
        )

    received_seed: list[torch.Tensor] = []

    def solve_trajopt(*args: Any, seed_config: torch.Tensor, **kwargs: Any) -> Any:
        received_seed.append(seed_config)
        return TrajOptSolverResult(
            success=torch.tensor([[True]]),
            total_time=3.0,
            solve_time=1.5,
            batch_size=1,
            num_seeds=1,
        )

    planner.ik_solver = SimpleNamespace(
        solve_pose=solve_ik,
        kinematics=_kinematics(limits),
        config=SimpleNamespace(num_seeds=1),
    )
    planner.trajopt_solver = SimpleNamespace(
        config=SimpleNamespace(num_seeds=1),
        solve_pose=solve_trajopt,
        kinematics=_kinematics(limits),
    )
    current = JointState.from_position(torch.zeros(1, 3))

    result = planner._plan_pose_single(
        SimpleNamespace(),
        current,
        max_attempts=1,
        enable_graph_attempt=0,
    )

    assert result is not None
    assert len(calls) == 2
    torch.testing.assert_close(calls[1], first_solution)
    torch.testing.assert_close(received_seed[0], refined_solution)
    assert result.total_time == 6.0
    assert result.solve_time == 3.0
