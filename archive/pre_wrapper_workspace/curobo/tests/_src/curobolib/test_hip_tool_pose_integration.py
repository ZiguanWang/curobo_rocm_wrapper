# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the ROCm tool-pose cost implementation."""

import math

import pytest
import torch

from curobo._src.curobolib.backends.hip_backend.cost_tool_pose import tool_pose_cost

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required",
)


def _run_cost(
    current_position: torch.Tensor,
    goal_position: torch.Tensor,
    *,
    current_quaternion: torch.Tensor | None = None,
    goal_quaternion: torch.Tensor | None = None,
    axes_weight: torch.Tensor | None = None,
    project: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a one-link tool-pose cost with concise test defaults."""
    device = current_position.device
    batch_size, horizon, _, _ = current_position.shape
    num_goalset = goal_position.shape[-2]
    if current_quaternion is None:
        current_quaternion = torch.zeros(
            (batch_size, horizon, 1, 4), dtype=torch.float32, device=device
        )
        current_quaternion[..., 0] = 1.0
    if goal_quaternion is None:
        goal_quaternion = torch.zeros((1, 1, num_goalset, 4), dtype=torch.float32, device=device)
        goal_quaternion[..., 0] = 1.0
    if axes_weight is None:
        axes_weight = torch.ones((1, 6), dtype=torch.float32, device=device)
    tolerance = torch.zeros((1, 2), dtype=torch.float32, device=device)
    return tool_pose_cost(
        current_position,
        current_quaternion,
        goal_position,
        goal_quaternion,
        torch.zeros((batch_size, 1), dtype=torch.int32, device=device),
        torch.ones(2, dtype=torch.float32, device=device),
        axes_weight,
        axes_weight,
        tolerance,
        tolerance,
        torch.tensor([[project]], dtype=torch.uint8, device=device),
        rotation_method=0,
    )


def test_tool_pose_selects_nearest_goalset() -> None:
    """Select the lowest combined position and rotation cost for each sample."""
    device = torch.device("cuda")
    current_position = torch.tensor(
        [[[[0.9, 0.0, 0.0]]], [[[0.1, 0.0, 0.0]]]],
        dtype=torch.float32,
        device=device,
    )
    goal_position = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]],
        dtype=torch.float32,
        device=device,
    )

    cost, linear_distance, angular_distance, goalset_idx = _run_cost(
        current_position, goal_position
    )

    assert torch.equal(goalset_idx[:, 0, 0], torch.tensor([1, 0], device=device))
    assert torch.allclose(linear_distance[:, 0, 0], torch.tensor([0.1, 0.1], device=device))
    assert torch.count_nonzero(angular_distance) == 0
    assert torch.allclose(cost[:, 0, 0], torch.tensor([0.005, 0.005], device=device))


def test_tool_pose_projects_position_axes_into_goal_frame() -> None:
    """Apply per-axis weights after transforming position error into the goal frame."""
    device = torch.device("cuda")
    current_position = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32, device=device)
    goal_position = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
    half_angle = 0.25 * math.pi
    goal_quaternion = torch.tensor(
        [[[[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]]]],
        dtype=torch.float32,
        device=device,
    )
    axes_weight = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
    )

    world_cost = _run_cost(
        current_position,
        goal_position,
        goal_quaternion=goal_quaternion,
        axes_weight=axes_weight,
        project=False,
    )[0]
    projected_cost = _run_cost(
        current_position,
        goal_position,
        goal_quaternion=goal_quaternion,
        axes_weight=axes_weight,
        project=True,
    )[0]

    assert torch.allclose(world_cost[..., 0], torch.tensor([[0.5]], device=device))
    assert torch.allclose(projected_cost[..., 0], torch.zeros((1, 1), device=device), atol=1e-6)


def test_tool_pose_position_gradient_matches_float32_finite_difference() -> None:
    """Match the differentiable ROCm position cost against a float32 central difference."""
    device = torch.device("cuda")
    current_position = torch.tensor(
        [[[[0.3, -0.2, 0.1]]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    goal_position = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)
    axes_weight = torch.tensor(
        [[1.0, 2.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
    )
    loss = _run_cost(current_position, goal_position, axes_weight=axes_weight)[0].sum()
    loss.backward()
    analytical = current_position.grad.detach().clone()

    epsilon = 1.0e-3
    numerical = torch.zeros_like(current_position)
    with torch.no_grad():
        for axis in range(3):
            positive = current_position.detach().clone()
            negative = current_position.detach().clone()
            positive[..., axis] += epsilon
            negative[..., axis] -= epsilon
            positive_cost = _run_cost(positive, goal_position, axes_weight=axes_weight)[0].sum()
            negative_cost = _run_cost(negative, goal_position, axes_weight=axes_weight)[0].sum()
            numerical[..., axis] = (positive_cost - negative_cost) / (2.0 * epsilon)

    assert torch.allclose(analytical, numerical, atol=2e-4, rtol=2e-3)
