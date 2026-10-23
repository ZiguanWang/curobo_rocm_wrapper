# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm integration tests for HIP trajectory kernels."""

import importlib
from types import ModuleType

import pytest
import torch


def _hip_trajectory() -> ModuleType:
    """Load the HIP trajectory extension when running on a ROCm GPU host."""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU is not available")
    try:
        return importlib.import_module(
            "curobo._src.curobolib.backends.hip_backend.trajectory"
        )
    except ModuleNotFoundError:
        pytest.skip("HIP trajectory extension is not built")


def test_hip_zero_acceleration_preserves_stationary_state() -> None:
    """Acceleration integration preserves a stationary initial state."""
    trajectory = _hip_trajectory()
    batch_size = 2
    horizon = 8
    dof = 3
    shape = (batch_size, horizon, dof)
    output_position = torch.zeros(shape, device="cuda")
    output_velocity = torch.zeros_like(output_position)
    output_acceleration = torch.zeros_like(output_position)
    output_jerk = torch.zeros_like(output_position)
    acceleration = torch.zeros_like(output_position)
    start_position = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.float32)
    start_velocity = torch.zeros_like(start_position)
    start_acceleration = torch.zeros_like(start_position)
    start_index = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    trajectory_dt = torch.full((horizon,), 0.1, device="cuda")

    trajectory.launch_integration_acceleration_kernel(
        output_position,
        output_velocity,
        output_acceleration,
        output_jerk,
        acceleration,
        start_position,
        start_velocity,
        start_acceleration,
        start_index,
        trajectory_dt,
        batch_size,
        horizon,
        dof,
        True,
    )
    torch.cuda.synchronize()

    expected_position = start_position[0].view(1, 1, -1).expand_as(output_position)
    torch.testing.assert_close(output_position, expected_position)
    assert torch.count_nonzero(output_velocity) == 0
    assert torch.count_nonzero(output_acceleration) == 0
    assert torch.count_nonzero(output_jerk) == 0


def _run_bspline_position_forward(
    trajectory: ModuleType,
    knots: torch.Tensor,
    start_position: torch.Tensor,
    goal_position: torch.Tensor,
    trajectory_dt: torch.Tensor,
    state_index: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    padded_horizon: int,
    degree: int,
) -> torch.Tensor:
    """Run B-spline interpolation and return positions for the supplied knots."""
    batch_size, n_knots, dof = knots.shape
    shape = (batch_size, padded_horizon, dof)
    output_position = torch.empty(shape, device="cuda")
    output_velocity = torch.empty_like(output_position)
    output_acceleration = torch.empty_like(output_position)
    output_jerk = torch.empty_like(output_position)
    output_dt = torch.empty((batch_size, padded_horizon), device="cuda")
    zero_boundary = torch.zeros_like(start_position)
    trajectory.launch_bspline_interpolation_forward_kernel(
        output_position,
        output_velocity,
        output_acceleration,
        output_jerk,
        output_dt,
        knots,
        start_position,
        zero_boundary,
        zero_boundary,
        zero_boundary,
        goal_position,
        zero_boundary,
        zero_boundary,
        zero_boundary,
        state_index,
        state_index,
        trajectory_dt,
        use_implicit_goal_state,
        batch_size,
        padded_horizon,
        dof,
        n_knots,
        degree,
    )
    return output_position


def test_hip_bspline_backward_matches_finite_difference() -> None:
    """Cubic B-spline position gradients match a float32 finite difference reference."""
    trajectory = _hip_trajectory()
    batch_size = 1
    degree = 3
    n_knots = 8
    dof = 2
    interpolation_steps = 2
    horizon = (n_knots + degree + 1) * interpolation_steps
    padded_horizon = horizon + 1
    knots = torch.linspace(-0.4, 0.9, n_knots * dof, device="cuda").view(
        batch_size, n_knots, dof
    )
    start_position = torch.tensor([[0.2, -0.3]], device="cuda")
    goal_position = torch.tensor([[0.8, 0.4]], device="cuda")
    trajectory_dt = torch.tensor([0.1], device="cuda")
    state_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    use_implicit_goal_state = torch.zeros(batch_size, dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(4)
    position_gradient = torch.randn(
        (batch_size, padded_horizon, dof), device="cuda", generator=generator
    )

    output_gradient = torch.empty_like(knots)
    zero_gradient = torch.zeros_like(position_gradient)
    trajectory.launch_bspline_interpolation_backward_kernel(
        output_gradient,
        position_gradient,
        zero_gradient,
        zero_gradient,
        zero_gradient,
        trajectory_dt,
        state_index,
        use_implicit_goal_state,
        batch_size,
        padded_horizon,
        dof,
        n_knots,
        degree,
        False,
    )

    epsilon = 1e-3
    numerical_gradient = torch.empty_like(knots)
    for knot_index in range(n_knots):
        for dof_index in range(dof):
            positive_knots = knots.clone()
            negative_knots = knots.clone()
            positive_knots[0, knot_index, dof_index] += epsilon
            negative_knots[0, knot_index, dof_index] -= epsilon
            positive_position = _run_bspline_position_forward(
                trajectory,
                positive_knots,
                start_position,
                goal_position,
                trajectory_dt,
                state_index,
                use_implicit_goal_state,
                padded_horizon,
                degree,
            )
            negative_position = _run_bspline_position_forward(
                trajectory,
                negative_knots,
                start_position,
                goal_position,
                trajectory_dt,
                state_index,
                use_implicit_goal_state,
                padded_horizon,
                degree,
            )
            numerical_gradient[0, knot_index, dof_index] = (
                ((positive_position - negative_position) * position_gradient).sum()
                / (2.0 * epsilon)
            )

    torch.cuda.synchronize()
    assert torch.isfinite(output_gradient).all()
    torch.testing.assert_close(output_gradient, numerical_gradient, atol=4e-4, rtol=3e-3)
