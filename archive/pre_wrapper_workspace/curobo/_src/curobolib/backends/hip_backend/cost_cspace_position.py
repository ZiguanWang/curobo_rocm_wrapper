# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""ROCm implementation of the position joint-space cost."""

from __future__ import annotations

import torch


def position_cspace_cost(
    position: torch.Tensor,
    joint_torque: torch.Tensor,
    target_position: torch.Tensor,
    target_indices: torch.Tensor,
    position_limits: torch.Tensor,
    effort_limits: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    target_weight: torch.Tensor,
    target_dof_weight: torch.Tensor,
    squared_l2_weight: torch.Tensor,
    current_position: torch.Tensor,
    current_velocity: torch.Tensor,
    current_indices: torch.Tensor,
    velocity_limits: torch.Tensor,
    state_dt: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable position cost for ``(batch, horizon, dof)`` tensors."""
    target_indices = target_indices.view(-1).to(dtype=torch.long)
    current_indices = current_indices.view(-1).to(dtype=torch.long)
    batch_size = position.shape[0]
    dof = position.shape[-1]

    target_values = target_position.view(-1, dof)
    current_values = current_position.view(-1, dof)
    current_velocity_values = current_velocity.view(-1, dof)
    dt_values = state_dt.view(-1)
    if target_values.shape[0] == 1:
        target = target_values.expand(batch_size, -1).view(batch_size, 1, dof)
    else:
        target = torch.index_select(target_values, 0, target_indices[:batch_size]).view(
            batch_size, 1, dof
        )
    if current_values.shape[0] == 1:
        current = current_values.expand(batch_size, -1).view(batch_size, 1, dof)
    else:
        current = torch.index_select(current_values, 0, current_indices[:batch_size]).view(
            batch_size, 1, dof
        )
    if current_velocity_values.shape[0] == 1:
        current_vel = current_velocity_values.expand(batch_size, -1).view(batch_size, 1, dof)
    else:
        current_vel = torch.index_select(
            current_velocity_values, 0, current_indices[:batch_size]
        ).view(batch_size, 1, dof)
    if dt_values.shape[0] == 1:
        dt = dt_values.expand(batch_size).view(batch_size, 1, 1)
    else:
        dt = torch.index_select(dt_values, 0, current_indices[:batch_size]).view(batch_size, 1, 1)

    position_range = position_limits[1] - position_limits[0]
    lower_position = position_limits[0] + activation_distance[0] * position_range
    upper_position = position_limits[1] - activation_distance[0] * position_range
    velocity_lower = current + velocity_limits[0].view(1, 1, -1) * dt
    velocity_upper = current + velocity_limits[1].view(1, 1, -1) * dt
    has_dt = dt > 0.0
    lower_position = torch.where(
        has_dt,
        torch.maximum(lower_position.view(1, 1, -1), velocity_lower),
        lower_position.view(1, 1, -1),
    )
    upper_position = torch.where(
        has_dt,
        torch.minimum(upper_position.view(1, 1, -1), velocity_upper),
        upper_position.view(1, 1, -1),
    )
    position_delta = torch.where(
        position < lower_position,
        position - lower_position,
        torch.where(position > upper_position, position - upper_position, 0.0),
    )
    cost = 0.5 * weight[0] * position_delta.square()

    effort_range = effort_limits[1] - effort_limits[0]
    lower_effort = effort_limits[0] + activation_distance[1] * effort_range
    upper_effort = effort_limits[1] - activation_distance[1] * effort_range
    effort_delta = torch.where(
        joint_torque < lower_effort,
        joint_torque - lower_effort,
        torch.where(joint_torque > upper_effort, joint_torque - upper_effort, 0.0),
    )
    cost = cost + 0.5 * weight[1] * effort_delta.square()

    position_error = position - target
    cost = cost + target_weight[0] * target_dof_weight.view(1, 1, -1) * position_error.square()

    safe_dt = torch.where(has_dt, dt, torch.ones_like(dt))
    implied_velocity = (position - current) / safe_dt
    velocity_weight = squared_l2_weight[0] * dt
    acceleration_weight = squared_l2_weight[1] * dt.square()
    implied_acceleration = (implied_velocity - current_vel) / safe_dt
    cost = cost + torch.where(
        has_dt,
        0.5 * velocity_weight * implied_velocity.square()
        + 0.5 * acceleration_weight * implied_acceleration.square(),
        0.0,
    )
    return cost
