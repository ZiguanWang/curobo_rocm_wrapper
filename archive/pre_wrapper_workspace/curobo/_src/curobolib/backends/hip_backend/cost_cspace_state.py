# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm implementation of the full joint-state C-space cost."""

from __future__ import annotations

import torch

from curobo._src.util.logging import log_and_raise


def _bound_cost(
    value: torch.Tensor,
    limits: torch.Tensor,
    activation_distance: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute a squared bound penalty after shrinking the valid interval."""
    full_range = limits[1] - limits[0]
    lower = limits[0] + activation_distance * full_range
    upper = limits[1] - activation_distance * full_range
    delta = torch.where(
        value < lower,
        value - lower,
        torch.where(value > upper, value - upper, torch.zeros_like(value)),
    )
    return 0.5 * weight * delta.square()


def _select_targets(
    target_position: torch.Tensor,
    target_indices: torch.Tensor,
    batch_size: int,
    dof: int,
) -> torch.Tensor:
    """Select or broadcast one target joint vector per optimization problem."""
    target_values = target_position.view(-1, dof)
    if target_values.shape[0] == 1:
        return target_values.expand(batch_size, -1).view(batch_size, 1, dof)
    return torch.index_select(
        target_values,
        0,
        target_indices.view(-1)[:batch_size].to(dtype=torch.long),
    ).view(batch_size, 1, dof)


def cspace_state_cost(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    joint_torque: torch.Tensor,
    state_dt: torch.Tensor,
    target_position: torch.Tensor,
    target_indices: torch.Tensor,
    position_limits: torch.Tensor,
    velocity_limits: torch.Tensor,
    acceleration_limits: torch.Tensor,
    jerk_limits: torch.Tensor,
    effort_limits: torch.Tensor,
    bound_weight: torch.Tensor,
    activation_distance: torch.Tensor,
    regularization_weight: torch.Tensor,
    target_weight: torch.Tensor,
    non_terminal_target_factor: torch.Tensor,
    target_dof_weight: torch.Tensor,
    retime_weights: bool,
    retime_regularization_weights: bool,
) -> torch.Tensor:
    """Compute the differentiable state C-space cost on an AMD GPU.

    All state tensors have shape ``(B, H, D)``. The returned tensor preserves
    that shape so existing horizon and degree-of-freedom reductions remain unchanged.
    """
    if position.ndim != 3:
        log_and_raise("position must have shape (B, H, D)")
    batch_size, horizon, dof = position.shape
    for name, value in (
        ("velocity", velocity),
        ("acceleration", acceleration),
        ("jerk", jerk),
        ("joint_torque", joint_torque),
    ):
        if value.shape != position.shape:
            log_and_raise(f"{name} must have shape {position.shape}")
    if state_dt.numel() != batch_size:
        log_and_raise(f"state_dt must contain {batch_size} values")

    dt = state_dt.view(batch_size, 1, 1)
    bound_scale = torch.ones((batch_size, 1, 5), dtype=position.dtype, device=position.device)
    if retime_weights:
        bound_scale = torch.cat(
            (
                torch.ones_like(dt),
                dt,
                dt.square(),
                dt.pow(3.0),
                torch.ones_like(dt),
            ),
            dim=-1,
        )
    scaled_bound_weight = bound_weight.view(1, 1, 5) * bound_scale

    regularization_scale = torch.ones_like(bound_scale)
    if retime_regularization_weights:
        regularization_scale = torch.cat(
            (dt, dt.square(), dt.pow(3.0), torch.ones_like(dt), dt), dim=-1
        )
    scaled_regularization_weight = regularization_weight.view(1, 1, 5) * regularization_scale

    cost = _bound_cost(
        position,
        position_limits,
        activation_distance[0],
        scaled_bound_weight[..., 0:1],
    )
    cost = cost + _bound_cost(
        velocity,
        velocity_limits,
        activation_distance[1],
        scaled_bound_weight[..., 1:2],
    )
    cost = cost + _bound_cost(
        acceleration,
        acceleration_limits,
        activation_distance[2],
        scaled_bound_weight[..., 2:3],
    )
    cost = cost + _bound_cost(
        jerk,
        jerk_limits,
        activation_distance[3],
        scaled_bound_weight[..., 3:4],
    )
    cost = cost + _bound_cost(
        joint_torque,
        effort_limits,
        activation_distance[4],
        scaled_bound_weight[..., 4:5],
    )

    selected_target = _select_targets(target_position, target_indices, batch_size, dof)
    horizon_target_scale = torch.full(
        (1, horizon, 1),
        non_terminal_target_factor[0],
        dtype=position.dtype,
        device=position.device,
    )
    horizon_target_scale[:, -1] = 1.0
    position_error = position - selected_target
    cost = cost + (
        target_weight[0]
        * horizon_target_scale
        * target_dof_weight.view(1, 1, dof)
        * position_error.square()
    )

    cost = cost + 0.5 * scaled_regularization_weight[..., 0:1] * velocity.square()
    cost = cost + 0.5 * scaled_regularization_weight[..., 1:2] * acceleration.square()
    cost = cost + 0.5 * scaled_regularization_weight[..., 2:3] * jerk.square()
    cost = cost + 0.5 * scaled_regularization_weight[..., 3:4] * joint_torque.square()
    energy = joint_torque * velocity * dt
    cost = cost + scaled_regularization_weight[..., 4:5] * energy.square()
    return cost
