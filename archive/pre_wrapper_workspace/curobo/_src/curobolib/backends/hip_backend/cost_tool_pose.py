# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm implementation of the tool-pose goal-set cost."""

from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.util.logging import log_and_raise


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply real-first quaternions with broadcasting over leading dimensions."""
    left_w, left_x, left_y, left_z = torch.unbind(left, dim=-1)
    right_w, right_x, right_y, right_z = torch.unbind(right, dim=-1)
    return torch.stack(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ),
        dim=-1,
    )


def _quaternion_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """Invert real-first quaternions without assuming exact unit norm."""
    conjugate = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    return conjugate / quaternion.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-15)


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by real-first quaternions."""
    quaternion_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quaternion_vector, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(quaternion_vector, twice_cross, dim=-1)
    )


def _select_horizon_criteria(
    terminal: torch.Tensor,
    non_terminal: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Expand link criteria across a horizon, using terminal values at the final step."""
    selected = terminal.unsqueeze(0).expand(horizon, -1, -1).clone()
    if horizon > 1:
        selected[:-1] = non_terminal
    return selected


def _rotation_cost(
    current_quaternion: torch.Tensor,
    goal_quaternion: torch.Tensor,
    axes_weight: torch.Tensor,
    rotation_weight: torch.Tensor,
    tolerance_squared: torch.Tensor,
    rotation_method: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return rotation cost and reported angular distance for all goal candidates."""
    quaternion_delta = _quaternion_multiply(
        current_quaternion, _quaternion_inverse(goal_quaternion)
    )
    scalar = quaternion_delta[..., :1]
    vector = quaternion_delta[..., 1:]

    if rotation_method == 0:
        weighted_vector = axes_weight * vector
        vector_norm = torch.linalg.vector_norm(weighted_vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vector_norm, torch.abs(scalar))
        angle = torch.where(rotation_weight > 0.0, angle, torch.zeros_like(angle))
        axis = torch.where(
            vector_norm > 1.0e-15,
            weighted_vector / vector_norm.clamp_min(1.0e-15),
            torch.zeros_like(weighted_vector),
        )
        tangent = angle * axis
        angular_distance = rotation_weight * tangent.square().sum(dim=-1, keepdim=True)
        reported_angle = angle
    elif rotation_method in (1, 2):
        hemisphere_sign = torch.where(
            scalar < 0.0, -torch.ones_like(scalar), torch.ones_like(scalar)
        )
        scalar = scalar * hemisphere_sign
        vector = vector * hemisphere_sign
        vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        half_angle = torch.atan2(vector_norm, torch.abs(scalar))
        half_angle = torch.where(rotation_weight > 0.0, half_angle, torch.zeros_like(half_angle))
        geometric_angle = 2.0 * half_angle
        standard_factor = geometric_angle / (2.0 * torch.sin(half_angle)).clamp_min(1.0e-15)
        near_identity = 2.0 * vector
        tangent = torch.where(vector_norm < 1.0e-10, near_identity, standard_factor * vector)
        weighted_tangent = axes_weight * tangent
        reported_angle = torch.linalg.vector_norm(weighted_tangent, dim=-1, keepdim=True)
        angular_distance = rotation_weight * weighted_tangent.square().sum(dim=-1, keepdim=True)
    else:
        log_and_raise(f"Unsupported ROCm tool-pose rotation method: {rotation_method}")

    angular_distance = torch.where(
        angular_distance < tolerance_squared,
        torch.zeros_like(angular_distance),
        angular_distance,
    )
    return angular_distance.squeeze(-1), reported_angle.squeeze(-1)


def tool_pose_cost(
    current_position: torch.Tensor,
    current_quaternion: torch.Tensor,
    goal_position: torch.Tensor,
    goal_quaternion: torch.Tensor,
    idxs_goal: torch.Tensor,
    position_orientation_weight: torch.Tensor,
    terminal_axes_weight: torch.Tensor,
    non_terminal_axes_weight: torch.Tensor,
    terminal_tolerance: torch.Tensor,
    non_terminal_tolerance: torch.Tensor,
    project_distance_to_goal: torch.Tensor,
    rotation_method: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the tool-pose cost on ROCm using differentiable PyTorch GPU operations.

    Args:
        current_position: Current positions with shape ``(B, H, L, 3)``.
        current_quaternion: Current real-first quaternions with shape ``(B, H, L, 4)``.
        goal_position: Goal positions with shape ``(N, L, G, 3)``.
        goal_quaternion: Goal real-first quaternions with shape ``(N, L, G, 4)``.
        idxs_goal: Goal-batch indices with shape ``(B, 1)``.
        position_orientation_weight: Position and rotation weights with shape ``(2,)``.
        terminal_axes_weight: Terminal per-link axis weights with shape ``(L, 6)``.
        non_terminal_axes_weight: Non-terminal per-link axis weights with shape ``(L, 6)``.
        terminal_tolerance: Terminal position and rotation tolerances with shape ``(L, 2)``.
        non_terminal_tolerance: Non-terminal tolerances with shape ``(L, 2)``.
        project_distance_to_goal: Per-link goal-frame projection flags with shape ``(L, 1)``.
        rotation_method: Rotation error method, matching the Warp implementation.

    Returns:
        Interleaved cost ``(B, H, 2L)``, linear distance ``(B, H, L)``, angular
        distance ``(B, H, L)``, and selected goal-set index ``(B, H, L)``.
    """
    batch_size, horizon, num_links, _ = current_position.shape
    if current_quaternion.shape != (batch_size, horizon, num_links, 4):
        log_and_raise("current_quaternion must have shape (B, H, L, 4)")
    if goal_position.ndim != 4 or goal_position.shape[1] != num_links:
        log_and_raise("goal_position must have shape (N, L, G, 3)")
    if goal_quaternion.shape != (*goal_position.shape[:-1], 4):
        log_and_raise("goal_quaternion must have shape (N, L, G, 4)")
    if idxs_goal.shape != (batch_size, 1):
        log_and_raise(f"idxs_goal must have shape ({batch_size}, 1)")

    goal_batch_indices = idxs_goal[:, 0].to(dtype=torch.long)
    selected_goal_position = goal_position.index_select(0, goal_batch_indices).unsqueeze(1)
    selected_goal_quaternion = goal_quaternion.index_select(0, goal_batch_indices).unsqueeze(1)
    expanded_current_position = current_position.unsqueeze(-2)
    expanded_current_quaternion = current_quaternion.unsqueeze(-2)

    axes_weight = (
        _select_horizon_criteria(terminal_axes_weight, non_terminal_axes_weight, horizon)
        .unsqueeze(0)
        .unsqueeze(-2)
    )
    tolerance_squared = (
        _select_horizon_criteria(terminal_tolerance, non_terminal_tolerance, horizon)
        .square()
        .unsqueeze(0)
        .unsqueeze(-2)
    )
    position_axes_weight = axes_weight[..., :3]
    rotation_axes_weight = axes_weight[..., 3:]

    projection_mask = project_distance_to_goal.to(dtype=torch.bool).view(1, 1, num_links, 1, 1)
    inverse_goal_quaternion = _quaternion_inverse(selected_goal_quaternion)
    projected_position = _quaternion_rotate(
        inverse_goal_quaternion, expanded_current_position - selected_goal_position
    )
    position_delta = torch.where(
        projection_mask,
        projected_position,
        expanded_current_position - selected_goal_position,
    )

    projected_current_quaternion = _quaternion_multiply(
        inverse_goal_quaternion, expanded_current_quaternion
    )
    rotation_current = torch.where(
        projection_mask, projected_current_quaternion, expanded_current_quaternion
    )
    rotation_goal = torch.where(
        projection_mask, torch.zeros_like(selected_goal_quaternion), selected_goal_quaternion
    )
    rotation_goal = torch.cat(
        (
            torch.where(
                projection_mask, torch.ones_like(rotation_goal[..., :1]), rotation_goal[..., :1]
            ),
            rotation_goal[..., 1:],
        ),
        dim=-1,
    )

    position_weight = position_orientation_weight[0]
    rotation_weight = position_orientation_weight[1]
    weighted_position_delta = position_axes_weight * position_delta
    position_cost = 0.5 * position_weight * weighted_position_delta.square().sum(dim=-1)
    position_cost = torch.where(
        position_cost < tolerance_squared[..., 0],
        torch.zeros_like(position_cost),
        position_cost,
    )
    rotation_cost, angular_distance_candidates = _rotation_cost(
        rotation_current,
        rotation_goal,
        rotation_axes_weight,
        rotation_weight,
        tolerance_squared[..., 1:2],
        rotation_method,
    )

    goalset_idx = torch.argmin(position_cost + rotation_cost, dim=-1)
    gather_idx = goalset_idx.unsqueeze(-1)
    selected_position_cost = torch.gather(position_cost, -1, gather_idx).squeeze(-1)
    selected_rotation_cost = torch.gather(rotation_cost, -1, gather_idx).squeeze(-1)
    angular_distance = torch.gather(angular_distance_candidates, -1, gather_idx).squeeze(-1)
    linear_distance = torch.where(
        position_weight > 0.0,
        torch.sqrt(
            (2.0 * selected_position_cost / position_weight.clamp_min(1.0e-15)).clamp_min(0.0)
        ),
        torch.zeros_like(selected_position_cost),
    )
    cost = torch.stack((selected_position_cost, selected_rotation_cost), dim=-1).reshape(
        batch_size, horizon, 2 * num_links
    )
    return cost, linear_distance, angular_distance, goalset_idx.to(dtype=torch.int32)
