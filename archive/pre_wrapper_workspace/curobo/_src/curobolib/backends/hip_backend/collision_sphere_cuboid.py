# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm sphere-cuboid collision cost implemented with GPU tensor operations."""

from __future__ import annotations

from typing import Tuple

import torch


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by real-first unit quaternions with broadcast dimensions."""
    quaternion_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quaternion_vector, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(quaternion_vector, twice_cross, dim=-1)
    )


def _cuboid_sdf_and_gradient(
    local_center: torch.Tensor, half_extents: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cuboid SDF and Warp-compatible penetration gradient in local space."""
    signed_components = torch.abs(local_center) - half_extents
    outside_components = torch.clamp_min(signed_components, 0.0)
    outside_distance = torch.linalg.vector_norm(outside_components, dim=-1)
    inside_distance = torch.clamp_max(torch.max(signed_components, dim=-1).values, 0.0)
    signed_distance = outside_distance + inside_distance

    coordinate_sign = torch.where(local_center < 0.0, -1.0, 1.0)
    outside_gradient = (
        -coordinate_sign
        * outside_components
        / outside_distance.clamp_min(1.0e-6).unsqueeze(-1)
    )
    nearest_axis = torch.argmax(signed_components, dim=-1)
    inside_gradient = torch.zeros_like(local_center).scatter_(
        -1,
        nearest_axis.unsqueeze(-1),
        -coordinate_sign.gather(-1, nearest_axis.unsqueeze(-1)),
    )
    local_gradient = torch.where(
        (outside_distance > 1.0e-6).unsqueeze(-1), outside_gradient, inside_gradient
    )
    return signed_distance, local_gradient


def _collision_activation(
    penetration: torch.Tensor, eta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the C1 collision activation cost and gradient scale."""
    safe_eta = eta.clamp_min(torch.finfo(penetration.dtype).eps)
    quadratic = (eta > 0.0) & (penetration <= eta)
    cost = torch.where(
        quadratic,
        0.5 * penetration.square() / safe_eta,
        penetration - 0.5 * eta,
    )
    gradient_scale = torch.where(quadratic, penetration / safe_eta, 1.0)
    return cost, gradient_scale


def _compute_cost_and_gradient(
    query_spheres: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_inv_pose: torch.Tensor,
    cuboid_enable: torch.Tensor,
    cuboid_count: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    env_query_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Warp-compatible cost and sphere gradient for selected environments."""
    batch_size, horizon, num_spheres, _ = query_spheres.shape
    max_cuboids = cuboid_dims.shape[1]
    env_indices = env_query_idx.to(dtype=torch.long)

    dims = torch.index_select(cuboid_dims, 0, env_indices)
    inv_pose = torch.index_select(cuboid_inv_pose, 0, env_indices)
    enabled = torch.index_select(cuboid_enable, 0, env_indices).to(dtype=torch.bool)
    count = torch.index_select(cuboid_count, 0, env_indices)
    obstacle_indices = torch.arange(max_cuboids, device=query_spheres.device)
    enabled = enabled & (obstacle_indices.view(1, -1) < count.view(-1, 1))

    center = query_spheres[..., :3].unsqueeze(-2)
    radius = query_spheres[..., 3].detach().unsqueeze(-1)
    inv_position = inv_pose[:, None, None, :, :3]
    inv_quaternion = inv_pose[:, None, None, :, 3:7]
    local_center = _quaternion_rotate(inv_quaternion, center) + inv_position

    half_extents = 0.5 * dims[:, None, None, :, :3]
    signed_distance, local_gradient = _cuboid_sdf_and_gradient(local_center, half_extents)

    forward_quaternion = torch.cat(
        (inv_quaternion[..., :1], -inv_quaternion[..., 1:]), dim=-1
    )
    world_gradient = _quaternion_rotate(forward_quaternion, local_gradient)

    eta = activation_distance.reshape(-1)[0]
    penetration = -signed_distance + radius + eta
    active = (
        (penetration > 0.0)
        & enabled[:, None, None, :]
        & (radius >= 0.0)
    )
    activation_cost, gradient_scale = _collision_activation(penetration, eta)
    activation_cost = torch.where(active, activation_cost, 0.0)
    gradient_scale = torch.where(active, gradient_scale, 0.0)

    scalar_weight = weight.reshape(-1)[0]
    cost = scalar_weight * activation_cost.sum(dim=-1)
    xyz_gradient = scalar_weight * (gradient_scale.unsqueeze(-1) * world_gradient).sum(dim=-2)
    sphere_gradient = torch.cat((xyz_gradient, torch.zeros_like(radius)), dim=-1)
    return cost.reshape(batch_size, horizon, num_spheres), sphere_gradient


def _compute_swept_cost_and_gradient(
    query_spheres: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_inv_pose: torch.Tensor,
    cuboid_enable: torch.Tensor,
    cuboid_count: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    speed_dt: torch.Tensor,
    enable_speed_metric: bool,
    env_query_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the three-step swept sphere cost used by the existing Warp kernel."""
    batch_size, horizon, num_spheres, _ = query_spheres.shape
    max_cuboids = cuboid_dims.shape[1]
    env_indices = env_query_idx.to(dtype=torch.long)
    dims = torch.index_select(cuboid_dims, 0, env_indices)
    inv_pose = torch.index_select(cuboid_inv_pose, 0, env_indices)
    enabled = torch.index_select(cuboid_enable, 0, env_indices).to(dtype=torch.bool)
    count = torch.index_select(cuboid_count, 0, env_indices)
    obstacle_indices = torch.arange(max_cuboids, device=query_spheres.device)
    enabled = enabled & (obstacle_indices.view(1, -1) < count.view(-1, 1))

    center = query_spheres[..., :3]
    radius = query_spheres[..., 3].detach().unsqueeze(-1)
    inv_position = inv_pose[:, None, None, :, :3]
    inv_quaternion = inv_pose[:, None, None, :, 3:7]
    local_current = _quaternion_rotate(
        inv_quaternion, center.unsqueeze(-2)
    ) + inv_position
    half_extents = 0.5 * dims[:, None, None, :, :3]
    eta = activation_distance.reshape(-1)[0]
    radius_adjusted = radius + eta
    base_active = enabled[:, None, None, :] & (radius >= 0.0)

    cost_sum = torch.zeros_like(local_current[..., 0])
    gradient_sum = torch.zeros_like(local_current)

    def accumulate(local_point: torch.Tensor, active_step: torch.Tensor) -> torch.Tensor:
        """Accumulate one adaptive sweep sample and return its penetration."""
        nonlocal cost_sum, gradient_sum
        signed_distance, local_gradient = _cuboid_sdf_and_gradient(
            local_point, half_extents
        )
        penetration = -signed_distance + radius_adjusted
        active = base_active & active_step & (penetration > 0.0)
        activation_cost, gradient_scale = _collision_activation(penetration, eta)
        cost_sum = cost_sum + torch.where(active, activation_cost, 0.0)
        gradient_sum = gradient_sum + torch.where(
            active.unsqueeze(-1), gradient_scale.unsqueeze(-1) * local_gradient, 0.0
        )
        return penetration

    accumulate(local_current, torch.ones_like(base_active))

    for direction in (-1, 1):
        neighbor_center = torch.roll(center, shifts=-direction, dims=1)
        local_neighbor = _quaternion_rotate(
            inv_quaternion, neighbor_center.unsqueeze(-2)
        ) + inv_position
        horizon_indices = torch.arange(horizon, device=query_spheres.device)
        valid_horizon = (
            horizon_indices > 0 if direction == -1 else horizon_indices < horizon - 1
        )
        valid_horizon = valid_horizon.view(1, horizon, 1, 1)
        half_distance = 0.5 * torch.linalg.vector_norm(
            local_neighbor - local_current, dim=-1
        )
        inverse_half_distance = half_distance.clamp_min(0.001).reciprocal()
        jump = torch.zeros_like(half_distance)
        for _ in range(3):
            active_step = valid_horizon & (jump < half_distance)
            interpolation = 1.0 - 0.5 * jump * inverse_half_distance
            local_point = (
                interpolation.unsqueeze(-1) * local_current
                + (1.0 - interpolation).unsqueeze(-1) * local_neighbor
            )
            penetration = accumulate(local_point, active_step)
            free_step = torch.maximum(-penetration, radius_adjusted)
            increment = torch.where(penetration > 0.0, penetration, free_step)
            jump = jump + torch.where(active_step, increment, 0.0)

    forward_quaternion = torch.cat(
        (inv_quaternion[..., :1], -inv_quaternion[..., 1:]), dim=-1
    )
    world_gradient = _quaternion_rotate(forward_quaternion, gradient_sum)
    scalar_weight = weight.reshape(-1)[0]
    cost = scalar_weight * cost_sum.sum(dim=-1)
    xyz_gradient = scalar_weight * world_gradient.sum(dim=-2)

    if enable_speed_metric and horizon > 2:
        dt = speed_dt.reshape(-1)[0].clamp_min(1.0e-6)
        velocity = 0.5 * (center[:, 2:] - center[:, :-2]) / dt
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        acceleration = (center[:, :-2] + center[:, 2:] - 2.0 * center[:, 1:-1]) / dt.square()
        safe_speed = speed.clamp_min(1.0e-3)
        normalized_velocity = velocity / safe_speed.unsqueeze(-1)
        curvature = acceleration / safe_speed.square().unsqueeze(-1)
        interior_gradient = xyz_gradient[:, 1:-1]
        orthogonal_gradient = interior_gradient - (
            normalized_velocity * interior_gradient
        ).sum(dim=-1, keepdim=True) * normalized_velocity
        orthogonal_curvature = curvature - (
            normalized_velocity * curvature
        ).sum(dim=-1, keepdim=True) * normalized_velocity
        apply_metric = (speed >= 1.0e-3) & (cost[:, 1:-1] > 0.0)
        metric_gradient = speed.unsqueeze(-1) * (
            orthogonal_gradient - cost[:, 1:-1].unsqueeze(-1) * orthogonal_curvature
        )
        xyz_gradient = xyz_gradient.clone()
        cost = cost.clone()
        xyz_gradient[:, 1:-1] = torch.where(
            apply_metric.unsqueeze(-1), metric_gradient, interior_gradient
        )
        cost[:, 1:-1] = torch.where(apply_metric, speed * cost[:, 1:-1], cost[:, 1:-1])

    sphere_gradient = torch.cat((xyz_gradient, torch.zeros_like(radius)), dim=-1)
    return cost.reshape(batch_size, horizon, num_spheres), sphere_gradient


class _SphereCuboidCollision(torch.autograd.Function):
    """Preserve the existing collision wrapper's precomputed-gradient contract."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query_spheres: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_inv_pose: torch.Tensor,
        cuboid_enable: torch.Tensor,
        cuboid_count: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: torch.Tensor,
        return_loss: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return collision cost and save its analytical sphere gradient."""
        cost, gradient = _compute_cost_and_gradient(
            query_spheres,
            cuboid_dims,
            cuboid_inv_pose,
            cuboid_enable,
            cuboid_count,
            weight,
            activation_distance,
            env_query_idx,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(gradient)
        ctx.mark_non_differentiable(gradient)
        return cost, gradient

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        _grad_gradient: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, ...]:
        """Return the saved sphere gradient using the existing loss-scaling rule."""
        sphere_gradient = None
        if ctx.needs_input_grad[0]:
            (saved_gradient,) = ctx.saved_tensors
            sphere_gradient = saved_gradient
            if ctx.return_loss:
                sphere_gradient = saved_gradient * grad_output.unsqueeze(-1)
        return sphere_gradient, None, None, None, None, None, None, None, None


class _SweptSphereCuboidCollision(torch.autograd.Function):
    """Preserve the swept collision wrapper's precomputed-gradient contract."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query_spheres: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_inv_pose: torch.Tensor,
        cuboid_enable: torch.Tensor,
        cuboid_count: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        enable_speed_metric: bool,
        env_query_idx: torch.Tensor,
        return_loss: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return swept collision cost and save its analytical sphere gradient."""
        cost, gradient = _compute_swept_cost_and_gradient(
            query_spheres,
            cuboid_dims,
            cuboid_inv_pose,
            cuboid_enable,
            cuboid_count,
            weight,
            activation_distance,
            speed_dt,
            enable_speed_metric,
            env_query_idx,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(gradient)
        ctx.mark_non_differentiable(gradient)
        return cost, gradient

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        _grad_gradient: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, ...]:
        """Return the saved sphere gradient using the existing loss-scaling rule."""
        sphere_gradient = None
        if ctx.needs_input_grad[0]:
            (saved_gradient,) = ctx.saved_tensors
            sphere_gradient = saved_gradient
            if ctx.return_loss:
                sphere_gradient = saved_gradient * grad_output.unsqueeze(-1)
        return (
            sphere_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def sphere_cuboid_collision(
    query_spheres: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_inv_pose: torch.Tensor,
    cuboid_enable: torch.Tensor,
    cuboid_count: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    env_query_idx: torch.Tensor,
    return_loss: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return collision cost and its precomputed gradient for an AMD GPU query."""
    cost, gradient = _SphereCuboidCollision.apply(
        query_spheres,
        cuboid_dims,
        cuboid_inv_pose,
        cuboid_enable,
        cuboid_count,
        weight,
        activation_distance,
        env_query_idx,
        return_loss,
    )
    return cost, gradient


def swept_sphere_cuboid_collision(
    query_spheres: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_inv_pose: torch.Tensor,
    cuboid_enable: torch.Tensor,
    cuboid_count: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    speed_dt: torch.Tensor,
    enable_speed_metric: bool,
    env_query_idx: torch.Tensor,
    return_loss: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return swept collision cost and its precomputed gradient on an AMD GPU."""
    return _SweptSphereCuboidCollision.apply(
        query_spheres,
        cuboid_dims,
        cuboid_inv_pose,
        cuboid_enable,
        cuboid_count,
        weight,
        activation_distance,
        speed_dt,
        enable_speed_metric,
        env_query_idx,
        return_loss,
    )
