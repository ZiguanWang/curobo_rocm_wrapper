# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for ROCm sphere-cuboid collision queries."""

from __future__ import annotations

import math

import pytest
import torch

from curobo._src.curobolib.backends.hip_backend.collision_sphere_cuboid import (
    sphere_cuboid_collision,
)
from curobo._src.runtime import AcceleratorType, detect_accelerator

pytestmark = pytest.mark.skipif(
    detect_accelerator() != AcceleratorType.ROCM or not torch.cuda.is_available(),
    reason="requires a ROCm GPU",
)


def _collision_query(
    query_spheres: torch.Tensor,
    dims: torch.Tensor,
    inv_pose: torch.Tensor,
    enable: torch.Tensor | None = None,
    count: torch.Tensor | None = None,
    env_query_idx: torch.Tensor | None = None,
    activation_distance: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a float32 collision query with standard test parameters."""
    device = query_spheres.device
    num_envs, max_cuboids = dims.shape[:2]
    if enable is None:
        enable = torch.ones((num_envs, max_cuboids), dtype=torch.uint8, device=device)
    if count is None:
        count = torch.full((num_envs,), max_cuboids, dtype=torch.int32, device=device)
    if env_query_idx is None:
        env_query_idx = torch.zeros(
            (query_spheres.shape[0],), dtype=torch.int32, device=device
        )
    return sphere_cuboid_collision(
        query_spheres,
        dims,
        inv_pose,
        enable,
        count,
        torch.ones((1,), dtype=torch.float32, device=device),
        torch.tensor([activation_distance], dtype=torch.float32, device=device),
        env_query_idx,
        True,
    )


def test_sphere_cuboid_outside_overlap_and_inside() -> None:
    """Match activation cost and penetration gradients across SDF regions."""
    device = torch.device("cuda")
    spheres = torch.tensor(
        [[[[0.9, 0.0, 0.0, 0.1], [0.7, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.1]]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    dims = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]], dtype=torch.float32, device=device)
    inv_pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )

    cost, gradient = _collision_query(spheres, dims, inv_pose)

    assert torch.allclose(cost, torch.tensor([[[0.0, 0.025, 0.7]]], device=device))
    assert torch.allclose(
        gradient[..., 0], torch.tensor([[[0.0, -0.5, -1.0]]], device=device)
    )
    assert torch.count_nonzero(gradient[..., 1:]) == 0
    cost.sum().backward()
    assert torch.allclose(spheres.grad, gradient)


def test_sphere_cuboid_rotation_and_multiple_obstacles() -> None:
    """Transform gradients to world space and accumulate obstacle contributions."""
    device = torch.device("cuda")
    half_angle = 0.25 * math.pi
    spheres = torch.tensor(
        [[[[0.0, 0.7, 0.0, 0.1]]]], dtype=torch.float32, device=device
    )
    dims = torch.tensor(
        [[[1.0, 0.4, 1.0, 0.0], [1.0, 0.4, 1.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    inv_pose = torch.tensor(
        [[
            [0.0, 0.0, 0.0, math.cos(half_angle), 0.0, 0.0, -math.sin(half_angle), 0.0],
            [0.0, 0.0, 0.0, math.cos(half_angle), 0.0, 0.0, -math.sin(half_angle), 0.0],
        ]],
        dtype=torch.float32,
        device=device,
    )

    cost, gradient = _collision_query(spheres, dims, inv_pose)

    assert torch.allclose(cost, torch.tensor([[[0.05]]], device=device), atol=1.0e-5)
    assert torch.allclose(
        gradient[..., :3],
        torch.tensor([[[[0.0, -1.0, 0.0]]]], device=device),
        atol=1.0e-5,
    )


def test_sphere_cuboid_environment_and_disable_masks() -> None:
    """Select per-batch environments and ignore disabled or out-of-count cuboids."""
    device = torch.device("cuda")
    spheres = torch.tensor(
        [
            [[[0.0, 0.0, 0.0, 0.1]]],
            [[[0.0, 0.0, 0.0, 0.1]]],
        ],
        dtype=torch.float32,
        device=device,
    )
    dims = torch.ones((2, 2, 4), dtype=torch.float32, device=device)
    dims[..., 3] = 0.0
    inv_pose = torch.zeros((2, 2, 8), dtype=torch.float32, device=device)
    inv_pose[..., 3] = 1.0
    inv_pose[1, 0, 0] = -2.0
    enable = torch.tensor([[1, 0], [1, 1]], dtype=torch.uint8, device=device)
    count = torch.tensor([2, 1], dtype=torch.int32, device=device)
    env_query_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)

    cost, _ = _collision_query(
        spheres,
        dims,
        inv_pose,
        enable=enable,
        count=count,
        env_query_idx=env_query_idx,
    )

    assert torch.allclose(cost[0], torch.tensor([[0.7]], device=device))
    assert torch.count_nonzero(cost[1]) == 0


def test_sphere_cuboid_disabled_sphere_and_finite_difference() -> None:
    """Disable negative-radius spheres and match float32 finite differences."""
    device = torch.device("cuda")
    spheres = torch.tensor(
        [[[[0.65, 0.13, -0.07, 0.1], [0.0, 0.0, 0.0, -0.1]]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    dims = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]], dtype=torch.float32, device=device)
    inv_pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    cost, gradient = _collision_query(spheres, dims, inv_pose)
    assert torch.count_nonzero(cost[..., 1]) == 0
    assert torch.count_nonzero(gradient[..., 1, :]) == 0

    epsilon = 1.0e-3
    plus = spheres.detach().clone()
    minus = spheres.detach().clone()
    plus[..., 0, 0] += epsilon
    minus[..., 0, 0] -= epsilon
    plus_cost, _ = _collision_query(plus, dims, inv_pose)
    minus_cost, _ = _collision_query(minus, dims, inv_pose)
    numerical = (plus_cost[..., 0] - minus_cost[..., 0]) / (2.0 * epsilon)
    assert torch.allclose(numerical, gradient[..., 0, 0], atol=2.0e-4, rtol=2.0e-3)
