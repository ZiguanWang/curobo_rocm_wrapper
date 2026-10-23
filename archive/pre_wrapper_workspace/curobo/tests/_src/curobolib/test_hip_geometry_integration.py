# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm integration tests for the HIP self-collision extension."""

import importlib
from types import ModuleType

import pytest
import torch


def _hip_geometry() -> ModuleType:
    """Load the HIP geometry extension when running on a ROCm GPU host."""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU is not available")
    try:
        return importlib.import_module("curobo._src.curobolib.backends.hip_backend.geometry")
    except ModuleNotFoundError:
        pytest.skip("HIP geometry extension is not built")


def test_hip_self_collision_distance_and_gradient() -> None:
    """Two overlapping spheres produce the expected cost and gradients."""
    geometry = _hip_geometry()
    device = torch.device("cuda")
    output_distance = torch.zeros((1, 1), device=device)
    output_gradient = torch.zeros((1, 1, 2, 4), device=device)
    pair_distance = torch.zeros((1, 1, 1), device=device)
    sparse_index = torch.zeros((1, 1, 2), device=device, dtype=torch.uint8)
    robot_spheres = torch.tensor(
        [[[[0, 0, 0, 1], [1, 0, 0, 1]]]],
        device=device,
        dtype=torch.float32,
    )
    sphere_padding = torch.zeros(2, device=device)
    weight = torch.ones(1, device=device)
    pair_locations = torch.tensor([[0, 1]], device=device, dtype=torch.int16)
    block_max_value = torch.zeros((1, 1, 1), device=device)
    block_max_index = torch.zeros((1, 1, 1, 2), device=device, dtype=torch.int16)

    geometry.self_collision_distance(
        output_distance,
        output_gradient,
        pair_distance,
        sparse_index,
        robot_spheres,
        sphere_padding,
        weight,
        pair_locations,
        block_max_value,
        block_max_index,
        1,
        128,
        1,
        1,
        2,
        1,
        True,
        True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output_distance, torch.tensor([[1.5]], device=device))
    torch.testing.assert_close(pair_distance, torch.tensor([[[3.0]]], device=device))
    torch.testing.assert_close(
        output_gradient,
        torch.tensor(
            [[[[1, 0, 0, -1], [-1, 0, 0, -1]]]],
            device=device,
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        sparse_index,
        torch.tensor([[[1, 1]]], device=device, dtype=torch.uint8),
    )
