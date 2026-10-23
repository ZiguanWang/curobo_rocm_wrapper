# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm integration tests for the HIP kinematics extension."""

import importlib
import math
from types import ModuleType

import pytest
import torch


def _hip_kinematics() -> ModuleType:
    """Load the HIP extension when running on a ROCm GPU host."""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU is not available")
    try:
        return importlib.import_module(
            "curobo._src.curobolib.backends.hip_backend.kinematics"
        )
    except ModuleNotFoundError:
        pytest.skip("HIP kinematics extension is not built")


def test_hip_kinematics_forward_and_backward() -> None:
    """A revolute link produces the expected pose and quaternion gradient."""
    kinematics = _hip_kinematics()
    device = torch.device("cuda")

    def tensor(data: object, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(data, device=device, dtype=dtype).contiguous()

    fixed_transforms = tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
        ]
    )
    link_map = tensor([0, 0], torch.int16)
    joint_map = tensor([0, 0], torch.int16)
    joint_map_type = tensor([-1, 5], torch.int8)
    tool_frame_map = tensor([1], torch.int16)
    joint_offset_map = tensor([[1, 0], [1, 0]])
    link_masses_com = torch.zeros((2, 4), device=device)
    joint_position = tensor([[math.pi / 2]])
    link_position = torch.zeros((1, 1, 3), device=device)
    link_quaternion = torch.zeros((1, 1, 4), device=device)
    center_of_mass = torch.zeros((1, 4), device=device)
    cumulative_transform = torch.zeros((1, 2, 3, 4), device=device)

    kinematics.launch_kinematics_forward(
        link_position,
        link_quaternion,
        center_of_mass,
        cumulative_transform,
        joint_position,
        fixed_transforms,
        link_masses_com,
        joint_map_type,
        joint_map,
        link_map,
        tool_frame_map,
        joint_offset_map,
        1,
        1,
        1,
        False,
    )

    expected_quaternion = tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    torch.testing.assert_close(link_position[0, 0], tensor([1, 0, 0]))
    torch.testing.assert_close(link_quaternion[0, 0], expected_quaternion)

    output_gradient = torch.zeros_like(joint_position)
    position_gradient = torch.zeros_like(link_position)
    quaternion_gradient = torch.zeros_like(link_quaternion)
    quaternion_gradient[..., 3] = 1
    empty_spheres = torch.empty((1, 0, 4), device=device)
    empty_sphere_map = torch.empty((0,), device=device, dtype=torch.int16)

    kinematics.launch_kinematics_backward(
        output_gradient,
        position_gradient,
        quaternion_gradient,
        empty_spheres,
        torch.zeros_like(center_of_mass),
        center_of_mass,
        position_gradient,
        cumulative_transform,
        empty_spheres,
        link_masses_com,
        link_map,
        joint_map,
        joint_map_type,
        tool_frame_map,
        empty_sphere_map,
        tensor([1], torch.int16),
        tensor([0, 0, 1], torch.int16),
        tensor([1], torch.int16),
        tensor([0, 1], torch.int16),
        tensor([True], torch.bool),
        joint_offset_map,
        tensor([0], torch.int32),
        1,
        1,
        1,
        1,
        0,
        False,
        False,
    )
    torch.cuda.synchronize()

    expected_gradient = 0.5 * math.cos(math.pi / 4)
    torch.testing.assert_close(
        output_gradient,
        tensor([[expected_gradient]]),
        atol=1e-6,
        rtol=1e-6,
    )
