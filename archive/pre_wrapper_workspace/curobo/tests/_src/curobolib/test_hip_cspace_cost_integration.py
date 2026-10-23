# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for ROCm joint-space cost implementations."""

import pytest
import torch

from curobo._src.curobolib.backends.hip_backend.cost_cspace_distance import (
    cspace_distance_cost,
)
from curobo._src.curobolib.backends.hip_backend.cost_cspace_state import (
    cspace_state_cost,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required",
)


def test_cspace_distance_uses_terminal_and_non_terminal_weights() -> None:
    """Apply the configured DOF weights to the correct horizon steps."""
    device = torch.device("cuda")
    current = torch.ones((1, 2, 2), dtype=torch.float32, device=device)
    target = torch.zeros((1, 2), dtype=torch.float32, device=device)
    cost = cspace_distance_cost(
        current,
        target,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
        torch.tensor([1.0, 3.0], dtype=torch.float32, device=device),
        torch.tensor([0.5, 0.0], dtype=torch.float32, device=device),
    )
    assert torch.allclose(cost, torch.tensor([[1.0, 8.0]], device=device))


def test_cspace_state_cost_and_gradient_match_reference() -> None:
    """Match bound, regularization, target, and energy terms with autograd enabled."""
    device = torch.device("cuda")
    position = torch.tensor(
        [[[1.5, -0.5], [0.5, 0.25]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    velocity = torch.full_like(position, 0.2, requires_grad=True)
    acceleration = torch.full_like(position, 0.1, requires_grad=True)
    jerk = torch.zeros_like(position, requires_grad=True)
    torque = torch.full_like(position, 0.3, requires_grad=True)
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32, device=device)
    cost = cspace_state_cost(
        position,
        velocity,
        acceleration,
        jerk,
        torque,
        torch.tensor([0.5], dtype=torch.float32, device=device),
        torch.zeros((1, 2), dtype=torch.float32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        limits,
        limits,
        limits,
        limits,
        limits,
        torch.ones(5, dtype=torch.float32, device=device),
        torch.zeros(5, dtype=torch.float32, device=device),
        torch.ones(5, dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
        torch.tensor([0.25], dtype=torch.float32, device=device),
        torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
        retime_weights=False,
        retime_regularization_weights=False,
    )
    loss = cost.sum()
    loss.backward()

    assert cost.shape == position.shape
    assert torch.isfinite(cost).all()
    assert position.grad is not None
    assert velocity.grad is not None
    assert acceleration.grad is not None
    assert jerk.grad is not None
    assert torque.grad is not None
    assert torch.allclose(
        position.grad,
        torch.tensor([[[2.0, -1.0], [2.0, 2.0]]], device=device),
    )
