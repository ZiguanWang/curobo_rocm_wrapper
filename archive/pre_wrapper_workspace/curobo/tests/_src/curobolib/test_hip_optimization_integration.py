# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm integration tests for HIP optimization kernels."""

import importlib
from types import ModuleType

import pytest
import torch


def _hip_optimization() -> ModuleType:
    """Load the HIP optimization extension when running on a ROCm GPU host."""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU is not available")
    try:
        return importlib.import_module(
            "curobo._src.curobolib.backends.hip_backend.optimization"
        )
    except ModuleNotFoundError:
        pytest.skip("HIP optimization extension is not built")


def _run_lbfgs(optimization: ModuleType, use_shared_buffers: bool) -> list[torch.Tensor]:
    """Run one deterministic L-BFGS update and return its mutated buffers."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    batch_size = 2
    vector_dimension = 64
    history = 5
    q = torch.randn((batch_size, vector_dimension), device="cuda", generator=generator)
    grad_q = torch.randn((batch_size, vector_dimension), device="cuda", generator=generator)
    x_0 = torch.zeros_like(q)
    grad_0 = torch.zeros_like(q)
    rho = torch.zeros((history, batch_size), device="cuda")
    y_buffer = torch.zeros((history, batch_size, vector_dimension), device="cuda")
    s_buffer = torch.zeros_like(y_buffer)
    step = torch.zeros_like(q)

    optimization.launch_lbfgs_step(
        step,
        rho,
        y_buffer,
        s_buffer,
        q,
        grad_q,
        x_0,
        grad_0,
        1e-6,
        batch_size,
        history,
        vector_dimension,
        False,
        use_shared_buffers,
    )
    torch.cuda.synchronize()
    return [step, rho, y_buffer, s_buffer, x_0, grad_0]


def test_hip_lbfgs_shared_and_stable_paths_match() -> None:
    """Shared-buffer and stable L-BFGS paths produce equal finite updates."""
    optimization = _hip_optimization()
    stable = _run_lbfgs(optimization, use_shared_buffers=False)
    shared = _run_lbfgs(optimization, use_shared_buffers=True)

    for stable_tensor, shared_tensor in zip(stable, shared, strict=True):
        assert torch.isfinite(stable_tensor).all()
        torch.testing.assert_close(stable_tensor, shared_tensor)


def test_hip_line_search_selects_last_strong_wolfe_candidate() -> None:
    """Line search selects and copies the final candidate satisfying strong Wolfe."""
    optimization = _hip_optimization()
    batch_size = 2
    n_linesearch = 4
    vector_dimension = 64
    search_magnitudes = torch.tensor([0.0, 0.25, 0.5, 1.0], device="cuda")
    search_cost = (1.0 - search_magnitudes).square().view(1, n_linesearch, 1)
    search_cost = search_cost.repeat(batch_size, 1, 1)
    search_action = torch.arange(
        batch_size * n_linesearch * vector_dimension,
        device="cuda",
        dtype=torch.float32,
    ).view(batch_size, n_linesearch, vector_dimension)
    search_gradient = (2.0 * (search_magnitudes - 1.0) / vector_dimension).view(
        1, n_linesearch, 1
    )
    search_gradient = search_gradient.expand(
        batch_size, n_linesearch, vector_dimension
    ).contiguous()
    step_direction = torch.ones((batch_size, 1, vector_dimension), device="cuda")

    best_cost = torch.full((batch_size,), 10.0, device="cuda")
    best_action = torch.zeros((batch_size, vector_dimension), device="cuda")
    best_iteration = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    current_iteration = torch.zeros_like(best_iteration)
    converged = torch.zeros(batch_size, dtype=torch.uint8, device="cuda")
    exploration_cost = torch.zeros(batch_size, device="cuda")
    exploration_action = torch.zeros_like(best_action)
    exploration_gradient = torch.zeros_like(best_action)
    exploration_idx = torch.zeros(
        (batch_size, n_linesearch), dtype=torch.int32, device="cuda"
    )
    selected_cost = torch.zeros(batch_size, device="cuda")
    selected_action = torch.zeros_like(best_action)
    selected_gradient = torch.zeros_like(best_action)
    selected_idx = torch.zeros_like(exploration_idx)

    optimization.launch_line_search(
        best_cost,
        best_action,
        best_iteration,
        current_iteration,
        converged,
        5,
        0.0,
        0.0,
        exploration_cost,
        exploration_action,
        exploration_gradient,
        exploration_idx.view(-1),
        selected_cost,
        selected_action,
        selected_gradient,
        selected_idx.view(-1),
        search_cost,
        search_action,
        search_gradient,
        step_direction,
        search_magnitudes,
        1e-4,
        0.9,
        True,
        False,
        n_linesearch,
        vector_dimension,
        batch_size,
    )
    torch.cuda.synchronize()

    expected_idx = torch.full_like(selected_idx, n_linesearch - 1)
    assert torch.equal(selected_idx, expected_idx)
    assert torch.equal(exploration_idx, expected_idx)
    torch.testing.assert_close(selected_cost, search_cost[:, -1, 0])
    torch.testing.assert_close(exploration_cost, search_cost[:, -1, 0])
    torch.testing.assert_close(selected_action, search_action[:, -1])
    torch.testing.assert_close(selected_gradient, search_gradient[:, -1])
    torch.testing.assert_close(best_action, search_action[:, -1])
    assert torch.equal(best_iteration, torch.ones_like(best_iteration))
    assert torch.equal(current_iteration, torch.ones_like(current_iteration))
    assert torch.count_nonzero(converged) == 0
