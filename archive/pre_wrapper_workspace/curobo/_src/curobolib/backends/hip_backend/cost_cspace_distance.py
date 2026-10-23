# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm implementation of the joint-space distance cost."""

from __future__ import annotations

import torch


def cspace_distance_cost(
    current: torch.Tensor,
    target: torch.Tensor,
    target_indices: torch.Tensor,
    weight: torch.Tensor,
    terminal_dof_weight: torch.Tensor,
    non_terminal_dof_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute a weighted squared joint-space distance entirely on the GPU."""
    batch_size, horizon, dof = current.shape
    target_values = target.view(-1, dof)
    if target_values.shape[0] == 1:
        selected_target = target_values.expand(batch_size, -1)
    else:
        selected_target = torch.index_select(
            target_values,
            0,
            target_indices.view(-1)[:batch_size].to(dtype=torch.long),
        )
    horizon_weight = terminal_dof_weight.view(1, 1, dof).expand(1, horizon, dof).clone()
    if horizon > 1:
        horizon_weight[:, :-1] = non_terminal_dof_weight.view(1, 1, dof)
    error = current - selected_target.view(batch_size, 1, dof)
    return (weight[0] * horizon_weight * error.square()).sum(dim=-1)
