# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm implementation of a batched Levenberg-Marquardt update."""

from __future__ import annotations

from typing import Tuple

import torch


def levenberg_marquardt_step(
    jacobian: torch.Tensor,
    jt_error: torch.Tensor,
    lambda_damping: torch.Tensor,
    joint_position: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve batched damped normal equations entirely on the AMD GPU."""
    action_dim = jacobian.shape[-1]
    normal_matrix = torch.bmm(jacobian.transpose(1, 2), jacobian)
    identity = torch.eye(action_dim, dtype=jacobian.dtype, device=jacobian.device)
    normal_matrix = normal_matrix + lambda_damping[:, None, None] * identity
    cholesky = torch.linalg.cholesky(normal_matrix)
    delta = torch.cholesky_solve(-jt_error.unsqueeze(-1), cholesky).squeeze(-1)
    updated_position = joint_position + delta
    predicted_reduction = 0.5 * (
        delta * (lambda_damping.unsqueeze(-1) * delta - jt_error)
    ).sum(dim=-1)
    return updated_position, predicted_reduction
