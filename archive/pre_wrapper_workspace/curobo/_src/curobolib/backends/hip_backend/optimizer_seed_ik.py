# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm adaptations for the SeedIK optimizer contract."""

from __future__ import annotations

import torch


def adapt_seed_ik_gradient(joint_gradient: torch.Tensor) -> torch.Tensor:
    """Convert PyTorch pose autograd output to the SeedIK LM descent convention."""
    return -joint_gradient
