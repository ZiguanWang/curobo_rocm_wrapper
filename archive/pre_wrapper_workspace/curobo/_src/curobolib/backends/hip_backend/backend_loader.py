# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load cuRobo PyTorch HIP extension modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

_MODULE_NAMES = ("kinematics", "optimization", "trajectory", "geometry")
_MODULE_PREFIX = "curobo._src.curobolib.backends.hip_backend"


def load_backend() -> Optional[dict[str, Any]]:
    """Load every HIP extension module currently installed.

    Returns:
        Mapping from backend operation group to compiled extension module, or ``None`` when no HIP
        extensions have been built. Missing operation groups remain unavailable through the backend
        capability boundary.
    """
    modules: dict[str, Any] = {}
    for module_name in _MODULE_NAMES:
        qualified_name = f"{_MODULE_PREFIX}.{module_name}"
        try:
            modules[module_name] = import_module(qualified_name)
        except ModuleNotFoundError as error:
            if error.name == qualified_name:
                continue
            raise
    return modules or None
