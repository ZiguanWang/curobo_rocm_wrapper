# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ROCm runtime preflight validation."""

from types import SimpleNamespace

import pytest

from curobo._src.curobolib.backends.hip_backend.runtime_preflight import (
    inspect_runtime,
    result_to_dict,
    validate_runtime,
)


def _fake_torch(
    hip_version: str | None = "6.3",
    gpu_available: bool = True,
    gfx_arch: str = "gfx1100:sramecc+:xnack-",
) -> SimpleNamespace:
    """Create the subset of PyTorch metadata used by preflight checks."""
    cuda = SimpleNamespace(
        is_available=lambda: gpu_available,
        get_device_name=lambda index: "AMD Radeon Test GPU",
        get_device_properties=lambda index: SimpleNamespace(gcnArchName=gfx_arch),
    )
    return SimpleNamespace(
        __version__="2.7.0+rocm6.3",
        version=SimpleNamespace(hip=hip_version),
        cuda=cuda,
    )


def test_inspect_runtime_extracts_base_gfx_arch() -> None:
    """Feature suffixes are removed from the architecture used for build matching."""
    result = inspect_runtime(
        _fake_torch(),
        environ={"PYTORCH_ROCM_ARCH": "gfx1100;gfx1101"},
    )
    assert result.rocm_version == "6.3"
    assert result.gfx_arch == "gfx1100"
    assert result.build_arches == ("gfx1100", "gfx1101")


def test_validate_runtime_accepts_matching_architecture() -> None:
    """A visible ROCm GPU matching the explicit build target passes validation."""
    result = inspect_runtime(_fake_torch(), environ={"PYTORCH_ROCM_ARCH": "gfx1100"})
    validate_runtime(result)


def test_validate_runtime_rejects_non_rocm_torch() -> None:
    """A CUDA or CPU PyTorch build cannot be used for a HIP build."""
    result = inspect_runtime(
        _fake_torch(hip_version=None),
        environ={"PYTORCH_ROCM_ARCH": "gfx1100"},
    )
    with pytest.raises(RuntimeError, match="does not provide ROCm"):
        validate_runtime(result)


def test_validate_runtime_rejects_architecture_mismatch() -> None:
    """The extension target must include the architecture of the visible GPU."""
    result = inspect_runtime(_fake_torch(), environ={"PYTORCH_ROCM_ARCH": "gfx1030"})
    with pytest.raises(RuntimeError, match="gfx1100.*gfx1030"):
        validate_runtime(result)


def test_result_to_dict_is_json_compatible() -> None:
    """Expose stable machine-readable runtime evidence for release aggregation."""
    result = inspect_runtime(_fake_torch(), environ={"PYTORCH_ROCM_ARCH": "gfx1100"})

    data = result_to_dict(result)

    assert data["accelerator"] == "rocm"
    assert data["backend"] == "hip"
    assert data["build_arches"] == ["gfx1100"]
