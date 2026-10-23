# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for accelerator-aware cuRobo kernel backend selection."""

from collections.abc import Iterator

import pytest

import curobo._src.curobolib.backends as backends
import curobo._src.runtime as runtime


@pytest.fixture(autouse=True)
def _reset_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore backend settings and cached selection after each test."""
    original_backend = runtime.kernel_backend
    backends.reset_backend()
    monkeypatch.delenv("CUROBO_KERNEL_BACKEND", raising=False)
    yield
    runtime.kernel_backend = original_backend
    backends.reset_backend()


def test_auto_selects_hip_for_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm builds select HIP without probing NVIDIA backends."""
    hip_modules = {"kinematics": object()}
    monkeypatch.setattr(backends, "detect_accelerator", lambda: runtime.AcceleratorType.ROCM)
    monkeypatch.setattr(backends, "_try_load_hip_backend", lambda: hip_modules)
    monkeypatch.setattr(
        backends,
        "_try_load_cuda_core_backend",
        lambda: pytest.fail("ROCm selection must not probe cuda.core"),
    )
    runtime.kernel_backend = "auto"

    assert backends.get_backend() is hip_modules
    assert backends.get_backend_name() == "hip"


def test_rocm_without_extensions_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm builds fail early when the HIP extensions are missing."""
    monkeypatch.setattr(backends, "detect_accelerator", lambda: runtime.AcceleratorType.ROCM)
    monkeypatch.setattr(backends, "_try_load_hip_backend", lambda: None)

    with pytest.raises(RuntimeError, match="ROCm.*HIP backend is not compiled"):
        backends.get_backend()


def test_environment_override_selects_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented environment variable overrides the runtime preference."""
    hip_modules = {"kinematics": object()}
    runtime.kernel_backend = "cuda_core"
    monkeypatch.setenv("CUROBO_KERNEL_BACKEND", "hip")
    monkeypatch.setattr(backends, "_try_load_hip_backend", lambda: hip_modules)

    assert backends.get_backend() is hip_modules
    assert backends.get_backend_name() == "hip"


def test_unknown_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid backend names report all supported values."""
    monkeypatch.setenv("CUROBO_KERNEL_BACKEND", "unknown")

    with pytest.raises(RuntimeError, match="auto, cuda_core, pybind, hip"):
        backends.get_backend()
