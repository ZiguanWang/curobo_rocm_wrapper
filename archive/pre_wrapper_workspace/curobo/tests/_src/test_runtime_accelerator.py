# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for accelerator detection and capabilities."""

from types import SimpleNamespace

import pytest

from curobo._src.runtime import (
    AcceleratorType,
    detect_accelerator,
    get_accelerator_capabilities,
    is_graph_capture_enabled,
)


def _fake_torch(cuda_version: str | None, hip_version: str | None) -> SimpleNamespace:
    """Create the torch metadata needed by accelerator detection."""
    return SimpleNamespace(version=SimpleNamespace(cuda=cuda_version, hip=hip_version))


def test_detect_rocm_before_cuda_namespace() -> None:
    """ROCm metadata takes precedence because ROCm also exposes torch.cuda APIs."""
    torch_module = _fake_torch(cuda_version=None, hip_version="6.3")
    assert detect_accelerator(torch_module) == AcceleratorType.ROCM


def test_detect_cuda() -> None:
    """CUDA metadata selects the CUDA accelerator."""
    torch_module = _fake_torch(cuda_version="12.8", hip_version=None)
    assert detect_accelerator(torch_module) == AcceleratorType.CUDA


def test_detect_cpu_build() -> None:
    """A PyTorch build without GPU runtime metadata selects CPU."""
    torch_module = _fake_torch(cuda_version=None, hip_version=None)
    assert detect_accelerator(torch_module) == AcceleratorType.CPU


def test_rocm_capabilities_do_not_enable_unvalidated_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROCm custom kernels are enabled while HIP graphs remain opt-in during bring-up."""
    import curobo._src.runtime as runtime

    monkeypatch.setattr(runtime, "rocm_graphs", False)
    torch_module = _fake_torch(cuda_version=None, hip_version="6.3")
    capabilities = get_accelerator_capabilities(torch_module)
    assert capabilities.accelerator == AcceleratorType.ROCM
    assert capabilities.supports_custom_kernels
    assert not capabilities.supports_graph_capture
    assert not capabilities.supports_cuda_core
    assert not is_graph_capture_enabled(torch_module)


def test_cuda_graph_capture_remains_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA retains its existing default graph behavior."""
    import curobo._src.runtime as runtime

    monkeypatch.setattr(runtime, "cuda_graphs", True)
    torch_module = _fake_torch(cuda_version="12.8", hip_version=None)
    assert is_graph_capture_enabled(torch_module)
