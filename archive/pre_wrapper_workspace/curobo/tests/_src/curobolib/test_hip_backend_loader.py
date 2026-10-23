# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for loading the isolated HIP extension namespace."""

from types import SimpleNamespace

import pytest

from curobo._src.curobolib.backends.hip_backend import backend_loader


def test_load_backend_returns_none_when_all_extensions_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty HIP build is not exposed as a usable backend."""
    monkeypatch.setattr(
        backend_loader,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
    )
    assert backend_loader.load_backend() is None


def test_load_backend_exposes_partial_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed operation group can be used before unrelated groups are ported."""
    kinematics = SimpleNamespace(name="kinematics")

    def _import_module(name: str) -> SimpleNamespace:
        if name.endswith(".kinematics"):
            return kinematics
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(backend_loader, "import_module", _import_module)
    assert backend_loader.load_backend() == {"kinematics": kinematics}


def test_load_backend_uses_hip_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """All compiled operation groups are loaded from the HIP-specific directory."""
    imported_names: list[str] = []

    def _import_module(name: str) -> SimpleNamespace:
        imported_names.append(name)
        return SimpleNamespace(name=name)

    monkeypatch.setattr(backend_loader, "import_module", _import_module)
    modules = backend_loader.load_backend()

    assert modules is not None
    assert set(modules) == {"kinematics", "optimization", "trajectory", "geometry"}
    assert all(".hip_backend." in name for name in imported_names)


def test_load_backend_preserves_binary_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broken extension dependencies are reported instead of looking like an absent module."""
    monkeypatch.setattr(
        backend_loader,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("undefined symbol: hip_test")),
    )
    with pytest.raises(ImportError, match="undefined symbol"):
        backend_loader.load_backend()
