# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for ROCm clean-container acceptance aggregation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "rocm_container_acceptance_summary.py"
    spec = importlib.util.spec_from_file_location("rocm_container_acceptance_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "accelerator": "rocm",
                "backend": "hip",
                "gpu_available": True,
                "gfx_arch": "gfx1201",
                "build_arches": ["gfx1201"],
                "torch_version": "2.10.0+rocm7.14.1",
                "rocm_version": "7.14.60850",
                "device_name": "AMD Radeon AI PRO R9700",
            }
        ),
        encoding="utf-8",
    )


def test_container_summary_accepts_214_clean_tests(tmp_path: Path) -> None:
    """Accept a matching gfx1201 runtime and complete clean-container suite."""
    module = _load_module()
    runtime = tmp_path / "runtime.json"
    junit = tmp_path / "tests.xml"
    _runtime(runtime)
    junit.write_text(
        '<testsuites><testsuite tests="214" failures="0" errors="0" skipped="0" '
        'time="476.64" /></testsuites>',
        encoding="utf-8",
    )

    result = module.summarize(runtime, junit, "sha256:image")

    assert result["passed"]
    assert result["tests_passed"] == 214
    assert result["tests_failed"] == 0


def test_container_summary_rejects_test_failure(tmp_path: Path) -> None:
    """Do not accept a nominal test count containing a failed test."""
    module = _load_module()
    runtime = tmp_path / "runtime.json"
    junit = tmp_path / "tests.xml"
    _runtime(runtime)
    junit.write_text(
        '<testsuite tests="214" failures="1" errors="0" skipped="0" time="10" />',
        encoding="utf-8",
    )

    result = module.summarize(runtime, junit, "sha256:image")

    assert not result["passed"]
    assert result["tests_failed"] == 1
