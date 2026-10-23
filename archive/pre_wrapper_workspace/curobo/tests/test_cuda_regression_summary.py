# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for NVIDIA CUDA regression acceptance aggregation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "cuda_regression_summary.py"
    spec = importlib.util.spec_from_file_location("cuda_regression_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path, rocm_success_rate: float = 1.0) -> tuple[Path, ...]:
    runtime = tmp_path / "runtime.json"
    junit = tmp_path / "tests.xml"
    cuda = tmp_path / "cuda.json"
    rocm = tmp_path / "rocm.json"
    runtime.write_text(
        json.dumps(
            {
                "accelerator": "cuda",
                "backend": "cuda",
                "torch_version": "2.8.0+cu129",
                "cuda_version": "12.9",
                "rocm_version": "",
                "device_name": "NVIDIA Test GPU",
            }
        ),
        encoding="utf-8",
    )
    junit.write_text(
        '<testsuite tests="3" failures="0" errors="0" skipped="0">'
        '<testcase name="test_simple_execution_with_graph" />'
        '<testcase name="test_use_cuda_graph_none_default" />'
        '<testcase name="test_motion" />'
        "</testsuite>",
        encoding="utf-8",
    )
    base = {
        "task": "stack_blocks_two",
        "seeds": [0, 1, 2, 3],
        "robotwin_revision": "robotwin-revision",
        "xpolicy_revision": "xpolicy-revision",
    }
    cuda.write_text(
        json.dumps(
            {
                **base,
                "accelerator": "cuda",
                "backend": "cuda",
                "success_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    rocm.write_text(
        json.dumps(
            {
                **base,
                "accelerator": "rocm",
                "backend": "hip",
                "success_rate": rocm_success_rate,
            }
        ),
        encoding="utf-8",
    )
    return runtime, junit, cuda, rocm


def test_cuda_regression_accepts_matching_success_and_graphs(tmp_path: Path) -> None:
    """Accept zero-failure CUDA tests and an equivalent frozen workload."""
    module = _load_module()

    result = module.summarize(*_write_inputs(tmp_path))

    assert result["passed"]
    assert result["cuda_graph_default_enabled"]
    assert result["cuda_graph_capture_replay_passed"]
    assert result["rocm_success_rate_delta_points"] == 0.0


def test_cuda_regression_rejects_rocm_success_gap(tmp_path: Path) -> None:
    """Reject ROCm when it trails the same CUDA workload by over two points."""
    module = _load_module()

    result = module.summarize(*_write_inputs(tmp_path, rocm_success_rate=0.75))

    assert not result["passed"]
    assert result["rocm_success_rate_delta_points"] == 25.0


def test_cuda_regression_rejects_skipped_graph_capture(tmp_path: Path) -> None:
    """A skipped CUDA Graph capture test is not release evidence."""
    module = _load_module()
    runtime, junit, cuda, rocm = _write_inputs(tmp_path)
    junit.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="1">'
        '<testcase name="test_simple_execution_with_graph"><skipped /></testcase>'
        '<testcase name="test_use_cuda_graph_none_default" />'
        "</testsuite>",
        encoding="utf-8",
    )

    result = module.summarize(runtime, junit, cuda, rocm)

    assert not result["passed"]
    assert not result["cuda_graph_capture_replay_passed"]
