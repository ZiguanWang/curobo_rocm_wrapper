# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the AMD RDNA release-level acceptance decision."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "rocm_release_acceptance.py"
    spec = importlib.util.spec_from_file_location("rocm_release_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence(module: ModuleType) -> dict[str, dict[str, Any]]:
    episode = {
        "passed": True,
        "accelerator": "rocm",
        "backend": "hip",
        "success_rate": 1.0,
        "invalid_runtime_seeds": [],
        "robotwin_revision": module.ROBOTWIN_REVISION,
        "xpolicy_revision": module.XPOLICY_REVISION,
    }
    return {
        "container_acceptance_summary.json": {
            "passed": True,
            "gfx_arch": "gfx1201",
            "tests_passed": 214,
            "tests_failed": 0,
            "source_built_wheel": True,
            "isolated_install": True,
        },
        "robotwin_stack_blocks_two_seeds0_3_current_summary.json": {
            **episode,
            "task": "stack_blocks_two",
        },
        "robotwin_stack_blocks_three_seeds0_3_current_summary.json": {
            **episode,
            "task": "stack_blocks_three",
        },
        "xpolicy_evaluation_summary.json": {
            "passed": True,
            "policy": "real_policy",
            "accelerator": "rocm",
            "backend": "hip",
            "episode_count": 4,
            "fallback_count": 0,
            "robotwin_revision": module.ROBOTWIN_REVISION,
            "xpolicy_revision": module.XPOLICY_REVISION,
        },
        "robotwin_stack_blocks_two_request_compare_strict_summary.json": {
            "passed": True,
            "enough_samples": True,
            "both_success_count": 5,
            "minimum_both_success": 5,
            "latency_seconds": {"rocm_speedup_over_mplib": 2.1},
        },
        "motion_soak_1000.json": {
            "accelerator": "rocm",
            "backend": "hip",
            "iterations": 1000,
            "failures": 0,
            "acceptance": {"zero_failures": True, "non_positive_memory_drift": True},
        },
        "mixed_task_soak_summary.json": {
            "passed": True,
            "accelerator": "rocm",
            "backend": "hip",
            "episode_count": 8,
            "fallback_count": 0,
            "gpu_fault_count": 0,
            "deadlock_count": 0,
        },
        "cuda_regression_summary.json": {
            "passed": True,
            "accelerator": "cuda",
            "backend": "cuda",
            "cuda_graph_default_enabled": True,
            "cuda_graph_capture_replay_passed": True,
            "rocm_success_rate_delta_points": 1.0,
        },
        "ci_history_summary.json": {
            "passed": True,
            "required_consecutive_passes": 3,
            "cuda_consecutive_passes": 3,
            "rocm_consecutive_passes": 3,
            "runs": [
                {"workflow_run_id": "1"},
                {"workflow_run_id": "2"},
                {"workflow_run_id": "3"},
            ],
        },
        "independent_review.json": {
            "approved": True,
            "reviewer": "reviewer@example.com",
            "review_date": "2026-10-22",
            "artifact_checksums_verified": True,
            "reviewer_independent": True,
            "reviewed_revision": "candidate",
        },
    }


def _write_evidence(root: Path, evidence: dict[str, dict[str, Any]]) -> None:
    review = evidence["independent_review.json"]
    for filename, data in evidence.items():
        if filename == "independent_review.json":
            continue
        (root / filename).write_text(json.dumps(data), encoding="utf-8")
    artifacts = []
    for filename in evidence:
        if filename == "independent_review.json":
            continue
        path = root / filename
        artifacts.append(
            {
                "path": filename,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    request_path = root / "independent_review_request.json"
    request_path.write_text(
        json.dumps(
            {
                "revision": "candidate",
                "artifacts": artifacts,
                "missing_artifacts": [],
                "ready_for_review": True,
            }
        ),
        encoding="utf-8",
    )
    review["review_request"] = request_path.name
    review["review_request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    (root / "independent_review.json").write_text(json.dumps(review), encoding="utf-8")


def test_release_acceptance_requires_every_gate(tmp_path: Path) -> None:
    """Accept only a complete set of conforming P0/P1/P2 evidence."""
    module = _load_module()
    _write_evidence(tmp_path, _evidence(module))

    result = module.evaluate(tmp_path)

    assert result["release_ready"]
    assert result["passed_gate_count"] == result["gate_count"] == 10
    assert result["failed_gates"] == []


def test_release_acceptance_rejects_missing_external_evidence(tmp_path: Path) -> None:
    """Keep CUDA, XPolicy, CI, soak, and review omissions visible."""
    module = _load_module()
    evidence = _evidence(module)
    del evidence["cuda_regression_summary.json"]
    del evidence["xpolicy_evaluation_summary.json"]
    _write_evidence(tmp_path, evidence)

    result = module.evaluate(tmp_path)

    assert not result["release_ready"]
    assert "p1_real_xpolicy" in result["failed_gates"]
    assert "p2_cuda_regression" in result["failed_gates"]


def test_release_acceptance_keeps_failed_long_task_in_denominator(tmp_path: Path) -> None:
    """Reject a 3/4 extended task even when every other gate passes."""
    module = _load_module()
    evidence = _evidence(module)
    extended = evidence["robotwin_stack_blocks_three_seeds0_3_current_summary.json"]
    extended["passed"] = False
    extended["success_rate"] = 0.75
    _write_evidence(tmp_path, evidence)

    result = module.evaluate(tmp_path)

    assert not result["release_ready"]
    assert result["failed_gates"] == ["p1_stack_blocks_three"]


def test_release_acceptance_rejects_stale_independent_review(tmp_path: Path) -> None:
    """Reject review approval after a checksummed artifact changes."""
    module = _load_module()
    evidence = _evidence(module)
    _write_evidence(tmp_path, evidence)
    path = tmp_path / "motion_soak_1000.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = module.evaluate(tmp_path)

    assert not result["release_ready"]
    assert result["failed_gates"] == ["p2_independent_review"]
