# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Aggregate AMD RDNA release evidence and enforce every P0/P1/P2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROBOTWIN_REVISION = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
XPOLICY_REVISION = "fa431ecd893ee706883e64fe5fe1464ec8cd928d"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _episode_summary(data: dict[str, Any], task: str) -> bool:
    return bool(
        data.get("passed")
        and data.get("task") == task
        and data.get("accelerator") == "rocm"
        and data.get("backend") == "hip"
        and data.get("success_rate") == 1.0
        and data.get("invalid_runtime_seeds") == []
        and data.get("robotwin_revision") == ROBOTWIN_REVISION
        and data.get("xpolicy_revision") == XPOLICY_REVISION
    )


def _performance(data: dict[str, Any]) -> bool:
    latency = data.get("latency_seconds", {})
    return bool(
        data.get("passed")
        and data.get("enough_samples")
        and data.get("both_success_count", 0) >= data.get("minimum_both_success", 1)
        and latency.get("rocm_speedup_over_mplib", 0.0) >= 2.0
    )


def _soak(data: dict[str, Any]) -> bool:
    acceptance = data.get("acceptance", {})
    return bool(
        data.get("accelerator") == "rocm"
        and data.get("backend") == "hip"
        and data.get("iterations", 0) >= 1000
        and data.get("failures") == 0
        and acceptance.get("zero_failures")
        and acceptance.get("non_positive_memory_drift")
    )


def _container(data: dict[str, Any]) -> bool:
    return bool(
        data.get("passed")
        and data.get("gfx_arch") == "gfx1201"
        and data.get("tests_passed", 0) >= 214
        and data.get("tests_failed") == 0
        and data.get("source_built_wheel")
        and data.get("isolated_install")
    )


def _xpolicy(data: dict[str, Any]) -> bool:
    return bool(
        data.get("passed")
        and data.get("policy") not in (None, "", "demo_policy")
        and data.get("accelerator") == "rocm"
        and data.get("backend") == "hip"
        and data.get("episode_count", 0) > 0
        and data.get("fallback_count") == 0
        and data.get("robotwin_revision") == ROBOTWIN_REVISION
        and data.get("xpolicy_revision") == XPOLICY_REVISION
    )


def _mixed_soak(data: dict[str, Any]) -> bool:
    return bool(
        data.get("passed")
        and data.get("accelerator") == "rocm"
        and data.get("backend") == "hip"
        and data.get("episode_count", 0) > 0
        and data.get("fallback_count") == 0
        and data.get("gpu_fault_count") == 0
        and data.get("deadlock_count") == 0
    )


def _cuda(data: dict[str, Any]) -> bool:
    return bool(
        data.get("passed")
        and data.get("accelerator") == "cuda"
        and data.get("backend") == "cuda"
        and data.get("cuda_graph_default_enabled")
        and data.get("cuda_graph_capture_replay_passed")
        and data.get("rocm_success_rate_delta_points", 100.0) <= 2.0
    )


def _ci_history(data: dict[str, Any]) -> bool:
    runs = data.get("runs", [])
    run_ids = [str(run.get("workflow_run_id", "")) for run in runs if isinstance(run, dict)]
    return bool(
        data.get("passed")
        and data.get("required_consecutive_passes", 0) >= 3
        and data.get("cuda_consecutive_passes", 0) >= 3
        and data.get("rocm_consecutive_passes", 0) >= 3
        and len(runs) >= 3
        and len(run_ids) == len(runs)
        and len(run_ids) == len(set(run_ids))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _review(data: dict[str, Any], evidence_root: Path) -> bool:
    request_name = data.get("review_request")
    if request_name != "independent_review_request.json":
        return False
    request_path = evidence_root / request_name
    if not request_path.is_file() or _sha256(request_path) != data.get("review_request_sha256"):
        return False
    request = json.loads(request_path.read_text(encoding="utf-8"))
    artifacts = request.get("artifacts", [])
    checksums_match = bool(artifacts)
    for artifact in artifacts:
        path = evidence_root / artifact.get("path", "")
        checksums_match = bool(
            checksums_match
            and path.is_file()
            and path.stat().st_size == artifact.get("size_bytes")
            and _sha256(path) == artifact.get("sha256")
        )
    return bool(
        data.get("approved")
        and data.get("reviewer")
        and data.get("review_date")
        and data.get("artifact_checksums_verified")
        and data.get("reviewer_independent")
        and data.get("reviewed_revision") == request.get("revision")
        and request.get("ready_for_review")
        and request.get("missing_artifacts") == []
        and checksums_match
    )


GateCheck = tuple[str, str, Callable[[dict[str, Any]], bool]]

GATES: tuple[GateCheck, ...] = (
    ("p0_clean_container", "container_acceptance_summary.json", _container),
    (
        "p1_stack_blocks_two",
        "robotwin_stack_blocks_two_seeds0_3_current_summary.json",
        lambda data: _episode_summary(data, "stack_blocks_two"),
    ),
    (
        "p1_stack_blocks_three",
        "robotwin_stack_blocks_three_seeds0_3_current_summary.json",
        lambda data: _episode_summary(data, "stack_blocks_three"),
    ),
    ("p1_real_xpolicy", "xpolicy_evaluation_summary.json", _xpolicy),
    (
        "p2_mplib_speedup",
        "robotwin_stack_blocks_two_request_compare_strict_summary.json",
        _performance,
    ),
    ("p2_primitive_soak", "motion_soak_1000.json", _soak),
    ("p2_mixed_task_soak", "mixed_task_soak_summary.json", _mixed_soak),
    ("p2_cuda_regression", "cuda_regression_summary.json", _cuda),
    ("p2_ci_history", "ci_history_summary.json", _ci_history),
    ("p2_independent_review", "independent_review.json", lambda data: bool(data)),
)


def evaluate(evidence_root: Path) -> dict[str, Any]:
    """Evaluate all release gates without treating missing evidence as success."""
    evidence_root = evidence_root.resolve()
    gates: dict[str, dict[str, Any]] = {}
    for gate_name, relative_path, check in GATES:
        path = evidence_root / relative_path
        if not path.is_file():
            gates[gate_name] = {
                "passed": False,
                "reason": "missing evidence",
                "evidence": str(path),
            }
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            passed = (
                _review(data, evidence_root)
                if gate_name == "p2_independent_review"
                else check(data)
            )
            reason = "passed" if passed else "evidence does not satisfy gate contract"
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            passed = False
            reason = f"invalid evidence: {type(error).__name__}: {error}"
        gates[gate_name] = {
            "passed": passed,
            "reason": reason,
            "evidence": str(path),
        }
    failed_gates = [name for name, result in gates.items() if not result["passed"]]
    return {
        "evidence_root": str(evidence_root),
        "gates": gates,
        "failed_gates": failed_gates,
        "passed_gate_count": len(gates) - len(failed_gates),
        "gate_count": len(gates),
        "release_ready": not failed_gates,
    }


def main() -> None:
    """Write the release decision and fail unless all gates pass."""
    args = _parse_args()
    result = evaluate(args.evidence_root)
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["release_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
