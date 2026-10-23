# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Aggregate consecutive CUDA and ROCm candidate CI run evidence."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--required-consecutive", default=3, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("completed_at must be an ISO-8601 string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "workflow_run_id",
        "revision",
        "completed_at",
        "cuda_passed",
        "rocm_passed",
        "source_url",
        "workflow_run_attempt",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    _timestamp(data["completed_at"])
    if not isinstance(data["cuda_passed"], bool) or not isinstance(data["rocm_passed"], bool):
        raise ValueError(f"{path} pass fields must be booleans")
    run_id = str(data["workflow_run_id"])
    source_url = data["source_url"]
    if not run_id.isdigit():
        raise ValueError(f"{path} workflow_run_id must be numeric")
    if not isinstance(data["workflow_run_attempt"], (str, int)) or not str(
        data["workflow_run_attempt"]
    ).isdigit():
        raise ValueError(f"{path} workflow_run_attempt must be positive")
    if int(data["workflow_run_attempt"]) < 1:
        raise ValueError(f"{path} workflow_run_attempt must be positive")
    expected_suffix = f"/actions/runs/{run_id}"
    if not isinstance(source_url, str) or not source_url.startswith("https://github.com/"):
        raise ValueError(f"{path} source_url must be a GitHub Actions URL")
    if not source_url.endswith(expected_suffix):
        raise ValueError(f"{path} source_url does not match workflow_run_id")
    if not isinstance(data["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", data["revision"]
    ):
        raise ValueError(f"{path} revision must be a full Git SHA")
    data["evidence"] = str(path.resolve())
    return data


def _consecutive_passes(runs: list[dict[str, Any]], field: str, revision: str) -> int:
    count = 0
    for run in reversed(runs):
        if run["revision"] != revision or not run[field]:
            break
        count += 1
    return count


def evaluate(
    evidence: list[Path], revision: str, required_consecutive: int = 3
) -> dict[str, Any]:
    """Return fail-closed consecutive CI evidence for one candidate revision."""
    if required_consecutive < 1:
        raise ValueError("required_consecutive must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full Git SHA")
    runs = sorted(
        (_load(path) for path in evidence),
        key=lambda run: _timestamp(run["completed_at"]),
    )
    run_ids = [str(run["workflow_run_id"]) for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("workflow_run_id values must be unique")
    cuda_passes = _consecutive_passes(runs, "cuda_passed", revision)
    rocm_passes = _consecutive_passes(runs, "rocm_passed", revision)
    passed = cuda_passes >= required_consecutive and rocm_passes >= required_consecutive
    return {
        "revision": revision,
        "required_consecutive_passes": required_consecutive,
        "cuda_consecutive_passes": cuda_passes,
        "rocm_consecutive_passes": rocm_passes,
        "run_count": len(runs),
        "runs": runs,
        "passed": passed,
    }


def main() -> None:
    """Write CI history evidence and fail until both platforms meet the threshold."""
    args = _parse_args()
    result = evaluate(args.evidence, args.revision, args.required_consecutive)
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
