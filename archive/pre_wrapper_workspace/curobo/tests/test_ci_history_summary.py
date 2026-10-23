# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for candidate CI history aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.ci_history_summary import evaluate


def _write_run(
    path: Path,
    run_id: int,
    revision: str,
    completed_at: str,
    cuda_passed: bool = True,
    rocm_passed: bool = True,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "workflow_run_id": run_id,
                "workflow_run_attempt": 1,
                "revision": revision,
                "completed_at": completed_at,
                "cuda_passed": cuda_passed,
                "rocm_passed": rocm_passed,
                "source_url": f"https://github.com/example/curobo/actions/runs/{run_id}",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_requires_three_consecutive_runs_for_same_revision(tmp_path: Path) -> None:
    """Three successful runs of one revision satisfy the gate."""
    revision = "a" * 40
    runs = [
        _write_run(tmp_path / f"run_{index}.json", index, revision, f"2026-10-2{index}T00:00:00Z")
        for index in range(1, 4)
    ]

    result = evaluate(runs, revision)

    assert result["passed"]
    assert result["cuda_consecutive_passes"] == 3
    assert result["rocm_consecutive_passes"] == 3


def test_latest_failure_resets_only_failed_platform(tmp_path: Path) -> None:
    """A latest failure resets the affected platform's consecutive count."""
    revision = "a" * 40
    runs = [
        _write_run(tmp_path / "run_1.json", 1, revision, "2026-10-21T00:00:00Z"),
        _write_run(tmp_path / "run_2.json", 2, revision, "2026-10-22T00:00:00Z"),
        _write_run(
            tmp_path / "run_3.json",
            3,
            revision,
            "2026-10-23T00:00:00Z",
            rocm_passed=False,
        ),
    ]

    result = evaluate(runs, revision)

    assert not result["passed"]
    assert result["cuda_consecutive_passes"] == 3
    assert result["rocm_consecutive_passes"] == 0


def test_rejects_duplicate_workflow_runs(tmp_path: Path) -> None:
    """Duplicate workflow run identifiers cannot inflate CI history."""
    runs = [
        _write_run(tmp_path / "run_1.json", 1, "a" * 40, "2026-10-21T00:00:00Z"),
        _write_run(tmp_path / "run_2.json", 1, "a" * 40, "2026-10-22T00:00:00Z"),
    ]

    with pytest.raises(ValueError, match="unique"):
        evaluate(runs, "a" * 40)


def test_rejects_non_github_run_source(tmp_path: Path) -> None:
    """A run ID without a matching GitHub Actions URL is invalid."""
    path = _write_run(tmp_path / "run.json", 1, "a" * 40, "2026-10-21T00:00:00Z")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_url"] = "https://example.com/actions/runs/1"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="GitHub Actions"):
        evaluate([path], "a" * 40)
