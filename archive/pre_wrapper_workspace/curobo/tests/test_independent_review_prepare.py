# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for independent review request preparation."""

from __future__ import annotations

from pathlib import Path

from benchmark.independent_review_prepare import REQUIRED_EVIDENCE, evaluate


def test_review_request_fails_closed_when_evidence_is_missing(tmp_path: Path) -> None:
    """A review request remains unsigned and blocked when evidence is missing."""
    result = evaluate(tmp_path, "candidate")

    assert not result["ready_for_review"]
    assert result["missing_artifacts"] == list(REQUIRED_EVIDENCE)
    assert not result["signature_template"]["approved"]


def test_review_request_checksums_all_required_evidence(tmp_path: Path) -> None:
    """Every required artifact receives a SHA-256 checksum."""
    for name in REQUIRED_EVIDENCE:
        (tmp_path / name).write_text(name, encoding="utf-8")

    result = evaluate(tmp_path, "candidate")

    assert result["ready_for_review"]
    assert not result["missing_artifacts"]
    assert len(result["artifacts"]) == len(REQUIRED_EVIDENCE)
    assert all(len(artifact["sha256"]) == 64 for artifact in result["artifacts"])
