# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Prepare checksummed AMD RDNA evidence for an independent reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_EVIDENCE = (
    "container_acceptance_summary.json",
    "robotwin_stack_blocks_two_seeds0_3_current_summary.json",
    "robotwin_stack_blocks_three_seeds0_3_current_summary.json",
    "xpolicy_evaluation_summary.json",
    "robotwin_stack_blocks_two_request_compare_strict_summary.json",
    "motion_soak_1000.json",
    "mixed_task_soak_summary.json",
    "cuda_regression_summary.json",
    "ci_history_summary.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(evidence_root: Path, revision: str) -> dict[str, Any]:
    """Return an unsigned review request with required artifact checksums."""
    evidence_root = evidence_root.resolve()
    artifacts = []
    missing = []
    for name in REQUIRED_EVIDENCE:
        path = evidence_root / name
        if not path.is_file():
            missing.append(name)
            continue
        artifacts.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "revision": revision,
        "evidence_root": str(evidence_root),
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "ready_for_review": not missing,
        "review_contract": {
            "reviewer_must_be_independent": True,
            "verify_artifact_checksums": True,
            "verify_no_cpu_mplib_or_warp_fallback": True,
            "verify_frozen_revisions_and_seeds": True,
            "verify_cuda_and_rocm_ci_history": True,
            "verify_documented_limitations": True,
        },
        "signature_template": {
            "approved": False,
            "reviewer": "",
            "review_date": "",
            "artifact_checksums_verified": False,
            "review_request_sha256": "fill after hashing this request file",
            "notes": "",
        },
    }


def main() -> None:
    """Write an unsigned review request and fail while required evidence is missing."""
    args = _parse_args()
    result = evaluate(args.evidence_root, args.revision)
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["ready_for_review"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
