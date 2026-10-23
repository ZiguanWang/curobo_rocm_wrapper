# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Validate assets and imports required for a real XPolicyLab evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROBOTWIN_REVISION = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
XPOLICY_REVISION = "fa431ecd893ee706883e64fe5fe1464ec8cd928d"
PLACEHOLDER_POLICIES = frozenset({"demo_policy"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--policy-python", default="python", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest(checkpoint: Path) -> list[dict[str, Any]]:
    files = (
        [checkpoint]
        if checkpoint.is_file()
        else sorted(path for path in checkpoint.rglob("*") if path.is_file())
    )
    root = checkpoint.parent if checkpoint.is_file() else checkpoint
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def evaluate(
    robotwin_root: Path,
    policy: str,
    checkpoint: Path,
    policy_python: Path,
) -> dict[str, Any]:
    """Return fail-closed readiness evidence for a real policy-server run."""
    robotwin_root = robotwin_root.resolve()
    xpolicy_root = robotwin_root / "XPolicyLab"
    policy_root = xpolicy_root / "policy" / policy
    checkpoint = checkpoint.resolve()
    resolved_python = shutil.which(str(policy_python))
    policy_python = (
        Path(resolved_python).absolute() if resolved_python else policy_python.absolute()
    )
    checks: dict[str, bool] = {
        "robotwin_revision": _revision(robotwin_root) == ROBOTWIN_REVISION,
        "xpolicy_revision": _revision(xpolicy_root) == XPOLICY_REVISION,
        "real_policy": policy not in PLACEHOLDER_POLICIES,
        "policy_directory": policy_root.is_dir(),
        "policy_model": (policy_root / "model.py").is_file(),
        "policy_deploy_config": (policy_root / "deploy.yml").is_file(),
        "policy_eval_entrypoint": (policy_root / "eval.sh").is_file(),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_nonempty": checkpoint.is_file()
        or (
            checkpoint.is_dir()
            and any(path.is_file() for path in checkpoint.rglob("*"))
        ),
        "policy_python": policy_python.is_file() and os.access(policy_python, os.X_OK),
    }
    import_diagnostic = "not attempted"
    if checks["policy_directory"] and checks["policy_python"]:
        environment = os.environ.copy()
        python_path = os.pathsep.join((str(robotwin_root), str(xpolicy_root)))
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (python_path, environment.get("PYTHONPATH", "")) if value
        )
        result = subprocess.run(
            [
                str(policy_python),
                "-c",
                f"from XPolicyLab.policy.{policy}.model import Model; print(Model.__name__)",
            ],
            capture_output=True,
            env=environment,
            text=True,
        )
        checks["policy_model_import"] = result.returncode == 0
        import_diagnostic = (result.stderr or result.stdout).strip()
    else:
        checks["policy_model_import"] = False

    checkpoint_manifest = (
        _checkpoint_manifest(checkpoint) if checks["checkpoint_exists"] else []
    )
    passed = all(checks.values())
    return {
        "robotwin_root": str(robotwin_root),
        "robotwin_revision": _revision(robotwin_root),
        "expected_robotwin_revision": ROBOTWIN_REVISION,
        "xpolicy_revision": _revision(xpolicy_root),
        "expected_xpolicy_revision": XPOLICY_REVISION,
        "policy": policy,
        "policy_root": str(policy_root),
        "policy_python": str(policy_python),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": checkpoint_manifest,
        "checks": checks,
        "policy_model_import_diagnostic": import_diagnostic,
        "ready_to_launch": passed,
    }


def main() -> None:
    """Write readiness evidence and fail when any prerequisite is missing."""
    args = _parse_args()
    evidence = evaluate(args.robotwin_root, args.policy, args.checkpoint, args.policy_python)
    output = json.dumps(evidence, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not evidence["ready_to_launch"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
