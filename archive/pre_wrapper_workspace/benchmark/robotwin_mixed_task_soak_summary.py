# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Aggregate a frozen multi-task RoboTwin ROCm soak and enforce its release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROBOTWIN_REVISION = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
XPOLICY_REVISION = "fa431ecd893ee706883e64fe5fe1464ec8cd928d"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--minimum-tasks", default=2, type=int)
    parser.add_argument("--minimum-episodes", default=8, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize(
    paths: list[Path], minimum_tasks: int = 2, minimum_episodes: int = 8
) -> dict[str, Any]:
    """Summarize mixed-task evidence without dropping failed tasks or seeds."""
    if minimum_tasks <= 0 or minimum_episodes <= 0:
        raise ValueError("minimum task and episode counts must be positive")
    raw_inputs = [path.read_bytes() for path in paths]
    episodes = [json.loads(data) for data in raw_inputs]
    if not episodes:
        raise ValueError("at least one episode result is required")

    identities = [(str(item["task"]), int(item["seed"])) for item in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("mixed-task evidence must contain unique task and seed pairs")
    tasks = sorted({task for task, _ in identities})
    invalid_revision = [
        identity
        for identity, item in zip(identities, episodes, strict=True)
        if item.get("robotwin_revision") != ROBOTWIN_REVISION
        or item.get("xpolicy_revision") != XPOLICY_REVISION
    ]
    invalid_runtime = [
        identity
        for identity, item in zip(identities, episodes, strict=True)
        if item.get("accelerator") != "rocm"
        or item.get("backend") != "hip"
        or item.get("left_planner") != "CuroboPlanner"
        or item.get("right_planner") != "CuroboPlanner"
        or not item.get("xpolicy_fail_closed", False)
    ]
    failed = [
        identity
        for identity, item in zip(identities, episodes, strict=True)
        if not item.get("task_success", False)
    ]
    gpu_fault_count = sum(int(item.get("gpu_fault_count", 0)) for item in episodes)
    deadlock_count = sum(int(item.get("deadlock_count", 0)) for item in episodes)
    fallback_count = len(invalid_runtime) + sum(
        int(item.get("fallback_count", 0)) for item in episodes
    )
    enough_coverage = len(tasks) >= minimum_tasks and len(episodes) >= minimum_episodes
    passed = bool(
        enough_coverage
        and not invalid_revision
        and not invalid_runtime
        and not failed
        and fallback_count == 0
        and gpu_fault_count == 0
        and deadlock_count == 0
    )
    return {
        "accelerator": "rocm",
        "backend": "hip",
        "robotwin_revision": ROBOTWIN_REVISION,
        "xpolicy_revision": XPOLICY_REVISION,
        "tasks": tasks,
        "task_count": len(tasks),
        "minimum_tasks": minimum_tasks,
        "episode_count": len(episodes),
        "minimum_episodes": minimum_episodes,
        "enough_coverage": enough_coverage,
        "episodes": [
            {"task": task, "seed": seed} for task, seed in sorted(identities)
        ],
        "failed_episodes": [
            {"task": task, "seed": seed} for task, seed in sorted(failed)
        ],
        "invalid_revision_episodes": [
            {"task": task, "seed": seed} for task, seed in sorted(invalid_revision)
        ],
        "invalid_runtime_episodes": [
            {"task": task, "seed": seed} for task, seed in sorted(invalid_runtime)
        ],
        "success_rate": (len(episodes) - len(failed)) / len(episodes),
        "fallback_count": fallback_count,
        "gpu_fault_count": gpu_fault_count,
        "deadlock_count": deadlock_count,
        "input_sha256": {
            str(path.resolve()): hashlib.sha256(data).hexdigest()
            for path, data in zip(paths, raw_inputs, strict=True)
        },
        "passed": passed,
    }


def main() -> None:
    """Write mixed-task soak evidence and fail unless the release gate passes."""
    args = _parse_args()
    result = summarize(args.inputs, args.minimum_tasks, args.minimum_episodes)
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
