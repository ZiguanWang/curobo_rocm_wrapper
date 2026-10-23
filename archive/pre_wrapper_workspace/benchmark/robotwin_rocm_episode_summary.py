# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Aggregate RoboTwin ROCm episode evidence and enforce success gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--minimum-success-rate", default=1.0, type=float)
    parser.add_argument("--accelerator", choices=("rocm", "cuda"), default="rocm")
    parser.add_argument("--backend", choices=("hip", "cuda"), default="hip")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(episodes: list[dict[str, Any]]) -> dict[str, float] | None:
    """Summarize latency for a non-empty episode subset."""
    if not episodes:
        return None
    total_seconds = [float(item["total_seconds"]) for item in episodes]
    play_seconds = [float(item["play_seconds"]) for item in episodes]
    return {
        "total_mean": statistics.fmean(total_seconds),
        "total_p50": _percentile(total_seconds, 0.50),
        "total_p90": _percentile(total_seconds, 0.90),
        "play_mean": statistics.fmean(play_seconds),
        "play_p50": _percentile(play_seconds, 0.50),
        "play_p90": _percentile(play_seconds, 0.90),
    }


def summarize(
    paths: list[Path],
    minimum_success_rate: float,
    accelerator: str = "rocm",
    backend: str = "hip",
) -> dict[str, Any]:
    """Aggregate episode JSON files and evaluate the configured acceptance gate."""
    if not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be between zero and one")
    raw_inputs = [path.read_bytes() for path in paths]
    episodes = [json.loads(data) for data in raw_inputs]
    if not episodes:
        raise ValueError("at least one episode result is required")

    task_revisions = {
        (item["task"], item["robotwin_revision"], item["xpolicy_revision"]) for item in episodes
    }
    if len(task_revisions) != 1:
        raise ValueError("all episode results must use one task and frozen upstream revisions")
    seeds = [int(item["seed"]) for item in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("episode results must contain unique seeds")
    invalid_runtime = [
        item["seed"]
        for item in episodes
        if item.get("accelerator") != accelerator
        or item.get("backend") != backend
        or item.get("left_planner") != "CuroboPlanner"
        or item.get("right_planner") != "CuroboPlanner"
        or not item.get("xpolicy_fail_closed", False)
    ]
    successful = [item for item in episodes if item.get("task_success", False)]
    failed = [item for item in episodes if not item.get("task_success", False)]
    success_rate = len(successful) / len(episodes)
    task, robotwin_revision, xpolicy_revision = next(iter(task_revisions))
    passed = not invalid_runtime and success_rate >= minimum_success_rate
    return {
        "task": task,
        "robotwin_revision": robotwin_revision,
        "xpolicy_revision": xpolicy_revision,
        "accelerator": accelerator,
        "backend": backend,
        "episode_count": len(episodes),
        "seeds": sorted(seeds),
        "successful_seeds": sorted(int(item["seed"]) for item in successful),
        "failed_seeds": sorted(int(item["seed"]) for item in failed),
        "invalid_runtime_seeds": sorted(int(seed) for seed in invalid_runtime),
        "success_rate": success_rate,
        "minimum_success_rate": minimum_success_rate,
        "latency_seconds": {
            "successful_episodes": _latency_summary(successful),
            "all_episode_exits": _latency_summary(episodes),
        },
        "passed": passed,
        "inputs": [str(path.resolve()) for path in paths],
        "input_sha256": {
            str(path.resolve()): hashlib.sha256(data).hexdigest()
            for path, data in zip(paths, raw_inputs, strict=True)
        },
    }


def main() -> None:
    """Write the summary and fail when its acceptance gate is not met."""
    args = _parse_args()
    evidence = summarize(
        args.inputs,
        args.minimum_success_rate,
        accelerator=args.accelerator,
        backend=args.backend,
    )
    output = json.dumps(evidence, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not evidence["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
