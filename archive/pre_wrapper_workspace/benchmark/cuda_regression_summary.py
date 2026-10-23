# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Combine CUDA runtime, JUnit, and RoboTwin evidence into the CUDA release gate."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

GRAPH_CAPTURE_TEST = "test_simple_execution_with_graph"
GRAPH_DEFAULT_TEST = "test_use_cuda_graph_none_default"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--cuda-episodes", required=True, type=Path)
    parser.add_argument("--rocm-episodes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _junit_evidence(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    testcases = [case for suite in suites for case in suite.findall("testcase")]

    def passed_test(name: str) -> bool:
        matches = [case for case in testcases if case.get("name") == name]
        return bool(matches) and all(
            case.find("failure") is None
            and case.find("error") is None
            and case.find("skipped") is None
            for case in matches
        )

    return {
        "tests": sum(int(suite.get("tests", "0")) for suite in suites),
        "failures": sum(int(suite.get("failures", "0")) for suite in suites),
        "errors": sum(int(suite.get("errors", "0")) for suite in suites),
        "skipped": sum(int(suite.get("skipped", "0")) for suite in suites),
        "cuda_graph_capture_replay_passed": passed_test(GRAPH_CAPTURE_TEST),
        "cuda_graph_default_enabled": passed_test(GRAPH_DEFAULT_TEST),
    }


def summarize(
    runtime_path: Path,
    junit_path: Path,
    cuda_episode_path: Path,
    rocm_episode_path: Path,
) -> dict[str, Any]:
    """Return CUDA regression evidence using the same frozen RoboTwin workload."""
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    cuda_episodes = json.loads(cuda_episode_path.read_text(encoding="utf-8"))
    rocm_episodes = json.loads(rocm_episode_path.read_text(encoding="utf-8"))
    junit = _junit_evidence(junit_path)
    same_workload = bool(
        cuda_episodes.get("task") == rocm_episodes.get("task")
        and cuda_episodes.get("seeds") == rocm_episodes.get("seeds")
        and cuda_episodes.get("robotwin_revision") == rocm_episodes.get("robotwin_revision")
        and cuda_episodes.get("xpolicy_revision") == rocm_episodes.get("xpolicy_revision")
    )
    cuda_success_rate = float(cuda_episodes.get("success_rate", 0.0))
    rocm_success_rate = float(rocm_episodes.get("success_rate", 0.0))
    delta_points = (cuda_success_rate - rocm_success_rate) * 100.0
    passed = bool(
        runtime.get("accelerator") == "cuda"
        and runtime.get("backend") == "cuda"
        and runtime.get("cuda_version")
        and not runtime.get("rocm_version")
        and junit["tests"] > 0
        and junit["failures"] == 0
        and junit["errors"] == 0
        and junit["cuda_graph_capture_replay_passed"]
        and junit["cuda_graph_default_enabled"]
        and cuda_episodes.get("accelerator") == "cuda"
        and cuda_episodes.get("backend") == "cuda"
        and rocm_episodes.get("accelerator") == "rocm"
        and rocm_episodes.get("backend") == "hip"
        and same_workload
        and delta_points <= 2.0
    )
    return {
        "passed": passed,
        "accelerator": "cuda",
        "backend": "cuda",
        "torch_version": runtime.get("torch_version"),
        "cuda_version": runtime.get("cuda_version"),
        "device_name": runtime.get("device_name"),
        "tests_run": junit["tests"],
        "tests_failed": junit["failures"] + junit["errors"],
        "tests_skipped": junit["skipped"],
        "cuda_graph_default_enabled": junit["cuda_graph_default_enabled"],
        "cuda_graph_capture_replay_passed": junit["cuda_graph_capture_replay_passed"],
        "same_workload": same_workload,
        "task": cuda_episodes.get("task"),
        "seeds": cuda_episodes.get("seeds"),
        "cuda_success_rate": cuda_success_rate,
        "rocm_success_rate": rocm_success_rate,
        "rocm_success_rate_delta_points": delta_points,
        "runtime_evidence": str(runtime_path.resolve()),
        "junit_evidence": str(junit_path.resolve()),
        "cuda_episode_evidence": str(cuda_episode_path.resolve()),
        "rocm_episode_evidence": str(rocm_episode_path.resolve()),
    }


def main() -> None:
    """Write CUDA regression evidence and fail unless the release gate passes."""
    args = _parse_args()
    result = summarize(args.runtime, args.junit, args.cuda_episodes, args.rocm_episodes)
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
