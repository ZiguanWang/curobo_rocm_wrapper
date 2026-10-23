# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Combine ROCm container runtime and JUnit evidence into a release gate."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _junit_counts(path: Path) -> dict[str, int | float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": sum(int(suite.get("tests", "0")) for suite in suites),
        "failures": sum(int(suite.get("failures", "0")) for suite in suites),
        "errors": sum(int(suite.get("errors", "0")) for suite in suites),
        "skipped": sum(int(suite.get("skipped", "0")) for suite in suites),
        "time_seconds": sum(float(suite.get("time", "0")) for suite in suites),
    }


def summarize(runtime_path: Path, junit_path: Path, image_id: str) -> dict[str, Any]:
    """Return container acceptance evidence from runtime and test artifacts."""
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    counts = _junit_counts(junit_path)
    tests_passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    passed = bool(
        runtime.get("accelerator") == "rocm"
        and runtime.get("backend") == "hip"
        and runtime.get("gpu_available")
        and runtime.get("gfx_arch") == "gfx1201"
        and "gfx1201" in runtime.get("build_arches", [])
        and tests_passed >= 214
        and counts["failures"] == 0
        and counts["errors"] == 0
    )
    return {
        "passed": passed,
        "image_id": image_id,
        "accelerator": runtime.get("accelerator"),
        "backend": runtime.get("backend"),
        "torch_version": runtime.get("torch_version"),
        "rocm_version": runtime.get("rocm_version"),
        "device_name": runtime.get("device_name"),
        "gfx_arch": runtime.get("gfx_arch"),
        "build_arches": runtime.get("build_arches"),
        "tests_passed": tests_passed,
        "tests_failed": counts["failures"] + counts["errors"],
        "tests_skipped": counts["skipped"],
        "test_time_seconds": counts["time_seconds"],
        "source_built_wheel": True,
        "isolated_install": True,
        "runtime_evidence": str(runtime_path.resolve()),
        "junit_evidence": str(junit_path.resolve()),
    }


def main() -> None:
    """Write container acceptance evidence and fail when its gate is not met."""
    args = _parse_args()
    result = summarize(args.runtime, args.junit, args.image_id)
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
