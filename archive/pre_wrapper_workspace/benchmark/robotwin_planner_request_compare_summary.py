# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Aggregate strict ROCm/MPLib request comparisons and enforce performance gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--minimum-speedup", default=2.0, type=float)
    parser.add_argument("--minimum-both-success", default=5, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(
    comparisons: list[dict[str, Any]],
    minimum_speedup: float = 2.0,
    minimum_both_success: int = 5,
) -> dict[str, Any]:
    """Aggregate matched requests without discarding failures or slow successful samples."""
    if not comparisons:
        raise ValueError("at least one comparison is required")
    if minimum_speedup <= 0.0:
        raise ValueError("minimum_speedup must be positive")
    if minimum_both_success < 1:
        raise ValueError("minimum_both_success must be positive")

    first = comparisons[0]
    identity_fields = ("task", "robotwin_revision", "xpolicy_revision")
    for comparison in comparisons[1:]:
        for field in identity_fields:
            if comparison.get(field) != first.get(field):
                raise ValueError(f"comparison inputs differ in {field}")

    matches = [match for comparison in comparisons for match in comparison.get("matches", [])]
    both_success = [match for match in matches if match.get("both_success")]
    rocm_seconds = [float(match["rocm_seconds"]) for match in both_success]
    mplib_seconds = [float(match["mplib_seconds"]) for match in both_success]
    latency = None
    speedup = None
    if both_success:
        rocm_p50 = statistics.median(rocm_seconds)
        mplib_p50 = statistics.median(mplib_seconds)
        speedup = mplib_p50 / rocm_p50
        latency = {
            "rocm_p50_seconds": rocm_p50,
            "rocm_p90_seconds": _percentile(rocm_seconds, 0.90),
            "rocm_p99_seconds": _percentile(rocm_seconds, 0.99),
            "mplib_p50_seconds": mplib_p50,
            "mplib_p90_seconds": _percentile(mplib_seconds, 0.90),
            "mplib_p99_seconds": _percentile(mplib_seconds, 0.99),
            "rocm_speedup_over_mplib": speedup,
        }

    matched_count = len(matches)
    rocm_success_count = sum(int(match.get("rocm_success", False)) for match in matches)
    mplib_success_count = sum(int(match.get("mplib_success", False)) for match in matches)
    enough_samples = len(both_success) >= minimum_both_success
    passed = enough_samples and speedup is not None and speedup >= minimum_speedup
    return {
        "task": first["task"],
        "robotwin_revision": first["robotwin_revision"],
        "xpolicy_revision": first["xpolicy_revision"],
        "seeds": sorted({int(comparison["seed"]) for comparison in comparisons}),
        "comparison_count": len(comparisons),
        "matched_request_count": matched_count,
        "rocm_success_count": rocm_success_count,
        "mplib_success_count": mplib_success_count,
        "both_success_count": len(both_success),
        "rocm_success_rate": rocm_success_count / matched_count if matched_count else None,
        "mplib_success_rate": mplib_success_count / matched_count if matched_count else None,
        "latency_seconds": latency,
        "minimum_speedup": minimum_speedup,
        "minimum_both_success": minimum_both_success,
        "enough_samples": enough_samples,
        "passed": passed,
    }


def main() -> None:
    """Write the aggregate comparison and fail when any configured gate is unmet."""
    args = _parse_args()
    result = summarize(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs],
        minimum_speedup=args.minimum_speedup,
        minimum_both_success=args.minimum_both_success,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
