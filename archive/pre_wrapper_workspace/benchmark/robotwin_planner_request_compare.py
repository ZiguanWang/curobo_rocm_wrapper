# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Compare semantically equivalent RoboTwin planner requests across backends."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", required=True, type=Path)
    parser.add_argument("--mplib", required=True, type=Path)
    parser.add_argument("--minimum-speedup", default=2.0, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _request_success(record: dict[str, Any]) -> bool:
    """Return whether a single request produced at least one successful result."""
    status = record.get("status")
    if isinstance(status, list):
        return any(value == "Success" for value in status)
    return status == "Success"


def _request_measurements(record: dict[str, Any]) -> list[tuple[bool, float]]:
    """Return each measured repetition, or the normal one-shot request measurement."""
    repeat_results = record.get("repeat_results")
    if repeat_results is None:
        return [(_request_success(record), float(record["elapsed_seconds"]))]
    measurements = []
    for result in repeat_results:
        measurements.append(
            (
                _request_success({"status": result.get("status")}),
                float(result["elapsed_seconds"]),
            )
        )
    return measurements


def _request_key(record: dict[str, Any]) -> str | None:
    """Build a stable key for an unconstrained request with a recorded start and targets."""
    if record.get("constraint_pose") is not None or record.get("constraint_ignored", False):
        return None
    current_qpos = record.get("current_qpos")
    if current_qpos is None:
        return None
    targets = record.get("target_poses")
    if targets is None:
        target = record.get("target_pose")
        if target is None:
            return None
        targets = [target]
    rounded_targets = [[round(float(component), 6) for component in target] for target in targets]
    rounded_start = [round(float(component), 6) for component in current_qpos]
    return json.dumps(
        {
            "arm": record.get("arm"),
            "kind": record.get("kind"),
            "current_qpos": rounded_start,
            "targets": rounded_targets,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _indexed_requests(
    records: list[dict[str, Any]],
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Group comparable requests by target key while preserving duplicate occurrences."""
    indexed: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        key = _request_key(record)
        if key is not None:
            indexed.setdefault(key, []).append((index, record))
    return indexed


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


def compare(
    rocm: dict[str, Any],
    mplib: dict[str, Any],
    minimum_speedup: float = 2.0,
) -> dict[str, Any]:
    """Match equivalent requests and evaluate the configured ROCm speedup gate."""
    if minimum_speedup <= 0.0:
        raise ValueError("minimum_speedup must be positive")
    identity_fields = ("task", "seed", "robotwin_revision", "xpolicy_revision")
    for field in identity_fields:
        if rocm.get(field) != mplib.get(field):
            raise ValueError(f"ROCm and MPLib evidence differ in {field}")
    if rocm.get("backend") != "hip":
        raise ValueError("ROCm evidence must use backend=hip")
    if mplib.get("backend") != "cpu_mplib":
        raise ValueError("MPLib evidence must use backend=cpu_mplib")

    rocm_index = _indexed_requests(rocm.get("planner_records", []))
    mplib_index = _indexed_requests(mplib.get("planner_records", []))
    matches: list[dict[str, Any]] = []
    for key in sorted(set(rocm_index) & set(mplib_index)):
        for (rocm_index_value, rocm_record), (mplib_index_value, mplib_record) in zip(
            rocm_index[key], mplib_index[key]
        ):
            rocm_measurements = _request_measurements(rocm_record)
            mplib_measurements = _request_measurements(mplib_record)
            if len(rocm_measurements) != len(mplib_measurements):
                raise ValueError("matched requests have different repetition counts")
            for repeat_index, (rocm_measurement, mplib_measurement) in enumerate(
                zip(rocm_measurements, mplib_measurements)
            ):
                rocm_success, rocm_seconds = rocm_measurement
                mplib_success, mplib_seconds = mplib_measurement
                matches.append(
                    {
                        "rocm_index": rocm_index_value,
                        "mplib_index": mplib_index_value,
                        "repeat_index": repeat_index,
                        "arm": rocm_record.get("arm"),
                        "kind": rocm_record.get("kind"),
                        "rocm_success": rocm_success,
                        "mplib_success": mplib_success,
                        "both_success": rocm_success and mplib_success,
                        "rocm_seconds": rocm_seconds,
                        "mplib_seconds": mplib_seconds,
                    }
                )

    both_success = [item for item in matches if item["both_success"]]
    speedup = None
    latency = None
    if both_success:
        rocm_seconds = [item["rocm_seconds"] for item in both_success]
        mplib_seconds = [item["mplib_seconds"] for item in both_success]
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
    rocm_success_count = sum(int(item["rocm_success"]) for item in matches)
    mplib_success_count = sum(int(item["mplib_success"]) for item in matches)
    passed = bool(both_success) and speedup is not None and speedup >= minimum_speedup
    return {
        "task": rocm["task"],
        "seed": rocm["seed"],
        "robotwin_revision": rocm["robotwin_revision"],
        "xpolicy_revision": rocm["xpolicy_revision"],
        "matched_request_count": matched_count,
        "rocm_success_count": rocm_success_count,
        "mplib_success_count": mplib_success_count,
        "both_success_count": len(both_success),
        "rocm_success_rate": rocm_success_count / matched_count if matched_count else None,
        "mplib_success_rate": mplib_success_count / matched_count if matched_count else None,
        "latency_seconds": latency,
        "minimum_speedup": minimum_speedup,
        "passed": passed,
        "matches": matches,
    }


def main() -> None:
    """Write the comparison and fail when the speedup gate is not met."""
    args = _parse_args()
    result = compare(
        json.loads(args.rocm.read_text(encoding="utf-8")),
        json.loads(args.mplib.read_text(encoding="utf-8")),
        args.minimum_speedup,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
