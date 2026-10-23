# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark and soak-test the cuRobo MotionPlanner ROCm path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from curobo._src.curobolib.backends import get_backend_name
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.runtime import AcceleratorType, detect_accelerator
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _parse_args() -> argparse.Namespace:
    """Parse soak-test command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--memory-sample-interval", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--scene", default="collision_test.yml")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a percentile from a non-empty latency sequence."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _create_planner(scene: str) -> MotionPlanner:
    """Create the fixed primitive-scene planner used by the soak test."""
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=scene,
        device_cfg=device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_goalset=4,
    )
    return MotionPlanner(config)


def _run_soak(args: argparse.Namespace) -> dict[str, Any]:
    """Run repeated nearby C-space plans and return machine-readable metrics."""
    if detect_accelerator() != AcceleratorType.ROCM:
        raise RuntimeError("This acceptance benchmark requires a PyTorch ROCm runtime")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.memory_sample_interval <= 0:
        raise ValueError("--memory-sample-interval must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    planner = _create_planner(args.scene)
    if get_backend_name() != "hip":
        raise RuntimeError("ROCm soak test requires the cuRobo 'hip' kernel backend")

    current_position = planner.default_joint_state.position.unsqueeze(0)
    current_state = JointState.from_position(current_position, joint_names=planner.joint_names)
    goal_position = current_position.clone()
    goal_position[..., 0] += 0.05
    goal_state = JointState.from_position(goal_position, joint_names=planner.joint_names)

    planner.warmup(enable_graph=True, num_warmup_iterations=args.warmup_iterations)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies_ms: list[float] = []
    memory_allocated: list[int] = []
    failures: list[int] = []
    start_time = time.perf_counter()
    for iteration in range(args.iterations):
        iteration_start = time.perf_counter()
        result = planner.plan_cspace(
            goal_state=goal_state,
            current_state=current_state,
            max_attempts=1,
            enable_graph_attempt=1,
        )
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - iteration_start) * 1000.0)
        if result is None or not bool(torch.any(result.success).item()):
            failures.append(iteration)
        if iteration % args.memory_sample_interval == 0 or iteration == args.iterations - 1:
            memory_allocated.append(torch.cuda.memory_allocated())
        completed = iteration + 1
        if completed % args.progress_interval == 0 or completed == args.iterations:
            print(
                f"completed={completed}/{args.iterations} failures={len(failures)} "
                f"allocated_bytes={memory_allocated[-1]}",
                file=sys.stderr,
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - start_time
    memory_drift_bytes = memory_allocated[-1] - memory_allocated[0]
    return {
        "accelerator": detect_accelerator().value,
        "backend": get_backend_name(),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "iterations": args.iterations,
        "failures": len(failures),
        "failure_iterations": failures,
        "success_rate": 1.0 - len(failures) / args.iterations,
        "elapsed_seconds": elapsed_seconds,
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": _percentile(latencies_ms, 50.0),
            "p90": _percentile(latencies_ms, 90.0),
            "p99": _percentile(latencies_ms, 99.0),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
        "memory": {
            "allocated_start_bytes": memory_allocated[0],
            "allocated_end_bytes": memory_allocated[-1],
            "allocated_drift_bytes": memory_drift_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "acceptance": {
            "zero_failures": len(failures) == 0,
            "non_positive_memory_drift": memory_drift_bytes <= 0,
        },
    }


def main() -> int:
    """Run the soak test and optionally save its JSON result."""
    args = _parse_args()
    result = _run_soak(args)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if all(result["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
