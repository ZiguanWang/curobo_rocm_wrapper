# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Run one RoboTwin episode with the CPU MPLib fallback as a comparison baseline."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from robotwin_rocm_episode import (
    _pose_list,
    _task_arguments,
    preserve_task_rng_during_planner_initialization,
)
from robotwin_rocm_sapien_smoke import _prepare_imports, _require_sapien_version, _revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--task", default="stack_blocks_two")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument(
        "--repeat-first-comparable-request",
        default=1,
        type=int,
        help="repeat the first unconstrained planner request for steady-state timing",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _install_mplib_baseline(robot_module: Any, planner_module: Any) -> None:
    """Force Robot.set_planner to construct CPU MPLib planners only."""
    mplib_type = planner_module.MplibPlanner
    original_plan_path = mplib_type.plan_path

    def compatible_plan_path(
        self: Any,
        now_qpos: Any,
        target_pose: Any,
        constraint_pose: Any = None,
        arms_tag: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del constraint_pose, kwargs
        return original_plan_path(self, now_qpos, target_pose, arms_tag=arms_tag)

    def compatible_plan_batch(
        self: Any,
        now_qpos: Any,
        target_pose_list: list[Any],
        constraint_pose: Any = None,
        arms_tag: Any = None,
    ) -> dict[str, Any]:
        statuses: list[str] = []
        positions: list[Any] = []
        velocities: list[Any] = []
        for pose in target_pose_list:
            result = compatible_plan_path(
                self,
                now_qpos,
                pose,
                constraint_pose=constraint_pose,
                arms_tag=arms_tag,
            )
            success = result.get("status") == "Success"
            statuses.append("Success" if success else "Failure")
            positions.append(result.get("position") if success else None)
            velocities.append(result.get("velocity") if success else None)
        return {"status": statuses, "position": positions, "velocity": velocities}

    mplib_type.plan_path = compatible_plan_path
    mplib_type.plan_batch = compatible_plan_batch

    def set_mplib_planner(self: Any, scene: Any = None) -> None:
        self.communication_flag = False
        self.left_planner = mplib_type(
            self.left_urdf_path,
            self.left_srdf_path,
            self.left_move_group,
            self.left_entity_origion_pose,
            self.left_entity,
            "mplib_RRT",
            scene,
        )
        self.right_planner = mplib_type(
            self.right_urdf_path,
            self.right_srdf_path,
            self.right_move_group,
            self.right_entity_origion_pose,
            self.right_entity,
            "mplib_RRT",
            scene,
        )
        self.left_mplib_planner = self.left_planner
        self.right_mplib_planner = self.right_planner

    robot_module.Robot.set_planner = set_mplib_planner


def _install_trace(
    robot: Any,
    records: list[dict[str, Any]],
    repeat_first_comparable_request: int = 1,
) -> None:
    """Trace MPLib calls and optionally repeat the first comparable request."""
    if repeat_first_comparable_request < 1:
        raise ValueError("repeat_first_comparable_request must be positive")
    repeated_request_claimed = False
    for arm in ("left", "right"):
        planner = getattr(robot, f"{arm}_planner")
        original_single = planner.plan_path
        original_batch = planner.plan_batch

        def traced_single(
            current_qpos: Any,
            target_pose: Any,
            constraint_pose: Any = None,
            arms_tag: Any = None,
            *,
            _arm: str = arm,
            _function: Any = original_single,
        ) -> dict[str, Any]:
            nonlocal repeated_request_claimed
            repeat_count = 1
            if constraint_pose is None and not repeated_request_claimed:
                repeat_count = repeat_first_comparable_request
                repeated_request_claimed = True
            repeated_results = []
            for _ in range(repeat_count):
                started = time.perf_counter()
                result = _function(current_qpos, target_pose, constraint_pose, arms_tag)
                repeated_results.append(
                    {
                        "status": result.get("status"),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            records.append(
                {
                    "kind": "single",
                    "arm": _arm,
                    "status": result.get("status"),
                    "elapsed_seconds": repeated_results[-1]["elapsed_seconds"],
                    "repeat_results": repeated_results,
                    "current_qpos": np.asarray(current_qpos, dtype=np.float32).tolist(),
                    "target_pose": _pose_list(target_pose),
                    "constraint_pose": (
                        None if constraint_pose is None else np.asarray(constraint_pose).tolist()
                    ),
                    "constraint_ignored": constraint_pose is not None,
                    "position_shape": None
                    if result.get("position") is None
                    else list(np.asarray(result["position"]).shape),
                }
            )
            return result

        def traced_batch(
            current_qpos: Any,
            target_poses: list[Any],
            constraint_pose: Any = None,
            arms_tag: Any = None,
            *,
            _arm: str = arm,
            _function: Any = original_batch,
        ) -> dict[str, Any]:
            nonlocal repeated_request_claimed
            repeat_count = 1
            if constraint_pose is None and not repeated_request_claimed:
                repeat_count = repeat_first_comparable_request
                repeated_request_claimed = True
            repeated_results = []
            for _ in range(repeat_count):
                started = time.perf_counter()
                result = _function(current_qpos, target_poses, constraint_pose, arms_tag)
                repeated_results.append(
                    {
                        "status": list(result.get("status", [])),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            records.append(
                {
                    "kind": "batch",
                    "arm": _arm,
                    "status": list(result.get("status", [])),
                    "elapsed_seconds": repeated_results[-1]["elapsed_seconds"],
                    "repeat_results": repeated_results,
                    "current_qpos": np.asarray(current_qpos, dtype=np.float32).tolist(),
                    "target_count": len(target_poses),
                    "target_poses": [_pose_list(pose) for pose in target_poses],
                    "constraint_pose": (
                        None if constraint_pose is None else np.asarray(constraint_pose).tolist()
                    ),
                    "constraint_ignored": constraint_pose is not None,
                }
            )
            return result

        planner.plan_path = traced_single
        planner.plan_batch = traced_batch


def run_episode(args: argparse.Namespace) -> dict[str, Any]:
    """Run a CPU MPLib episode and return machine-readable comparison evidence."""
    root = args.robotwin_root.resolve()
    _prepare_imports(root)
    # Keep CUROBO_REQUIRE_ROCM set to disable the NVIDIA-only OIDN path on
    # this AMD host. Planner selection is explicitly overridden below.
    sapien_version = _require_sapien_version()
    planner_module = importlib.import_module("envs.robot.planner")
    robot_module = importlib.import_module("envs.robot.robot")
    _install_mplib_baseline(robot_module, planner_module)
    preserve_task_rng_during_planner_initialization(robot_module.Robot)
    task_module = importlib.import_module(f"envs.{args.task}")
    task_type = getattr(task_module, args.task)

    task = task_type()
    started = time.perf_counter()
    task.setup_demo(
        now_ep_num=0,
        seed=args.seed,
        **_task_arguments(root, args.task, args.embodiment),
    )
    setup_seconds = time.perf_counter() - started
    if type(task.robot.left_planner).__name__ != "MplibPlanner":
        raise RuntimeError("left planner is not MplibPlanner")
    if type(task.robot.right_planner).__name__ != "MplibPlanner":
        raise RuntimeError("right planner is not MplibPlanner")

    records: list[dict[str, Any]] = []
    _install_trace(
        task.robot,
        records,
        repeat_first_comparable_request=getattr(args, "repeat_first_comparable_request", 1),
    )
    play_started = time.perf_counter()
    info = task.play_once()
    play_seconds = time.perf_counter() - play_started
    evidence = {
        "robotwin_revision": _revision(root),
        "xpolicy_revision": _revision(root / "XPolicyLab"),
        "sapien_version": sapien_version,
        "backend": "cpu_mplib",
        "mplib_version": getattr(importlib.import_module("mplib"), "__version__", "unknown"),
        "task": args.task,
        "seed": args.seed,
        "repeat_first_comparable_request": getattr(
            args, "repeat_first_comparable_request", 1
        ),
        "left_planner": type(task.robot.left_planner).__name__,
        "right_planner": type(task.robot.right_planner).__name__,
        "plan_success": bool(task.plan_success),
        "task_success": bool(task.plan_success and task.check_success()),
        "setup_seconds": setup_seconds,
        "play_seconds": play_seconds,
        "total_seconds": time.perf_counter() - started,
        "constraint_semantics": "ignored_by_mplib",
        "ignored_constraint_requests": sum(
            int(record["constraint_ignored"]) for record in records
        ),
        "planner_records": records,
        "task_info": info,
    }
    task.close_env()
    return evidence


def main() -> None:
    """Run and persist one CPU MPLib baseline episode."""
    args = _parse_args()
    args.output = args.output.resolve()
    evidence = run_episode(args)
    output = json.dumps(evidence, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not evidence["task_success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
