# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Run one fail-closed RoboTwin episode and capture every planner request."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from robotwin_rocm_sapien_smoke import (
    _prepare_imports,
    _require_sapien_version,
    _revision,
    _validate_robotwin_adapter,
    _validate_xpolicy_fail_closed,
)

from curobo.robotwin import validate_robotwin_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--task", default="stack_blocks_two")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument(
        "--required-accelerator",
        choices=("rocm", "cuda"),
        default="rocm",
        help="fail unless RoboTwin uses the selected GPU backend",
    )
    parser.add_argument(
        "--repeat-first-comparable-request",
        default=1,
        type=int,
        help="repeat the first unconstrained planner request for steady-state timing",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _task_arguments(root: Path, task_name: str, embodiment: str) -> dict[str, Any]:
    robot_dir = (root / "assets" / "embodiments" / embodiment).resolve()
    config = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
    return {
        "task_name": task_name,
        "save_path": str(root / "build" / "robotwin_episode"),
        "render_freq": 0,
        "save_freq": 15,
        "data_type": {
            "rgb": False,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        },
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1.0,
            "random_head_camera_dis": 0,
            "random_table_height": 0.0,
            "random_light": False,
            "crazy_random_light_rate": 0.0,
            "random_embodiment": False,
        },
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": False,
            "collect_wrist_camera": False,
        },
        "left_robot_file": str(robot_dir),
        "right_robot_file": str(robot_dir),
        "left_embodiment_config": config,
        "right_embodiment_config": config,
        "dual_arm_embodied": True,
        "embodiment_dis": 0.0,
        "embodiment_name": embodiment,
        "need_plan": True,
        "collect_data": False,
        "eval_mode": False,
    }


def _pose_list(pose: Any) -> list[float]:
    return np.concatenate((np.asarray(pose.p), np.asarray(pose.q))).astype(np.float32).tolist()


def _task_final_state(task: Any) -> dict[str, Any]:
    """Capture final task geometry needed to explain a failed success predicate."""
    actor_poses = {}
    for name in ("block1", "block2", "block3"):
        actor = getattr(task, name, None)
        if actor is not None:
            actor_poses[name] = _pose_list(actor.get_pose())
    return {
        "actor_poses": actor_poses,
        "left_gripper_open": bool(task.is_left_gripper_open()),
        "right_gripper_open": bool(task.is_right_gripper_open()),
    }


def _successful_batch_trajectories(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Serialize successful batch trajectories for independent acceptance checks."""
    statuses = np.asarray(result.get("status"), dtype=object).reshape(-1)
    if "position" not in result:
        return []
    positions = np.asarray(result["position"], dtype=np.float32)
    velocities = (
        None
        if "velocity" not in result
        else np.asarray(result["velocity"], dtype=np.float32)
    )
    if positions.ndim != 3 or positions.shape[0] != statuses.shape[0]:
        raise ValueError(
            "batch trajectory shape must be (batch, horizon, dof) and match status: "
            f"position={positions.shape}, status={statuses.shape}"
        )
    if velocities is not None and velocities.shape != positions.shape:
        raise ValueError(
            f"batch velocity shape {velocities.shape} does not match position {positions.shape}"
        )
    last_tsteps = result.get("last_tstep")
    if last_tsteps is not None:
        last_tsteps = np.asarray(last_tsteps, dtype=np.int64).reshape(-1)
        if last_tsteps.shape != statuses.shape:
            raise ValueError(
                f"batch last_tstep shape {last_tsteps.shape} does not match status "
                f"{statuses.shape}"
            )

    trajectories = []
    for index, status in enumerate(statuses):
        if status != "Success":
            continue
        last_tstep = positions.shape[1]
        if last_tsteps is not None:
            last_tstep = int(last_tsteps[index])
            if last_tstep <= 0 or last_tstep > positions.shape[1]:
                raise ValueError(
                    f"batch slot {index} last_tstep {last_tstep} is outside horizon "
                    f"{positions.shape[1]}"
                )
        trajectories.append(
            {
                "slot": index,
                "last_tstep": last_tstep,
                "position": positions[index, :last_tstep].tolist(),
                "velocity": (
                    None
                    if velocities is None
                    else velocities[index, :last_tstep].tolist()
                ),
            }
        )
    return trajectories


def preserve_task_rng_during_planner_initialization(robot_type: type[Any]) -> None:
    """Prevent planner warmup from changing deterministic task generation."""
    original = robot_type.set_planner
    if getattr(original, "_curobo_preserves_task_rng", False):
        return

    def preserving_set_planner(self: Any, scene: Any = None) -> Any:
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            return original(self, scene=scene)
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    preserving_set_planner._curobo_preserves_task_rng = True
    robot_type.set_planner = preserving_set_planner


def _install_planner_trace(
    robot: Any,
    records: list[dict[str, Any]],
    repeat_first_comparable_request: int = 1,
) -> None:
    """Trace planner calls and optionally repeat the first comparable request."""
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
            _function: Callable[..., dict[str, Any]] = original_single,
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
                    "constraint_pose": None
                    if constraint_pose is None
                    else np.asarray(constraint_pose, dtype=np.float32).tolist(),
                    "position_shape": None
                    if "position" not in result
                    else list(result["position"].shape),
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
            _function: Callable[..., dict[str, Any]] = original_batch,
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
                        "status": np.asarray(result.get("status")).tolist(),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            successful_trajectories = _successful_batch_trajectories(result)
            records.append(
                {
                    "kind": "batch",
                    "arm": _arm,
                    "status": np.asarray(result.get("status")).tolist(),
                    "elapsed_seconds": repeated_results[-1]["elapsed_seconds"],
                    "repeat_results": repeated_results,
                    "current_qpos": np.asarray(current_qpos, dtype=np.float32).tolist(),
                    "target_poses": [_pose_list(pose) for pose in target_poses],
                    "constraint_pose": None
                    if constraint_pose is None
                    else np.asarray(constraint_pose, dtype=np.float32).tolist(),
                    "position_shape": None
                    if "position" not in result
                    else list(result["position"].shape),
                    "successful_trajectories": successful_trajectories,
                }
            )
            return result

        planner.plan_path = traced_single
        planner.plan_batch = traced_batch


def run_episode(args: argparse.Namespace) -> dict[str, Any]:
    """Run one episode and return machine-readable evidence."""
    root = args.robotwin_root.resolve()
    _prepare_imports(root, required_accelerator=args.required_accelerator)
    planner_path = _validate_robotwin_adapter(root)
    wrapper_path = _validate_xpolicy_fail_closed(root)
    sapien_version = _require_sapien_version()

    planner_module = importlib.import_module("envs.robot.planner")
    robot_module = importlib.import_module("envs.robot.robot")
    task_module = importlib.import_module(f"envs.{args.task}")
    preserve_task_rng_during_planner_initialization(robot_module.Robot)
    planner_type = planner_module.CuroboPlanner
    task_type = getattr(task_module, args.task)

    runtime = validate_robotwin_runtime(
        required_accelerator=getattr(args, "required_accelerator", "rocm")
    )
    task = task_type()
    started = time.perf_counter()
    task.setup_demo(
        now_ep_num=0,
        seed=args.seed,
        **_task_arguments(root, args.task, args.embodiment),
    )
    setup_seconds = time.perf_counter() - started
    if not isinstance(task.robot.left_planner, planner_type):
        raise RuntimeError("left planner is not CuroboPlanner")
    if not isinstance(task.robot.right_planner, planner_type):
        raise RuntimeError("right planner is not CuroboPlanner")

    records: list[dict[str, Any]] = []
    _install_planner_trace(
        task.robot,
        records,
        repeat_first_comparable_request=getattr(args, "repeat_first_comparable_request", 1),
    )
    play_started = time.perf_counter()
    info = task.play_once()
    play_seconds = time.perf_counter() - play_started
    task_success = bool(task.plan_success and task.check_success())
    evidence = {
        "robotwin_revision": _revision(root),
        "xpolicy_revision": _revision(root / "XPolicyLab"),
        "robotwin_planner": str(planner_path),
        "xpolicy_wrapper": str(wrapper_path),
        "xpolicy_fail_closed": True,
        "sapien_version": sapien_version,
        "accelerator": runtime.accelerator,
        "backend": runtime.backend,
        "device_name": runtime.device_name,
        "gfx_arch": runtime.gfx_arch,
        "rocm_version": runtime.rocm_version,
        "cuda_version": runtime.cuda_version,
        "task": args.task,
        "seed": args.seed,
        "repeat_first_comparable_request": getattr(
            args, "repeat_first_comparable_request", 1
        ),
        "left_planner": type(task.robot.left_planner).__name__,
        "right_planner": type(task.robot.right_planner).__name__,
        "plan_success": bool(task.plan_success),
        "task_success": task_success,
        "setup_seconds": setup_seconds,
        "play_seconds": play_seconds,
        "total_seconds": time.perf_counter() - started,
        "planner_records": records,
        "task_info": info,
        "task_final_state": _task_final_state(task),
    }
    task.close_env()
    return evidence


def main() -> None:
    """Run the episode, save all evidence, and fail when the task fails."""
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
