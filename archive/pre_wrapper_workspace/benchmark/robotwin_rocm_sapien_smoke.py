# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Run a fail-closed RoboTwin/SAPIEN planning smoke test on ROCm.

This script uses an external RoboTwin checkout and its official embodiment
assets.  It loads the Aloha-AgileX articulation in SAPIEN, constructs the
RoboTwin planners, and plans one left-arm motion from the simulated state.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from packaging.version import Version

from curobo.robotwin import validate_rocm_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument("--position-offset", default=0.03, type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_sapien_version() -> str:
    version = importlib.metadata.version("sapien")
    if Version(version) < Version("3.0.3"):
        raise RuntimeError(
            "RoboTwin ROCm smoke test requires sapien>=3.0.3; older 3.0.0b1 "
            "fails to identify RDNA Vulkan devices"
        )
    return version


def _prepare_imports(
    robotwin_root: Path, *, required_accelerator: str = "rocm"
) -> None:
    if not (robotwin_root / "envs" / "robot" / "robot.py").is_file():
        raise FileNotFoundError(f"invalid RoboTwin checkout: {robotwin_root}")
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    if required_accelerator == "rocm":
        os.environ["CUROBO_REQUIRE_ROCM"] = "1"
    elif required_accelerator == "cuda":
        os.environ.pop("CUROBO_REQUIRE_ROCM", None)
    else:
        raise ValueError("required_accelerator must be one of: rocm, cuda")
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)


def _validate_xpolicy_fail_closed(robotwin_root: Path) -> Path:
    wrapper_path = (
        robotwin_root
        / "XPolicyLab"
        / "policy"
        / "OpenWAM"
        / "OpenWAM"
        / "benchmarks"
        / "robotwin"
        / "eval_policy_wrapper.py"
    )
    source = wrapper_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(wrapper_path))
    guarded_raise = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "safe_set_planner":
            continue
        for handler in (child for child in ast.walk(node) if isinstance(child, ast.ExceptHandler)):
            has_guard = any(
                isinstance(child, ast.Constant) and child.value == "CUROBO_REQUIRE_ROCM"
                for child in ast.walk(handler)
            )
            has_raise = any(isinstance(child, ast.Raise) for child in ast.walk(handler))
            guarded_raise = guarded_raise or (has_guard and has_raise)
    if not guarded_raise:
        raise RuntimeError(
            "XPolicyLab OpenWAM wrapper is not fail-closed: add a "
            "CUROBO_REQUIRE_ROCM guard that re-raises planner initialization failures"
        )
    if "planner fallback engaged" not in source:
        raise RuntimeError("unexpected XPolicyLab wrapper: fallback audit marker is missing")
    return wrapper_path


def _validate_robotwin_adapter(robotwin_root: Path) -> Path:
    planner_path = robotwin_root / "envs" / "robot" / "planner.py"
    source = planner_path.read_text(encoding="utf-8")
    required_markers = (
        "from curobo.robotwin import",
        "validate_robotwin_runtime",
        'required_accelerator="rocm"',
        "CUROBO_REQUIRE_ROCM",
        "plan_single_repeated",
        "interpolate_joint_action",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(
            f"RoboTwin planner is missing ROCm adapter integration: {', '.join(missing)}"
        )
    return planner_path


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the SAPIEN plus ROCm planner smoke test and return its evidence."""
    root = args.robotwin_root.resolve()
    _prepare_imports(root)
    robotwin_planner = _validate_robotwin_adapter(root)
    xpolicy_wrapper = _validate_xpolicy_fail_closed(root)
    sapien_version = _require_sapien_version()

    import sapien
    import yaml
    from envs.robot.planner import CuroboPlanner
    from envs.robot.robot import Robot

    if CuroboPlanner is None:
        raise RuntimeError("RoboTwin did not expose CuroboPlanner")

    runtime = validate_rocm_runtime()
    robot_dir = root / "assets" / "embodiments" / args.embodiment
    config_path = robot_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    scene = sapien.Scene()
    robot = Robot(
        scene,
        need_topp=False,
        left_embodiment_config=config,
        right_embodiment_config=config,
        left_robot_file=str(robot_dir),
        right_robot_file=str(robot_dir),
        dual_arm_embodied=True,
        embodiment_dis=0.0,
    )
    robot.init_joints()
    robot.move_to_homestate()
    for _ in range(10):
        scene.step()

    planner_started = time.perf_counter()
    robot.set_planner(scene=scene)
    planner_initialization_seconds = time.perf_counter() - planner_started
    if not isinstance(robot.left_planner, CuroboPlanner):
        raise RuntimeError("left arm silently selected a non-cuRobo planner")
    if not isinstance(robot.right_planner, CuroboPlanner):
        raise RuntimeError("right arm silently selected a non-cuRobo planner")

    current_qpos = robot.left_entity.get_qpos()
    current_pose = np.asarray(robot.get_left_ee_pose(), dtype=np.float32)
    target_position = current_pose[:3] + np.asarray(
        [args.position_offset, 0.0, 0.0], dtype=np.float32
    )
    target_pose = sapien.Pose(target_position, current_pose[3:])

    planning_started = time.perf_counter()
    result = robot.left_planner.plan_path(current_qpos, target_pose, arms_tag="left")
    planning_seconds = time.perf_counter() - planning_started
    if result.get("status") != "Success":
        raise RuntimeError(f"RoboTwin cuRobo planning failed: {result.get('status')}")

    joint_target = np.asarray(current_qpos[:6], dtype=np.float32).copy()
    joint_target[0] += 0.01
    joint_action_started = time.perf_counter()
    joint_action = robot.left_planner.interpolate_joint_action(
        np.asarray(current_qpos[:6], dtype=np.float32), joint_target
    )
    joint_action_seconds = time.perf_counter() - joint_action_started
    if joint_action.get("status") != "Success":
        raise RuntimeError("RoboTwin cuRobo joint-action interpolation failed")

    return {
        "robotwin_revision": _revision(root),
        "robotwin_planner": str(robotwin_planner),
        "xpolicy_revision": _revision(root / "XPolicyLab"),
        "xpolicy_fail_closed": True,
        "xpolicy_wrapper": str(xpolicy_wrapper),
        "sapien_version": sapien_version,
        "accelerator": runtime.accelerator,
        "backend": runtime.backend,
        "torch_version": runtime.torch_version,
        "rocm_version": runtime.rocm_version,
        "device_name": runtime.device_name,
        "gfx_arch": runtime.gfx_arch,
        "embodiment": args.embodiment,
        "sapien_active_joint_count": len(robot.left_entity.get_active_joints()),
        "left_joint_names": robot.left_planner.active_joints_name,
        "right_joint_names": robot.right_planner.active_joints_name,
        "status": result["status"],
        "position_shape": list(result["position"].shape),
        "velocity_shape": list(result["velocity"].shape),
        "planner_initialization_seconds": planner_initialization_seconds,
        "planning_seconds": planning_seconds,
        "joint_action_backend": "curobo",
        "joint_action_mplib_used": False,
        "joint_action_position_shape": list(joint_action["position"].shape),
        "joint_action_velocity_shape": list(joint_action["velocity"].shape),
        "joint_action_seconds": joint_action_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    """Run the smoke test and emit machine-readable acceptance evidence."""
    args = _parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
    evidence = run_smoke(args)
    output = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
