# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Prepare a frozen RoboTwin/XPolicyLab checkout for fail-closed ROCm evaluation."""

# ruff: noqa: E501 -- exact upstream source strings preserve frozen revision formatting.

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROBOTWIN_REVISION = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
XPOLICY_REVISION = "fa431ecd893ee706883e64fe5fe1464ec8cd928d"


@dataclass(frozen=True)
class Replacement:
    """Describe one exact, idempotent source replacement."""

    relative_path: Path
    original: str
    migrated: str
    legacy_originals: tuple[str, ...] = ()


REPLACEMENTS = (
    Replacement(
        Path("envs/robot/planner.py"),
        """    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level=\"error\", logger_name=\"curobo\")
""",
        """    from curobo.types import JointState, Pose as CuroboPose
    import time
    from curobo.robotwin import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
        validate_robotwin_runtime,
    )
    import curobo.logging as logger
    import os
    import torch
    import yaml
    validate_robotwin_runtime(
        required_accelerator="rocm"
        if os.environ.get("CUROBO_REQUIRE_ROCM") == "1"
        else None
    )
    logger.setup_logger(level=\"error\", logger_name=\"curobo\")
""",
        legacy_originals=(
            """    from curobo.types import JointState, Pose as CuroboPose
    import time
    from curobo.robotwin import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
        validate_rocm_runtime,
    )
    import curobo.logging as logger
    import torch
    import yaml
    validate_rocm_runtime()
    logger.setup_logger(level="error", logger_name="curobo")
""",
        ),
    ),
    Replacement(
        Path("envs/robot/planner.py"),
        "            self.motion_gen_batch = MotionGen(motion_gen_config)\n"
        "            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)\n\n"
        "        def plan_path(\n",
        "            self.motion_gen_batch = MotionGen(motion_gen_config)\n"
        "            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)\n\n"
        "        def interpolate_joint_action(self, curr_joint_pos, target_joint_pos):\n"
        "            \"\"\"Interpolate a qpos policy action with cuRobo device limits.\"\"\"\n"
        "            start_joint_state = JointState.from_position(\n"
        "                torch.as_tensor(\n"
        "                    curr_joint_pos,\n"
        "                    device=self.motion_gen.tensor_args.device,\n"
        "                    dtype=self.motion_gen.tensor_args.dtype,\n"
        "                ).unsqueeze(0),\n"
        "                joint_names=self.active_joints_name,\n"
        "            )\n"
        "            target_joint_state = JointState.from_position(\n"
        "                torch.as_tensor(\n"
        "                    target_joint_pos,\n"
        "                    device=self.motion_gen.tensor_args.device,\n"
        "                    dtype=self.motion_gen.tensor_args.dtype,\n"
        "                ).unsqueeze(0),\n"
        "                joint_names=self.active_joints_name,\n"
        "            )\n"
        "            trajectory = self.motion_gen.interpolate_joint_action(\n"
        "                start_joint_state,\n"
        "                target_joint_state,\n"
        "                control_dt=1 / 250,\n"
        "            )\n"
        "            return {\n"
        "                \"status\": \"Success\",\n"
        "                \"position\": trajectory.position.detach().cpu().numpy(),\n"
        "                \"velocity\": trajectory.velocity.detach().cpu().numpy(),\n"
        "            }\n\n"
        "        def plan_path(\n",
    ),
    Replacement(
        Path("envs/robot/planner.py"),
        "            result = self.motion_gen.plan_single("
        "start_joint_states, goal_pose_of_ee, plan_config)\n\n"
        "            # output\n",
        "            result = self.motion_gen_batch.reuse_cached_single(\n"
        "                start_joint_states,\n"
        "                goal_pose_of_ee,\n"
        "                plan_config,\n"
        "            )\n"
        "            if result is None:\n"
        "                result = self.motion_gen.plan_single(\n"
        "                    start_joint_states, goal_pose_of_ee, plan_config\n"
        "                )\n\n"
        "            if not result.success.item():\n"
        "                result = self.motion_gen_batch.plan_single_repeated(\n"
        "                    start_joint_states,\n"
        "                    goal_pose_of_ee,\n"
        "                    plan_config,\n"
        "                )\n\n"
        "            # output\n",
        legacy_originals=(
            "            result = self.motion_gen.plan_single("
            "start_joint_states, goal_pose_of_ee, plan_config)\n\n"
            "            if not result.success.item():\n"
            "                result = self.motion_gen_batch.plan_single_repeated(\n"
            "                    start_joint_states,\n"
            "                    goal_pose_of_ee,\n"
            "                    plan_config,\n"
            "                )\n\n"
            "            # output\n",
            "            result = self.motion_gen_batch.reuse_cached_single(\n"
            "                start_joint_states,\n"
            "                goal_pose_of_ee,\n"
            "                plan_config,\n"
            "            )\n"
            "            if result is None:\n"
            "                result = self.motion_gen.plan_single(\n"
            "                    start_joint_states, goal_pose_of_ee, plan_config\n"
            "                )\n\n"
            "            if not result.success.item():\n"
            "                result = self.motion_gen_batch.plan_single_repeated(\n"
            "                    start_joint_states,\n"
            "                    goal_pose_of_ee,\n"
            "                    plan_config,\n"
            "                )\n\n"
            "            # output\n",
        ),
    ),
    Replacement(
        Path("envs/robot/planner.py"),
        "            if np.all(res_result[\"status\"] == \"Failure\"):\n"
        "                return res_result\n\n"
        "            res_result[\"position\"] = np.array("
        "result.interpolated_plan.position.to(\"cpu\"))\n"
        "            res_result[\"velocity\"] = np.array("
        "result.interpolated_plan.velocity.to(\"cpu\"))\n"
        "            return res_result\n",
        "            if np.all(res_result[\"status\"] == \"Failure\"):\n"
        "                return res_result\n\n"
        "            res_result[\"position\"] = np.array("
        "result.interpolated_plan.position.to(\"cpu\"))\n"
        "            res_result[\"velocity\"] = np.array("
        "result.interpolated_plan.velocity.to(\"cpu\"))\n"
        "            if result.interpolated_last_tstep is not None:\n"
        "                res_result[\"last_tstep\"] = np.array(\n"
        "                    result.interpolated_last_tstep.to(\"cpu\")\n"
        "                )\n"
        "            return res_result\n",
    ),
    Replacement(
        Path("envs/_base_task.py"),
        '        sapien.render.set_ray_tracing_denoiser("oidn")\n',
        """        if os.environ.get("CUROBO_REQUIRE_ROCM") != "1":
            sapien.render.set_ray_tracing_denoiser("oidn")
""",
    ),
    Replacement(
        Path("envs/robot/robot.py"),
        "        if self.need_topp:\n"
        "            self.left_mplib_planner = MplibPlanner(\n",
        "        if self.need_topp and os.environ.get(\"CUROBO_REQUIRE_ROCM\") != \"1\":\n"
        "            self.left_mplib_planner = MplibPlanner(\n",
    ),
    Replacement(
        Path("envs/_base_task.py"),
        "            # ========== TOPP ==========\n"
        "            # TODO\n"
        "            topp_left_flag, topp_right_flag = True, True\n\n"
        "            try:\n"
        "                times, left_pos, left_vel, acc, duration = (self.robot.left_mplib_planner.TOPP(left_path,\n"
        "                                                                                            1 / 250,\n"
        "                                                                                            verbose=True))\n"
        "                left_result = dict()\n"
        "                left_result[\"position\"], left_result[\"velocity\"] = left_pos, left_vel\n"
        "                left_n_step = left_result[\"position\"].shape[0]\n"
        "            except Exception as e:\n"
        "                # print(\"left arm TOPP error: \", e)\n"
        "                topp_left_flag = False\n"
        "                left_n_step = 50  # fixed\n\n"
        "            if left_n_step == 0:\n"
        "                topp_left_flag = False\n"
        "                left_n_step = 50  # fixed\n\n"
        "            try:\n"
        "                times, right_pos, right_vel, acc, duration = (self.robot.right_mplib_planner.TOPP(right_path,\n"
        "                                                                                                1 / 250,\n"
        "                                                                                                verbose=True))\n"
        "                right_result = dict()\n"
        "                right_result[\"position\"], right_result[\"velocity\"] = right_pos, right_vel\n"
        "                right_n_step = right_result[\"position\"].shape[0]\n"
        "            except Exception as e:\n"
        "                # print(\"right arm TOPP error: \", e)\n"
        "                topp_right_flag = False\n"
        "                right_n_step = 50  # fixed\n\n"
        "            if right_n_step == 0:\n"
        "                topp_right_flag = False\n"
        "                right_n_step = 50  # fixed\n",
        "            # ========== Time parameterization ==========\n"
        "            topp_left_flag, topp_right_flag = True, True\n"
        "            if os.environ.get(\"CUROBO_REQUIRE_ROCM\") == \"1\":\n"
        "                left_result = self.robot.left_planner.interpolate_joint_action(\n"
        "                    left_current_qpos, left_arm_actions[0]\n"
        "                )\n"
        "                right_result = self.robot.right_planner.interpolate_joint_action(\n"
        "                    right_current_qpos, right_arm_actions[0]\n"
        "                )\n"
        "                left_n_step = left_result[\"position\"].shape[0]\n"
        "                right_n_step = right_result[\"position\"].shape[0]\n"
        "            else:\n"
        "                try:\n"
        "                    times, left_pos, left_vel, acc, duration = (self.robot.left_mplib_planner.TOPP(left_path,\n"
        "                                                                                                1 / 250,\n"
        "                                                                                                verbose=True))\n"
        "                    left_result = dict()\n"
        "                    left_result[\"position\"], left_result[\"velocity\"] = left_pos, left_vel\n"
        "                    left_n_step = left_result[\"position\"].shape[0]\n"
        "                except Exception as e:\n"
        "                    topp_left_flag = False\n"
        "                    left_n_step = 50\n\n"
        "                if left_n_step == 0:\n"
        "                    topp_left_flag = False\n"
        "                    left_n_step = 50\n\n"
        "                try:\n"
        "                    times, right_pos, right_vel, acc, duration = (self.robot.right_mplib_planner.TOPP(right_path,\n"
        "                                                                                                    1 / 250,\n"
        "                                                                                                    verbose=True))\n"
        "                    right_result = dict()\n"
        "                    right_result[\"position\"], right_result[\"velocity\"] = right_pos, right_vel\n"
        "                    right_n_step = right_result[\"position\"].shape[0]\n"
        "                except Exception as e:\n"
        "                    topp_right_flag = False\n"
        "                    right_n_step = 50\n\n"
        "                if right_n_step == 0:\n"
        "                    topp_right_flag = False\n"
        "                    right_n_step = 50\n",
    ),
    Replacement(
        Path("XPolicyLab/policy/OpenWAM/OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py"),
        """        except Exception as exc:
            print(f"[eval_policy_wrapper] planner fallback engaged: {type(exc).__name__}: {exc}")
""",
        """        except Exception as exc:
            if os.environ.get("CUROBO_REQUIRE_ROCM") == "1":
                raise RuntimeError(
                    "ROCm cuRobo planner initialization failed; MPLib fallback is disabled"
                ) from exc
            print(f"[eval_policy_wrapper] planner fallback engaged: {type(exc).__name__}: {exc}")
""",
    ),
    Replacement(
        Path("envs/stack_blocks_three.py"),
        "        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))  # arm_tag\n\n"
        "        self.last_gripper = arm_tag\n",
        "        if block is not self.block3:\n"
        "            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))  # arm_tag\n\n"
        "        self.last_gripper = arm_tag\n",
    ),
    Replacement(
        Path("scripts/eval_policy_xpolicylab.py"),
        "                    if inference_count <= 3:\n"
        "                        action_array_value = np.asarray(action_chunk, dtype=np.float32)\n"
        "                        print(\n"
        "                            \"XPOLICY_ACTION \"\n"
        "                            f\"inference={inference_count} shape={list(action_array_value.shape)} \"\n"
        "                            f\"min={float(action_array_value.min()):.6f} \"\n"
        "                            f\"max={float(action_array_value.max()):.6f} \"\n"
        "                            f\"left_gripper=[{float(action_array_value[:, 6].min()):.6f},\"\n"
        "                            f\"{float(action_array_value[:, 6].max()):.6f}] \"\n"
        "                            f\"right_gripper=[{float(action_array_value[:, 13].min()):.6f},\"\n"
        "                            f\"{float(action_array_value[:, 13].max()):.6f}]\"\n"
        "                        )\n",
        "                    action_array_value = np.asarray(action_chunk, dtype=np.float32)\n"
        "                    print(\n"
        "                        \"XPOLICY_ACTION \"\n"
        "                        f\"inference={inference_count} shape={list(action_array_value.shape)} \"\n"
        "                        f\"min={float(action_array_value.min()):.6f} \"\n"
        "                        f\"max={float(action_array_value.max()):.6f} \"\n"
        "                        f\"left_gripper=[{float(action_array_value[:, 6].min()):.6f},\"\n"
        "                        f\"{float(action_array_value[:, 6].max()):.6f}] \"\n"
        "                        f\"right_gripper=[{float(action_array_value[:, 13].min()):.6f},\"\n"
        "                        f\"{float(action_array_value[:, 13].max()):.6f}]\"\n"
        "                    )\n",
    ),
    Replacement(
        Path("scripts/eval_policy_xpolicylab.py"),
        "        print(\n"
        "            \"XPOLICY_EPISODE \"\n"
        "            f\"seed={now_seed} success={succ} rollout_steps={rollout_steps} \"\n"
        "            f\"inference_count={inference_count} \"\n"
        "            f\"final_state={np.asarray(final_observation['joint_action']['vector']).tolist()}\"\n"
        "        )\n",
        "        final_actor_positions = {\n"
        "            name: getattr(task_env, name).get_pose().p.tolist()\n"
        "            for name in (\"block1\", \"block2\", \"block3\")\n"
        "            if hasattr(task_env, name)\n"
        "        }\n"
        "        print(\n"
        "            \"XPOLICY_EPISODE \"\n"
        "            f\"seed={now_seed} success={succ} rollout_steps={rollout_steps} \"\n"
        "            f\"inference_count={inference_count} \"\n"
        "            f\"final_state={np.asarray(final_observation['joint_action']['vector']).tolist()} \"\n"
        "            f\"final_actor_positions={final_actor_positions}\"\n"
        "        )\n",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the validated changes; without this flag only verify applicability",
    )
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


def migrate_source(source: str, replacement: Replacement) -> tuple[str, bool]:
    """Return migrated source and whether this replacement changed the file."""
    originals = (replacement.original, *replacement.legacy_originals)
    original_counts = [source.count(original) for original in originals]
    migrated_count = source.count(replacement.migrated)
    if migrated_count == 1:
        source_without_migrated = source.replace(replacement.migrated, "", 1)
        if not any(original in source_without_migrated for original in originals):
            return source, False
    if sum(original_counts) == 1 and migrated_count == 0:
        matched_original = originals[original_counts.index(1)]
        return source.replace(matched_original, replacement.migrated), True
    raise RuntimeError(
        f"unexpected source shape for {replacement.relative_path}: "
        f"originals={original_counts}, migrated={migrated_count}"
    )


def prepare_checkout(root: Path, *, apply: bool) -> list[Path]:
    """Validate and optionally migrate the frozen RoboTwin checkout."""
    root = root.resolve()
    xpolicy_root = root / "XPolicyLab"
    if _revision(root) != ROBOTWIN_REVISION:
        raise RuntimeError(f"RoboTwin must be checked out at {ROBOTWIN_REVISION}")
    if _revision(xpolicy_root) != XPOLICY_REVISION:
        raise RuntimeError(f"XPolicyLab must be checked out at {XPOLICY_REVISION}")

    staged: dict[Path, str] = {}
    changed: list[Path] = []
    for replacement in REPLACEMENTS:
        path = root / replacement.relative_path
        source = staged.get(path, path.read_text(encoding="utf-8"))
        migrated, did_change = migrate_source(source, replacement)
        staged[path] = migrated
        if did_change and path not in changed:
            changed.append(path)

    if apply:
        for path, source in staged.items():
            path.write_text(source, encoding="utf-8")
    return changed


def main() -> None:
    """Validate or apply the reproducible RoboTwin ROCm integration overlay."""
    args = _parse_args()
    changed = prepare_checkout(args.robotwin_root, apply=args.apply)
    action = "updated" if args.apply else "would update"
    if not changed:
        print("RoboTwin ROCm integration is already prepared")
        return
    for path in changed:
        print(f"{action}: {path}")


if __name__ == "__main__":
    main()
