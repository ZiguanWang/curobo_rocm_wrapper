# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Replay the last RoboTwin partial-pose failure without starting SAPIEN."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import transforms3d as t3d
import yaml

from curobo._src.motion.motion_planner import (
    _make_local_ik_seed_config,
    _normalize_revolute_seed_config,
    _replace_failed_seed_config,
)
from curobo._src.state.state_robot import RobotState
from curobo.robotwin import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
    validate_rocm_runtime,
)
from curobo.types import JointState, Pose

_ALOHA_ACTIVE_QPOS_INDICES = {
    "left": [6, 14, 18, 22, 26, 30],
    "right": [7, 15, 19, 23, 27, 31],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument(
        "--record-arm",
        choices=("left", "right"),
        help="Source-record arm when testing the same world target with the opposite planner.",
    )
    parser.add_argument("--record-index", type=int, default=-1)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--local-ik-seeds", type=int)
    parser.add_argument("--num-trajopt-seeds", type=int, default=1)
    parser.add_argument("--graph-iterations", type=int)
    parser.add_argument("--graph-new-nodes", type=int)
    parser.add_argument("--graph-neighbors", type=int)
    parser.add_argument("--tolerance-goalset-size", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _tensor_value(value: Optional[torch.Tensor]) -> Any:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _constraint_summary(metrics: Any) -> Optional[list[dict[str, Any]]]:
    """Summarize each rollout constraint at its first, last, and worst timestep."""
    if metrics is None or metrics.costs_and_constraints is None:
        return None
    collection = metrics.costs_and_constraints.constraints
    hybrid = metrics.costs_and_constraints.hybrid_costs_constraints
    entries: list[dict[str, Any]] = []
    for category, values, names in (
        ("constraint", collection.values, collection.names),
        ("hybrid", hybrid.values, hybrid.names),
    ):
        for name, value in zip(names, values):
            detached = value.detach()
            entries.append(
                {
                    "category": category,
                    "name": name,
                    "shape": list(detached.shape),
                    "maximum": float(torch.max(detached).item()),
                    "per_batch_maximum": torch.amax(detached.reshape(detached.shape[0], -1), dim=1)
                    .cpu()
                    .tolist(),
                    "first_maximum": float(torch.max(detached[:, 0]).item()),
                    "last_maximum": float(torch.max(detached[:, -1]).item()),
                }
            )
    return entries


def _collision_constraints(planner: Any, positions: torch.Tensor) -> list[dict[str, Any]]:
    """Evaluate self- and scene-collision constraints for joint positions."""
    rollout = planner.trajopt_solver.auxiliary_rollout
    if positions.ndim == 2:
        flat_positions = positions.unsqueeze(1).contiguous()
    else:
        flat_positions = positions.reshape(-1, positions.shape[-2], positions.shape[-1])
    rollout.update_batch_size(flat_positions.shape[0])
    joint_state = JointState.from_position(flat_positions, joint_names=planner.joint_names)
    state = RobotState(
        joint_state=joint_state,
        cuda_robot_model_state=planner.compute_kinematics(joint_state),
    )
    manager = rollout.metrics_constraint_manager
    disabled = [
        name
        for name in manager.get_enabled_costs()
        if name not in ("self_collision", "scene_collision")
    ]
    try:
        for name in disabled:
            manager.disable_cost_component(name)
        collection = manager.compute_costs(state, goal=None)
    finally:
        for name in disabled:
            manager.enable_cost_component(name)
    entries: list[dict[str, Any]] = []
    for name, value in zip(collection.names, collection.values):
        detached = value.detach()
        per_timestep = torch.amax(
            detached.reshape(detached.shape[0], detached.shape[1], -1), dim=2
        )
        flat_maximum_index = int(torch.argmax(per_timestep).item())
        entries.append(
            {
                "name": name,
                "shape": list(detached.shape),
                "maximum": float(torch.max(detached).item()),
                "maximum_batch_index": flat_maximum_index // per_timestep.shape[1],
                "maximum_horizon_index": flat_maximum_index % per_timestep.shape[1],
                "first_maximum": float(torch.max(detached[:, 0]).item()),
                "last_maximum": float(torch.max(detached[:, -1]).item()),
                "per_batch_maximum": torch.amax(detached.reshape(detached.shape[0], -1), dim=1)
                .cpu()
                .tolist(),
            }
        )
    return entries


def _graph_seed_results(
    motion_gen: MotionGen,
    start_state: JointState,
    normalized_seeds: torch.Tensor,
    ik_success: torch.Tensor,
    maximum_targets: int = 4,
) -> list[dict[str, Any]]:
    """Query PRM separately for successful IK targets and summarize each result."""
    planner = motion_gen._require_planner()
    if planner.graph_planner is None:
        return [{"error": "graph planner is not configured"}]

    flattened_seeds = normalized_seeds.reshape(-1, normalized_seeds.shape[-1])
    successful_indices = torch.nonzero(ik_success.reshape(-1), as_tuple=False).flatten()
    entries: list[dict[str, Any]] = []
    for seed_index_tensor in successful_indices[:maximum_targets]:
        seed_index = int(seed_index_tensor.item())
        goal = flattened_seeds[seed_index].reshape(1, -1)
        started = time.perf_counter()
        try:
            result = planner.graph_planner.find_path(
                start_state.position.clone(),
                goal.clone(),
                interpolate_waypoints=True,
                interpolation_steps=100,
                validate_interpolated_trajectory=True,
            )
        except (RuntimeError, ValueError) as error:
            entries.append(
                {
                    "seed_index": seed_index,
                    "success": [False],
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": str(error),
                }
            )
            break
        interpolated = result.interpolated_waypoints
        entries.append(
            {
                "seed_index": seed_index,
                "success": _tensor_value(result.success),
                "valid_query": result.valid_query,
                "debug_info": result.debug_info,
                "solve_time": result.solve_time,
                "elapsed_seconds": time.perf_counter() - started,
                "path_length": _tensor_value(result.path_length),
                "plan_waypoint_shapes": [
                    None if path is None else list(path.shape)
                    for path in (result.plan_waypoints or [])
                ],
                "interpolated_waypoints_shape": (
                    None if interpolated is None else list(interpolated.shape)
                ),
                "interpolated_collision_constraints": (
                    None if interpolated is None else _collision_constraints(planner, interpolated)
                ),
            }
        )
    return entries


def _analytic_self_collision_summary(
    planner: Any,
    positions: torch.Tensor,
) -> dict[str, Any]:
    """Identify the most-overlapping configured sphere pair with PyTorch operations."""
    if positions.ndim == 4:
        positions = positions.reshape(-1, positions.shape[-2], positions.shape[-1])
    joint_state = JointState.from_position(positions, joint_names=planner.joint_names)
    spheres = planner.compute_kinematics(joint_state).robot_spheres
    collision_config = planner.kinematics.get_self_collision_config()
    pairs = collision_config.collision_pairs.to(dtype=torch.long)
    padding = collision_config.sphere_padding
    first = spheres[..., pairs[:, 0], :]
    second = spheres[..., pairs[:, 1], :]
    radius_sum = first[..., 3] + second[..., 3] + padding[pairs[:, 0]] + padding[pairs[:, 1]]
    center_distance = torch.linalg.vector_norm(first[..., :3] - second[..., :3], dim=-1)
    overlap = radius_sum - center_distance
    flat_index = int(torch.argmax(overlap).item())
    pair_index = flat_index % overlap.shape[-1]
    horizon_index = (flat_index // overlap.shape[-1]) % overlap.shape[-2]
    batch_index = flat_index // (overlap.shape[-2] * overlap.shape[-1])
    sphere_indices = pairs[pair_index]

    kinematics_config = planner.kinematics.config.kinematics_config
    link_indices = kinematics_config.link_sphere_idx_map[sphere_indices].to(dtype=torch.long)
    index_to_link_name = {
        index: name for name, index in kinematics_config.link_name_to_idx_map.items()
    }
    link_names = [index_to_link_name[int(index.item())] for index in link_indices]
    return {
        "maximum_overlap": float(overlap[batch_index, horizon_index, pair_index].item()),
        "batch_index": batch_index,
        "horizon_index": horizon_index,
        "pair_index": pair_index,
        "sphere_indices": _tensor_value(sphere_indices),
        "link_indices": _tensor_value(link_indices),
        "link_names": link_names,
        "center_distance": float(center_distance[batch_index, horizon_index, pair_index].item()),
        "radius_sum_with_padding": float(
            radius_sum[batch_index, horizon_index, pair_index].item()
        ),
        "joint_position": _tensor_value(positions[batch_index, horizon_index]),
    }


def _continuation_ik_result(
    motion_gen: MotionGen,
    start_state: JointState,
    start_pose: Pose,
    goal_pose: Pose,
    num_steps: int = 8,
) -> dict[str, Any]:
    """Follow a Cartesian pose interpolation while retaining the nearest IK branch."""
    planner = motion_gen._require_planner()
    seed = start_state.position.unsqueeze(1)
    previous = start_state.position
    start_quaternion = start_pose.quaternion
    goal_quaternion = goal_pose.quaternion
    quaternion_sign = torch.where(
        torch.sum(start_quaternion * goal_quaternion, dim=-1, keepdim=True) < 0.0,
        -1.0,
        1.0,
    )
    goal_quaternion = quaternion_sign * goal_quaternion
    entries: list[dict[str, Any]] = []
    final_success = False

    for step in range(1, num_steps + 1):
        fraction = step / num_steps
        position = torch.lerp(start_pose.position, goal_pose.position, fraction)
        quaternion = torch.lerp(start_quaternion, goal_quaternion, fraction)
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        intermediate_pose = Pose(position=position, quaternion=quaternion)
        result = planner.ik_solver.solve_pose(
            motion_gen._goal_from_pose(intermediate_pose),
            seed_config=seed,
            return_seeds=planner.trajopt_solver.config.num_seeds,
            current_state=JointState.from_position(previous, joint_names=planner.joint_names),
        )
        successful_indices = torch.nonzero(result.success[0], as_tuple=False).flatten()
        selected_index = None
        if successful_indices.numel() > 0:
            candidates = _normalize_revolute_seed_config(
                result.solution,
                JointState.from_position(previous, joint_names=planner.joint_names),
                planner.kinematics,
            )[0, successful_indices]
            selected_offset = int(
                torch.argmin(torch.linalg.vector_norm(candidates - previous, dim=-1)).item()
            )
            selected_index = int(successful_indices[selected_offset].item())
            previous = candidates[selected_offset].reshape(1, -1)
            seed = previous.unsqueeze(1)
            final_success = step == num_steps
        else:
            final_success = False

        entries.append(
            {
                "step": step,
                "fraction": fraction,
                "success": _tensor_value(result.success),
                "position_error": _tensor_value(result.position_error),
                "rotation_error": _tensor_value(result.rotation_error),
                "selected_index": selected_index,
                "selected_position": None if selected_index is None else _tensor_value(previous),
            }
        )
        if selected_index is None:
            break

    graph_results = []
    if final_success:
        graph_results = _graph_seed_results(
            motion_gen,
            start_state,
            previous.unsqueeze(1),
            torch.ones((1, 1), device=previous.device, dtype=torch.bool),
            maximum_targets=1,
        )
    return {
        "success": final_success,
        "steps": entries,
        "final_position": _tensor_value(previous) if final_success else None,
        "final_constraints": (
            _collision_constraints(planner, previous) if final_success else None
        ),
        "graph_seed_results": graph_results,
    }


def _load_record(source: Path, record_index: int, arm: str) -> dict[str, Any]:
    evidence = json.loads(source.read_text(encoding="utf-8"))
    records = evidence.get("planner_records", evidence.get("records"))
    if not isinstance(records, list):
        raise ValueError("replay source does not contain planner_records or records")
    record = records[record_index]
    if record.get("kind") != "single" or record.get("arm") != arm:
        raise ValueError(
            f"selected record must be a single {arm} request; received "
            f"kind={record.get('kind')}, arm={record.get('arm')}"
        )
    return record


def _internal_goal_pose(
    target_pose: list[float],
    robot_pose: list[float],
    frame_bias: list[float],
    arm: str,
) -> list[float]:
    """Apply RoboTwin's world-to-base and Aloha planner-frame transforms."""
    target = np.asarray(target_pose, dtype=np.float64)
    base = np.asarray(robot_pose, dtype=np.float64)
    world_from_base = t3d.quaternions.quat2mat(base[3:])
    world_from_target = t3d.quaternions.quat2mat(target[3:])
    base_position = world_from_base.T @ (target[:3] - base[:3])
    base_rotation = world_from_base.T @ world_from_target
    target_transform = t3d.affines.compose(base_position, base_rotation, [1.0] * 3)
    bias_transform = t3d.affines.compose(frame_bias, np.eye(3), [1.0] * 3)
    angle = -0.02 if arm == "left" else -0.01
    rotation_transform = t3d.affines.compose(
        [0.0] * 3,
        t3d.axangles.axangle2mat([0.0, 0.0, 1.0], angle),
        [1.0] * 3,
    )
    transformed = rotation_transform @ bias_transform @ target_transform
    return transformed[:3, 3].tolist() + t3d.quaternions.mat2quat(transformed[:3, :3]).tolist()


def _case_result(
    motion_gen: MotionGen,
    start_state: JointState,
    goal_pose: Pose,
    name: str,
    legacy_weight: Optional[list[float]],
    max_attempts: int,
) -> dict[str, Any]:
    metric = None
    if legacy_weight is not None:
        metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=torch.tensor(
                legacy_weight,
                device=start_state.position.device,
                dtype=torch.float32,
            ),
        )
    started = time.perf_counter()
    try:
        result = motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(max_attempts=max_attempts, pose_cost_metric=metric),
        )
    except (RuntimeError, ValueError) as error:
        return {
            "case": name,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error),
            "success": False,
        }
    return {
        "case": name,
        "elapsed_seconds": time.perf_counter() - started,
        "success": bool(result.success.item()),
        "trajectory_shape": (
            None
            if result.interpolated_plan is None
            else list(result.interpolated_plan.position.shape)
        ),
    }


def _normalized_seed_result(
    motion_gen: MotionGen,
    start_state: JointState,
    goal_pose: Pose,
    ik_solution: torch.Tensor,
    ik_success: torch.Tensor,
    hold_orientation: bool = True,
) -> dict[str, Any]:
    """Run TrajOpt after mapping revolute IK solutions nearest to the start state."""
    planner = motion_gen._require_planner()
    normalized = _normalize_revolute_seed_config(
        ik_solution,
        start_state,
        planner.kinematics,
    )
    if not torch.any(ik_success):
        return {
            "error": "IK returned no successful seed",
            "input_seed": _tensor_value(ik_solution),
            "normalized_seed": _tensor_value(normalized),
            "normalized_seed_constraints": _collision_constraints(planner, normalized),
            "analytic_self_collision": _analytic_self_collision_summary(planner, normalized),
            "success": False,
        }
    normalized = _replace_failed_seed_config(normalized, ik_success)
    metric = None
    if hold_orientation:
        metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=torch.tensor(
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                device=start_state.position.device,
                dtype=torch.float32,
            ),
        )
    motion_gen._update_pose_metric(metric, start_state, goal_pose)
    started = time.perf_counter()
    result = planner.trajopt_solver.solve_pose(
        motion_gen._goal_from_pose(goal_pose),
        start_state,
        seed_config=normalized,
        use_implicit_goal=True,
    )
    trajectory_constraints = None
    analytic_self_collision = None
    if result.interpolated_trajectory is not None:
        trajectory = result.interpolated_trajectory.reorder(planner.joint_names)
        last_tstep = int(torch.max(result.interpolated_last_tstep).item())
        active_trajectory = trajectory.position[..., : last_tstep + 1, :]
        trajectory_constraints = _collision_constraints(
            planner,
            active_trajectory,
        )
        analytic_self_collision = _analytic_self_collision_summary(
            planner,
            active_trajectory,
        )
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "input_seed": _tensor_value(ik_solution),
        "normalized_seed": _tensor_value(normalized),
        "start_constraints": _collision_constraints(
            planner,
            start_state.position.expand(normalized.shape[1], -1),
        ),
        "normalized_seed_constraints": _collision_constraints(planner, normalized),
        "trajectory_collision_constraints": trajectory_constraints,
        "analytic_self_collision": analytic_self_collision,
        "success": _tensor_value(result.success),
        "feasible": _tensor_value(result.feasible),
        "position_error": _tensor_value(result.position_error),
        "rotation_error": _tensor_value(result.rotation_error),
        "trajectory_shape": (
            None
            if result.interpolated_trajectory is None
            else list(result.interpolated_trajectory.position.shape)
        ),
        "constraints": _constraint_summary(result.metrics),
        "interpolated_constraints": _constraint_summary(result.interpolated_metrics),
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Replay one recorded request and return machine-readable diagnostics."""
    root = args.robotwin_root.resolve()
    os.environ["ASSETS_PATH"] = str(root)
    runtime = validate_rocm_runtime()
    record = _load_record(args.source, args.record_index, args.record_arm or args.arm)
    robot_dir = root / "assets" / "embodiments" / "aloha-agilex"
    embodiment = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
    robot_poses = embodiment["robot_pose"]
    robot_pose = robot_poses[0 if len(robot_poses) == 1 or args.arm == "left" else 1]
    robot_config_path = robot_dir / f"curobo_{args.arm}.yml"
    robot_config = yaml.safe_load(robot_config_path.read_text(encoding="utf-8"))
    frame_bias = robot_config["planner"]["frame_bias"]
    internal_goal = _internal_goal_pose(record["target_pose"], robot_pose, frame_bias, args.arm)
    qpos_indices = _ALOHA_ACTIVE_QPOS_INDICES[args.arm]
    active_qpos = [record["current_qpos"][index] for index in qpos_indices]
    device = torch.device("cuda")
    world_config = {
        "cuboid": {
            "table": {
                "dims": [0.7, 2.0, 0.04],
                "pose": [robot_pose[1], 0.0, 0.74 - robot_pose[2], 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config(
            str(robot_config_path),
            world_config,
            interpolation_dt=1.0 / 250.0,
            num_ik_seeds=args.num_ik_seeds,
            num_trajopt_seeds=args.num_trajopt_seeds,
        )
    )
    warmup_started = time.perf_counter()
    warmup_success = motion_gen.warmup()
    warmup_seconds = time.perf_counter() - warmup_started
    planner = motion_gen._require_planner()
    if planner.graph_planner is not None:
        if args.graph_iterations is not None:
            planner.graph_planner.config.max_path_finding_iterations = args.graph_iterations
        if args.graph_new_nodes is not None:
            planner.graph_planner.config.new_nodes_per_iteration = args.graph_new_nodes
        if args.graph_neighbors is not None:
            planner.graph_planner.config.neighbors_per_node = args.graph_neighbors
        if any(
            value is not None
            for value in (args.graph_iterations, args.graph_new_nodes, args.graph_neighbors)
        ):
            planner.graph_planner.reset_buffer()
    graph_target_count = 1 if args.graph_iterations is not None else 4
    start_state = JointState.from_position(
        torch.tensor(active_qpos, device=device, dtype=torch.float32).reshape(1, -1),
        joint_names=planner.joint_names,
    )
    goal_pose = Pose.from_list(internal_goal)
    hold_orientation = args.record_arm is None or args.record_arm == args.arm
    tool_frame = planner.tool_frames[0]
    start_pose = planner.compute_kinematics(start_state).tool_poses[tool_frame]
    continuation_ik = _continuation_ik_result(
        motion_gen,
        start_state,
        start_pose,
        goal_pose,
    )
    ik_result = planner.ik_solver.solve_pose(
        motion_gen._goal_from_pose(goal_pose),
        return_seeds=planner.trajopt_solver.config.num_seeds,
        current_state=start_state,
    )
    normalized_seed = _normalized_seed_result(
        motion_gen,
        start_state,
        goal_pose,
        ik_result.solution,
        ik_result.success,
        hold_orientation=hold_orientation,
    )
    normalized_seed_tensor = _normalize_revolute_seed_config(
        ik_result.solution,
        start_state,
        planner.kinematics,
    )
    if torch.any(ik_result.success):
        normalized_seed_tensor = _replace_failed_seed_config(
            normalized_seed_tensor,
            ik_result.success,
        )
    graph_seed_results = _graph_seed_results(
        motion_gen,
        start_state,
        normalized_seed_tensor,
        ik_result.success,
        maximum_targets=graph_target_count,
    )
    refined_ik_result = planner.ik_solver.solve_pose(
        motion_gen._goal_from_pose(goal_pose),
        seed_config=normalized_seed_tensor,
        return_seeds=planner.trajopt_solver.config.num_seeds,
        current_state=start_state,
    )
    refined_normalized_seed_tensor = _normalize_revolute_seed_config(
        refined_ik_result.solution,
        start_state,
        planner.kinematics,
    )
    refined_normalized_seed = _normalized_seed_result(
        motion_gen,
        start_state,
        goal_pose,
        refined_ik_result.solution,
        refined_ik_result.success,
        hold_orientation=hold_orientation,
    )
    if torch.any(refined_ik_result.success):
        refined_normalized_seed_tensor = _replace_failed_seed_config(
            refined_normalized_seed_tensor,
            refined_ik_result.success,
        )
    refined_graph_seed_results = _graph_seed_results(
        motion_gen,
        start_state,
        refined_normalized_seed_tensor,
        refined_ik_result.success,
        maximum_targets=graph_target_count,
    )
    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(0)
    local_ik_seeds = args.local_ik_seeds or args.num_ik_seeds
    local_seed = start_state.position.unsqueeze(1).repeat(1, local_ik_seeds, 1)
    local_noise = torch.randn(
        local_seed.shape,
        generator=local_generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=local_seed.device)
    local_seed = local_seed + 0.35 * local_noise
    position_limits = planner.kinematics.kinematics_config.joint_limits.position.to(
        device=local_seed.device
    )
    local_seed = torch.clamp(
        local_seed,
        min=position_limits[0].view(1, 1, -1),
        max=position_limits[1].view(1, 1, -1),
    )
    local_ik_result = planner.ik_solver.solve_pose(
        motion_gen._goal_from_pose(goal_pose),
        seed_config=local_seed,
        return_seeds=planner.trajopt_solver.config.num_seeds,
        current_state=start_state,
    )
    local_normalized_seed = _normalized_seed_result(
        motion_gen,
        start_state,
        goal_pose,
        local_ik_result.solution,
        local_ik_result.success,
        hold_orientation=hold_orientation,
    )
    local_chunk_results: list[dict[str, Any]] = []
    for chunk_index in range(16):
        chunk_seed = _make_local_ik_seed_config(
            start_state,
            planner.kinematics,
            planner.ik_solver.config.num_seeds,
            chunk_index,
        )
        chunk_result = planner.ik_solver.solve_pose(
            motion_gen._goal_from_pose(goal_pose),
            seed_config=chunk_seed,
            return_seeds=planner.trajopt_solver.config.num_seeds,
            current_state=start_state,
        )
        local_chunk_results.append(
            {
                "chunk_index": chunk_index,
                "success": _tensor_value(chunk_result.success),
                "position_error": _tensor_value(chunk_result.position_error),
                "rotation_error": _tensor_value(chunk_result.rotation_error),
            }
        )
        if torch.any(chunk_result.success):
            break
    collision_escape_results: list[dict[str, Any]] = []
    collision_escape_seed_result: Optional[dict[str, Any]] = None
    escape_center = local_ik_result.solution[:, :1].detach()
    for scale_index, noise_scale in enumerate((0.01, 0.03, 0.05, 0.1, 0.2, 0.5)):
        best_escape_result = None
        best_escape_score = None
        for chunk_index in range(16):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1000 + scale_index)
            noise = torch.randn(
                (
                    escape_center.shape[0],
                    (chunk_index + 1) * planner.ik_solver.config.num_seeds,
                    escape_center.shape[-1],
                ),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )[:, -planner.ik_solver.config.num_seeds :].to(
                device=escape_center.device,
                dtype=escape_center.dtype,
            )
            escape_seed = torch.clamp(
                escape_center + noise_scale * noise,
                min=position_limits[0].view(1, 1, -1),
                max=position_limits[1].view(1, 1, -1),
            )
            escape_result = planner.ik_solver.solve_pose(
                motion_gen._goal_from_pose(goal_pose),
                seed_config=escape_seed,
                return_seeds=planner.trajopt_solver.config.num_seeds,
                current_state=start_state,
            )
            collision_escape_results.append(
                {
                    "noise_scale": noise_scale,
                    "chunk_index": chunk_index,
                    "success": _tensor_value(escape_result.success),
                    "position_error": _tensor_value(escape_result.position_error),
                    "rotation_error": _tensor_value(escape_result.rotation_error),
                }
            )
            if torch.any(escape_result.success):
                score = float(
                    torch.min(escape_result.position_error + escape_result.rotation_error).item()
                )
                if best_escape_score is None or score < best_escape_score:
                    best_escape_score = score
                    best_escape_result = escape_result
        if best_escape_result is not None:
            collision_escape_seed_result = _normalized_seed_result(
                motion_gen,
                start_state,
                goal_pose,
                best_escape_result.solution,
                best_escape_result.success,
                hold_orientation=False,
            )
            break
    tolerance_goal_results: list[dict[str, Any]] = []
    tolerance_offsets = (
        (0.004, 0.0, 0.0),
        (-0.004, 0.0, 0.0),
        (0.0, 0.004, 0.0),
        (0.0, -0.004, 0.0),
        (0.0, 0.0, 0.004),
        (0.0, 0.0, -0.004),
    )
    for offset in tolerance_offsets:
        offset_tensor = torch.tensor(
            offset,
            device=goal_pose.position.device,
            dtype=goal_pose.position.dtype,
        ).view(1, 3)
        offset_goal = Pose(goal_pose.position + offset_tensor, goal_pose.quaternion.clone())
        started = time.perf_counter()
        offset_result = motion_gen.plan_single(
            start_state,
            offset_goal,
            MotionGenPlanConfig(max_attempts=1),
        )
        endpoint_position_error = None
        endpoint_rotation_error = None
        if offset_result.success.item() and offset_result.interpolated_plan is not None:
            endpoint_state = JointState.from_position(
                offset_result.interpolated_plan.position[-1:].clone(),
                joint_names=planner.joint_names,
            )
            endpoint_pose = planner.compute_kinematics(endpoint_state).tool_poses[tool_frame]
            endpoint_position_error = float(
                torch.linalg.vector_norm(endpoint_pose.position - goal_pose.position).item()
            )
            endpoint_rotation_error = float(endpoint_pose.angular_distance(goal_pose).item())
        tolerance_goal_results.append(
            {
                "offset": list(offset),
                "elapsed_seconds": time.perf_counter() - started,
                "success": bool(offset_result.success.item()),
                "endpoint_position_error": endpoint_position_error,
                "endpoint_rotation_error": endpoint_rotation_error,
                "trajectory_shape": (
                    None
                    if offset_result.interpolated_plan is None
                    else list(offset_result.interpolated_plan.position.shape)
                ),
            }
        )
        if (
            offset_result.success.item()
            and endpoint_position_error is not None
            and endpoint_position_error <= 0.005
        ):
            break
    if not any(result["success"] for result in tolerance_goal_results):
        for axis_index in range(3):
            for angle in (0.04, -0.04):
                half_angle = 0.5 * angle
                delta_quaternion = torch.zeros(
                    (1, 4),
                    device=goal_pose.quaternion.device,
                    dtype=goal_pose.quaternion.dtype,
                )
                delta_quaternion[:, 0] = torch.cos(
                    torch.tensor(half_angle, device=delta_quaternion.device)
                )
                delta_quaternion[:, axis_index + 1] = torch.sin(
                    torch.tensor(half_angle, device=delta_quaternion.device)
                )
                left_w = delta_quaternion[:, :1]
                left_xyz = delta_quaternion[:, 1:]
                right_w = goal_pose.quaternion[:, :1]
                right_xyz = goal_pose.quaternion[:, 1:]
                offset_quaternion = torch.cat(
                    (
                        left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True),
                        left_w * right_xyz
                        + right_w * left_xyz
                        + torch.linalg.cross(left_xyz, right_xyz),
                    ),
                    dim=-1,
                )
                offset_goal = Pose(goal_pose.position.clone(), offset_quaternion)
                started = time.perf_counter()
                offset_result = motion_gen.plan_single(
                    start_state,
                    offset_goal,
                    MotionGenPlanConfig(max_attempts=1),
                )
                endpoint_position_error = None
                endpoint_rotation_error = None
                if offset_result.success.item() and offset_result.interpolated_plan is not None:
                    endpoint_state = JointState.from_position(
                        offset_result.interpolated_plan.position[-1:].clone(),
                        joint_names=planner.joint_names,
                    )
                    endpoint_pose = planner.compute_kinematics(endpoint_state).tool_poses[
                        tool_frame
                    ]
                    endpoint_position_error = float(
                        torch.linalg.vector_norm(
                            endpoint_pose.position - goal_pose.position
                        ).item()
                    )
                    endpoint_rotation_error = float(
                        endpoint_pose.angular_distance(goal_pose).item()
                    )
                tolerance_goal_results.append(
                    {
                        "rotation_axis": axis_index,
                        "rotation_angle": angle,
                        "elapsed_seconds": time.perf_counter() - started,
                        "success": bool(offset_result.success.item()),
                        "endpoint_position_error": endpoint_position_error,
                        "endpoint_rotation_error": endpoint_rotation_error,
                        "trajectory_shape": (
                            None
                            if offset_result.interpolated_plan is None
                            else list(offset_result.interpolated_plan.position.shape)
                        ),
                    }
                )
                if (
                    offset_result.success.item()
                    and endpoint_position_error is not None
                    and endpoint_position_error <= 0.005
                    and endpoint_rotation_error is not None
                    and endpoint_rotation_error <= 0.05
                ):
                    break
            if tolerance_goal_results[-1]["success"]:
                break
    tolerance_goalset_result: dict[str, Any] = {"success": False}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)
    goalset_size = args.tolerance_goalset_size
    if goalset_size < 1:
        raise ValueError("tolerance_goalset_size must be positive")
    position_directions = torch.randn(
        (goalset_size, 3),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    position_directions = position_directions / torch.linalg.vector_norm(
        position_directions, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    position_radii = 0.004 * torch.rand(
        (goalset_size, 1), generator=generator, device="cpu", dtype=torch.float32
    ).pow(1.0 / 3.0)
    position_offsets = (position_directions * position_radii).to(
        device=goal_pose.position.device,
        dtype=goal_pose.position.dtype,
    )
    rotation_axes = torch.randn(
        (goalset_size, 3),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    rotation_axes = rotation_axes / torch.linalg.vector_norm(
        rotation_axes, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    rotation_angles = 0.04 * torch.rand(
        (goalset_size, 1), generator=generator, device="cpu", dtype=torch.float32
    )
    delta_quaternions = torch.cat(
        (
            torch.cos(0.5 * rotation_angles),
            rotation_axes * torch.sin(0.5 * rotation_angles),
        ),
        dim=-1,
    ).to(device=goal_pose.quaternion.device, dtype=goal_pose.quaternion.dtype)
    goal_quaternions = goal_pose.quaternion.expand(goalset_size, -1)
    left_w = delta_quaternions[:, :1]
    left_xyz = delta_quaternions[:, 1:]
    right_w = goal_quaternions[:, :1]
    right_xyz = goal_quaternions[:, 1:]
    goalset_quaternions = torch.cat(
        (
            left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True),
            left_w * right_xyz
            + right_w * left_xyz
            + torch.linalg.cross(left_xyz, right_xyz),
        ),
        dim=-1,
    ).unsqueeze(0)
    goalset_pose = Pose(
        goal_pose.position + position_offsets,
        goalset_quaternions.reshape(-1, 4),
    )
    goalset_start = JointState.from_position(
        start_state.position.repeat(goalset_size, 1),
        joint_names=planner.joint_names,
    )
    tolerance_motion_gen = MotionGen(motion_gen.config)
    goalset_warmup_started = time.perf_counter()
    goalset_warmup_success = tolerance_motion_gen.warmup(batch=goalset_size)
    goalset_warmup_seconds = time.perf_counter() - goalset_warmup_started
    started = time.perf_counter()
    goalset_plan = tolerance_motion_gen.plan_batch(
        goalset_start,
        goalset_pose,
        MotionGenPlanConfig(max_attempts=args.max_attempts),
    )
    tolerance_goalset_result = {
        "goalset_size": goalset_size,
        "max_attempts": args.max_attempts,
        "maximum_position_offset": float(
            torch.linalg.vector_norm(position_offsets, dim=-1).max().item()
        ),
        "maximum_rotation_offset": float(rotation_angles.max().item()),
        "warmup_success": goalset_warmup_success,
        "warmup_seconds": goalset_warmup_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "success": bool(torch.any(goalset_plan.success).item()),
    }
    if (
        torch.any(goalset_plan.success)
        and goalset_plan.interpolated_plan is not None
    ):
        selected = int(torch.nonzero(goalset_plan.success, as_tuple=False)[0].item())
        last_tstep = goalset_plan.interpolated_plan.position.shape[-2]
        if goalset_plan.interpolated_last_tstep is not None:
            last_tstep = int(goalset_plan.interpolated_last_tstep[selected].item())
        endpoint_state = JointState.from_position(
            goalset_plan.interpolated_plan.position[selected, last_tstep - 1].reshape(1, -1),
            joint_names=planner.joint_names,
        )
        endpoint_pose = planner.compute_kinematics(endpoint_state).tool_poses[tool_frame]
        active_trajectory = goalset_plan.interpolated_plan.position[
            selected : selected + 1, :last_tstep
        ]
        tolerance_goalset_result.update(
            {
                "selected": selected,
                "last_tstep": last_tstep,
                "endpoint_position_error": float(
                    torch.linalg.vector_norm(endpoint_pose.position - goal_pose.position).item()
                ),
                "endpoint_rotation_error": float(
                    endpoint_pose.angular_distance(goal_pose).item()
                ),
                "trajectory_collision_constraints": _collision_constraints(
                    planner, active_trajectory
                ),
            }
        )
    cases = [
        _case_result(motion_gen, start_state, goal_pose, "unconstrained", None, args.max_attempts),
        _case_result(
            motion_gen,
            start_state,
            goal_pose,
            "hold_orientation",
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            args.max_attempts,
        ),
        _case_result(
            motion_gen,
            start_state,
            goal_pose,
            "hold_position_control",
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            args.max_attempts,
        ),
    ]
    return {
        "accelerator": runtime.accelerator,
        "backend": runtime.backend,
        "device_name": runtime.device_name,
        "gfx_arch": runtime.gfx_arch,
        "source": str(args.source.resolve()),
        "arm": args.arm,
        "source_record_arm": args.record_arm or args.arm,
        "num_ik_seeds": args.num_ik_seeds,
        "local_ik_seeds": local_ik_seeds,
        "num_trajopt_seeds": args.num_trajopt_seeds,
        "graph_config_override": {
            "max_path_finding_iterations": args.graph_iterations,
            "new_nodes_per_iteration": args.graph_new_nodes,
            "neighbors_per_node": args.graph_neighbors,
        },
        "qpos_indices": qpos_indices,
        "active_qpos": active_qpos,
        "incoming_target_pose": record["target_pose"],
        "internal_goal_pose": internal_goal,
        "start_pose": {
            "position": _tensor_value(start_pose.position),
            "quaternion": _tensor_value(start_pose.quaternion),
            "goal_angular_distance": _tensor_value(start_pose.angular_distance(goal_pose)),
        },
        "continuation_ik": continuation_ik,
        "warmup_success": warmup_success,
        "warmup_seconds": warmup_seconds,
        "ik": {
            "success": _tensor_value(ik_result.success),
            "feasible": _tensor_value(ik_result.feasible),
            "position_error": _tensor_value(ik_result.position_error),
            "rotation_error": _tensor_value(ik_result.rotation_error),
            "solution": _tensor_value(ik_result.solution),
        },
        "refined_ik": {
            "success": _tensor_value(refined_ik_result.success),
            "feasible": _tensor_value(refined_ik_result.feasible),
            "position_error": _tensor_value(refined_ik_result.position_error),
            "rotation_error": _tensor_value(refined_ik_result.rotation_error),
            "solution": _tensor_value(refined_ik_result.solution),
            "graph_seed_results": refined_graph_seed_results,
        },
        "local_ik": {
            "success": _tensor_value(local_ik_result.success),
            "feasible": _tensor_value(local_ik_result.feasible),
            "position_error": _tensor_value(local_ik_result.position_error),
            "rotation_error": _tensor_value(local_ik_result.rotation_error),
            "solution": _tensor_value(local_ik_result.solution),
            "normalized_ik_seed_trajopt": local_normalized_seed,
            "chunk_results": local_chunk_results,
        },
        "collision_escape_ik": {
            "center": _tensor_value(escape_center),
            "attempts": collision_escape_results,
            "normalized_ik_seed_trajopt": collision_escape_seed_result,
        },
        "tolerance_goal_scan": tolerance_goal_results,
        "tolerance_goalset": tolerance_goalset_result,
        "normalized_ik_seed_trajopt": normalized_seed,
        "refined_normalized_ik_seed_trajopt": refined_normalized_seed,
        "graph_seed_results": graph_seed_results,
        "cases": cases,
    }


def main() -> None:
    """Run the replay and save its acceptance evidence."""
    args = _parse_args()
    evidence = run_replay(args)
    output = json.dumps(evidence, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
