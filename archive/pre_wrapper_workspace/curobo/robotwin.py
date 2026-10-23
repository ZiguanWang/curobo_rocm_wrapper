# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""RoboTwin compatibility adapter for the current cuRobo planning API.

RoboTwin currently consumes the small subset of the historical ``MotionGen``
API represented here.  The adapter intentionally contains no ROCm-specific
implementation; accelerator selection remains inside cuRobo's runtime and
kernel backend layers.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.curobolib.backends import get_backend_name
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.runtime import AcceleratorType, detect_accelerator
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.config_io import join_path, resolve_config
from curobo._src.util.logging import log_and_raise
from curobo.content import get_robot_configs_path

RobotConfig = Union[str, Dict[str, Any]]

_LEGACY_KINEMATICS_FIELDS = (
    "isaac_usd_path",
    "usd_flip_joint_limits",
    "usd_flip_joints",
    "usd_path",
    "usd_robot_root",
    "use_usd_kinematics",
)


def _normalize_robot_config(robot: RobotConfig) -> Dict[str, Any]:
    """Translate the RoboTwin-pinned v0.7.8 robot schema to the current schema."""
    resolved = resolve_config(join_path(get_robot_configs_path(), robot))
    if not isinstance(resolved, dict):
        log_and_raise("RoboTwin robot configuration must resolve to a dictionary")
    normalized = deepcopy(resolved)
    robot_data = normalized.get("robot_cfg", normalized)
    kinematics = robot_data.get("kinematics")
    if not isinstance(kinematics, dict):
        log_and_raise("RoboTwin robot configuration is missing robot_cfg.kinematics")

    for field_name in _LEGACY_KINEMATICS_FIELDS:
        kinematics.pop(field_name, None)
    legacy_tool_frame = kinematics.pop("ee_link", None)
    if "tool_frames" not in kinematics and legacy_tool_frame is not None:
        kinematics["tool_frames"] = [legacy_tool_frame]
    if kinematics.get("asset_root_path") is None:
        kinematics["asset_root_path"] = ""
    for path_field in ("asset_root_path", "collision_spheres", "urdf_path"):
        path_value = kinematics.get(path_field)
        if isinstance(path_value, str):
            kinematics[path_field] = os.path.expandvars(path_value)

    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict) and "retract_config" in cspace:
        cspace.setdefault("default_joint_position", cspace.pop("retract_config"))
    return normalized


def _convert_legacy_hold_weight(weight: torch.Tensor) -> list[float]:
    """Convert v0.7.8 ``[rotation, position]`` axes to current ordering."""
    legacy_weight = weight.detach().view(-1)
    return torch.cat((legacy_weight[3:], legacy_weight[:3])).tolist()


@dataclass(frozen=True)
class RoboTwinRuntimeInfo:
    """Fail-closed accelerator information emitted before RoboTwin starts."""

    accelerator: str
    backend: str
    torch_version: str
    rocm_version: str
    cuda_version: str
    device_name: str
    gfx_arch: str


def validate_robotwin_runtime(
    required_accelerator: Optional[str] = None,
) -> RoboTwinRuntimeInfo:
    """Require a visible supported GPU and its matching cuRobo backend.

    Args:
        required_accelerator: Optional explicit ``"rocm"`` or ``"cuda"``
            requirement. When omitted, either supported GPU accelerator is
            accepted.

    Returns:
        Runtime metadata suitable for recording in an evaluation log.
    """
    accelerator = detect_accelerator()
    if required_accelerator not in (None, "rocm", "cuda"):
        log_and_raise(
            "required_accelerator must be one of: rocm, cuda",
            exception_type=RuntimeError,
        )
    if accelerator not in (AcceleratorType.ROCM, AcceleratorType.CUDA):
        log_and_raise(
            f"RoboTwin evaluation requires a supported GPU, got {accelerator.value}",
            exception_type=RuntimeError,
        )
    if required_accelerator is not None and accelerator.value != required_accelerator:
        log_and_raise(
            f"RoboTwin evaluation requires accelerator={required_accelerator}, "
            f"got {accelerator.value}",
            exception_type=RuntimeError,
        )
    implementation_backend = get_backend_name()
    expected_backends = (
        {"hip"} if accelerator == AcceleratorType.ROCM else {"cuda_core", "pybind"}
    )
    if implementation_backend not in expected_backends:
        log_and_raise(
            f"RoboTwin {accelerator.value} evaluation requires cuRobo "
            f"backend in {sorted(expected_backends)}, got {implementation_backend}",
            exception_type=RuntimeError,
        )
    backend = "hip" if accelerator == AcceleratorType.ROCM else "cuda"
    rocm_version = ""
    cuda_version = ""
    gfx_arch = ""
    if accelerator == AcceleratorType.ROCM:
        from curobo._src.curobolib.backends.hip_backend.runtime_preflight import (
            inspect_runtime as inspect_hip_runtime,
        )
        from curobo._src.curobolib.backends.hip_backend.runtime_preflight import (
            validate_runtime as validate_hip_runtime,
        )

        hip_info = inspect_hip_runtime()
        validate_hip_runtime(hip_info, require_build_arch=False)
        rocm_version = hip_info.rocm_version
        device_name = hip_info.device_name
        gfx_arch = hip_info.gfx_arch
    else:
        cuda_version_value = getattr(getattr(torch, "version", None), "cuda", None)
        if not cuda_version_value or not torch.cuda.is_available():
            log_and_raise(
                "RoboTwin CUDA evaluation requires a visible CUDA PyTorch GPU",
                exception_type=RuntimeError,
            )
        cuda_version = str(cuda_version_value)
        device_name = str(torch.cuda.get_device_name(0))
    return RoboTwinRuntimeInfo(
        accelerator=accelerator.value,
        backend=backend,
        torch_version=str(torch.__version__),
        rocm_version=rocm_version,
        cuda_version=cuda_version,
        device_name=device_name,
        gfx_arch=gfx_arch,
    )


def validate_rocm_runtime() -> RoboTwinRuntimeInfo:
    """Require a visible ROCm GPU and the cuRobo HIP kernel backend."""
    return validate_robotwin_runtime(required_accelerator="rocm")


def validate_cuda_runtime() -> RoboTwinRuntimeInfo:
    """Require a visible NVIDIA GPU and the cuRobo CUDA kernel backend."""
    return validate_robotwin_runtime(required_accelerator="cuda")


@dataclass
class PoseCostMetric:
    """Pose constraint fields used by RoboTwin's planner calls."""

    hold_partial_pose: bool = False
    hold_vec_weight: Optional[torch.Tensor] = None


@dataclass
class MotionGenPlanConfig:
    """Per-request options used by RoboTwin."""

    max_attempts: int = 5
    pose_cost_metric: Optional[PoseCostMetric] = None


@dataclass
class MotionGenResult:
    """RoboTwin-facing planning result."""

    success: torch.Tensor
    interpolated_plan: Optional[JointState] = None
    interpolated_last_tstep: Optional[torch.Tensor] = None
    selected_goal_index: Optional[int] = None
    goal_position_error: Optional[float] = None
    goal_rotation_error: Optional[float] = None


@dataclass
class MotionGenConfig:
    """Deferred planner configuration used by :class:`MotionGen`."""

    robot: RobotConfig
    world_config: Optional[Dict[str, Any]]
    interpolation_dt: float
    interpolation_buffer_size: int
    num_ik_seeds: int
    num_trajopt_seeds: int
    device_cfg: DeviceCfg

    @classmethod
    def load_from_robot_config(
        cls,
        robot: RobotConfig,
        world_config: Optional[Dict[str, Any]] = None,
        *,
        interpolation_dt: float = 0.025,
        interpolation_buffer_size: int = 5000,
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 1,
        num_graph_seeds: int = 1,
        device_cfg: DeviceCfg = DeviceCfg(),
        **_: Any,
    ) -> MotionGenConfig:
        """Create the RoboTwin subset of the historical MotionGen config.

        Args:
            robot: Robot YAML path or parsed robot configuration.
            world_config: Scene dictionary using cuRobo's obstacle schema.
            interpolation_dt: Dense output trajectory timestep in seconds.
            interpolation_buffer_size: Maximum dense trajectory length. The
                default accommodates RoboTwin's 250 Hz execution trajectories.
            num_ik_seeds: Number of inverse-kinematics seeds per target.
            num_trajopt_seeds: Number of trajectory-optimization seeds.
            num_graph_seeds: Historical graph seed count. RoboTwin uses one;
                other values are rejected because the current planner owns
                graph seed generation internally.
            device_cfg: Tensor device and dtype configuration.

        Returns:
            Deferred configuration that is sized during ``warmup``.
        """
        if num_graph_seeds != 1:
            log_and_raise(
                f"RoboTwin adapter supports num_graph_seeds=1; received {num_graph_seeds}"
            )
        if interpolation_buffer_size < 1:
            log_and_raise(
                f"interpolation_buffer_size must be positive; received {interpolation_buffer_size}"
            )
        if num_ik_seeds < 1:
            log_and_raise(f"num_ik_seeds must be positive; received {num_ik_seeds}")
        return cls(
            robot=_normalize_robot_config(robot),
            world_config=world_config,
            interpolation_dt=interpolation_dt,
            interpolation_buffer_size=interpolation_buffer_size,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            device_cfg=device_cfg,
        )


class MotionGen:
    """Compatibility facade backed by current motion planner classes."""

    def __init__(self, config: MotionGenConfig) -> None:
        """Create a deferred planner from a RoboTwin compatibility config."""
        self.config = config
        self.tensor_args = config.device_cfg
        self._planner: Optional[Union[MotionPlanner, BatchMotionPlanner]] = None
        self._batch_size = 0
        self._cached_batch_start: Optional[torch.Tensor] = None
        self._cached_batch_goal: Optional[Pose] = None
        self._cached_batch_result: Optional[MotionGenResult] = None
        self._tolerance_planners: dict[int, MotionGen] = {}

    def _create_planner(self, batch_size: int) -> None:
        planner_cfg = MotionPlannerCfg.create(
            robot=self.config.robot,
            scene_model=self.config.world_config,
            device_cfg=self.config.device_cfg,
            num_ik_seeds=self.config.num_ik_seeds,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            interpolation_dt=self.config.interpolation_dt,
            interpolation_buffer_size=self.config.interpolation_buffer_size,
            max_batch_size=batch_size,
        )
        self._planner = (
            MotionPlanner(planner_cfg) if batch_size == 1 else BatchMotionPlanner(planner_cfg)
        )
        self._batch_size = batch_size

    def warmup(self, *, batch: Optional[int] = None) -> bool:
        """Allocate and warm a single or fixed-size batch planner.

        RoboTwin batch requests use batched IK and trajectory optimization,
        without graph fallback.  The batch path therefore avoids building a
        PRM roadmap that is not consumed by :meth:`plan_batch`.  Single
        requests retain the normal graph-planner warm-up.
        """
        batch_size = 1 if batch is None else batch
        if batch_size < 1:
            log_and_raise(f"batch must be positive, received {batch_size}")
        if self._planner is None or self._batch_size != batch_size:
            self._create_planner(batch_size)
        planner = self._require_planner()
        if isinstance(planner, BatchMotionPlanner):
            return planner.warmup(enable_graph=False)
        return planner.warmup()

    def _require_planner(self) -> Union[MotionPlanner, BatchMotionPlanner]:
        if self._planner is None:
            log_and_raise("MotionGen.warmup() must be called before planning")
        return self._planner

    def interpolate_joint_action(
        self,
        start_state: JointState,
        goal_state: JointState,
        *,
        control_dt: float = 1.0 / 250.0,
    ) -> JointState:
        """Interpolate one joint target on the planner device without MPLib.

        The cubic time law starts and ends at zero velocity. Its duration is
        selected from the robot velocity and acceleration limits, so every
        generated waypoint remains within the planner's active-joint bounds.

        Args:
            start_state: Current active-joint position with shape ``[1, dof]``.
            goal_state: Target active-joint position with shape ``[1, dof]``.
            control_dt: Simulator control period in seconds.

        Returns:
            Device-resident positions and velocities with shape ``[steps, dof]``.
        """
        if control_dt <= 0.0:
            log_and_raise("control_dt must be greater than zero")
        planner = self._require_planner()
        if start_state.position.shape != goal_state.position.shape:
            log_and_raise(
                "joint-action start and goal shapes must match; received "
                f"{start_state.position.shape} and {goal_state.position.shape}"
            )
        if start_state.position.ndim != 2 or start_state.position.shape[0] != 1:
            log_and_raise("joint-action interpolation requires shape [1, dof]")
        if start_state.joint_names != planner.joint_names:
            start_state = start_state.reorder(planner.joint_names)
        if goal_state.joint_names != planner.joint_names:
            goal_state = goal_state.reorder(planner.joint_names)

        limits = planner.kinematics.get_joint_limits()
        start = start_state.position[0]
        goal = goal_state.position[0]
        if torch.any(goal < limits.position[0]) or torch.any(goal > limits.position[1]):
            log_and_raise("joint-action target exceeds robot position limits")

        delta = goal - start
        max_velocity = torch.minimum(torch.abs(limits.velocity[0]), limits.velocity[1])
        max_acceleration = torch.minimum(
            torch.abs(limits.acceleration[0]), limits.acceleration[1]
        )
        if torch.any(max_velocity <= 0.0) or torch.any(max_acceleration <= 0.0):
            log_and_raise("joint-action interpolation requires positive motion limits")

        velocity_duration = torch.max(1.5 * torch.abs(delta) / max_velocity)
        acceleration_duration = torch.max(
            torch.sqrt(6.0 * torch.abs(delta) / max_acceleration)
        )
        duration = max(float(velocity_duration.item()), float(acceleration_duration.item()))
        steps = max(2, math.ceil(duration / control_dt) + 1)
        actual_duration = (steps - 1) * control_dt
        phase = torch.linspace(
            0.0,
            1.0,
            steps,
            device=start.device,
            dtype=start.dtype,
        ).unsqueeze(-1)
        blend = phase.square() * (3.0 - 2.0 * phase)
        blend_velocity = 6.0 * phase * (1.0 - phase) / actual_duration
        return JointState(
            position=start.unsqueeze(0) + blend * delta.unsqueeze(0),
            velocity=blend_velocity * delta.unsqueeze(0),
            joint_names=planner.joint_names,
            dt=torch.tensor(control_dt, device=start.device, dtype=start.dtype),
        )

    def _goal_from_pose(self, goal_pose: Pose) -> GoalToolPose:
        planner = self._require_planner()
        tool_frame = planner.tool_frames[0]
        return GoalToolPose.from_poses(
            {tool_frame: goal_pose},
            ordered_tool_frames=[tool_frame],
        )

    def _cache_batch_plan(
        self,
        start_state: JointState,
        goal_pose: Pose,
        result: MotionGenResult,
        request: MotionGenPlanConfig,
    ) -> None:
        """Cache unconstrained successful batch trajectories for exact later reuse."""
        if (
            request.pose_cost_metric is not None
            or result.interpolated_plan is None
            or not torch.any(result.success)
        ):
            return
        self._cached_batch_start = start_state.position.detach().clone()
        self._cached_batch_goal = goal_pose.detach().clone()
        self._cached_batch_result = MotionGenResult(
            success=result.success.detach().clone(),
            interpolated_plan=result.interpolated_plan.detach().clone(),
            interpolated_last_tstep=(
                None
                if result.interpolated_last_tstep is None
                else result.interpolated_last_tstep.detach().clone()
            ),
        )

    def _reuse_cached_batch_plan(
        self,
        start_state: JointState,
        goal_pose: Pose,
        request: MotionGenPlanConfig,
    ) -> Optional[MotionGenResult]:
        """Return a cached successful slot matching the same start and goal."""
        cached = self._cached_batch_result
        cached_start = self._cached_batch_start
        cached_goal = self._cached_batch_goal
        if (
            request.pose_cost_metric is not None
            or cached is None
            or cached_start is None
            or cached_goal is None
            or cached.interpolated_plan is None
        ):
            return None
        start_matches = torch.all(
            torch.isclose(
                cached_start,
                start_state.position.expand_as(cached_start),
                atol=1.0e-5,
                rtol=0.0,
            ),
            dim=-1,
        )
        position_matches = torch.linalg.vector_norm(
            cached_goal.position - goal_pose.position,
            dim=-1,
        ) <= 1.0e-5
        quaternion_dot = torch.abs(
            torch.sum(cached_goal.quaternion * goal_pose.quaternion, dim=-1)
        )
        orientation_matches = quaternion_dot >= 1.0 - 1.0e-6
        matching = torch.nonzero(
            cached.success & start_matches & position_matches & orientation_matches,
            as_tuple=False,
        ).view(-1)
        if matching.numel() == 0:
            return None

        selected = int(matching[0].item())
        trajectory = cached.interpolated_plan
        last_tstep = trajectory.position.shape[-2] - 1
        if cached.interpolated_last_tstep is not None:
            last_tstep = int(cached.interpolated_last_tstep[selected].item())

        def select(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value[selected, : last_tstep + 1].clone()

        selected_dt = None
        if trajectory.dt is not None:
            selected_dt = trajectory.dt[selected].clone()

        selected_trajectory = JointState(
            position=trajectory.position[selected, : last_tstep + 1].clone(),
            velocity=select(trajectory.velocity),
            acceleration=select(trajectory.acceleration),
            jerk=select(trajectory.jerk),
            joint_names=trajectory.joint_names,
            dt=selected_dt,
            control_space=trajectory.control_space,
        )
        return MotionGenResult(
            success=cached.success[selected : selected + 1].clone(),
            interpolated_plan=selected_trajectory,
            interpolated_last_tstep=torch.tensor(
                [last_tstep],
                device=cached.success.device,
                dtype=torch.long,
            ),
        )

    def _update_pose_metric(
        self,
        metric: Optional[PoseCostMetric],
        start_state: JointState,
        goal_pose: Pose,
    ) -> None:
        planner = self._require_planner()
        if metric is not None and metric.hold_partial_pose:
            if metric.hold_vec_weight is None or metric.hold_vec_weight.numel() != 6:
                log_and_raise("hold_partial_pose requires hold_vec_weight with six xyz/rpy values")
            legacy_weight = metric.hold_vec_weight.detach().view(-1)
            tool_frame = planner.tool_frames[0]
            start_pose = planner.compute_kinematics(start_state).tool_poses[tool_frame]
            if torch.any(legacy_weight[:3] > 0.0):
                orientation_error = start_pose.angular_distance(goal_pose)
                if torch.max(orientation_error) > 0.05:
                    log_and_raise(
                        "RoboTwin partial-pose orientation constraint requires the "
                        "constrained start and goal orientation to match"
                    )
            position_mask = legacy_weight[3:] > 0.0
            if torch.any(position_mask):
                position_error = torch.abs(goal_pose.position - start_pose.position)
                if torch.any(position_error[..., position_mask] > 0.005):
                    log_and_raise(
                        "RoboTwin partial-pose position constraint requires the "
                        "constrained start and goal position to match"
                    )
            non_terminal_weight = _convert_legacy_hold_weight(metric.hold_vec_weight)
        else:
            non_terminal_weight = [0.0] * 6
        criteria = {
            frame: ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=non_terminal_weight,
                device_cfg=self.config.device_cfg,
            )
            for frame in planner.tool_frames
        }
        planner.ik_solver.update_tool_pose_criteria(criteria)
        planner.trajopt_solver.update_tool_pose_criteria(criteria)

    @staticmethod
    def _squeeze_seed_dimension(trajectory: JointState) -> JointState:
        """Remove the selected-seed axis from a ``[B, 1, H, D]`` trajectory."""

        def select(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value[:, 0]

        return JointState(
            position=trajectory.position[:, 0],
            velocity=select(trajectory.velocity),
            acceleration=select(trajectory.acceleration),
            jerk=select(trajectory.jerk),
            joint_names=trajectory.joint_names,
            dt=select(trajectory.dt),
            control_space=trajectory.control_space,
        )

    @staticmethod
    def _adapt_result(
        result: Optional[TrajOptSolverResult],
        batch_size: int,
        device: torch.device,
        active_joint_names: list[str],
    ) -> MotionGenResult:
        if result is None:
            return MotionGenResult(
                success=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        success = result.success
        if success.ndim > 1:
            success = success.any(dim=-1)
        trajectory = result.interpolated_trajectory
        last_tstep = result.interpolated_last_tstep
        if last_tstep is not None and last_tstep.ndim > 1 and last_tstep.shape[-1] == 1:
            last_tstep = last_tstep[..., 0]
        if trajectory is not None:
            trajectory = trajectory.reorder(active_joint_names)
            if trajectory.position.ndim == 4 and trajectory.position.shape[1] == 1:
                trajectory = MotionGen._squeeze_seed_dimension(trajectory)
            if result.interpolated_last_tstep is not None:
                maximum_last_tstep = int(torch.max(result.interpolated_last_tstep).item())
                trajectory = trim_joint_state_trajectory(trajectory, 0, maximum_last_tstep)
            if batch_size == 1 and trajectory.position.ndim == 3:
                trajectory = trajectory[0]
        return MotionGenResult(
            success=success,
            interpolated_plan=trajectory,
            interpolated_last_tstep=last_tstep,
        )

    def plan_single(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: Optional[MotionGenPlanConfig] = None,
    ) -> MotionGenResult:
        """Plan one RoboTwin arm request."""
        planner = self._require_planner()
        if not isinstance(planner, MotionPlanner):
            log_and_raise("plan_single requires warmup() without a batch argument")
        request = plan_config or MotionGenPlanConfig()
        self._update_pose_metric(request.pose_cost_metric, start_state, goal_pose)
        result = planner.plan_pose(
            self._goal_from_pose(goal_pose),
            start_state,
            max_attempts=request.max_attempts,
        )
        return self._adapt_result(result, 1, start_state.position.device, planner.joint_names)

    def plan_batch(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: Optional[MotionGenPlanConfig] = None,
    ) -> MotionGenResult:
        """Plan a fixed RoboTwin batch request."""
        planner = self._require_planner()
        if not isinstance(planner, BatchMotionPlanner):
            log_and_raise("plan_batch requires warmup(batch=N)")
        if start_state.position.shape[0] != self._batch_size:
            log_and_raise(
                f"start-state batch {start_state.position.shape[0]} does not match "
                f"warmup batch {self._batch_size}"
            )
        request = plan_config or MotionGenPlanConfig(max_attempts=1)
        self._update_pose_metric(request.pose_cost_metric, start_state, goal_pose)
        result = planner.plan_pose(
            self._goal_from_pose(goal_pose),
            start_state,
            max_attempts=request.max_attempts,
            success_ratio=1.0 / self._batch_size,
        )
        adapted = self._adapt_result(
            result,
            self._batch_size,
            start_state.position.device,
            planner.joint_names,
        )
        self._cache_batch_plan(start_state, goal_pose, adapted, request)
        return adapted

    def plan_single_repeated(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: Optional[MotionGenPlanConfig] = None,
    ) -> MotionGenResult:
        """Plan one request across every slot of a warmed batch planner.

        This compatibility path gives RoboTwin a larger deterministic seed
        pool after its single planner fails.  The first successful batch slot
        is returned with the normal single-request trajectory shape.
        """
        planner = self._require_planner()
        if not isinstance(planner, BatchMotionPlanner):
            log_and_raise("plan_single_repeated requires warmup(batch=N)")
        if start_state.position.shape[0] != 1:
            log_and_raise(
                "plan_single_repeated requires one start state; received "
                f"{start_state.position.shape[0]}"
            )
        request = plan_config or MotionGenPlanConfig(max_attempts=1)
        cached_result = self._reuse_cached_batch_plan(start_state, goal_pose, request)
        if cached_result is not None:
            return cached_result
        repeated_start = JointState.from_position(
            start_state.position.repeat(self._batch_size, 1),
            joint_names=start_state.joint_names,
        )
        repeated_goal = Pose(
            goal_pose.position.repeat(self._batch_size, 1),
            goal_pose.quaternion.repeat(self._batch_size, 1),
        )
        batch_result = self.plan_batch(repeated_start, repeated_goal, request)
        successful = torch.nonzero(batch_result.success, as_tuple=False).view(-1)
        if successful.numel() == 0 or batch_result.interpolated_plan is None:
            return MotionGenResult(
                success=torch.zeros(1, dtype=torch.bool, device=start_state.position.device)
            )

        selected = int(successful[0].item())
        batch_trajectory = batch_result.interpolated_plan

        def select(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value[selected]

        trajectory = JointState(
            position=batch_trajectory.position[selected],
            velocity=select(batch_trajectory.velocity),
            acceleration=select(batch_trajectory.acceleration),
            jerk=select(batch_trajectory.jerk),
            joint_names=batch_trajectory.joint_names,
            dt=select(batch_trajectory.dt),
            control_space=batch_trajectory.control_space,
        )
        return MotionGenResult(
            success=batch_result.success[selected : selected + 1],
            interpolated_plan=trajectory,
            interpolated_last_tstep=(
                None
                if batch_result.interpolated_last_tstep is None
                else batch_result.interpolated_last_tstep[selected : selected + 1]
            ),
        )

    def plan_single_with_tolerance(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: Optional[MotionGenPlanConfig] = None,
        *,
        goalset_size: int = 256,
        position_sample_radius: float = 0.005,
        rotation_sample_radius: float = 0.05,
        maximum_position_error: float = 0.005,
        maximum_rotation_error: float = 0.05,
        sampling_seed: int = 2026,
    ) -> MotionGenResult:
        """Plan to a deterministic collision-checked goal set within task tolerances."""
        if start_state.position.shape[0] != 1:
            log_and_raise("plan_single_with_tolerance requires one start state")
        if goalset_size < 1:
            log_and_raise("goalset_size must be positive")
        request = plan_config or MotionGenPlanConfig(max_attempts=1)
        if request.pose_cost_metric is not None:
            return MotionGenResult(
                success=torch.zeros(1, dtype=torch.bool, device=start_state.position.device)
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(sampling_seed)
        position_directions = torch.randn(
            (goalset_size, 3), generator=generator, dtype=torch.float32
        )
        position_directions /= torch.linalg.vector_norm(
            position_directions, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        position_radii = position_sample_radius * torch.rand(
            (goalset_size, 1), generator=generator, dtype=torch.float32
        ).pow(1.0 / 3.0)
        position_offsets = (position_directions * position_radii).to(
            device=goal_pose.position.device, dtype=goal_pose.position.dtype
        )

        rotation_axes = torch.randn(
            (goalset_size, 3), generator=generator, dtype=torch.float32
        )
        rotation_axes /= torch.linalg.vector_norm(
            rotation_axes, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        rotation_angles = rotation_sample_radius * torch.rand(
            (goalset_size, 1), generator=generator, dtype=torch.float32
        )
        delta_quaternions = torch.cat(
            (
                torch.cos(0.5 * rotation_angles),
                rotation_axes * torch.sin(0.5 * rotation_angles),
            ),
            dim=-1,
        ).to(device=goal_pose.quaternion.device, dtype=goal_pose.quaternion.dtype)
        goal_quaternions = goal_pose.quaternion.expand(goalset_size, -1)
        delta_w, delta_xyz = delta_quaternions[:, :1], delta_quaternions[:, 1:]
        goal_w, goal_xyz = goal_quaternions[:, :1], goal_quaternions[:, 1:]
        sampled_quaternions = torch.cat(
            (
                delta_w * goal_w - torch.sum(delta_xyz * goal_xyz, dim=-1, keepdim=True),
                delta_w * goal_xyz
                + goal_w * delta_xyz
                + torch.linalg.cross(delta_xyz, goal_xyz),
            ),
            dim=-1,
        )
        sampled_goals = Pose(
            goal_pose.position + position_offsets,
            sampled_quaternions,
        )
        repeated_start = JointState.from_position(
            start_state.position.repeat(goalset_size, 1),
            joint_names=start_state.joint_names,
        )
        tolerance_planner = self._tolerance_planners.get(goalset_size)
        if tolerance_planner is None:
            tolerance_planner = MotionGen(self.config)
            tolerance_planner.warmup(batch=goalset_size)
            self._tolerance_planners[goalset_size] = tolerance_planner
        batch_result = tolerance_planner.plan_batch(repeated_start, sampled_goals, request)
        successful = torch.nonzero(batch_result.success, as_tuple=False).view(-1)
        if successful.numel() == 0 or batch_result.interpolated_plan is None:
            return MotionGenResult(
                success=torch.zeros(1, dtype=torch.bool, device=start_state.position.device)
            )

        trajectory = batch_result.interpolated_plan
        planner = tolerance_planner._require_planner()
        selected: Optional[int] = None
        last_tstep = 0
        position_error = float("inf")
        rotation_error = float("inf")
        best_self_collision_clearance = float("-inf")
        for candidate_tensor in successful:
            candidate = int(candidate_tensor.item())
            candidate_last_tstep = trajectory.position.shape[-2]
            if batch_result.interpolated_last_tstep is not None:
                candidate_last_tstep = int(
                    batch_result.interpolated_last_tstep[candidate].item()
                )
            endpoint_state = JointState.from_position(
                trajectory.position[candidate, candidate_last_tstep - 1].reshape(1, -1),
                joint_names=trajectory.joint_names,
            )
            endpoint_kinematics = planner.compute_kinematics(endpoint_state)
            endpoint_pose = endpoint_kinematics.tool_poses[planner.tool_frames[0]]
            candidate_position_error = float(
                torch.linalg.vector_norm(endpoint_pose.position - goal_pose.position).item()
            )
            candidate_rotation_error = float(endpoint_pose.angular_distance(goal_pose).item())
            if not (
                candidate_position_error <= maximum_position_error
                and candidate_rotation_error <= maximum_rotation_error
            ):
                continue
            spheres = endpoint_kinematics.robot_spheres
            collision_config = planner.kinematics.get_self_collision_config()
            pairs = collision_config.collision_pairs.to(dtype=torch.long)
            first = spheres[..., pairs[:, 0], :]
            second = spheres[..., pairs[:, 1], :]
            radius_sum = (
                first[..., 3]
                + second[..., 3]
                + collision_config.sphere_padding[pairs[:, 0]]
                + collision_config.sphere_padding[pairs[:, 1]]
            )
            clearance = float(
                torch.min(
                    torch.linalg.vector_norm(first[..., :3] - second[..., :3], dim=-1)
                    - radius_sum
                ).item()
            )
            if clearance > best_self_collision_clearance:
                selected = candidate
                last_tstep = candidate_last_tstep
                position_error = candidate_position_error
                rotation_error = candidate_rotation_error
                best_self_collision_clearance = clearance
        if selected is None:
            return MotionGenResult(
                success=torch.zeros(1, dtype=torch.bool, device=start_state.position.device),
            )

        def select(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value[selected, :last_tstep].clone()

        selected_trajectory = JointState(
            position=trajectory.position[selected, :last_tstep].clone(),
            velocity=select(trajectory.velocity),
            acceleration=select(trajectory.acceleration),
            jerk=select(trajectory.jerk),
            joint_names=trajectory.joint_names,
            dt=None if trajectory.dt is None else trajectory.dt[selected].clone(),
            control_space=trajectory.control_space,
        )
        return MotionGenResult(
            success=batch_result.success[selected : selected + 1].clone(),
            interpolated_plan=selected_trajectory,
            interpolated_last_tstep=torch.tensor(
                [last_tstep], device=start_state.position.device, dtype=torch.long
            ),
            selected_goal_index=selected,
            goal_position_error=position_error,
            goal_rotation_error=rotation_error,
        )

    def reuse_cached_single(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: Optional[MotionGenPlanConfig] = None,
    ) -> Optional[MotionGenResult]:
        """Return a matching validated batch trajectory without starting a solve."""
        if start_state.position.shape[0] != 1:
            log_and_raise(
                "reuse_cached_single requires one start state; received "
                f"{start_state.position.shape[0]}"
            )
        request = plan_config or MotionGenPlanConfig(max_attempts=1)
        return self._reuse_cached_batch_plan(start_state, goal_pose, request)


__all__ = [
    "MotionGen",
    "MotionGenConfig",
    "MotionGenPlanConfig",
    "MotionGenResult",
    "PoseCostMetric",
    "RoboTwinRuntimeInfo",
    "validate_cuda_runtime",
    "validate_rocm_runtime",
    "validate_robotwin_runtime",
]
