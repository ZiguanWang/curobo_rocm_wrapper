# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the RoboTwin planning adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.runtime import AcceleratorType
from curobo.robotwin import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
    _convert_legacy_hold_weight,
    validate_robotwin_runtime,
    validate_rocm_runtime,
)
from curobo.types import JointState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU planning tests require torch.cuda"
)


def _world_config() -> dict:
    return {
        "cuboid": {
            "table": {
                "dims": [0.7, 2.0, 0.04],
                "pose": [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }


def test_robotwin_adapter_rejects_unmapped_graph_seed_count() -> None:
    """Reject historical options without a current semantic equivalent."""
    with pytest.raises(ValueError, match="num_graph_seeds=1"):
        MotionGenConfig.load_from_robot_config("franka.yml", _world_config(), num_graph_seeds=2)


def test_robotwin_adapter_normalizes_v078_robot_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate fields present in RoboTwin's pinned cuRobo v0.7.8 YAML files."""
    monkeypatch.setenv("ROBOTWIN_ASSET_ROOT", "/tmp/robotwin-assets")
    config = MotionGenConfig.load_from_robot_config(
        {
            "robot_cfg": {
                "kinematics": {
                    "use_usd_kinematics": False,
                    "urdf_path": "${ROBOTWIN_ASSET_ROOT}/robot.urdf",
                    "asset_root_path": None,
                    "base_link": "base",
                    "ee_link": "tool",
                    "cspace": {
                        "joint_names": ["joint"],
                        "retract_config": [0.0],
                    },
                }
            }
        },
        _world_config(),
    )

    kinematics = config.robot["robot_cfg"]["kinematics"]
    assert "use_usd_kinematics" not in kinematics
    assert "ee_link" not in kinematics
    assert kinematics["tool_frames"] == ["tool"]
    assert kinematics["urdf_path"] == "/tmp/robotwin-assets/robot.urdf"
    assert kinematics["asset_root_path"] == ""
    assert kinematics["cspace"]["default_joint_position"] == [0.0]
    assert "retract_config" not in kinematics["cspace"]


def test_robotwin_adapter_converts_legacy_pose_constraint_order() -> None:
    """Map v0.7.8 rotation/position weights to current position/rotation order."""
    legacy_weight = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    assert _convert_legacy_hold_weight(legacy_weight) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_robotwin_adapter_sizes_dense_trajectory_buffer() -> None:
    """Reserve enough interpolation storage for RoboTwin's 250 Hz paths."""
    config = MotionGenConfig.load_from_robot_config("franka.yml", _world_config())
    assert config.interpolation_buffer_size == 5000
    assert config.num_ik_seeds == 32

    overridden = MotionGenConfig.load_from_robot_config(
        "franka.yml", _world_config(), interpolation_buffer_size=2048, num_ik_seeds=48
    )
    assert overridden.interpolation_buffer_size == 2048
    assert overridden.num_ik_seeds == 48


def test_robotwin_batch_warmup_skips_unused_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not build a PRM roadmap for RoboTwin's graph-free batch path."""
    motion_gen = MotionGen(MotionGenConfig.load_from_robot_config("franka.yml", _world_config()))

    class WarmupPlanner:
        def warmup(self, *, enable_graph: bool = True) -> bool:
            assert not enable_graph
            return True

    warmup_planner = WarmupPlanner()
    monkeypatch.setattr(motion_gen, "_create_planner", lambda batch_size: None)
    monkeypatch.setattr(motion_gen, "_require_planner", lambda: warmup_planner)
    monkeypatch.setattr(
        "curobo.robotwin.BatchMotionPlanner",
        WarmupPlanner,
    )

    assert motion_gen.warmup(batch=2)


def test_robotwin_adapter_plans_single_pose() -> None:
    """Run the RoboTwin single-pose contract through the current planner."""
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config(
            "franka.yml",
            _world_config(),
            interpolation_dt=0.025,
            num_trajopt_seeds=2,
        )
    )
    assert motion_gen.warmup()
    planner = motion_gen._require_planner()
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    goal_state = start.clone()
    goal_state.position[..., 0] += 0.1
    goal_pose = planner.compute_kinematics(goal_state).tool_poses[planner.tool_frames[0]]

    with pytest.raises(ValueError, match="orientation constraint"):
        motion_gen.plan_single(
            start,
            goal_pose,
            MotionGenPlanConfig(
                max_attempts=1,
                pose_cost_metric=PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=torch.tensor(
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        device=start.position.device,
                    ),
                ),
            ),
        )

    result = motion_gen.plan_single(
        start,
        goal_pose,
        MotionGenPlanConfig(
            max_attempts=2,
            pose_cost_metric=PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=torch.zeros(6, device=start.position.device),
            ),
        ),
    )

    assert result.success.shape == (1,)
    assert result.success.item()
    assert result.interpolated_plan is not None
    assert result.interpolated_plan.position.ndim == 2
    assert result.interpolated_plan.position.shape[-1] == len(planner.joint_names)


def test_robotwin_joint_action_interpolation_respects_limits() -> None:
    """Generate a GPU-compatible qpos path without invoking MPLib TOPP."""
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config("franka.yml", _world_config())
    )
    assert motion_gen.warmup()
    planner = motion_gen._require_planner()
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    goal = start.clone()
    goal.position[..., 0] += 0.1

    path = motion_gen.interpolate_joint_action(start, goal, control_dt=0.01)

    limits = planner.kinematics.get_joint_limits()
    assert path.position.ndim == 2
    assert path.position.shape == path.velocity.shape
    assert torch.allclose(path.position[0], start.position[0])
    assert torch.allclose(path.position[-1], goal.position[0])
    assert torch.allclose(path.velocity[[0, -1]], torch.zeros_like(path.velocity[[0, -1]]))
    assert torch.all(torch.abs(path.velocity) <= limits.velocity[1] + 1.0e-5)


def test_robotwin_joint_action_interpolation_rejects_position_limit() -> None:
    """Fail closed instead of clipping an invalid policy joint target."""
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config("franka.yml", _world_config())
    )
    assert motion_gen.warmup()
    planner = motion_gen._require_planner()
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    goal = start.clone()
    goal.position[..., 0] = planner.kinematics.get_joint_limits().position[1, 0] + 0.1

    with pytest.raises(ValueError, match="position limits"):
        motion_gen.interpolate_joint_action(start, goal)


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm runtime required")
def test_robotwin_runtime_preflight_is_fail_closed() -> None:
    """Record ROCm and HIP backend identity before an evaluation starts."""
    runtime = validate_rocm_runtime()
    assert runtime.accelerator == "rocm"
    assert runtime.backend == "hip"
    assert runtime.rocm_version
    assert runtime.gfx_arch.startswith("gfx")


@pytest.mark.parametrize("implementation_backend", ["cuda_core", "pybind"])
def test_robotwin_runtime_preflight_accepts_cuda_backend(
    monkeypatch: pytest.MonkeyPatch,
    implementation_backend: str,
) -> None:
    """Keep the shared RoboTwin adapter usable on an NVIDIA CUDA runtime."""
    fake_torch = SimpleNamespace(
        __version__="2.8.0+cu129",
        version=SimpleNamespace(cuda="12.9", hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA Test GPU",
        ),
    )
    monkeypatch.setattr("curobo.robotwin.torch", fake_torch)
    monkeypatch.setattr(
        "curobo.robotwin.detect_accelerator", lambda: AcceleratorType.CUDA
    )
    monkeypatch.setattr(
        "curobo.robotwin.get_backend_name", lambda: implementation_backend
    )

    runtime = validate_robotwin_runtime()

    assert runtime.accelerator == "cuda"
    assert runtime.backend == "cuda"
    assert runtime.cuda_version == "12.9"
    assert runtime.rocm_version == ""
    assert runtime.device_name == "NVIDIA Test GPU"


def test_robotwin_runtime_preflight_keeps_explicit_rocm_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject CUDA when a ROCm acceptance run explicitly requires ROCm."""
    monkeypatch.setattr(
        "curobo.robotwin.detect_accelerator", lambda: AcceleratorType.CUDA
    )

    with pytest.raises(RuntimeError, match="requires accelerator=rocm"):
        validate_robotwin_runtime(required_accelerator="rocm")


def test_robotwin_adapter_plans_pose_batch() -> None:
    """Run the fixed-size RoboTwin batch contract through BatchMotionPlanner."""
    batch_size = 2
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config(
            "franka.yml",
            _world_config(),
            interpolation_dt=0.025,
            num_trajopt_seeds=2,
        )
    )
    assert motion_gen.warmup(batch=batch_size)
    planner = motion_gen._require_planner()
    positions = planner.default_joint_state.position.unsqueeze(0).repeat(batch_size, 1)
    start = JointState.from_position(positions, joint_names=planner.joint_names)
    goal_state = start.clone()
    goal_state.position[..., 0] += 0.1
    goal_pose = planner.compute_kinematics(goal_state).tool_poses[planner.tool_frames[0]]
    success_ratios: list[float] = []
    original_plan_pose = planner.plan_pose

    def traced_plan_pose(*args: object, **kwargs: object) -> object:
        success_ratios.append(float(kwargs["success_ratio"]))
        return original_plan_pose(*args, **kwargs)

    planner.plan_pose = traced_plan_pose

    result = motion_gen.plan_batch(
        start,
        goal_pose,
        MotionGenPlanConfig(max_attempts=2),
    )

    assert result.success.shape == (batch_size,)
    assert torch.all(result.success)
    assert result.interpolated_plan is not None
    assert result.interpolated_plan.position.ndim == 3
    assert result.interpolated_plan.position.shape[0] == batch_size
    assert result.interpolated_plan.position.shape[-1] == len(planner.joint_names)
    assert result.interpolated_last_tstep is not None
    assert result.interpolated_last_tstep.shape == (batch_size,)
    assert success_ratios == [0.5]

    motion_gen._cache_batch_plan(
        start,
        goal_pose,
        MotionGenResult(success=torch.zeros(batch_size, dtype=torch.bool, device="cuda")),
        MotionGenPlanConfig(max_attempts=2),
    )

    def fail_if_batch_solver_runs(*args: object, **kwargs: object) -> object:
        raise AssertionError("matching successful batch trajectory should be reused")

    motion_gen.plan_batch = fail_if_batch_solver_runs

    cached_single = motion_gen.reuse_cached_single(
        JointState.from_position(positions[:1], joint_names=planner.joint_names),
        planner.compute_kinematics(goal_state).tool_poses[planner.tool_frames[0]][:1],
        MotionGenPlanConfig(max_attempts=2),
    )
    assert cached_single is not None
    assert cached_single.success.item()

    repeated_single = motion_gen.plan_single_repeated(
        JointState.from_position(positions[:1], joint_names=planner.joint_names),
        planner.compute_kinematics(goal_state).tool_poses[planner.tool_frames[0]][:1],
        MotionGenPlanConfig(max_attempts=2),
    )
    assert repeated_single.success.shape == (1,)
    assert repeated_single.success.item()
    assert repeated_single.interpolated_plan is not None
    assert repeated_single.interpolated_plan.position.ndim == 2
    assert repeated_single.interpolated_plan.position.shape[-1] == len(planner.joint_names)
    assert repeated_single.interpolated_last_tstep is not None
    assert repeated_single.interpolated_last_tstep.shape == (1,)
    assert success_ratios == [0.5]
