# AMD RDNA/ROCm 支持矩阵

本页定义 cuRobo 首个 AMD RDNA/ROCm 交付版本的已验证平台、能力边界和兼容性状态。
迁移操作和验收流程见 {doc}`../guides/amd_rdna_migration_zh`。

## 发布支持组合

| 项目 | 支持值 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 |
| GPU 架构 | `gfx1201` |
| 操作系统 | Ubuntu 24.04 LTS |
| 系统 ROCm/驱动 | ROCm 7.2 |
| PyTorch | 2.13.0+rocm7.1 |
| PyTorch HIP runtime | ROCm 7.1 |
| Python | 3.12 |
| 规划 dtype | `float32` |
| 构建架构 | `PYTORCH_ROCM_ARCH=gfx1201` |
| world collision | primitive cuboid |
| cuRobo accelerator/backend | `rocm` / `hip` |

系统 ROCm/驱动 7.2 与 PyTorch 自带 HIP runtime 7.1 的组合是当前完整 RoboTwin、性能和
稳定性证据对应的冻结发布组合。版本差异是已验证配置的一部分，不代表其他任意
ROCm/PyTorch 跨版本组合均受支持。

## 兼容性验证组合

| 项目 | 验证值 |
| --- | --- |
| 容器基础镜像 | `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| PyTorch | 2.10.0+rocm7.14.1 |
| ROCm | 7.14.1 |
| Python | 3.12 |
| GPU/架构 | AMD Radeon AI PRO R9700 / `gfx1201` |
| 验证范围 | wheel 源码构建、隔离安装、HIP 核心算子、非 ESDF motion |
| 验证结果 | 214 passed、0 failed，472.112 秒 |
| 已复验镜像 | `sha256:2c41199ef74505b9b9e6e868bbc1c60b209aed6b54fa1f0503fe8dba7619ccf1` |

该组合属于构建和非 ESDF motion 兼容性验证组合，不属于首个版本的完整 RoboTwin 发布支持
组合。它没有替代发布组合上的完整 episode、性能和稳定性证据。

## 功能支持状态

| 能力 | 状态 | 边界 |
| --- | --- | --- |
| Kinematics forward/backward | 支持 | HIP GPU，`float32` |
| Robot self-collision | 支持 | sphere model |
| Sphere-cuboid collision | 支持 | discrete、swept、多环境和 mask |
| ToolPose cost | 支持 | HIP GPU |
| C-space position/state/distance cost | 支持 | HIP GPU |
| SeedIK、LM、L-BFGS、line search | 支持 | HIP GPU |
| Trajectory integration、B-spline、interpolation | 支持 | HIP GPU |
| IK、TrajOpt、graph planner | 支持 | 已验证规划路径 |
| MotionPlanner、BatchMotionPlanner | 支持 | single 和 batch |
| RoboTwin single request | 支持 | 冻结 revision |
| RoboTwin batch request | 支持 | 冻结 revision |
| RoboTwin goal-set request | 支持 | 冻结 revision |
| RoboTwin partial-pose request | 支持 | 冻结 revision |
| Active mesh collision | 不支持 | 运行时 fail-closed |
| Voxel/ESDF collision | 不支持 | 运行时 fail-closed |
| `float64` planning | 不支持 | 首个交付版本不包含 |
| 其他 AMD GPU 架构 | 未验证 | 不属于发布支持范围 |

## 后端与回退约束

| 条件 | 规定行为 |
| --- | --- |
| ROCm 自动选择 | `accelerator=rocm`，`backend=hip` |
| 不支持的 collision capability | 在规划前明确报错 |
| HIP extension 构建或加载失败 | 明确报错并保留诊断 |
| Warp 仅可使用 CPU | 禁止作为规划回退 |
| MPLib 可用 | 禁止由 cuRobo 初始化或规划失败触发自动回退 |
| 规划热路径 CPU 实现 | 禁止 |

## 上游版本

| 上游 | 冻结 revision |
| --- | --- |
| RoboTwin | `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` |
| XPolicyLab | `fa431ecd893ee706883e64fe5fe1464ec8cd928d` |

未冻结 revision 的接口和行为不属于当前兼容性承诺。

## NVIDIA CUDA 状态

NVIDIA CUDA 后端保留，CUDA Graph 在正常运行和性能路径中保持启用。当前迁移分支尚未完成
NVIDIA GPU 主机上的数值、规划、性能和 CUDA Graph 发布回归，因此双平台发布门禁仍未通过。
共享 RoboTwin adapter 已移除无条件 ROCm-only preflight：通用入口会接受匹配的 CUDA/CUDA
组合，显式 ROCm 验收仍通过 `required_accelerator="rocm"` fail-closed。该分派契约已通过模拟
CUDA 单元测试和真实 ROCm/SAPIEN smoke，但不能替代 NVIDIA GPU 主机回归。
