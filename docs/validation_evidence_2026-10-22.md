# 双版本 ROCm 验证证据（2026-10-22）

本页记录 2026-10-22 的历史 patch 与结果。`v0.8.0` 后续修复了 Python/HIP kinematics
分派并新增回归测试；当前 patch 摘要及容器复测结果见
[2026-10-23 报告](container_validation_2026-10-23.md)。

## 验证平台

- GPU：AMD Radeon AI PRO R9700
- 目标架构：`gfx1201`
- Python：3.12
- PyTorch：2.13.0+rocm7.1
- ROCm runtime：7.1.52802
- 编译器：ROCm 7.2 `amdclang++`，通过隔离的 `/tmp/rocm-fixed/bin/hipcc` shim 调用
- 强制环境：`CUROBO_USE_HIP=1`、`CUROBO_KERNEL_BACKEND=hip`、
  `CUROBO_REQUIRE_ROCM=1`、`PYTORCH_ROCM_ARCH=gfx1201`、`HIP_VISIBLE_DEVICES=0`

系统 `hipcc` 存在递归 wrapper。本次验证没有修改系统 ROCm 文件，也没有终止其他用户进程。

## v0.7.8

- 上游 commit：`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- patch SHA-256：`8574b4d51908738b5f72c7062e1eb3f706150a0dc4fc0c646a6fd1fd80a2661a`
- 第二轮 wheel SHA-256：`e200ce79a8ff5f175f1606f2188e7d87846facb0a1794743641682bcc1c729ae`
- 第二轮结果：`7 passed in 5.30s`
- 第一轮结果：`7 passed in 7.73s`

七类测试实际执行 kinematics、self-collision、sphere/OBB、pose-distance、C-space tensor step、
L-BFGS/line-search 和 acceleration trajectory HIP kernel。两轮均从独立 wheel 安装目录运行，
无 skip、无 CPU fallback。

## v0.8.0

- 上游 commit：`4ea77366ca48ee453e7df139e39fa6532af49f3b`
- patch SHA-256：`ec60a6762bf8bb23469a5cdac52087c58c4f42de67517df359758c961ea2fcc1`
- 第二轮 wheel SHA-256：`72be77e0f2351e04f916dc2328d1af527fa7735c9161b04385303fd86d6418d9`
- 第二轮结果：`15 passed in 0.86s`
- 第一轮结果：`15 passed in 0.94s`

七个 manifest 文件覆盖 kinematics、geometry/self-collision、sphere-cuboid、ToolPose、C-space、
optimization 和 trajectory/B-spline，共 15 个真实 HIP 用例。两轮均从独立 wheel 安装目录运行，
无 skip、无 CPU fallback。

## Wrapper 合同

- wrapper 单元测试：`4 passed`
- Ruff：通过
- shell 语法检查：通过
- 两份最终 patch 均在对应精确 commit 的全新 clone 上通过 `git apply --check`
- 两份 patch 均通过 `git diff --check`
- `versions.toml` 的 patch SHA-256 与实际文件一致
- 两个测试 manifest 均严格包含七类测试文件

## 结论

v0.7.8 和 v0.8.0 均完成两轮独立的精确 commit、patch、wheel 构建、隔离安装和 AMD GPU 执行验证。
技术验证目标已经完成。许可证边界保持不变：v0.8.0 可作为正式发布主线；v0.7.8 只能作为
legacy/non-commercial profile，不得据此发布商业 wheel 或容器。
