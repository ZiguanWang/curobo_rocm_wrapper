# cuRobo v0.8.0 官方全套：gfx1201 / ROCm 7.14.1

## 结论

从上游 v0.8.0 固定 commit 和本仓库当前 profile 补丁建立全新 checkout、构建
wheel、隔离安装后，同 tag 原版官方测试 **3966 项：3928 passed、38 skipped、
0 failed、0 errors**，退出码为 0，耗时 1378.12 秒。七类 HIP 迁移测试
**16/16 passed**；连同原版 GraphExecutor 文件及原版 `motion_planning --test`
示例的定向运行 **39/39 passed**。这两个定向用例曾是上一轮官方全套的失败项。

测试副本中的 **199 个 Python 文件**已逐一按 SHA-256 与上游固定 commit
的 `curobo/tests/` 文件比较，内容一致。测试使用 `--import-mode=importlib`，
`curobo` 从隔离安装的 wheel 导入；没有修改官方测试的断言或跳过规则。

## 版本、环境与产物

| 项目 | 值 |
| --- | --- |
| cuRobo tag / commit | `v0.8.0` / `4ea77366ca48ee453e7df139e39fa6532af49f3b` |
| cuRobo profile patch SHA-256 | `089495f877e7d6b1f77331be95ca70e067ad8e8610302291bb6ae89090b6ac4e` |
| 已验证 wheel SHA-256 | `3a71e86b43d0b62e147a93ef59b4a38bf8a93fa79141a1922af0cb5e3857887e` |
| AMD Warp commit | `74014999a31471228e0aaa213bbeb2e5daba95f0` |
| Warp patch SHA-256 | `0a1ee165f214649201f00ca24ed093d3cffae64c2f48621a15e6d165b2a8347d` |
| GPU | Radeon AI PRO R9700，`gfx1201` |
| 镜像 / PyTorch | `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.11.0` / `2.11.0+rocm7.14.1` |

隔离源码、wheel、依赖、安装目录及日志位于
`/tmp/curobo-v080-rocm714-validation/`。最终 wheel 在 `dist_motion_final/`；
官方全套 JUnit 与原始输出分别是
`logs/official-full-motion-final-headers-714.xml` 和同名前缀的 `.log`；
定向 JUnit 是 `logs/targeted-motion-final-714.xml`。这些 `/tmp` 产物尚未
归档进仓库。根仓库两个 submodule 工作树未用于本轮构建。

运行设置 `CUROBO_USE_HIP=1`、`CUROBO_KERNEL_BACKEND=hip`、
`CUROBO_REQUIRE_ROCM=1`、`PYTORCH_ROCM_ARCH=gfx1201`、
`HIP_VISIBLE_DEVICES=0` 和 `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`；未设置
`AMD_SERIALIZE_KERNEL`。容器挂载同版 ROCm 开发头文件到 `/opt/rocm/include`，
供 Warp 的 HIPRTC JIT 编译使用。测试插件仅在 pytest 收集前初始化 Warp。

## 这轮修复的两项失败

- 通用 `GraphExecutor` 的默认图开关现在读取公开的 `cuda_graphs` 配置；
  solver 通过 `create_graph_executor` 时仍受 ROCm 图捕获安全开关约束。
  原版 GraphExecutor 文件和整套官方测试均通过。
- 原版 `motion_planning --test` 的 pose、grasp 和 lift 链路已通过。
  ROCm 单目标规划在场景 IK 陷入局部解时，先生成无场景 IK 候选，再用全新
  场景 IK 实例验证约束，按与当前关节状态的距离挑选轨迹种子。原有场景
  约束和抓取的线性运动约束继续由求解器执行；原版示例的目标、障碍物和
  断言均未修改。先前失败的 IK 实例会保留影响后续评估的状态，因此验证
  使用新的实例。

## 跳过范围与结论边界

38 项跳过包括：缺少 Pinocchio 的 28 项、未实现的
`AccelerationTensorStepIdxKernel` backward 4 项、收集阶段跳过 3 项，
以及 Warp voxel overload、`KinematicsParams` 大规模配置和 Inf 梯度
数值稳定性问题各 1 项。通过数不覆盖这些跳过功能。

这份记录证明上述固定 ROCm 环境、调试版 AMD Warp 和当前补丁下的
cuRobo v0.8.0 官方单元测试通过；不构成优化构建性能、其他 GPU/ROCm
版本或 LingBot 指定 RoboTwin commit 端到端替换评测的证明。
