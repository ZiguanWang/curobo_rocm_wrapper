# LingBot-VLA-v2 指定 RoboTwin 版本的 cuRobo 替换状态

核对日期：2026-10-24。本文用于界定本仓库初始提交可以声明的范围。

## 对照目标

[LingBot-VLA-v2 的 RoboTwin 说明](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/README.md)
要求检出 RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`，其数据准备示例使用
`policy/pi0`，仿真评测使用 `start_robotwin_infer_and_eval.sh`。该说明没有为
XPolicyLab 单独给出一个冻结 commit；不能把本地其它 RoboTwin/XPolicyLab 组合的验收
结果当成这份说明指定组合的结果。

在 RoboTwin `13c3c47...` 的
`envs/robot/planner.py` 中，`CuroboPlanner` 可选导入
`curobo.types.math.Pose`、`curobo.types.robot.JointState`、
`curobo.wrap.reacher.motion_gen.MotionGen` 等旧版接口。该文件捕获 cuRobo 导入异常，
之后仍保留 `MplibPlanner`；评测脚本能够启动本身不能证明实际规划使用了 cuRobo。

## 当前 wrapper 的实测结果

| Profile | 对上述旧版导入块的隔离 wheel 冒烟测试 | 进一步验证 |
| --- | --- | --- |
| `v0_8_0` | **失败**：`No module named 'curobo.types.math'; 'curobo.types' is not a package` | 使用新的 `curobo.types` API；RoboTwin 旧 planner 需要显式适配 |
| `v0_7_8` | **通过**：所列旧接口能导入 | 官方 IK/TrajOpt 配置测试在当前 ROCm 容器中仍因 Warp 无 GPU 设备而无法完整收集；尚未运行此 RoboTwin commit 的完整 episode |

这些 wheel 的 GPU 迁移测试结果见
[容器验证报告](container_validation_2026-10-23.md)：`v0.8.0` 为 16/16，
`v0.7.8` 为 7/7。迁移测试覆盖与 RoboTwin 完整 planner 初始化、SAPIEN 执行、
策略推理及 task success 不同，不能替代端到端验收。

迁移前工作区曾对另一组冻结 revision 做过 XPolicyLab 验收：RoboTwin
`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`、XPolicyLab
`fa431ecd893ee706883e64fe5fe1464ec8cd928d`。本机历史结果的真实策略
`stack_blocks_two` 两个 episode 为 1 成功、1 失败，零 fallback。历史适配文件位于
`archive/pre_wrapper_workspace/`，不会被当前 CLI 应用；那份验收既不是
`13c3c47...` 的证据，也不是当前 wrapper wheel 的端到端验收。

## 初始提交与替换声明

当前仓库可以作为**版本固定的 cuRobo ROCm wrapper 和验证记录**进行初始提交。
不能在初始提交中声明它是上述 LingBot/RoboTwin commit、XPolicyLab 或两者的
**开箱即用 cuRobo 替换件**。

若要对指定评测组合声明可替换，至少需要：

1. 固定 RoboTwin、实际采用的策略代码、checkpoint、资产与环境版本；确认所用
   XPolicyLab revision，若该流程确实使用 XPolicyLab。
2. 选定一个 cuRobo profile；对 `v0.8.0` 适配旧 planner 的导入与调用 API，
   对 `v0.7.8` 处理实际规划路径中的 Warp 依赖，并遵守其非商业许可边界。
3. 在目标 ROCm 容器中使用**由本仓库当前 patch 构建的 wheel**，跑 planner 初始化、
   single/batch/goal-set/partial-pose 请求及真实 RoboTwin episode。
4. 记录 `accelerator=rocm`、`backend=hip`、左右臂实际 planner、每个 episode
   的任务成功/失败和失败原因；对 cuRobo 初始化失败、Warp CPU、MPLib 回退
   进行显式检测，不把评测进程正常退出当作 task success。
5. 按该 LingBot 说明要求的任务数量、episode 数、推理精度和预定成功率门槛
   出具可追溯结果，再限定范围地声明兼容。
