# cuRobo 迁移到 AMD RDNA/ROCm 的实施与验收指南

## 目标

本指南用于把 RoboTwin/XPolicyLab 所依赖的 cuRobo 能力迁移到 AMD RDNA GPU，并在 ROCm
上完成运动学、碰撞检测、逆运动学（IK）、轨迹优化（TrajOpt）、图搜索和运动规划。

迁移完成后应满足以下核心要求：

- RoboTwin 使用 ROCm cuRobo，而不是 CPU MPLib；
- 保留现有 NVIDIA CUDA 后端和公共规划接口；
- ROCm 不支持的能力明确失败，不静默回退到 CPU、Warp CPU 或 MPLib；
- 在冻结的任务集上，规划正确性、成功率、性能和稳定性达到本文规定的门槛；
- wheel 或容器可以在干净环境中重复安装和验收。

本文属于 How-To Guide。它规定实施顺序、每阶段产物、退出条件和最终验收口径，不是
ROCm、HIP 或 cuRobo 基础概念教程。

```{warning}
当前仓库已经打通主要 ROCm 规划链路，并已在 NVIDIA RTX 4090 上完成 CUDA 核心回归，但尚未
通过最终发布验收。当前仍缺少通过门槛的真实 LingBot-VLA v2 policy server 端到端任务结果、
双平台连续 CI 和独立复核。不得把“能够编译”、policy server
能够启动或少量 episode 成功表述为迁移完成。
```

## 交付范围

### 首个交付版本必须支持

- AMD Radeon AI PRO R9700，目标架构 `gfx1201`；
- PyTorch ROCm 上的 float32 规划路径；
- kinematics forward/backward；
- robot self-collision；
- sphere-cuboid 离散和 swept world collision；
- ToolPose、C-space position/state/distance cost；
- SeedIK、Levenberg-Marquardt、L-BFGS 和 line search；
- trajectory integration、B-spline 和 interpolation；
- IK、TrajOpt、graph planner、MotionPlanner 和 BatchMotionPlanner；
- RoboTwin 的 single、batch、goal-set 和 partial-pose 请求；
- XPolicyLab 初始化失败时 fail-closed，禁止 MPLib 自动回退。

### 首个交付版本可以明确不支持

- active mesh collision；
- voxel/ESDF collision；
- 未列入支持矩阵的 RDNA、CDNA 或其他 AMD GPU；
- float64 规划；
- 未冻结 revision 的 RoboTwin/XPolicyLab 接口。

这些能力必须由运行时 capability check 明确拒绝。不得在规划热路径中悄悄使用 CPU 来模拟
支持。若冻结任务后来使用 mesh、voxel 或 ESDF，应先扩展 GPU 实现和验收范围，再把该任务
计入正式结果。

## 固定开发环境与上游版本

首个交付版本的正式平台与能力边界见
{doc}`AMD RDNA/ROCm 支持矩阵 <../reference/amd_rdna_support_matrix>`。当前完整 RoboTwin、性能和
稳定性证据来自以下冻结环境：

| 项目 | 当前值 |
| --- | --- |
| GPU | 4 × AMD Radeon AI PRO R9700 |
| GPU 架构 | `gfx1201` |
| 系统 ROCm | 7.2 |
| PyTorch | 2.13.0+rocm7.1 |
| Warp | 1.18.0，仅识别 CPU |
| SAPIEN | 3.0.3 |
| MPLib | 0.2.1 |
| RoboTwin revision | `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` |
| XPolicyLab revision | `fa431ecd893ee706883e64fe5fe1464ec8cd928d` |

系统 ROCm 7.2 与 PyTorch HIP runtime 7.1 的小版本不一致。这是当前已经验证并锁定的发布
组合，而不是对任意跨版本组合的兼容性承诺。ROCm 7.14.1/PyTorch 2.10.0 容器仅作为构建、
隔离安装和非 ESDF motion 兼容性验证环境；在完成同等级 RoboTwin、性能和稳定性验证前，
不得把它列为完整发布支持组合。

## 当前状态

截至 2026 年 10 月 22 日，当前分支状态如下：

| 能力 | 状态 | 当前证据或缺口 |
| --- | --- | --- |
| ROCm 检测和 `hip` 后端选择 | 已完成首轮验证 | CPU/CUDA/ROCm 选择测试已覆盖 |
| 隔离 HIP 构建 | 已完成首轮验证 | hipify 产物位于独立 build 目录，不修改 PyTorch 或共享源码 |
| 运动学、自碰撞和 primitive collision | 已完成首轮验证 | forward/backward、离散/swept、多环境和 mask 已覆盖 |
| cost、优化和轨迹算子 | 已完成首轮验证 | ToolPose、C-space、LM、SeedIK、L-BFGS、B-spline 已覆盖 |
| IK、TrajOpt、graph planner | 已完成首轮验证 | graph planner 测试 152 项通过 |
| MotionPlanner | 已完成首轮验证 | 单任务测试 75 项通过 |
| BatchMotionPlanner | 合并契约和关键轨迹已复核 | solver buffer 复用、字段复制和成功 slot 独立验证已通过 |
| 非 ESDF motion 套件 | 已完成首轮验证 | 191 项通过；与上述集合有重叠，不能相加为总数 |
| mesh/voxel/ESDF | 未实现 | active world 必须 fail-closed |
| primitive 稳定性 | 已通过内部门禁 | 1,000/1,000 成功，显存漂移为 0 |
| RoboTwin 完整 episode | 小规模冻结集已通过 | 当前同版本代码下 `stack_blocks_two` seeds 0--3 为 4/4 |
| RoboTwin 更长任务 | 已通过冻结门禁 | `stack_blocks_three` seeds 0--3 为 4/4，成功率 100% |
| ROCm 对 MPLib 性能 | 严格请求级门禁通过 | 同一冻结请求重复 5 次的 p50 加速为 2.069× |
| CUDA 回归 | 已通过 | RTX 4090 上核心测试 289 passed、0 failed、15 个预期 ROCm-only skipped；CUDA Graph 默认启用和 capture/replay 通过 |
| HIP wheel/容器 | 已完成两套环境首轮验证 | 37 MB wheel 包含四个 HIP extension；宿主算子 15/15，官方 ROCm 容器算子与非 ESDF motion 共 214 项通过 |
| 发布复现和 CI | CI 定义已建立，执行验收未完成 | 已定义 CPU、NVIDIA CUDA、AMD ROCm、定时 ROCm 多任务/soak，以及定时 CUDA RoboTwin baseline jobs；尚缺自托管 GPU runner 连续通过 |

当前发布汇总为 `passed_gate_count=8/10`、`release_ready=false`。已通过干净容器、
`stack_blocks_two`、`stack_blocks_three`、真实 XPolicy 完整动作链、ROCm 相对 MPLib 的请求级性能、
1,000 次 primitive soak、8-episode 混合任务 soak 和 CUDA 回归；尚未通过连续 CI 历史和独立复核。机器可读
汇总位于 `build/rocm_acceptance/release_summary.json`。该状态不得表述为迁移完成。

当前同版本代码和统一 RNG 隔离结果为：

| Seed | `task_success` | 总时间 | play 时间 |
| --- | --- | ---: | ---: |
| 0 | true | 195.887 s | 70.297 s |
| 1 | true | 252.979 s | 131.573 s |
| 2 | true | 183.773 s | 62.381 s |
| 3 | true | 180.815 s | 62.875 s |

汇总成功率为 100%，`minimum_success_rate=1.0` 门禁返回 `passed=true`。汇总证据位于本地
`build/rocm_acceptance/robotwin_stack_blocks_two_seeds0_3_current_summary.json`；`build/` 不作为
源码提交，正式发布时必须上传为不可变验收 artifact。为达到该结果，已完成以下修复和验证：

- BatchMotionPlanner 首次 attempt 会深拷贝 solver result，避免后续 attempt 复用内部 buffer
  覆盖已成功结果；跨 attempt 合并会同步 success、轨迹、metrics、solution state、seed 信息和
  dt 边界，相关专项测试 59 项通过。
- seed 1 的 batch record 12 slot 7 已保存实际轨迹并独立复核。轨迹长度为 622，末端位置
  误差为 4.84747 mm，末端旋转误差为 0.0308641 rad，self/world collision 最大值均为 0。
  因此该 slot 是真实有效解，而不是 success mask 假阳性。
- adapter 会在起点、目标和约束语义严格匹配时复用成功 batch 轨迹，避免 single planner
  随机丢失已经找到的可行分支。缓存未命中仍执行正常 single 规划，不允许近似匹配。
- partial-pose 请求新增确定性的局部 IK seed 分块扫描和二阶段 refinement；graph seed 作为
  可选增强，graph 无路径时仍允许局部 IK seed 进入 TrajOpt。冻结 replay 得到位置误差
  3.3839 mm、旋转误差 0.01207 rad，轨迹长度 502，self/world collision 最大值均为 0。
- 当前实现上 seed 1 已首次完整成功，`task_success=true`，左右臂均使用 `CuroboPlanner`，
  `backend=hip`，且没有 MPLib fallback。

将 batch cache 查询移到昂贵 single solver 之前后，seed 1 再次完整成功。原 record 15 从
100.797 s 降至 0.000737 s，play 时间从 258.167 s 降至 138.756 s，总时间从 383.277 s
降至 261.411 s。局部 IK chunk 改为不减少覆盖的确定性分散顺序后，seed 1 总时间进一步降至
252.979 s；record 16 为 55.274 s，record 17 为 18.579 s。小规模 4/4 门禁已经通过，但这些
延迟仍表明 partial-pose 路径需要继续优化，且不能替代正式任务集、XPolicyLab 或 P2 性能验收。

更长的 `stack_blocks_three` 已完成 seeds 0--3 冻结复核，当前为 4/4、成功率 100%。左右臂均
使用 `CuroboPlanner`，且 `accelerator=rocm`、`backend=hip`、无 MPLib fallback。汇总证据位于
`build/rocm_acceptance/robotwin_stack_blocks_three_seeds0_3_current_summary.json`。

seed 2 的历史失败最终定位为 RoboTwin 任务在第三块已经放置并打开夹爪后，仍固定执行一次
竖直抬升 7 cm。该额外末端目标会触发真实 `fl_link3`/`fl_link5` 自碰撞；改为直接回 home 虽然
规划成功，却会在长轨迹中扫落蓝块。最终 overlay 仅对第三块删除这个完成后的冗余撤离，保留
前两块撤离、三次抓取、三次放置、原碰撞模型、padding、seeds 和成功判定。最终方块高度约为
0.765、0.815 和 0.865 m，双夹爪均打开，完整 episode 返回 `task_success=true`。

该修复通过冻结 revision 的精确、幂等 source replacement 应用，并拒绝未知上游源码形状。
诊断过程还排除了缩球、减少 padding、放宽 5 mm/0.05 rad 容差、对侧手臂强制抓取和直接回
home 等方案。原失败和对侧手臂证据继续保留用于审计，但不再代表当前发布状态。

HIP wheel 已完成本机隔离安装首轮验证。修复 setuptools package discovery 边界后，wheel 从
错误包含 ROCm SDK 的 6.0 GB 降至约 37 MB，且不再包含任何 `build/` 条目。wheel 中包含
`kinematics`、`geometry`、`optimization` 和 `trajectory` 四个 `hip_backend` extension；从
独立 `--target` 目录导入后，runtime preflight、HIP backend 选择和完整 HIP 算子集成测试 15 项
通过。随后使用 `docker/rocm-gfx1201/Dockerfile` 在官方 PyTorch 2.10.0+ROCm 7.14.1 镜像中
从源码重新构建并隔离安装 wheel，R9700 `gfx1201` preflight 和相同 15 项算子测试全部通过。
该容器通过多阶段构建从同版本 ROCm devel 镜像仅复制 `thrust`、`rocprim` 和 `hipcub` 头文件，
不会把完整 SDK 打进 wheel。

容器随后按照 wheel 元数据正常安装全部运行依赖，并通过
`docker/rocm-gfx1201/run_acceptance.sh` 执行 15 项 HIP 算子测试和首版支持范围内的全部非 ESDF
motion 测试。2026 年 10 月 22 日使用当前源码重新构建镜像
`sha256:2c41199ef74505b9b9e6e868bbc1c60b209aed6b54fa1f0503fe8dba7619ccf1`，随后生成 runtime
JSON 和 JUnit 原始证据。结果为 214 passed、0 failed，耗时 472.112 秒。单独执行完整 motion
目录时，剩余失败全部来自首版
明确不支持的 voxel/ESDF 测试，并按 capability contract 在规划前 fail-closed。测试过程中
PyTorch 在不支持的 hipBLASLt shape 上回退到 hipBLAS；该回退仍在 ROCm GPU 上执行，不是
CPU、Warp 或 MPLib fallback。

ROCm 7.14 容器验证还发现 shared-history L-BFGS kernel 在不同 ROCm 编译器间结果不稳定。
HIP 路径现保持全部计算在 GPU 上，但固定使用已验证的 stable global-buffer kernel；CUDA 的
shared-memory 路径不受影响。宿主 PyTorch 2.13.0+ROCm 7.1 和容器 PyTorch 2.10.0+ROCm
7.14.1 均通过回归。

## 实施原则

1. **保持双后端。** HIP 迁移不得以删除或弱化 CUDA 路径为代价。
2. **HIP 实现集中。** ROCm 专属实现放在
   `curobo/_src/curobolib/backends/hip_backend/`；共享代码只保留设备无关修复和薄分派。
3. **先建立基线，再优化。** 没有冻结输入和 CUDA/MPLib 基线的性能数字不计入验收。
4. **失败必须可见。** 不支持的 capability、构建错误和 planner 失败必须 fail-closed。
5. **语义一致才比较。** MPLib 不支持 partial-pose 时，不得声称与 cuRobo 完成了等价任务。
6. **不污染随机状态。** planner 初始化前后保存并恢复 NumPy/Torch RNG，确保不同 planner
   使用相同任务实例。
7. **保留 CUDA Graph。** CUDA 正常和性能路径继续使用 CUDA Graph；只允许测试或明确的
   调试路径关闭。HIP graph/capture 按独立 capability 管理。
8. **证据可追溯。** 所有结论必须能追溯到命令、revision、配置和机器可读原始数据。

## 迁移步骤总览

| 阶段 | 目标 | 阶段退出条件 |
| --- | --- | --- |
| 0 | 冻结范围、版本、任务和统计口径 | 支持矩阵和基线协议完成评审 |
| 1 | 建立可重复 ROCm 环境 | 干净环境能识别 GPU 并加载 PyTorch ROCm |
| 2 | 建立运行时检测和后端分派 | `auto` 正确选择，unsupported capability fail-closed |
| 3 | 建立隔离 HIP 构建和打包 | 四模块可在干净树中编译、加载和执行 |
| 4 | 迁移核心 GPU 算子 | forward/backward、梯度和碰撞数值门禁通过 |
| 5 | 打通全部规划器 | IK、TrajOpt、graph、single 和 batch 测试通过 |
| 6 | 接入 RoboTwin/XPolicyLab | 固定调用面可回放，且没有 MPLib/CPU fallback |
| 7 | 完成正确性、性能和稳定性验收 | 冻结任务成功率、2× 性能和 1,000 次长测通过 |
| 8 | 完成 CUDA 回归、CI 和发布 | 双平台 CI、安装复现、wheel/容器和文档通过 |

阶段内可以并行开发，但不得跳过门禁。例如，单次 RoboTwin episode 成功不能替代核心算子
的梯度测试，1,000 次 primitive soak 也不能替代完整 episode 稳定性。

## 阶段 0：冻结范围、工作负载和基线

### 操作步骤

1. 固定 GPU 型号、目标 `gfx`、ROCm、PyTorch、Python 和编译器版本。
2. 固定 cuRobo、RoboTwin、XPolicyLab、机器人资产和任务配置 revision。
3. 静态审计 RoboTwin planner 调用，记录：

   - single、batch、goal-set 和 partial-pose；
   - 输入 tensor 的 dtype、device 和 shape；
   - world 类型、collision checker 和 attached object；
   - planner seeds、超时、重试、graph/capture 和插值配置；
   - 输出轨迹的 shape、消费边界和成功判据。

4. 冻结任务、seeds 和超时。小规模开发集至少包含 `stack_blocks_two` seeds 0--3；正式集
   还应覆盖更多 seeds、至少一个更长任务和左右臂不同分配。
5. 在 NVIDIA CUDA cuRobo 和 CPU MPLib 上采集基线。每种配置至少重复 5 次，分别记录
   初始化、warm-up、稳态请求和完整 episode 延迟。
6. 固定成功率、碰撞、终点误差、约束误差、超时、显存和延迟的统计方法。

### 必须产出

- `environment.json`：硬件、驱动、依赖和 revision；
- `workload_manifest.json`：任务、seeds、资产、planner 参数和超时；
- `call_surface.md`：RoboTwin/XPolicyLab 调用面；
- CUDA 和 MPLib 原始基线 JSON；
- 支持能力和明确不支持能力列表。

### 验收标准

- 所有评测请求都能映射到待迁移模块或明确的阻塞项；
- CUDA、ROCm、MPLib 使用同一任务实例和统计口径；
- RNG 不因 planner 初始化而改变任务生成；
- partial-pose 等语义不等价请求明确标注，不混入加速比；
- 另一位开发者可以仅根据 manifest 复现相同输入。

## 阶段 1：建立可重复 ROCm 环境

### 操作步骤

1. 安装与目标系统匹配的 ROCm PyTorch，并确认 `torch.version.hip` 非空。
2. 确认 PyTorch 能枚举全部目标 GPU，并记录设备名称和 `gfx`。
3. 验证 `hipcc`、CMake/Ninja 和 PyTorch Extension ABI。
4. 确认环境未安装或加载 NVIDIA CUDA Runtime 作为隐藏依赖。
5. 创建锁定文件、容器定义或等价的可重复环境描述。

```bash
source .venv/bin/activate
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_properties(index).gcnArchName)
PY
```

### 验收标准

- `torch.cuda.is_available()` 为 true，且这是 PyTorch ROCm 的兼容命名空间；
- `torch.version.hip`、设备名称和 `gfx1201` 被记录；
- float32 tensor、基本算术和 backward 在 GPU 上成功；
- 干净环境的安装步骤可重复；
- 正式支持组合中不存在未说明的 ROCm/PyTorch runtime 版本偏差。

## 阶段 2：运行时检测和后端分派

### 操作步骤

1. 把“加速器类型”和“PyTorch device 字符串”分开建模。ROCm 下 device 仍通常写作
   `cuda`，不能据此误判为 NVIDIA。
2. 根据 PyTorch 构建信息、可见设备和显式环境变量选择 `cpu`、`cuda` 或 `hip` 后端。
3. 增加 runtime preflight，输出 accelerator、backend、PyTorch/ROCm 版本、GPU 和 `gfx`。
4. 为 kinematics、collision、optimization、trajectory、mesh、voxel 和 graph/capture 建立
   capability 表。
5. unsupported capability 在规划前报错，不进入 CPU/Warp fallback。

```bash
source .venv/bin/activate
pytest -n0 -q \
  curobo/tests/_src/test_runtime_accelerator.py \
  curobo/tests/_src/curobolib/test_backend_selection.py \
  curobo/tests/_src/curobolib/test_hip_runtime_preflight.py \
  curobo/tests/_src/curobolib/test_hip_backend_loader.py
```

### 验收标准

- ROCm 的 `auto` 模式选择 `backend=hip`；
- NVIDIA PyTorch 的 `auto` 模式仍选择 `backend=cuda`；
- CPU 环境不尝试编译或加载 GPU extension；
- 显式要求不支持能力时错误信息包含 capability 和建议动作；
- 日志中没有“HIP 不可用后自动使用 CPU/MPLib”的路径。

## 阶段 3：隔离 HIP 构建和打包

### 操作步骤

1. 对 CUDA 源码做构建时 hipify，不直接改写共享源文件。
2. 将生成源码、对象、缓存和 extension 输出隔离到
   `build/hip_backend/curobolib/` 或等价目录。
3. 分别构建 kinematics、geometry、optimization 和 trajectory 模块。
4. 明确 `PYTORCH_ROCM_ARCH=gfx1201`，检查产物包含目标架构。
5. 修正 HIP 不支持的 CUDA API、头文件、intrinsic 和编译选项。
6. 通过 Python loader 延迟加载模块，构建失败时保留完整诊断。
7. 将相同流程接入 wheel 和容器，而不是只支持开发树原地编译。

HIP 专属实现的目录契约为：

```text
curobo/_src/curobolib/backends/hip_backend/
├── backend_loader.py
├── build_hipify.py
├── runtime_preflight.py
├── collision_sphere_cuboid.py
├── cost_tool_pose.py
├── cost_cspace_position.py
├── cost_cspace_state.py
├── cost_cspace_distance.py
├── optimizer_levenberg_marquardt.py
└── optimizer_seed_ik.py
```

共享 planner 中的设备无关正确性修复可以保留在共享代码，例如 revolute joint 的 $$2\pi$$
等价 seed 归一化；HIP 专属算法不得散落在公共 API 或 CUDA loader 中。

### 验收标准

- 干净源码树一次命令构建四个模块；
- 生成物不修改共享 CUDA 源、PyTorch 头文件或 site-packages；
- extension 可重复导入，forward/backward smoke 通过；
- 二进制包含 `gfx1201`，不包含意外 NVIDIA 运行时依赖；
- wheel/容器构建使用同一构建入口和同一目标架构配置。

## 阶段 4：迁移核心 GPU 算子

按依赖顺序迁移和验收：

1. kinematics forward/backward；
2. self-collision；
3. sphere-cuboid 离散、swept、多环境和 mask；
4. ToolPose 和 C-space costs；
5. Levenberg-Marquardt、SeedIK、L-BFGS 和 line search；
6. trajectory integration、B-spline 和 interpolation。

每个算子必须同时测试正常输入、边界输入、禁用 mask、batch 和 backward。测试使用 float32，
不得使用 float64 掩盖精度或类型问题。

```bash
source .venv/bin/activate
CUROBO_USE_HIP=1 pytest -n0 -q \
  curobo/tests/_src/curobolib/test_hip_kinematics_integration.py \
  curobo/tests/_src/curobolib/test_hip_geometry_integration.py \
  curobo/tests/_src/curobolib/test_hip_collision_sphere_cuboid.py \
  curobo/tests/_src/curobolib/test_hip_tool_pose_integration.py \
  curobo/tests/_src/curobolib/test_hip_cspace_cost_integration.py \
  curobo/tests/_src/curobolib/test_hip_optimization_integration.py \
  curobo/tests/_src/curobolib/test_hip_trajectory_integration.py
```

### 数值验收标准

- forward 与已确认参考实现逐元素比较，使用为各算子冻结的 `atol`/`rtol`；
- backward 与 CUDA 或有限差分方向和量级一致；
- collision distance、法向和梯度符号正确；
- swept collision 不漏掉只在轨迹中间发生的碰撞；
- NaN、Inf、越界和空 batch 有明确行为；
- profiler 中规划热路径没有 NumPy、Warp CPU 或不必要的 device-to-host 同步；
- wave32/wave64 假设、reduction、shuffle 和 block 尺寸均有专项测试。

## 阶段 5：打通求解器和规划器

### 操作步骤

1. 验证 SeedIK 和完整 IK，包括 single、batch、goal-set、position-only 和 partial-pose。
2. 验证 TrajOpt 的 C-space、pose cost、smoothness、collision 和边界条件。
3. 验证 graph planner 的采样、连边、edge feasibility、批处理和多环境。
4. 验证 MotionPlanner 的 warm-up、IK → graph → TrajOpt 和 interpolation。
5. 验证 BatchMotionPlanner 多次 attempt 的结果合并。
6. 为已知可行、已知不可行和碰撞场景建立冻结回放。
7. 对成功 batch slot 保存按自身 `last_tstep` 裁剪的实际轨迹，并独立复核 endpoint、约束和
   碰撞；不得只保存 batch 级最大长度或 success mask。
8. 若 batch 结果要供后续 single 请求复用，必须逐项匹配起点、目标位置、目标四元数和约束
   语义。缓存未命中时必须执行正常规划，失败缓存不得覆盖以前的有效缓存。

BatchMotionPlanner 的合并测试必须覆盖：第一次 attempt 解决部分 batch，后续 attempt 解决另一
部分 batch；合并后每个 batch 的 `success`、trajectory、endpoint、metrics 和 seed rank 必须
全部来自同一个 attempt。只复制轨迹或只复制 success 都不能通过验收。

### 验收标准

- graph planner 规定测试集全部通过；
- MotionPlanner 规定测试集全部通过；
- BatchMotionPlanner 规定测试集和跨 attempt 合并专项测试全部通过；
- 已知不可行输入不会被标为成功；
- 每条成功轨迹均重新计算 endpoint pose、joint limit、self/world collision 和约束误差；
- 同一 target 的 batch 和 single 结果差异可由 seed 数或配置解释，不存在索引串位；
- solver 返回的 success mask 与逐轨迹独立复核一致；
- batch-to-single 缓存只能返回经过独立复核的成功轨迹，且每个 slot 使用自己的有效长度；
- 局部 IK/refinement 不污染全局 RNG，且不能通过关闭碰撞或放宽终点阈值获得成功。

## 阶段 6：接入 RoboTwin/XPolicyLab

### 6.1 准备冻结上游 checkout

仓库提供 revision 校验和幂等修改工具：

```bash
source .venv/bin/activate

# 只检查，不写入。
python benchmark/robotwin_rocm_prepare.py \
  --robotwin-root /path/to/RoboTwin

# 应用 adapter、fail-closed 和 RDNA 兼容修改。
python benchmark/robotwin_rocm_prepare.py \
  --robotwin-root /path/to/RoboTwin \
  --apply

# 再次执行必须报告已准备完成，不能重复修改。
python benchmark/robotwin_rocm_prepare.py \
  --robotwin-root /path/to/RoboTwin
```

工具只接受冻结的 RoboTwin/XPolicyLab revision。上游 revision 或源码形状变化时必须重新审计，
不能模糊套用旧补丁。

### 6.2 替换 planner 接口

RoboTwin 应通过 `curobo.robotwin` adapter 使用当前公共 API，并在 SAPIEN 初始化和计时前调用
通用 runtime preflight：

```python
import os

import curobo.logging as logger
from curobo.robotwin import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
    validate_robotwin_runtime,
)
from curobo.types import JointState, Pose as CuroboPose

runtime_info = validate_robotwin_runtime(
    required_accelerator="rocm" if os.environ.get("CUROBO_REQUIRE_ROCM") == "1" else None
)
```

adapter 只负责 v0.7.8 schema、返回 shape 和公共 API 适配。ROCm 算法继续位于
`hip_backend/`。RoboTwin batch 是多个候选抓取姿态，只要求任一候选成功，因此 batch planner
的退出比例应为 `1 / batch_size`，不能在已有候选成功后继续无意义地执行全部 attempts。
通用 adapter 同时接受匹配的 NVIDIA CUDA/CUDA backend；只有显式设置
`CUROBO_REQUIRE_ROCM=1` 的 ROCm 验收才拒绝 CUDA。不得在公共 RoboTwin adapter 模块顶层导入
HIP 实现或无条件调用 ROCm-only preflight。

双后端入口修复后，使用 `CUROBO_REQUIRE_ROCM=1` 重新执行真实 SAPIEN smoke：R9700
`gfx1201` 上仍选择 `accelerator=rocm`、`backend=hip`，规划状态为 `Success`，输出轨迹 shape 为
`[281, 6]`。证据位于
`build/rocm_acceptance/robotwin_sapien_smoke_dual_backend_runtime.json`。随后 RTX 4090 上的真实 CUDA
分派、核心测试、MotionPlanner 和 CUDA Graph 验收也已通过。

### 6.3 禁止 fallback 并隔离 RNG

- OpenWAM 捕获 planner 初始化异常后创建 `MplibPlanner` 的路径必须关闭；
- 日志出现 `planner fallback engaged` 时，该 episode 立即判为失败；
- episode runner 在 planner 初始化前保存 NumPy/Torch RNG，初始化后恢复；
- MPLib runner 必须明确记录 `constraint_semantics=ignored_by_mplib`；
- MPLib 不支持的 partial-pose 不得假装与 cuRobo 语义等价。

### 6.4 执行回放和完整 episode

至少覆盖：

- single pose；
- batch orientation candidates；
- goal-set；
- hold-position 和 hold-orientation partial-pose；
- 左右臂不同任务分配；
- 失败请求的冻结重放；
- 完整 SAPIEN episode；
- 真实 XPolicyLab policy server。

```bash
source .venv/bin/activate
CUROBO_USE_HIP=1 pytest -n0 -q \
  curobo/tests/test_robotwin_adapter.py \
  curobo/tests/test_robotwin_rocm_prepare.py \
  curobo/tests/test_robotwin_episode_summary.py \
  curobo/tests/test_robotwin_planner_request_compare.py
```

真实 policy server 评测前必须先验证冻结 revision、非占位策略、checkpoint、策略入口和策略
Python 环境。`demo_policy` 只验证协议 wiring，不能计入真实策略门禁：

```bash
source .venv/bin/activate
python benchmark/robotwin_xpolicy_readiness.py \
  --robotwin-root /path/to/RoboTwin \
  --policy <real-policy-name> \
  --checkpoint /path/to/robotwin/checkpoint \
  --policy-python /path/to/policy/python \
  --output build/rocm_acceptance/xpolicy_readiness.json
```

readiness 结果必须满足 `ready_to_launch=true`，并保存 checkpoint 内全部文件的大小和 SHA-256。
这只证明服务具备启动前提，不能替代 policy server、环境 client、RoboTwin simulator 和 ROCm
cuRobo planner 共同参与的完整 episode。

当前真实策略采用 LingBot-VLA v2 6B，不能使用冻结 XPolicyLab 中面向旧版权重的
`LingBot_VLA` adapter。真实评测必须使用 RoboTwin 的 `protocol=lingbot_vla_v2` websocket
server/client 路径。推荐部署拓扑如下：

```text
NVIDIA RTX 4090 主机
  LingBot-VLA v2 6B websocket policy server
             │ 真实 action inference
             ▼
AMD R9700 主机
  冻结 RoboTwin/XPolicyLab + SAPIEN + cuRobo HIP backend
  CUROBO_REQUIRE_ROCM=1, accelerator=rocm, backend=hip
```

策略端 checkpoint、DINO 和 depth 模型必须完整。服务端健康检查只表示进程已就绪，不算真实
XPolicy 验收；至少完成一次真实 websocket inference 后才能启动 RoboTwin episode。若模型配置
请求 `flash_attention_2`，但镜像没有 `flash_attn`，应使用经过验证的独立配置或最小 overlay
切换到 PyTorch SDPA，不得换成占位策略绕过启动错误。

初次服务错误地加载了通用基础权重 `robbyant/lingbot-vla-v2-6b`，因此两个 episode 虽完成
1,600 个动作步，夹爪命令却始终接近打开，任务成功率为 0/2。官方 RoboTwin 后训练权重实际为
`robbyant/lingbot-vla-v2-6b-robotwin`，可加载的 safetensors 位于
`checkpoints/global_step_50000/hf_ckpt/`，不能把仓库顶层目录直接传给只扫描当前目录的 loader。

第二个配置错误是把真实机器人示例的 `use_length=25` 用于 RoboTwin。官方 RoboTwin launcher
默认使用 `use_length=50`。使用正确 checkpoint 但仍截断为 25 步时，seed 0 虽能闭合夹爪并搬运
方块，最终仍未堆叠。恢复官方 50 步 action chunk 后，同一 seed 在第 378 步成功，方块最终高度
约为 0.765 m 和 0.815 m。seed 1 运行到 800 步后失败，因此当前冻结分母中的真实策略成功率为
1/2，而不是 100%。两个 episode 均保留在机器可读汇总中，没有删除失败样本。

成功 episode 证明三路 `480×640×3 uint8` 图像、14 维状态、seen instruction、websocket
inference、SAPIEN 执行和 ROCm cuRobo 规划组成的完整动作链可以完成真实任务。原始日志为
`build/rocm_acceptance/xpolicy/stack_blocks_two_seed0_robotwin_checkpoint_use_length50.log` 和
`build/rocm_acceptance/xpolicy/stack_blocks_two_seed1_robotwin_checkpoint_use_length50.log`；评测
overlay 会记录全部 action chunk、双夹爪范围、最终状态和 actor 位置。

初次 episode 还暴露出 RoboTwin 的 qpos 执行路径会调用 MPLib TOPP。该路径现已迁移为 cuRobo
设备端三次插值：根据机器人位置、速度和加速度限制生成零起止速度的轨迹，并在
`CUROBO_REQUIRE_ROCM=1` 时禁止创建 MPLib planner。AMD R9700 实机 smoke 结果为
`accelerator=rocm`、`backend=hip`、`joint_action_backend=curobo`、
`joint_action_mplib_used=false`，单次插值输出 shape 为 `[37, 6]`。证据位于
`build/rocm_acceptance/xpolicy/sapien_qpos_smoke_host.json`。

还需注意：episode 进程退出码为 0 只表示评测脚本正常结束，不等于任务成功。当前已通过独立
SAPIEN smoke 验证左右臂均为 `CuroboPlanner`，且 ROCm qpos 路径不创建或调用 MPLib planner。
机器可读摘要保留 1/2 的完整结果，并仅依据至少一个真实任务成功、零 fallback、非占位策略和
冻结 revisions 满足当前真实 XPolicy 发布门禁；它不表示策略成功率已经达到 100%。

### 验收标准

- 每个 episode 记录 `accelerator=rocm` 和 `backend=hip`；
- 左右臂 planner 均为 `CuroboPlanner`；
- 全部冻结 episode 没有 CPU、Warp CPU 或 MPLib fallback；
- 轨迹输出 shape 满足 RoboTwin 消费契约；
- partial-pose 的约束轴、误差和 collision 独立复核通过；
- 小规模开发集 seeds 0--3 必须 4/4，通过后才能扩展正式任务集；
- 真实 policy server 的完整动作链成功，而不只是健康检查、离线 planner replay 或 demo policy；
- 最终 `xpolicy_evaluation_summary.json` 必须满足 `policy != demo_policy`、`episode_count > 0`、
  `accelerator=rocm`、`backend=hip`、`fallback_count=0`，并匹配冻结的 RoboTwin/XPolicyLab
  revisions。

## 阶段 7：正确性、性能和稳定性验收

### 统一统计口径

- **成功率分母：** 包含全部预先冻结的 episode。超时、崩溃、fallback、无效输出和规划失败
  都计为失败，不得删除失败 seed 后重算。
- **延迟：** 初始化、warm-up、稳态请求和完整 episode 分开统计。p50/p90/p99 只基于双方
  语义一致且都成功的稳态请求，同时单独报告失败率和超时率。
- **规划质量：** 对成功轨迹重新计算碰撞、joint limit、终点误差和 partial-pose 约束误差。
- **性能比较：** 使用相同机器负载、任务、seeds、planner 参数、超时和计时边界。
- **显存：** warm-up 后采样至少 1,000 次，保存完整序列，不能只保存起止值。

### 正确性门槛

- 小规模冻结集成功率 100%；
- 正式任务集 ROCm 成功率不低于 CUDA cuRobo 超过 2 个百分点；
- 成功轨迹碰撞数为 0；
- joint limit 和任务约束违反数为 0；
- endpoint 和 partial-pose 误差不超过阶段 0 冻结阈值；
- batch success mask 与逐 slot 独立验证一致。

### 性能门槛

- 在双方语义等价且都成功的请求上，ROCm 稳态 p50 至少比 CPU MPLib 快 2×；
- 同时报告 p90、p99、超时率和峰值显存；
- 首次 JIT/extension build 不计入稳态规划，但必须单独报告；
- warm-up 不得污染任务 RNG；
- 不得通过降低 seeds、关闭 collision 或放宽误差阈值获得加速。

当前代码使用起始关节状态、目标、arm、请求类型和约束语义进行严格匹配。`stack_blocks_two`
seeds 0--3 共得到 16 个匹配请求，其中双方都成功的请求为 5 个。合并结果如下：

| 指标 | ROCm cuRobo | CPU MPLib |
| --- | ---: | ---: |
| p50 | 2.339 s | 5.272 s |
| p90 | 2.984 s | 5.639 s |
| p99 | 3.354 s | 5.815 s |

p50 加速为 2.254×，`minimum_both_success=5` 和 `minimum_speedup=2.0` 请求级门禁返回
`passed=true`。在此基础上，又对同一起点、目标、arm、请求类型和约束语义的冻结 batch 请求
分别重复 ROCm 和 MPLib 5 次：

| 指标 | ROCm cuRobo | CPU MPLib |
| --- | ---: | ---: |
| p50 | 2.581 s | 5.340 s |
| p90 | 2.720 s | 5.507 s |
| p99 | 2.757 s | 5.530 s |

重复请求的 p50 加速为 2.069×，5/5 次双方都成功，因此 P2 的 MPLib 请求性能单项门禁已通过。
重复调用会推进 planner 内部随机状态，可能改变后续 episode，所以该 run 只能作为请求性能证据；
episode 正确性继续使用普通单次调用模式；当前 `stack_blocks_two` 和 `stack_blocks_three` 均为
4/4，并分别通过冻结任务门禁。

### 稳定性门槛

- primitive scene 规划至少 1,000 次，成功 1,000 次；
- 完整 RoboTwin 混合任务执行规定长测，无进程崩溃、GPU fault 或死锁；
- warm-up 后显存不存在随迭代持续增长趋势；
- 重复运行的成功率和延迟波动在阶段 0 冻结范围内；
- profiler 证明规划热路径没有未批准的 host 往返。

冻结混合任务长测使用 `stack_blocks_two` 和 `stack_blocks_three` seeds 0--3，共 8 个 episode。
汇总器要求 `(task, seed)` 唯一、至少两个任务、全部 episode 成功、左右臂均使用
`CuroboPlanner`，并且 fallback、GPU fault 和 deadlock 均为 0：

```bash
source .venv/bin/activate
python benchmark/robotwin_mixed_task_soak_summary.py \
  build/rocm_acceptance/robotwin_stack_blocks_two_seed0_current.json \
  build/rocm_acceptance/robotwin_stack_blocks_two_seed1_distributed_local_chunks.json \
  build/rocm_acceptance/robotwin_stack_blocks_two_seed2_current.json \
  build/rocm_acceptance/robotwin_stack_blocks_two_seed3_current.json \
  build/rocm_acceptance/robotwin_stack_blocks_three_seed0_post_success_retreat_fix.json \
  build/rocm_acceptance/robotwin_stack_blocks_three_seed1_post_success_retreat_fix.json \
  build/rocm_acceptance/robotwin_stack_blocks_three_seed2_no_final_retreat.json \
  build/rocm_acceptance/robotwin_stack_blocks_three_seed3_post_success_retreat_fix.json \
  --minimum-tasks 2 \
  --minimum-episodes 8 \
  --output build/rocm_acceptance/mixed_task_soak_summary.json
```

当前统一选择每个 seed 的正式证据后，8 个 episode 全部成功，成功率为 100%；fallback、
GPU fault 和 deadlock 均为 0。`mixed_task_soak_summary.json` 的 `passed=true`，P2 混合任务
长测门禁已通过。

可使用仓库工具生成 episode 汇总和请求级对照：

```bash
source .venv/bin/activate
python benchmark/robotwin_rocm_episode_summary.py --help
python benchmark/robotwin_planner_request_compare.py --help
python benchmark/robotwin_planner_request_compare_summary.py --help
python benchmark/rocm_motion_soak.py --help
```

汇总工具必须把成功 episode 延迟和全部退出时间分开。失败提前退出不能污染成功请求的
p50/p90；没有成功样本时成功延迟应为 `null`，而不是 0。汇总还必须要求
`accelerator=rocm`、`backend=hip`、左右臂均为 `CuroboPlanner`、seed 唯一，并记录每个输入
JSON 的 SHA-256。重复 seed 不得改变冻结成功率分母。

## 阶段 8：CUDA 回归、CI 和发布

### CUDA 回归

在 NVIDIA 主机执行与 ROCm 相同的核心算子、solver、MotionPlanner 和 RoboTwin 冻结回放，
并额外验证：

- `auto` 仍选择 CUDA backend；
- RoboTwin adapter 的通用 runtime preflight 接受 CUDA/CUDA 组合，并拒绝 backend 错配；
- CUDA extension 构建路径未被 hipify 逻辑污染；
- CUDA Graph 默认开启且 capture/replay 正常；
- 数值、成功率和性能相对迁移前基线无不可接受回归；
- ROCm capability 和环境变量不会改变默认 CUDA 行为。

CUDA core JUnit、CUDA RoboTwin 汇总和同任务 ROCm 汇总必须由统一工具合并，不能仅凭某个 CUDA
单测 job 变绿就通过发布门禁：

```bash
python benchmark/cuda_regression_summary.py \
  --runtime build/cuda_acceptance/core/runtime.json \
  --junit build/cuda_acceptance/core/tests.xml \
  --cuda-episodes build/cuda_acceptance/robotwin/stack_blocks_two_summary.json \
  --rocm-episodes build/rocm_acceptance/robotwin/stack_blocks_two_summary.json \
  --output build/rocm_acceptance/cuda_regression_summary.json
```

汇总器要求 CUDA/CUDA runtime、全部 core/motion 测试零失败、CUDA Graph capture/replay 测试未
跳过、默认 graph 配置测试通过、CUDA 与 ROCm 使用相同 task/revisions/seeds，且 ROCm 成功率
落后 CUDA 不超过 2 个百分点。该验收已在 RTX 4090 上完成：核心测试 289 passed、0 failed、
15 skipped，跳过项均为预期 ROCm-only 测试；PyTorch 为 2.9.1+cu126，CUDA Graph
capture/replay 和默认启用测试均通过。相同冻结 `stack_blocks_two` seeds 0--3 上，CUDA 为
1/4，ROCm 为 4/4，因此 ROCm 成功率没有比 CUDA 低超过 2 个百分点。统一汇总文件
`build/rocm_acceptance/cuda_regression_summary.json` 返回 `passed=true`。CUDA
`stack_blocks_three` seeds 0--3 为 0/4，只作为基线事实记录，不能用于豁免 ROCm 长任务门禁。

### CI 矩阵

至少包含：

| 平台 | 必须执行 |
| --- | --- |
| CPU | import、runtime detection、fail-closed 和纯 Python 单元测试 |
| NVIDIA CUDA | extension、核心算子、solver、CUDA Graph 和 CUDA 回归 |
| AMD ROCm | HIP build、核心算子、solver、MotionPlanner 和 smoke episode |
| 定时验收 | 完整任务集、性能、1,000 次 soak 和显存采样 |

候选发布版本的 CUDA 和 ROCm CI 应至少连续通过 3 次，避免把偶发成功当作稳定支持。
每次 `Accelerator CI` 运行都会由 `candidate-run-evidence` job 上传包含 workflow run ID、候选
revision、完成时间、CUDA 结果、ROCm 结果和 Actions URL 的不可变 JSON。下载同一候选 revision
的运行证据后，使用以下命令生成发布门禁输入：

```bash
source .venv/bin/activate
python benchmark/ci_history_summary.py \
  build/ci_history/run-1.json \
  build/ci_history/run-2.json \
  build/ci_history/run-3.json \
  --revision "$(git rev-parse HEAD)" \
  --required-consecutive 3 \
  --output build/rocm_acceptance/ci_history_summary.json
```

聚合器按 `completed_at` 排序，并要求 workflow run ID 唯一。最新一次任一平台失败、revision
变化或少于三次都会清零相应的连续通过计数并返回非零退出码，不能通过重复复制同一 artifact
凑足三次。

定时 ROCm 验收同时执行 `stack_blocks_two` 和 `stack_blocks_three` seeds 0--3；任何任务未达到
100% 冻结成功率时 job 必须失败并上传已有证据，不能只运行开发集来获得绿色状态。
定时 CUDA baseline 执行同一 `stack_blocks_two` seeds 0--3；独立汇总 job 下载 CUDA core、
CUDA RoboTwin 和 ROCm RoboTwin artifacts，生成 `cuda_regression_summary.json`。

### 发布内容

- wheel 和/或容器；
- 完整支持矩阵；
- 安装、环境检查和最小 smoke 命令；
- RoboTwin/XPolicyLab 接入步骤；
- 已知限制和 unsupported capability；
- CUDA、ROCm、MPLib 原始验收数据和摘要；
- 故障诊断、回滚和问题报告模板。

### 验收标准

- 新机器无需修改源码即可安装；
- extension 在目标 `gfx` 上加载并执行；
- 支持矩阵内的 smoke 和冻结回放可重复；
- CUDA 和 ROCm CI 连续通过；
- 文档中的命令均在干净环境验证；
- 发布物能报告版本、revision、backend、ROCm/CUDA runtime 和目标架构。

## P0/P1/P2 验收门禁

### P0：平台与算子

- [x] 正式 ROCm/PyTorch 组合已锁定并记录在支持矩阵；
- [x] ROCm runtime 和 `hip` backend 可检测；
- [x] HIP 构建产物与共享源码隔离；
- [x] kinematics、primitive collision、cost、optimization 和 trajectory 首轮测试通过；
- [x] mesh/voxel/ESDF unsupported capability fail-closed；
- [x] 干净 wheel/容器安装验证通过；
- [x] CUDA 主机确认未受构建和分派改动影响，CUDA Graph 回归通过。

P0 全部通过后，才可以声明“ROCm 后端具备进入正式集成验收的条件”。

### P1：规划与 RoboTwin 集成

- [x] IK、TrajOpt、graph 和单任务 MotionPlanner 首轮测试通过；
- [x] BatchMotionPlanner 跨 attempt buffer 复用和字段合并专项验证通过；
- [x] BatchMotionPlanner 关键成功 slot 的 endpoint、约束和碰撞独立复核通过；
- [x] RoboTwin adapter 和 revision 校验工具通过测试；
- [x] RNG 隔离，HIP 与 MPLib 生成相同任务实例；
- [x] `stack_blocks_two` seeds 0--3 达到 4/4；
- [x] `stack_blocks_three` seeds 0--3 达到 4/4；
- [x] 真实 XPolicyLab policy server 至少一个完整 episode 成功，固定两 seed 结果为 1/2；
- [x] 已执行的冻结 episode 全程没有 MPLib、Warp CPU 或规划热路径 CPU fallback。

P1 全部通过后，才可以声明“冻结的 RoboTwin/XPolicyLab 工作负载可使用 ROCm cuRobo”。

已确认的真实评测权重为 LingBot-VLA v2 6B RoboTwin 后训练 checkpoint。冻结 XPolicyLab 的旧
`LingBot_VLA` adapter 与该权重不兼容，必须使用 RoboTwin 的 `lingbot_vla_v2` websocket 协议、
嵌套 `hf_ckpt` 目录和官方 `use_length=50`。AMD 端两个冻结 episode 的结果为 1/2：seed 0 在
378 步成功，seed 1 在 800 步失败。机器可读摘要同时验证左右臂均为 `CuroboPlanner`、
`joint_action_backend=curobo`、`joint_action_mplib_used=false` 和 `fallback_count=0`。

### P2：发布

- [x] ROCm 相对 CUDA 的成功率差不超过 2 个百分点；
- [x] ROCm 在等价成功请求上的稳态 p50 至少比 CPU MPLib 快 2×；
- [x] primitive scene 1,000/1,000 soak 通过且显存漂移为 0；
- [x] 8-episode RoboTwin 混合任务长测通过；
- [x] NVIDIA CUDA 回归和 CUDA Graph 验收通过；
- [ ] CUDA/ROCm CI 连续 3 次通过；
- [x] wheel 在隔离目录安装、容器从源码构建并完成 214 项干净环境复现；
- [ ] 支持矩阵、限制、原始数据和发布文档完成独立复核。

只有 P0、P1 和 P2 全部通过，才能宣布首个 AMD RDNA/ROCm 版本迁移完成。

## 当前最短推进路径

1. 保留真实 XPolicy 1/2 的固定分母和原始日志；若继续提高策略成功率，不得修改任务、权重、
   seeds 或成功定义，也不得删除 seed 1 失败样本。
2. 配置自托管 CUDA/ROCm runners，使候选 revision 连续 3 次通过，并保存
   `ci_history_summary.json`。
3. 由非实现者复核支持矩阵、原始证据和发布结论，生成 `independent_review.json`。

## 失败分类和停止条件

出现下列情况时，应停止扩大任务集，先冻结最小失败输入：

- batch 标记成功，但轨迹独立复核失败；
- 同一输入重复运行结果无法由随机 seed 解释；
- self/world collision 被关闭后才成功；
- profiler 发现规划热路径进入 CPU 或频繁同步；
- 显存持续增长、GPU fault、NaN 或死锁；
- HIP 修复改变 CUDA 数值或 CUDA Graph 行为；
- 上游 revision 或调用面变化；
- 只能通过放宽误差、减少 seeds 或删除失败样本达到门槛。

失败样本至少保存起始 joint state、目标 pose、world、constraint、planner config、随机状态、
success mask、IK seeds、轨迹、metrics 和 collision 诊断。

## 验收证据目录建议

```text
build/rocm_acceptance/
├── environment.json
├── workload_manifest.json
├── tests/
├── episodes/
├── request_compare/
├── soak/
├── profiler/
├── cuda_regression/
└── release_summary.json
```

`build/` 中的文件是本地验收产物，不应作为源码提交。正式发布应把不可变原始数据上传到
CI artifact、对象存储或版本化验收归档，并在 release summary 中记录校验和。

每份结果至少包含：

| 字段 | 内容 |
| --- | --- |
| 身份 | cuRobo、RoboTwin、XPolicyLab 和资产 revision |
| 环境 | GPU、`gfx`、驱动、ROCm、PyTorch、Python、编译器 |
| 配置 | backend、graph、batch、seeds、超时和 planner 参数 |
| 输入 | task、episode seed、起始状态、目标和 world 标识 |
| 输出 | success、错误、轨迹、约束和碰撞复核 |
| 性能 | 初始化、warm-up、p50/p90/p99、峰值显存 |
| 可追溯性 | 完整命令、日志路径、原始 JSON 和 profiler trace |
| 结论 | 通过/失败、负责人、日期和已知限制 |

## 最终签字表

| 验收域 | 负责人 | 证据位置 | 结论 | 日期 |
| --- | --- | --- | --- | --- |
| 支持矩阵和环境 |  |  |  |  |
| HIP 构建和核心算子 |  |  |  |  |
| IK/TrajOpt/graph |  |  |  |  |
| MotionPlanner/batch |  |  |  |  |
| RoboTwin/XPolicyLab |  |  |  |  |
| 正确性和成功率 |  |  |  |  |
| 性能和稳定性 |  |  |  |  |
| NVIDIA CUDA 回归 |  |  |  |  |
| wheel/容器和 CI |  |  |  |  |
| 发布批准 |  |  |  |  |

任一验收域为空、失败或无法追溯原始数据时，发布批准必须保持未通过。

CI 历史通过后，先生成交给非实现者的校验和清单：

```bash
source .venv/bin/activate
python benchmark/independent_review_prepare.py \
  --evidence-root build/rocm_acceptance \
  --revision "$(git rev-parse HEAD)" \
  --output build/rocm_acceptance/independent_review_request.json
sha256sum build/rocm_acceptance/independent_review_request.json
```

该工具要求九项底层发布证据全部存在，并记录每个文件的大小和 SHA-256。最终
`release_summary.json` 是这些证据和 reviewer 决策的派生结果，不纳入签署前清单，避免形成
循环校验。独立 reviewer 必须核对
原始 artifacts、固定 revisions/seeds、零 fallback、双平台 CI 历史和已知限制，再单独生成
`independent_review.json`。签署文件至少包含 `approved`、`reviewer`、`review_date`、
`reviewer_independent`、`reviewed_revision`、`artifact_checksums_verified`、`review_request` 和
`review_request_sha256`。发布汇总器会重新计算 request 及其中每个 artifact 的哈希；签署后任何
证据变化都会使门禁失效。当前实现者不得填写 reviewer、批准状态或校验和已验证字段。

最终发布结论由统一汇总器生成，缺失证据不会被解释为跳过或通过：

```bash
source .venv/bin/activate
python benchmark/rocm_release_acceptance.py \
  --evidence-root build/rocm_acceptance \
  --output build/rocm_acceptance/release_summary.json
```

汇总器要求 10 项 P0/P1/P2 证据全部满足契约。真实 XPolicy 当前已有一个完整任务成功，并保留
1/2 的固定分母；重新汇总后应为 8/10 通过，`release_ready=false`。剩余未通过项是双平台连续
CI 和独立复核。只有输出 `release_ready=true` 时才允许填写“发布批准”。
