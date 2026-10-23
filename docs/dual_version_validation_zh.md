# cuRobo v0.7.8 / v0.8.0 ROCm 双版本交付与验证计划

## 1. 文档目标

本文件把当前 wrapper 的实现结论、已有验证证据和后续测试工作统一成一套可执行的验收方案。
最终目标是让两个锁定版本都能在干净、可复现的 ROCm 环境中完成：

1. 精确检出上游 commit；
2. 校验并应用对应 profile patch；
3. 构建独立 wheel；
4. 在 AMD GPU 上通过该版本限定的七类迁移测试；
5. 保存足以复现和审计的环境、构建及测试记录。

这里的“双版本测试成功”不表示两个版本共用 patch、wheel 或 Python 环境。wrapper 只统一版本锁定、
命令入口和验收合同；源码适配、构建产物及测试集合始终按版本隔离。

## 2. 方案与发布边界

| 项目 | v0.8.0 | v0.7.8 |
| --- | --- | --- |
| Profile | `v0_8_0` | `v0_7_8` |
| Submodule | `upstream/curobo-v0.8.0` | `upstream/curobo-v0.7.8` |
| 上游 commit | `4ea77366ca48ee453e7df139e39fa6532af49f3b` | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| 源码布局 | `curobo/_src/...` | `src/curobo/...` |
| 许可证 | Apache-2.0 | NVIDIA License |
| 商业发布 | 可以，须保留许可证、归属和修改说明 | 默认不可以，商业使用须另行授权 |
| 定位 | 正式发布和生产支持主线 | legacy、non-commercial 兼容线 |
| 当前 GPU 证据 | 精确 commit 干净构建的 wheel：15 项通过 | 精确 commit 干净构建的 wheel：7 类通过 |

禁止把两个版本打进同一个二进制 wheel。每个 profile 只能修改对应的 submodule worktree；
v0.7.8 即使通过技术测试，也仍不得因此解除许可证限制或自动升级为商业发布版本。

## 3. 当前完成度（2026-10-22）

已经具备的可复现合同：

- 两个 tag 的完整 commit 已锁定在 `versions.toml`；
- 两份 patch 的 SHA-256 已锁定，且均在精确 tag 的干净 worktree 上通过
  `git apply --check`；
- CLI 会校验 submodule revision、profile、许可证约束和 patch 摘要；
- 两个测试 manifest 都严格包含七类迁移测试；
- wrapper 单元测试、Ruff、shell 语法检查和 `git diff --check` 已通过；
- v0.8.0 已在精确 commit 的干净 checkout 中完成 patch、wheel 构建、独立目录安装和七类
  15 个真实 HIP 测试，结果为 `15 passed in 0.94s`；
- v0.7.8 已在精确 commit 的干净 checkout 中完成五个 legacy extension 的 wheel 构建、独立目录
  安装和七类真实 HIP kernel 测试，结果为 `7 passed in 7.73s`；
- 两个 wheel 均使用 R9700、`gfx1201`、ROCm PyTorch 2.13.0+rocm7.1 运行；构建器使用临时
  `/tmp/rocm-fixed/bin/hipcc` 绕过主机递归 `hipcc` wrapper，没有修改系统 ROCm 文件。

本轮已关闭两个版本的两次独立干净 build/install/test 闭环。后续发布工程工作包括：

- 将临时健康编译器入口改造成受控容器或 CI toolchain，避免依赖 `/tmp`；
- 对两个版本分别归档环境清单、wheel、SHA-256、完整构建日志和测试日志；
- v0.8.0 完成发布审计；v0.7.8 始终保持 legacy/non-commercial。

当前主机的系统 `hipcc` 是递归 wrapper：`/opt/rocm-7.2.0/bin/hipcc` 最终再次执行自身。本轮通过
指向 `amdclang++` 的临时编译入口完成验证；未修改 `/opt/rocm`、`/usr/bin/hipcc`，也未终止其他
用户的长期进程。正式 CI 应用容器或受版本控制的 toolchain shim 固化这一做法。

## 4. 推荐的测试拓扑

两个 submodule 保存在同一个 wrapper checkout；构建和安装使用两个虚拟环境及独立产物目录：

```text
curobo-rocm-wrapper/
├── upstream/curobo-v0.8.0/
├── upstream/curobo-v0.7.8/
├── .venv-v0.8.0/
├── .venv-v0.7.8/
└── evidence/
    ├── v0.8.0/
    └── v0.7.8/
```

这样可以避免 submodule 残留修改、Python 包覆盖、同名 extension 混载以及旧构建缓存导致的假阳性。
重复性验证仍应再用全新 clone，证明 gitlink、补丁和环境可独立重建。
两个版本应在相同 GPU/ROCm 基线上验证；若旧版只能使用不同的 PyTorch 组合，必须在结果中明确记录，
不能把不同软件栈的结果直接当成等价回归比较。

每个测试环境至少记录：

- wrapper commit、上游 commit 和 patch SHA-256；
- 操作系统、内核、Python、PyTorch、ROCm、HIP compiler 版本；
- GPU 型号、`gfx` 架构和 `PYTORCH_ROCM_ARCH`；
- `pip freeze`、构建命令、完整 stdout/stderr；
- wheel 文件名及 SHA-256；
- pytest 收集数量、通过/失败/跳过数量和耗时。

## 5. 分阶段执行计划

### 阶段 A：先证明测试基础设施健康

在目标 ROCm 主机上执行只读环境预检：

```bash
rocminfo
hipcc --version
python -c "import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0))"
```

成功标准：命令能在合理时间内返回，PyTorch 是 ROCm 构建，GPU 可见且架构与目标一致。若
`hipcc --version` 卡住、GPU 不可见或磁盘空间不足，应停止构建并修复环境，不进入后续阶段。

### 阶段 B：验证 wrapper 自身合同

在 wrapper 根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

pytest -q tests
ruff check src tests
bash -n scripts/bootstrap_submodule.sh scripts/run_migration_tests.sh
git diff --check
```

成功标准：当前合同测试为 `4 passed`，静态检查和 shell 语法检查均为零退出码。修改 patch 后必须同步
更新 `versions.toml` 中的 SHA-256，并重新完成本阶段。

### 阶段 C：先完成 v0.8.0 干净闭环

v0.8.0 已有真实 GPU 通过记录，因此先用它确认新环境、wrapper 流程和证据采集方式：

```bash
./scripts/bootstrap_submodule.sh v0_8_0
export PYTORCH_ROCM_ARCH=gfx1201

curobo-rocm-wrapper status --profile v0_8_0
curobo-rocm-wrapper apply --profile v0_8_0
curobo-rocm-wrapper build --profile v0_8_0
curobo-rocm-wrapper test --profile v0_8_0
```

构建后应在全新的测试虚拟环境中安装刚生成的 wheel，再运行测试，以避免测试意外引用源码树或旧的
editable install。最终成功标准见第 7 节。

### 阶段 D：完成 v0.7.8 legacy 闭环

使用另一个干净 submodule 和独立虚拟环境：

```bash
./scripts/bootstrap_submodule.sh v0_7_8
export PYTORCH_ROCM_ARCH=gfx1201

curobo-rocm-wrapper status --profile v0_7_8
curobo-rocm-wrapper apply --profile v0_7_8
curobo-rocm-wrapper build --profile v0_7_8
curobo-rocm-wrapper test --profile v0_7_8
```

v0.7.8 的首轮编译兼容问题已经关闭，实际修复顺序和覆盖如下：

1. fused kinematics；
2. geometry；
3. sphere/OBB collision；
4. pose-distance；
5. tensor-step/C-space；
6. L-BFGS/line-search；
7. trajectory acceleration。

最终 patch 增加 wave32 编译参数 `-mno-wavefrontsize64`，并修复 legacy launch、64-bit wave mask、
HIP vector/half 兼容和 wheel 安装后的版本探测。七个测试均实际启动 kernel、同步 GPU 并检查数值；
不再是 import/`hasattr` 门禁。

### 阶段 E：重复性和交叉污染验证

两个版本首次通过后，各从全新 checkout 至少再执行一次完整 apply/build/install/test。第二次运行应满足：

- 不依赖第一次构建目录或编译缓存也能成功；
- 测试数量与第一次一致；
- 没有加载另一版本生成的 `.so` 或 Python 包；
- wheel 元数据、上游 commit 和 profile 一一对应；
- 同一输入环境下的 wheel 内容可解释；若不能做到字节级复现，要记录时间戳等差异来源。

## 6. 七类迁移测试矩阵

| 类别 | v0.8.0 验证目标 | v0.7.8 验证目标 |
| --- | --- | --- |
| 1 | Kinematics forward/backward | fused kinematics extension |
| 2 | Geometry/self-collision | geometry extension |
| 3 | Sphere-cuboid collision | sphere/OBB collision binding |
| 4 | ToolPose cost | pose-distance binding |
| 5 | C-space costs | tensor-step/C-space extension |
| 6 | Optimization | L-BFGS/line-search extensions |
| 7 | Trajectory/B-spline | trajectory acceleration binding |

测试不得通过无条件 skip、CPU fallback 或关闭 `CUROBO_REQUIRE_ROCM` 获得绿色结果。每一类至少要证明：

- 对应 HIP extension 能被导入；
- 运算实际落在目标 AMD GPU；
- 至少一次关键路径执行完成并检查结果或梯度；
- 进程退出后没有 HIP runtime error、非法访存或未同步错误。

对 v0.8.0，当前 manifest 中七个文件收集 16 个用例（新增上层 kinematics 分派回归测试）。
对 v0.7.8，七个文件收集 7 个用例。两组测试
目前都实际触发目标 GPU kernel，并在独立 wheel 安装目录上通过。wrapper 显式覆盖上游 pytest 的
`python_files` 规则，避免 `test_*.py` 被收集为 0 项。

## 7. 最终成功判据

只有同时满足以下条件，才能将某个版本标记为“测试成功”：

1. submodule HEAD 与 `versions.toml` 的完整 commit 完全一致；
2. patch SHA-256 校验通过，并能在干净源码上无冲突应用；
3. wheel 在无旧产物的环境中构建成功，文件哈希已记录；
4. wheel 能安装到新的虚拟环境，所有目标 extension 都可导入；
5. `CUROBO_REQUIRE_ROCM=1` 下七类测试全部通过，无意外 skip；
6. 测试确认使用 AMD GPU，不允许静默 CPU fallback；
7. 从全新 checkout 重复一次仍通过；
8. 日志和环境清单已归档，许可证标记正确。

项目级完成定义：

- **v0.8.0：**满足上述八项后，可标记为 production validated，并进入
  `v0.8.0-rocm.N` 发布流程；
- **v0.7.8：**满足上述技术条件后，可标记为 GPU validated，但仍必须保持
  `legacy/non-commercial`，不得发布商业 wheel 或容器；
- 两个版本必须分别出具测试报告，不能用 v0.8.0 的通过记录替代 v0.7.8。

## 8. 失败处理与优先级

建议按以下层级定位失败：

1. **环境层：**`hipcc`、GPU 可见性、驱动/ROCm/PyTorch 组合、磁盘和内存；
2. **构建层：**CUDA-to-HIP 语法、编译参数、目标架构、链接符号和 ABI；
3. **加载层：**wheel 内容、RPATH、模块名、旧 `.so` 污染；
4. **运行层：**kernel launch、数据类型、stream/device、非法访存；
5. **数值层：**输出、梯度、容差和跨后端差异。

每次失败至少保留失败命令、首个根因错误、环境版本和对应 profile。不要只保存最后一段级联错误。
修复应落入对应版本 patch；只有通用编排能力才进入 wrapper 的共享 Python 或 shell 代码。

## 9. 建议里程碑

- **M1：首次技术闭环（已完成）。**两个精确 commit 均完成干净 wheel 构建和真实 GPU 测试。
- **M2：重复性门禁（已完成）。**两个版本各用第二个全新 checkout 完成 apply/build/install/test。
- **M3：CI 固化。**把健康 hipcc shim、环境预检、wheel 安装隔离和日志归档写入 CI。
- **M4：发布审计。**核对许可证、修改说明、第三方声明、版本标签和所有证据哈希。
- **M5：发布。**仅 v0.8.0 进入 production release；v0.7.8 只保留源码 profile/patch。

M1 与 M2 已完成，验证证据见 `docs/validation_evidence_2026-10-22.md`。后续进入 CI 固化和发布审计；
v0.7.8 的 legacy/non-commercial 定位保持不变。
