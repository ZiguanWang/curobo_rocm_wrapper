# cuRobo ROCm wrapper

这个仓库为 NVIDIA [cuRobo](https://github.com/NVlabs/curobo) 的两个固定版本提供 ROCm 适配。
wrapper 保存版本锁定信息、补丁、构建与测试入口；两个上游版本分别固定在
`upstream/curobo-v0.8.0/` 和 `upstream/curobo-v0.7.8/`。应用补丁时只修改所选版本的
submodule 工作树。`v0.8.0` 是推荐的发布基础，`v0.7.8` 仅用于受原许可证约束的旧版兼容。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `versions.toml` | 上游地址、两个完整 commit、许可证、补丁及其 SHA-256、测试清单 |
| `upstream/curobo-v0.8.0/`、`upstream/curobo-v0.7.8/` | 两个独立的 Git submodule；各自固定上游 commit |
| `profiles/v0.8.0/`、`profiles/v0.7.8/` | 各版本独立的 ROCm patch、`series` 和七项迁移测试清单 |
| `src/curobo_rocm_wrapper/` | profile 校验与 `status`、`apply`、`build`、`test` CLI |
| `scripts/` | 初始化上游 submodule 和运行迁移测试的入口 |
| `tests/` | wrapper 自身的版本、许可、补丁摘要和测试清单检查；不执行 GPU 内核 |
| `docs/` | 中文迁移指南、双版本验证流程及已有验证记录 |
| `archive/pre_wrapper_workspace/` | 迁移前工作区的历史文件，仅供追溯；CLI 不应用这些文件 |
| `MODIFICATIONS.md`、`THIRD_PARTY_NOTICES.md`、`STATUS.md` | 修改范围、第三方许可和验证状态 |
| `LICENSE`、`licenses/` | wrapper 的 Apache-2.0 全文与 v0.7.8 上游 NVIDIA License 副本 |

`archive/` 不是第三套适配源码。实际交付的补丁以 `profiles/` 中的文件为准。

## 版本与许可

| Profile | 上游 tag / 固定 commit | 上游源码布局 | 许可与用途 |
| --- | --- | --- | --- |
| `v0_8_0` | `v0.8.0` / `4ea77366ca48ee453e7df139e39fa6532af49f3b` | `curobo/_src/` | Apache-2.0；推荐发布基础 |
| `v0_7_8` | `v0.7.8` / `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | `src/curobo/` | NVIDIA License；仅非商业研究或评估，除非另获授权 |

两个版本的源码布局、扩展模块和许可证不同，因此使用不同的 patch。wrapper 的 Apache-2.0
许可证不改变上游许可证；详情见 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 初始化与使用

需要 Git、Python 3.11+；构建和 GPU 测试还需要可用的 ROCm 工具链、匹配的 ROCm PyTorch、
目标 AMD GPU 及相应的上游依赖。以下命令在 wrapper 仓库根目录执行。

初始提交包含 wrapper 文件、`.gitmodules` 和两个固定 gitlink。从已有提交克隆时
初始化两个 submodule：

```bash
git submodule update --init --recursive
```

`git submodule status` 应显示两个固定 commit。维护者需要重新初始化或重锁某个版本时，
可运行 `./scripts/bootstrap_submodule.sh v0_8_0` 或 `./scripts/bootstrap_submodule.sh v0_7_8`；
脚本从 `versions.toml` 读取路径和 commit，暂存 `.gitmodules` 和所选版本的 gitlink。两个 submodule
互不切换位置。

建议每个版本使用独立虚拟环境，以隔离互不兼容的上游依赖和同名 Python 包。以
`v0.8.0` 为例，先安装 wrapper 的开发依赖：

```bash
python3.12 -m venv .venv-v0.8.0
source .venv-v0.8.0/bin/activate
python -m pip install -e ".[dev]"
```

在 ROCm 主机上，先按照所用 ROCm 版本安装匹配的 PyTorch 和 cuRobo 上游依赖，再运行：

```bash
export PYTORCH_ROCM_ARCH=gfx1201  # 按实际 GPU 修改
curobo-rocm-wrapper status --profile v0_8_0
curobo-rocm-wrapper apply --profile v0_8_0
curobo-rocm-wrapper build --profile v0_8_0
curobo-rocm-wrapper test --profile v0_8_0
```

`apply` 要求上游工作树干净且 HEAD 与 profile 的完整 commit 一致，并在应用 patch 前执行
`git apply --check`。`build` 使用 HIP 后端构建 wheel；`test` 强制设置 `CUROBO_USE_HIP=1`、
`CUROBO_KERNEL_BACKEND=hip`、`CUROBO_REQUIRE_ROCM=1`，执行对应清单中的七类测试。
`test` 命令使用 pytest-xdist 的 `-n0` 选项；`[dev]` 依赖已包含该插件。也可以用
`./scripts/run_migration_tests.sh v0_8_0` 运行 GPU 测试。

wrapper 自身的检查无需 GPU：

```bash
python -m pytest -q tests
ruff check src tests
bash -n scripts/*.sh
```

### 同时维护两个版本

两个 submodule 可以在同一个 wrapper checkout 中同时存在，CLI 根据 `--profile` 选择目录。
构建和安装仍应使用各自的虚拟环境，避免两个 API 不兼容的 `curobo` 包互相覆盖：

```bash
# 新终端；先在这个环境中安装匹配的 ROCm PyTorch 和上游依赖
python3.12 -m venv .venv-v0.7.8
source .venv-v0.7.8/bin/activate
python -m pip install -e ".[dev]"
export PYTORCH_ROCM_ARCH=gfx1201
curobo-rocm-wrapper status --profile v0_7_8
curobo-rocm-wrapper apply --profile v0_7_8
curobo-rocm-wrapper build --profile v0_7_8
curobo-rocm-wrapper test --profile v0_7_8
```

每个 profile 只修改自己的 submodule。重跑 `apply` 前需要恢复该 submodule 的干净状态；
不要把两个版本装进同一个环境或打进同一个 wheel。发布时分别记录两个 gitlink、patch
SHA-256、ROCm/PyTorch 版本和各自的测试结果。`v0.7.8` 的非商业限制始终适用。

## 验证记录与文档

两个 profile 曾在 Radeon AI PRO R9700 (`gfx1201`) 上分别完成两轮干净 checkout、wheel
构建、隔离安装和 HIP 测试：`v0.8.0` 每轮 15 项通过，`v0.7.8` 每轮 7 项通过，均无 skip。
这是 [2026-10-22 的验证记录](docs/validation_evidence_2026-10-22.md)，不代表当前机器或
本次编辑已重新运行 GPU 测试。

当前冻结 patch 在 2026-10-23 的容器复测中，`v0.8.0` 七个迁移文件 **16 项通过**，
另有官方 kinematics 文件 **6 项通过**；`v0.7.8` 七项迁移测试通过。两版的其他
官方 GPU 测试仍受 Warp 缺少 ROCm 设备支持影响，详见下方容器报告。

本仓库当前不宣称是 LingBot-VLA-v2 指定 RoboTwin commit 的开箱即用 cuRobo 替换件：
其旧 planner API 与 `v0.8.0` 不匹配，`v0.7.8` 虽能导入旧接口但尚无该 commit
的端到端评测证据。具体对照见
[LingBot/RoboTwin 兼容状态](docs/lingbot_robotwin_compatibility_zh.md)。

- [迁移与发布指南](docs/migration_zh.md)
- [双版本验证计划](docs/dual_version_validation_zh.md)
- [当前验证状态](STATUS.md)
- [2026-10-23 容器双版本验证报告](docs/container_validation_2026-10-23.md)
- [LingBot/RoboTwin 指定版本兼容状态](docs/lingbot_robotwin_compatibility_zh.md)
- [上游修改说明](MODIFICATIONS.md)
