# cuRobo ROCm wrapper

这个仓库为 NVIDIA [cuRobo](https://github.com/NVlabs/curobo) 的两个固定版本提供 ROCm 适配。
wrapper 保存版本锁定信息、补丁、构建与测试入口；两个 cuRobo 上游版本分别固定在
`upstream/curobo-v0.8.0/` 和 `upstream/curobo-v0.7.8/`，共用的 ROCm Warp 固定在
`upstream/warp-rocm/`。`v0.8.0` 是推荐的发布基础，`v0.7.8` 仅用于受原许可证约束的旧版兼容。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `versions.toml` | cuRobo 与 Warp 的上游地址、完整 commit、补丁及其 SHA-256；另记录 cuRobo 许可证和测试清单 |
| `upstream/curobo-v0.8.0/`、`upstream/curobo-v0.7.8/` | 两个独立的 Git submodule；各自固定上游 commit |
| `upstream/warp-rocm/` | 两版 cuRobo 共用的 AMD Warp Git submodule；固定上游 commit |
| `profiles/v0.8.0/`、`profiles/v0.7.8/` | 各版本独立的 ROCm patch、`series` 和七项迁移测试清单 |
| `profiles/warp/` | 锁定 AMD Warp commit 的 gfx1201/HIP 单元测试修复补丁 |
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

仓库包含 wrapper 文件、`.gitmodules` 和三个固定 gitlink。从已有提交克隆时
初始化三个 submodule：

```bash
git submodule update --init --recursive
```

`git submodule status` 应显示两个 cuRobo commit 和 Warp commit
`74014999a31471228e0aaa213bbeb2e5daba95f0`。维护者需要重新初始化或重锁 cuRobo 版本时，
可运行 `./scripts/bootstrap_submodule.sh v0_8_0` 或 `./scripts/bootstrap_submodule.sh v0_7_8`；
脚本从 `versions.toml` 读取路径和 commit，暂存 `.gitmodules` 和所选版本的 gitlink。

建议每个版本使用独立虚拟环境，以隔离互不兼容的上游依赖和同名 Python 包。以
`v0.8.0` 为例，先安装 wrapper 的开发依赖：

```bash
python3.12 -m venv .venv-v0.8.0
source .venv-v0.8.0/bin/activate
python -m pip install -e ".[dev]"
```

### 安装共享的 ROCm Warp

先在选定 cuRobo 版本的虚拟环境中安装匹配的 ROCm PyTorch、ROCm 开发工具链
和上游其余依赖。当前在 `gfx1201` 上验证的是固定的 AMD Warp commit 加
`profiles/warp/` 补丁；测试过的 PyPI `warp-lang` wheel 在该容器中只发现 CPU。
两版 cuRobo 共用同一个 Warp 源码和补丁，但须分别安装到各自的虚拟环境。

在 wrapper 根目录执行以下命令。`ROCM_PATH` 必须指向与运行时同版、同时包含
HIP 编译器、设备库和开发头文件的 SDK；普通 ROCm 开发环境通常是 `/opt/rocm`。
这组命令构建已验证的 **debug** 版本：

```bash
git submodule update --init upstream/warp-rocm
mkdir -p build dist/warp-rocm
git -C upstream/warp-rocm worktree add --detach "$PWD/build/warp-rocm-src" 74014999a31471228e0aaa213bbeb2e5daba95f0
git -C build/warp-rocm-src apply --check "$PWD/profiles/warp/0001-gfx1201-unittest-fixes.patch"
git -C build/warp-rocm-src apply "$PWD/profiles/warp/0001-gfx1201-unittest-fixes.patch"
(
  cd build/warp-rocm-src
  HIP_ARCH=gfx1201 ROCM_PATH="${ROCM_PATH:-/opt/rocm}" PYTHON=python \
    bash build_amd.sh --mode=debug --no-use-libmathdx -j 4
  python -m pip wheel . --no-build-isolation --no-deps -w ../../dist/warp-rocm
)
python -m pip install --no-deps --force-reinstall dist/warp-rocm/warp_lang-1.17.0.dev4-*.whl
python scripts/verify_amd_warp.py
```

`verify_amd_warp.py` 检查 PyTorch GPU 与 Warp HIP 设备映射，并实际启动 GPU kernel。
cuRobo 上游依赖包含 `warp-lang`；安装 cuRobo wheel 时使用 `--no-deps`，或在安装
其余依赖后重新安装上述 Warp wheel，避免 PyPI wheel 覆盖它。构建工作树位于
忽略的 `build/` 下，共享 submodule 保持干净。在本仓库验证使用的 ROCm
PyTorch 7.14.1 容器中，SDK 安装在
`/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core`；执行构建命令前还要设置：

```bash
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
export CPATH=/opt/rocm/include
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ln -sfn libamdhip64.so.7 "$ROCM_PATH/lib/libamdhip64.so"
ln -sfn libhiprtc.so.7 "$ROCM_PATH/lib/libhiprtc.so"
ln -sfn libhiprtc-builtins.so.7 "$ROCM_PATH/lib/libhiprtc-builtins.so"
```

上述三个开发库链接只需在一次性构建容器内创建；该镜像原本只包含带 `.so.7`
后缀的运行库。构建和运行 Warp JIT 时还需将同版 ROCm 开发头文件挂载到
`/opt/rocm/include`。普通 ROCm 开发镜像通常已提供这些文件和链接；
本仓库的完整 GPU 测试结果限于 ROCm 7.14.1、`gfx1201` 和 debug Warp。

2026-10-26 从新增 submodule 的固定 commit 和当前 Warp 补丁重建的 wheel
已隔离安装成功（SHA-256：`a73fe52691a94a9ab6204379e84626fb4aae2b3f58574054b6864399f4ff5526`）。
GPU/Torch 共享冒烟通过；用该 wheel 运行的 v0.7.8 官方 Warp mesh 文件 **3/3**、
v0.8.0 官方 mesh SDF alignment 文件 **13/13** 通过。完整 cuRobo 官方全套
的结果仍以此前两份版本报告为准。

### 应用 cuRobo profile

完成 Warp 安装后，在 ROCm 主机上运行：

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

2026-10-23 的容器复测中，`v0.8.0` 七个迁移文件 **16 项通过**，
另有官方 kinematics 文件 **6 项通过**；`v0.7.8` 七项迁移测试通过。
2026-10-24 换用源码构建的 AMD Warp 后，先前的 Warp GPU 设备映射错误已在两版
定向测试中消失。旧版 IK 仍有 legacy self-collision 原生异常；新版官方全套
在本次 300 秒时限内未跑完。新版补丁另修复了 Warp mesh SDF alignment
尾部 block 的计数问题；四个相关官方文件共 84 项通过。详见下方两份报告。

2026-10-25，锁定的 AMD Warp 源码加本仓库 Warp 补丁在 `gfx1201`、
ROCm 7.14.1 的隔离 PyTorch 容器中完成全套 unittest：**7479 项，
0 失败、0 error、238 跳过**。其中新增的 1 项跳过明确对应 HIP graph
event 计时不可用；运行使用调试构建和文档记录的 ROCm 设置。详见
[Warp 全套报告](docs/warp_full_suite_rocm714_2026-10-24.md)。

随后按 v0.7.8、v0.8.0 的顺序完成隔离复测。v0.7.8 从当前冻结补丁
重新构建 wheel 后，
同 tag 官方全套 **257 项通过、11 项跳过、0 失败**，七类迁移测试 **7/7**。
旧版 self-collision GPU 异常已修复；结果与跳过边界见
[v0.7.8 官方全套报告](docs/curobo_v078_official_rocm714_2026-10-25.md)。
v0.8.0 从当前补丁重新构建 wheel 后，同 tag 原版官方全套 **3928 项通过、
38 项跳过、0 失败**，七类迁移测试 **16/16**；曾失败的 GraphExecutor
和 `motion_planning --test` 均通过。详情及跳过范围见
[v0.8.0 官方全套报告](docs/curobo_v080_official_rocm714_2026-10-25.md)。

本仓库当前不宣称是 LingBot-VLA-v2 指定 RoboTwin commit 的开箱即用 cuRobo 替换件：
其旧 planner API 与 `v0.8.0` 不匹配，`v0.7.8` 虽能导入旧接口但尚无该 commit
的端到端评测证据。具体对照见
[LingBot/RoboTwin 兼容状态](docs/lingbot_robotwin_compatibility_zh.md)。

- [迁移与发布指南](docs/migration_zh.md)
- [双版本验证计划](docs/dual_version_validation_zh.md)
- [当前验证状态](STATUS.md)
- [2026-10-23 容器双版本验证报告](docs/container_validation_2026-10-23.md)
- [2026-10-24 AMD Warp 源码验证记录](docs/amd_warp_source_validation_2026-10-24.md)
- [2026-10-24 Warp / ROCm 7.14.1 全套测试进展](docs/warp_full_suite_rocm714_2026-10-24.md)
- [2026-10-25 v0.7.8 官方全套验证](docs/curobo_v078_official_rocm714_2026-10-25.md)
- [2026-10-25 v0.8.0 官方全套验证](docs/curobo_v080_official_rocm714_2026-10-25.md)
- [LingBot/RoboTwin 指定版本兼容状态](docs/lingbot_robotwin_compatibility_zh.md)
- [上游修改说明](MODIFICATIONS.md)
