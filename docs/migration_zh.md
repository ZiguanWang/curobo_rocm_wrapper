# cuRobo 双版本 ROCm wrapper 迁移指南

## 目标

本 wrapper 统一管理 cuRobo `v0.7.8` 和 `v0.8.0`，但不修改 wrapper 仓库之外的上游源码：

```text
curobo-rocm-wrapper/
├── upstream/curobo-v0.8.0/      # v0.8.0 Git submodule
├── upstream/curobo-v0.7.8/      # v0.7.8 Git submodule
├── profiles/v0.8.0/             # Apache-2.0 modern profile
├── profiles/v0.7.8/             # NVIDIA License legacy profile
├── src/curobo_rocm_wrapper/      # shared orchestration
├── tests/                        # wrapper contract tests
└── scripts/                      # submodule and migration commands
```

## 为什么统一 wrapper、分开 patch

两个版本可以使用同一个仓库、CLI、构建约定和测试数量，但不能使用同一份 patch：

- `v0.7.8` 使用 `src/curobo/...` 和五个旧 pybind extension；
- `v0.8.0` 使用 `curobo/_src/...`、新的 backend 分派和四个 extension；
- `v0.7.8` 受 NVIDIA 非商业许可限制；
- `v0.8.0` 使用 Apache-2.0。

因此版本差异全部放在 `profiles/<tag>/`，共享脚本不得通过模糊路径猜测版本。

## 许可要求

### v0.8.0

可以在 Apache-2.0 条件下修改和再分发。发布时必须保留上游许可证和归属，并明确标注修改。
不得暗示 NVIDIA 为 wrapper 或 ROCm 版本背书。

### v0.7.8

只能用于非商业研究或评估，除非另行取得 NVIDIA 授权。再分发时必须包含该 tag 原始 NVIDIA
License，保留所有归属，并让衍生 patch 继续受相同使用限制。不要把 v0.7.8 wheel 放入商业
产品镜像，也不要与 v0.8.0 Apache 发布物合并为一个二进制包。

资产继续由上游 `LICENSE_ASSETS` 管理；wrapper 不复制机器人和示例资产。

## 初始化子仓库

两个固定版本可以在同一个 wrapper checkout 中并存。新克隆后初始化两个 submodule：

```bash
git submodule update --init --recursive
```

需要重新固定某个版本时，可分别运行：

```bash
./scripts/bootstrap_submodule.sh v0_8_0
./scripts/bootstrap_submodule.sh v0_7_8
```

脚本只操作所选 profile 的 submodule，并检出 `versions.toml` 中的完整 commit，而不是依赖可移动 tag。
两个版本构建和安装时仍须使用不同虚拟环境，防止同名 Python 包与扩展库互相覆盖。

## 应用适配

```bash
source .venv/bin/activate
curobo-rocm-wrapper status --profile v0_8_0
curobo-rocm-wrapper apply --profile v0_8_0
```

应用前要求 submodule 干净且 HEAD 与 profile 完全一致。patch 会先执行 `git apply --check`，任何
上游漂移、重复应用或上下文不匹配都会停止，不允许部分应用。

## 编译

先安装与目标 ROCm 匹配的 PyTorch，然后执行：

```bash
export CUROBO_USE_HIP=1
export CUROBO_KERNEL_BACKEND=hip
export CUROBO_REQUIRE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx1201

curobo-rocm-wrapper build --profile v0_8_0
```

`v0.8.0` 构建 kinematics、geometry、optimization 和 trajectory 四个 HIP extension。
`v0.7.8` 使用其旧模块名构建五个 hipified extension。两个版本的 wheel 必须分别构建和命名，
不能在同一 Python 环境中安装。

## 保留的七项迁移测试

### v0.8.0

1. Kinematics forward/backward
2. Geometry/self-collision
3. Sphere-cuboid collision
4. ToolPose cost
5. C-space costs
6. Optimization
7. Trajectory integration/B-spline

### v0.7.8

旧版没有现代 backend 和同名测试，因此保留七个对应的 legacy extension/binding 门禁：

1. fused kinematics extension
2. geometry extension
3. sphere/OBB collision binding
4. pose-distance binding
5. tensor-step/C-space extension
6. L-BFGS 与 line-search extensions
7. trajectory acceleration binding

运行：

```bash
export PYTORCH_ROCM_ARCH=gfx1201
./scripts/run_migration_tests.sh v0_8_0
# 或非商业 legacy profile：
./scripts/run_migration_tests.sh v0_7_8
```

wrapper 单元测试只验证版本锁定、许可边界、patch 合同和七项 manifest；真正 GPU 测试必须在
ROCm 主机上对已经应用 patch 的 submodule 执行。

## 发布建议

- 主发布线使用 `v0.8.0` profile；
- `v0.7.8` 只发布 patch/source compatibility kit，不发布商业容器或公共 wheel；
- wheel 和容器标签包含 upstream tag、ROCm、Python 和 `gfx`；
- 每个发布物记录 submodule commit、wrapper commit、patch SHA-256 和镜像 digest；
- 不把上游资产复制进 wrapper；需要资产时由 submodule 自己提供并沿用 `LICENSE_ASSETS`。

建议命名：

```text
curobo-rocm-wrapper-v0.1.0
curobo-v0.8.0-rocm7.1-py312-gfx1201.whl
curobo-rocm:v0.8.0-rocm7.1-gfx1201
```

`v0.7.8` 发布物必须显式包含 `noncommercial`：

```text
curobo-v0.7.8-rocm-noncommercial-source-patch.tar.gz
```
