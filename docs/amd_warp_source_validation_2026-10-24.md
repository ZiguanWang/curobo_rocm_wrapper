# AMD Warp 源码构建与双版本复测（2026-10-24）

## 当前结论

从 [AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp) 的
`amd-integration-halo` 分支固定 commit
`74014999a31471228e0aaa213bbeb2e5daba95f0`，在 PyTorch/ROCm 容器内以
`gfx1201` 调试模式构建并安装后，Warp 能识别当前 Radeon GPU、映射 PyTorch
ROCm 张量并执行真实 HIP kernel。该分支还使初版 fork 会在 cost→MorphIt
顺序下中止的测试 **213/213 通过**。两版先前的“无 GPU 设备映射”错误均已消失，
但官方全套测试仍未完成。

这一更新分支公开验证的 GPU 是 `gfx1151`，推荐 ROCm 7.14。
本次 `gfx1201`、ROCm 7.2.4 是实测的**额外组合**，不能推断它对全部
ROCm/Warp API 的正式支持。此前先测试过 `amd-integration`
`2e2673bb7b90eaabb9cb89dc1f55a76f61c13aab`：该分支明列支持
`gfx942`，虽解决设备映射，但有下面记录的 mesh 描述符故障；所以当前锁定
更新分支。

| 验证项 | `v0.7.8` | `v0.8.0` |
| --- | --- | --- |
| Warp GPU 设备映射 | 同一容器确认 `cuda:0` 为 HIP/`gfx1201` | 同左 |
| cuRobo HIP 迁移测试 | **7/7 通过** | **16/16 通过** |
| 先前失败的 Warp 相关官方测试 | 完整收集 **264 项**，几何 6/6、Warp mesh 3/3 通过 | 球体拟合、姿态、夹持球、mesh SDF alignment 四个官方文件合计 **84/84**；cost→MorphIt **213/213**；voxel Warp 文件 **36 passed、1 skipped** |
| 后续故障 | IK 执行中 legacy self-collision HIP 扩展仍触发 GPU 硬件异常 | 官方全套执行约 16% 后达到 300 秒上限，未获得全套结果 |

## 构建环境和可复现步骤

- 镜像：`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`，
  本地 image ID
  `sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`。
- PyTorch `2.10.0+rocm7.2.4.git3d3aa833`，ROCm HIP runtime `7.2.53211`，
  Python 3.12，`HIP_VISIBLE_DEVICES=0`，`PYTORCH_ROCM_ARCH=gfx1201`。
- Docker 映射 `/dev/kfd` 与 `/dev/dri`，加入主机 video/render 数字组，使用
  `--ipc=host --security-opt seccomp=unconfined`。以当前用户运行并把源码/产物目录
  挂载到容器，未向主机系统 Python 环境安装 Warp 或 cuRobo。

当前锁定的 `amd-integration-halo` 在容器内、已进入源码根目录时执行：

```bash
HIP_ARCH=gfx1201 ROCM_PATH=/opt/rocm PYTHON=python \
  bash build_amd.sh --mode=debug --no-use-libmathdx -j 4
python -m pip wheel --no-build-isolation --no-deps --wheel-dir /work/dist .
python -m pip install --no-build-isolation --no-deps --target /work/warp-install .
PYTHONPATH=/work/warp-install python /wrapper/scripts/verify_amd_warp.py
```

源码先用 `git clone --depth 1 --branch amd-integration-halo` 获取，再按上述完整 commit
检出；如远端分支已移动，先对该 commit 执行 `git fetch --depth 1 origin <commit>`。
应优先使用 `versions.toml` 中的锁定值，而不是随分支 HEAD 漂移。

当前锁定分支的调试构建 wheel：
`warp_lang-1.17.0.dev4-py3-none-manylinux_2_28_x86_64.whl`，
SHA-256 `81946b6493e1a4e5714e7d7de308a8de11a66d4d8835d3678190e76f263a6d71`。
`verify_amd_warp.py` 观察到 `['cpu', 'cuda:0']`、`device.is_hip=True`、
`device.arch_str='gfx1201'`，并让 Warp kernel 把 PyTorch GPU 张量
`[0,1,2,3]` 改为 `[1,2,3,4]`。

部分同 tag 官方测试文件在 `@wp.kernel` 定义或直接 `wp.from_torch` 时没有先
调用 `wp.init()`。单独运行此类文件需要加载
`scripts/pytest_amd_warp_plugin.py`，例如将 `scripts/` 加入 `PYTHONPATH`，
再传 `-p pytest_amd_warp_plugin`。该插件只初始化 Warp，不更改断言、结果或
跳过规则。用它运行原版 `test_voxel_collision.py` 的结果为
**36 passed、1 skipped**；未加载插件时该文件在收集后的直接张量映射处因
Warp runtime 尚未初始化而失败。

此前 `amd-integration` 的优化构建（`--mode release`）在单个
`warp/native/warp.cu` 的 `gfx1201` 编译上持续超过 22 分钟后停止；
`--hipcc-options=-O1` 又持续超过 8 分钟后停止，均无可测试的 wheel。
当前更新分支仅验证了调试构建；结果不能作为性能或正式发布构建的证据。

## v0.7.8：设备映射修复与剩余故障

同 tag 上游的 `src/curobo/geom/sdf/world_mesh.py` 使用
`wp.torch.device_from_torch`；AMD fork 将转换函数导出在公共
`wp.device_from_torch`。当前 `profiles/v0.7.8/` 补丁只增加这处兼容调整，
其余旧版 HIP 改动保持原有逻辑。新 patch SHA-256 为
`508369700e60ca75efe0a8cbe47ed665272231287f919b926a8bf3930bec17e7`。

从全新上游 commit、该 patch 构建的 wheel SHA-256 为
`386c888dc32e81e2e277fe510d9f6fdfa21d26740ea7751eb277f7578836e147`。
从隔离安装目录导入该 wheel 与当前锁定的 AMD Warp 后：

- 七个 ROCm 迁移测试 **7 passed**；
- 同 tag 官方测试由此前的“收集 256 项、2 个 Warp 设备错误”进展为
  **264 tests collected，无收集错误**；
- 官方 `geom_types_test.py` **6 passed**，`warp_mesh_test.py` **3 passed**。

随后执行官方 `ik_config_test.py::test_eval[config0-True]`，进程在
`curobo.curobolib.geom` 的 legacy self-collision HIP 扩展路径触发
`HSA_STATUS_ERROR_EXCEPTION`（GPU hardware exception，code `0x1016`）。
更新 Warp 分支后同一调用链仍中止。这是跨过 Warp 映射之后暴露的
**另一处原生 GPU 故障**，不能把 IK/TrajOpt
完整运行标记为通过。

## v0.8.0：设备映射修复与剩余故障

先沿用 `v0.8.0` kinematics 修复后的 wheel 验证 AMD Warp 的设备映射。随后
原版 `test_wp_mesh_sdf_alignment.py` 有两个不足完整 256 线程 block 的边界用例：
有效点数量分别算成 0（应为 1）和 160（应为 0）。该 Warp kernel 原来对尾部
不完整 block 也执行 tile reduction；当前 cuRobo patch 改为只对尾部活跃线程
使用原子累加，完整 block 仍保持 tile reduction。现行 patch SHA-256 为
`a22d7dfdf84206846ba62092ded3766f6944f0ae5df695890cec93cc3815f179`。

从精确上游 commit 新建 checkout、应用现行 patch 后构建的 wheel SHA-256 为
`f7ee4c6d9d4e364c9d5b148ad4552c9c0e30c5c6c8c88dd2da2140c17e399d87`。
隔离安装此 wheel，并使用当前锁定的 AMD Warp 后：

- 七个 ROCm 迁移文件 **16 passed**；
- 同 tag 官方 `test_sphere_fit.py`、`test_pose.py`、
  `test_attachment_manager.py` 和 `test_wp_mesh_sdf_alignment.py`
  四个文件合计 **84 passed**；
- 原版 `test_voxel_collision.py` 在显式 Warp 初始化的测试进程中
  **36 passed、1 skipped**；
- 曾会在 MorphIt 中止的 cost→`test_geom_types.py` 顺序 **213 passed**，
  无需关闭 Warp HIP 内存池；
- 官方全套以 `-x` 串行运行，越过上述故障点，约运行到 16% 后达到
  300 秒上限；截至超时未记录到测试失败，但**没有全套通过结果**。

对照：初版 `amd-integration` fork 的官方全套曾达到
**345 passed、4 skipped**，随后在
`SphereFitType.MORPHIT` 的反复 Warp mesh SDF 查询期间出现原生中止。
同一单项在独立进程通过，简单 box mesh 的 200 次连续 SDF 查询也通过；
进一步缩小到前置 cost 测试与 MorphIt 的同进程组合：collision 测试加
`test_geom_types.py` **33 passed**，单独的两半 cost 测试加该文件分别
**136 passed**、**81 passed**，但合并 cost 测试后在 MorphIt 处仍可复现中止。
初版 fork 的 HIP 内存池关闭后，该组合 **213 passed**，但官方全套下一处
较大球体拟合仍发生原生中止。更新分支的
`KNOWN_ISSUES-AMD.md` 记载了 mesh/BVH 描述符地址复用造成空地址 GPU fault
及其修复；本次 `gfx1201` 测试结果与该修复方向一致，但没有证明它覆盖了全部路径。

## 后续验证门槛

1. 对旧版 IK 的 self-collision 原生异常做最小复现与 GPU 调试；不能以
   Warp 测试通过替代规划器执行结果。
2. 给新版官方全套测试足够的运行时间，并按类别记录通过/失败/跳过；
   对持续较久的用例设独立时限，保留原始日志。
3. 得到可重复的优化版 AMD Warp 构建；在相同 wheel、镜像和目标 GPU 上复跑
   两版相关官方测试，再扩展至 RoboTwin 真实评测。

本次源码与日志暂存于 `/tmp/amd-warp-halo-validation/`、
`/tmp/amd-warp-rocm-debug-validation/`、
`/tmp/curobo-amdwarp-v078.unDztw/` 和
`/tmp/curobo-kinematics-final-check.I8lXmG/`。这些临时目录不是提交产物。
