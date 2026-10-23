# cuRobo v0.7.8 官方全套：gfx1201 / ROCm 7.14.1

## 结论

从上游 v0.7.8 固定 commit 和本仓库当前 profile 补丁建立全新 checkout、
构建 wheel、隔离安装后，同 tag 原版官方测试 **268 项：257 passed、
11 skipped、0 failed、0 errors**；七类 HIP 迁移测试 **7/7 passed**。
原版官方文件通过 `--import-mode=importlib` 运行，`curobo` 导入来自安装的
wheel。隔离测试副本中的 44 个 Python 文件已逐一与固定上游 commit 比较，
内容一致。结果不覆盖 v0.8.0，也不构成 RoboTwin 端到端评测。

## 可追溯版本与环境

| 项目 | 值 |
| --- | --- |
| cuRobo 上游 tag / commit | `v0.7.8` / `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| cuRobo profile patch SHA-256 | `0bcaafb80b5983fd24e195bc52112f90e38bc9bc7729d134ff0be72a8e5b9a2e` |
| AMD Warp commit | `74014999a31471228e0aaa213bbeb2e5daba95f0` |
| Warp patch SHA-256 | `0a1ee165f214649201f00ca24ed093d3cffae64c2f48621a15e6d165b2a8347d` |
| 已验证 wheel SHA-256 | `3e56a82787d7eab954c782cdad30a60f17b4115b6e2575c664e9ecd8a3a11066` |
| GPU / 架构 | Radeon AI PRO R9700 / `gfx1201` |
| 镜像 / PyTorch | `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.11.0` / `2.11.0+rocm7.14.1` |

源码、依赖、wheel、安装目录、原版测试和完整日志均在隔离目录
`/tmp/curobo-v078-rocm714-validation/`。最终 wheel 位于 `dist_verified/`，
JUnit 位于 `logs/official-full-verified-714.xml`，原始输出位于同名前缀的
`.log`。这些 `/tmp` 产物尚未归档进仓库。

ROCm PyTorch 镜像的编译器来自镜像内 SDK；构建时单独挂载同版本开发镜像的
`/opt/rocm/include`，设置 `CPATH=/opt/rocm/include`，并在隔离目录提供
`libamdhip64.so` 指向镜像内版本化库的链接入口。未修改系统 ROCm、主机
Python 环境或两个上游 submodule。构建和测试依赖安装在此版本专用目录。

运行时设置 `CUROBO_USE_HIP=1`、`CUROBO_KERNEL_BACKEND=hip`、
`CUROBO_REQUIRE_ROCM=1`、`PYTORCH_ROCM_ARCH=gfx1201`、
`HIP_VISIBLE_DEVICES=0` 和 `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`。
**未设置** `AMD_SERIALIZE_KERNEL`：PyTorch graph 捕获路径不应使用先前
Warp 全套资格运行中的强制同步设置。测试插件仅在收集前初始化 Warp，
不改断言或跳过规则。带插件的原版官方测试收集 **268 项**；较早不带
插件的收集得到 264 项，不能把两组收集数混为同一运行。

## 故障修复与验证范围

- 旧版 `self_collision_distance_kernel7` 原先在非完整 32 线程 wave 上执行
  shuffle 归约，实际机器人用例触发 GPU hardware exception。HIP launch
  补齐 block 后，原版 `self_collision_test.py` **6/6 通过**。
- 旧版 line search 将线程数补齐到 32 的倍数，却让额外线程在 block barrier
  前返回。HIP 补齐线程现在参与归约并贡献零，只让有效线程读写向量。
- 旧版融合 L-BFGS kernel 在规划器图执行中仍发生 GPU 异常；ROCm profile
  改用同版本 cuRobo 已有的 PyTorch L-BFGS 路径。原生扩展仍构建，
  七类迁移测试保持 **7/7 通过**。这可能影响性能，未在本轮测量。
- AMD Warp 的 `from_torch` 在当前 Torch stream 捕获 graph 时不再执行
  HIP 主机侧 stream synchronize；cuRobo 随后把 Warp kernel 放在同一
  Torch stream 上。原版 `cuda_graph_test.py::test_motion_gen_mpc` 已通过。

官方全套的 11 项跳过分别来自：USD 测试 2 项、Nvblox/渲染依赖 2 项，
以及当前 PyTorch `CUDAGraph.reset()` 不可用的 7 项。GPU 内核仍真实执行；
本轮没有把这些跳过项解释为对应功能已验证。

v0.7.8 仍受上游 NVIDIA License 的非商业使用限制。当前结果只证明此固定
ROCm 环境下 cuRobo 官方单元测试的上述范围；LingBot 指定 RoboTwin commit
及其评测策略仍需单独端到端验证。
