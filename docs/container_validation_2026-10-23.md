# cuRobo 双版本容器验证报告（2026-10-23）

本页是使用 PyPI NVIDIA Warp 的历史基线。2026-10-24 已在同类容器中从
AMD-Ecosystem/warp 源码构建出能识别 `gfx1201` 的 HIP 版本，并复测两个 cuRobo profile；
后续结果见 [AMD Warp 源码验证记录](amd_warp_source_validation_2026-10-24.md)。

## 结论

在当前 AMD GPU 上，两个固定版本的 ROCm patch 都能在同一 PyTorch 容器镜像中构建 wheel，
并从隔离安装目录通过各自的七类 HIP 迁移测试。初次运行发现的 `v0.8.0` kinematics
扩展参数不匹配已在同日修复，并从全新 checkout 重建 wheel 验证。**两个版本仍未与同 tag
官方 cuRobo 的全套单元测试完全对齐**：官方测试中的 Warp GPU 路径在该容器内不可运行。

| Profile | 容器 wheel 构建 | ROCm 迁移测试 | 同 tag 官方测试 | 完全对齐？ |
| --- | --- | --- | --- | --- |
| `v0_8_0` | 通过，4 个 HIP 扩展 | **16/16**，通过且无 skip | 原版测试可收集 3963 项；官方 kinematics 文件 6/6 通过；首个夹持球测试随后在 Warp mesh 查询失败；初版补丁另选 175 项为 172 通过、3 失败 | 否 |
| `v0_7_8` | 通过，5 个 HIP 扩展 | 7/7，通过且无 skip | 原版测试收集到 256 项后有 2 个收集错误；另选 9 项为 3 通过、6 失败 | 否 |

官方全套测试没有完成，因此不能把表中的通过数外推为官方全套的通过率，也不能断言未运行的
API 均兼容。

## 环境与隔离方式

- GPU：当前机器的 `gfx1201`，容器内 `torch.cuda.get_device_properties(0).gcnArchName`
  为 `gfx1201`；`HIP_VISIBLE_DEVICES=0`。容器内 8 元素 GPU 张量求和返回 `8.0`。
- 镜像：`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`，
  本地 image ID `sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`。
- Python 3.12；PyTorch `2.10.0+rocm7.2.4.git3d3aa833`；
  `torch.version.hip=7.2.53211`；pytest `8.4.2`；`warp-lang=1.18.0`。
- 容器以当前用户 UID/GID 启动，映射 `/dev/kfd`、`/dev/dri`，加入 video/render
  数字组，使用 `--ipc=host` 和 `--security-opt seccomp=unconfined`。
- 两版**串行**执行：先完成 `v0.8.0` 构建、安装和测试，再执行 `v0.7.8`。每版从对应
  submodule 的精确 commit 建立临时克隆，应用对应冻结 patch，在独立的容器挂载目录中
  `pip wheel ".[hip]" --no-build-isolation --no-deps`，再安装到该版本自己的
  `/work/install`。测试前验证 `curobo.__file__` 来自 `/work/install/curobo/`。
- 官方测试源码由对应 submodule 的 `git archive HEAD` 提取，未使用经过 patch 修改的
  官方测试文件。运行官方测试时以 `--import-mode=importlib` 加载这些原版文件，包导入
  指向已安装的 ROCm wheel。
- 没有向主机既有 Python 环境安装 cuRobo wheel，没有改动既有 `build/` 或运行中进程。
  本报告的测试结果均来自上述容器；此前主机上的构建不计入结果。

本次镜像的 PyTorch/ROCm 组合与
[2026-10-22 的验证记录](validation_evidence_2026-10-22.md)不同。软件栈差异可能影响
官方测试失败原因，尤其是 Warp 的设备支持；报告只对本次容器环境作结论。

## 精确版本、构建产物与迁移测试

| Profile | 上游 HEAD | Patch SHA-256 | 本次 wheel SHA-256 |
| --- | --- | --- | --- |
| `v0_8_0` | `4ea77366ca48ee453e7df139e39fa6532af49f3b` | `e6a832b1093116cabcb208443fc56ff59ca0f8f5c61b7a901d4c3b0c2f2c33f8` | `c9bcb204f18c0c90bfd26eb315d242a6990fcb8579d2ec1a395581cc436936f3` |
| `v0_7_8` | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | `8574b4d51908738b5f72c7062e1eb3f706150a0dc4fc0c646a6fd1fd80a2661a` | `1bc38a17124c86bda3aa4d7f2ccb9e622b16f53f649649e7af70db51f1871d28` |

两版均设置 `CUROBO_USE_HIP=1`、`CUROBO_KERNEL_BACKEND=hip`、
`CUROBO_REQUIRE_ROCM=1` 和 `PYTORCH_ROCM_ARCH=gfx1201`。实际结果：

| 迁移类别 | `v0.8.0` | `v0.7.8` |
| --- | --- | --- |
| Kinematics | 扩展 forward/backward 与上层 Python 分派 2 项通过 | fused kinematics 1 项通过 |
| Geometry / self collision | 1 项通过 | 1 项通过 |
| Collision | sphere/cuboid 4 项通过 | sphere/OBB 1 项通过 |
| Tool pose | 3 项通过 | pose distance 1 项通过 |
| C-space | 2 项通过 | position step 1 项通过 |
| Optimization | L-BFGS 与 line search 2 项通过 | L-BFGS 与 line search 1 项通过 |
| Trajectory | acceleration 与 B-spline 2 项通过 | acceleration step 1 项通过 |
| 合计 | **16 passed in 3.21s** | **7 passed in 1.73s** |

以上是 `profiles/<version>/tests.toml` 指定的七个文件，不是官方全套单元测试。
`v0.8.0` wheel 中确认存在 kinematics、geometry、optimization、trajectory 四个 HIP
扩展；`v0.7.8` wheel 中确认存在五个旧模块名扩展。

## 与官方单元测试的对照

### v0.8.0

官方 tag 含 165 个 `test_*.py` 文件。原版测试文件在本次容器和已安装 ROCm wheel 下
`--collect-only` 得到 **3963 tests collected**。首次使用旧 patch 的全套串行执行
设置 `-x`，在
`curobo/tests/_src/collision/test_attachment_manager.py::TestFitSpheres::test_fit_returns_tensor_with_4_columns`
的 fixture 中停止：`launch_kinematics_forward(): incompatible function arguments`
（`TypeError`），结果为 **3 skipped, 1 error**。当时的 patch SHA-256 为
`ec60a6762bf8bb23469a5cdac52087c58c4f42de67517df359758c961ea2fcc1`，
wheel SHA-256 为 `511198cae168941f519cff4fb13f1a075388cb0c2a2f99133937cc9ab37a19fc`。

随后将 Python kinematics 分派与补丁带入的三个 HIP forward 入口及 fused backward
参数合同对齐，加入上层机器人模型回归测试。最终 patch 从新的精确 commit checkout
重新构建并隔离安装后，七个迁移文件 **16 passed**，原版官方
`curobo/tests/_src/robot/kinematics/test_kinematics.py` **6 passed**。
上述夹持球测试不再出现 kinematics `TypeError`，继续运行到 Warp mesh 查询后因
`warp._src.torch.device_from_torch` 的 GPU 设备列表为空而失败。因此仍不能给出
官方全套通过率。

初版补丁另选未修改的官方 `test_cost_types.py`、`test_pose.py` 和 `test_tool_pose.py`
运行，共 **172 passed, 3 failed**。失败均在 `test_pose.py` 的 GPU 变换测试：
`test_transform_points[cuda:0]`、`test_pose_transform_point[cuda:0]` 和
`test_pose_transform_point_grad[cuda:0]`。调用链进入 `warp._src.torch.device_from_torch`
后因 Warp 的 `cuda_devices` 无对应设备而报 `IndexError: list index out of range`。
容器内 PyTorch 对同一 GPU 的张量运算已通过，因此这是当前 Warp 路径的独立缺口。

### v0.7.8

官方 tag 含 42 个测试文件。原版官方测试收集到 **256 项**后，在
`tests/ik_config_test.py` 和 `tests/trajopt_config_test.py` 出现 2 个收集错误；
两处均进入 `warp._src.torch.device_from_torch` 后因缺少对应 Warp 设备而报
`IndexError: list index out of range`。256 是报错前的收集数，**不是完整测试总数**。

另选原版 `curobo_version_test.py`、`geom_types_test.py`、`goal_test.py` 运行，
**3 passed, 6 failed**。通过的是版本测试及 `goal_test.py` 的
`test_repeat_seeds`、`test_horizon_repeat_seeds`；6 个失败均在
`geom_types_test.py` 的 mesh/bounding-volume 路径，原因同为 Warp 设备映射失败。

## API 对齐判断

比较基准是**各自相同 tag 的官方源码**，不是把 `v0.7.8` 和 `v0.8.0` 两套本就不同的
API 强行比较。对当前 patch 修改的 Python 文件做 AST 名称和参数检查：

- `v0.8.0` 未发现原有公开 Python 函数/类签名被删除或改名，但
  `curobo/runtime.py` 的 `__all__` 增加
  `AcceleratorCapabilities`、`AcceleratorType`、`detect_accelerator`、
  `get_accelerator_capabilities`、`is_graph_capture_enabled`、`rocm_graphs`；
  还增加 HIP 后端模块。补丁修改了 4 个原有官方测试文件，新增 11 个测试文件。
  初版补丁的扩展调用不匹配已由本次分派修复，并通过 6 项原版官方 kinematics 测试；
  Warp 相关路径仍未覆盖。
- `v0.7.8` 未发现原有公开 Python 函数/类签名变化；新增
  `curobo.curobolib.rocm_preflight`，且 `curobo.__version__` 在缺失包元数据时的
  回退值由 `v0.7.0-no-tag` 改为 `v0.7.8-no-tag`。原有官方测试文件未改，
  另增 7 个 ROCm 测试文件。官方测试在 Warp GPU 路径无法完整收集或通过。

静态签名检查不覆盖动态导出、C++/HIP 扩展 ABI 和所有运行时语义。结合实际的官方测试
失败，**当前不能宣称任一版本与官方 API 或官方单元测试完全对齐**。迁移测试通过证明的是
指定 HIP 路径在本次环境中可运行，不证明其它官方路径均可用。

## 后续对齐工作

1. 当前 `warp-lang 1.18.0` 初始化时只列出 `cpu`，内部 `cuda_devices=[]`；
   它在本次环境中没有可用于 `gfx1201` 的 GPU 设备。不能通过改变可见设备编号修复。
   `v0.7.8` 源码约有 8 个文件直接调用 Warp；实验性地为 quaternion/matrix 和
   pose 运算加 PyTorch 回退后，官方收集错误仍转移到 mesh collision 的
   `wp.torch.device_from_torch`。这份不完整的实验修改未合入冻结 patch。
   要继续对齐，须按功能提供 HIP/PyTorch 实现，或采用经验证可运行的 Warp ROCm 移植。
2. 补齐上述路径后，分别完整执行两个版本的原版官方测试集，记录通过、失败、跳过及
   环境依赖；当前结果不足以给出全套通过率。

本次原始日志与 wheel 暂存于
`/tmp/curobo-container-v080.6iM8xC/` 和
`/tmp/curobo-container-v078.9yksQP/` 的 `logs/`、`dist/`；
最终 `v0.8.0` 修复复测产物在 `/tmp/curobo-kinematics-final-check.I8lXmG/`。
这些临时目录不是
仓库发布物。报告中的命令、版本、wheel 摘要和失败用例用于复核本次结论。
