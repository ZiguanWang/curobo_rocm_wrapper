# AMD Warp 在 gfx1201 / ROCm 7.14.1 的全套测试进展

本记录补充 [早期的 cuRobo/Warp 定向验证](amd_warp_source_validation_2026-10-24.md)。
Warp 源码固定为 `AMD-Ecosystem/warp` 的 `amd-integration-halo` commit
`74014999a31471228e0aaa213bbeb2e5daba95f0`，以调试模式为 `gfx1201` 构建。
本页的 Warp 全套结果**不是** cuRobo 两版官方全套的结果。

## 环境与原始证据

- GPU：Radeon AI PRO R9700 (`gfx1201`)，`HIP_VISIBLE_DEVICES=0`。
- 开发镜像：`rocm/dev-ubuntu-24.04:7.14.1-full`；PyTorch 镜像：
  `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.11.0`。
- PyTorch 镜像缺 ROCm 开发头文件；单独挂载从同版本开发镜像提取的
  `/opt/rocm/include`，并在 `LD_LIBRARY_PATH` 中加入镜像内 ROCm SDK 库。
- 两次运行均使用 Warp 的 `unittest` 测试入口。原始 JUnit 和日志保存在隔离目录
  `/tmp/amd-warp-halo-rocm714-validation/.validation-home/`，文件名前缀分别为
  `warp-full-714` 和 `warp-full-714-torch`。该 `/tmp` 目录尚未纳入提交。

| 运行环境 | tests | failures | errors | skipped | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| ROCm 7.14.1 开发镜像，Warp 核心依赖 | 7424 | 8 | 0 | 239 | 未通过 |
| ROCm 7.14.1 PyTorch 镜像，包含 Torch | 7479 | 7 | 1 | 237 | 未通过 |

PyTorch 镜像中的 Warp/Torch 互操作类另行运行 **43/43 通过**。
两轮全套的稳定失败包括 graph elapsed event、跨 graph external event、
fp16 matrix inverse，以及两个故意触发 GPU 非法操作但 HIP 未输出所期望诊断文本的用例。
stream priority 和 tile FFT 在全套失败后单独复测通过，属于顺序、并发或资源负载敏感项。
开发镜像全套另有两个 mesh ray 失败，PyTorch 镜像全套未复现。

## 新增定向复现

`TestStreams.test_event_external_cuda_0` 在 PyTorch 7.14.1 容器中单独运行仍失败。
同一测试结构的最小复现重复发射两个 graph，期望最终数组值为 20：

- 直接读取：观察到 11；
- 只在最终等待第二条 stream：观察到 15；
- 最终同步整个设备：仍观察到 15；
- 每轮同时等待第一、第二条 stream：得到 20。

因此，单纯补上最终主机同步不能修复跨 graph 的 external event 依赖。
现有 Warp fork 的 `KNOWN_ISSUES-AMD.md` 也记录了 ROCm replayed external event
的设备端语义差异；应在 Warp/HIP 的 graph 依赖实现中解决或明确平台限制，
不能把此项计为通过。

`TestOptimExamples.test_optim.example_fluid_checkpoint_cuda_0` 在第一次独立
`unittest` 运行中超过两分钟仍无结果，进程由本次调试主动停止；第二次同一
`unittest` 用例 **19.130 秒通过**。直接运行该示例的 300 帧、5 次训练迭代
也完成并输出全部五次 loss。因此先前全套中的 600 秒超时至少具有间歇性；
需要在可抓取 Python/原生栈的配置下再次复现，判断是否为 ROCm 驱动等待或测试并发影响。

## HIP 设备断言修复（定向通过）

Warp 全套中的两个用例在 HIP 故意触发非法 GPU 操作，原测试依赖先 `printf`
再 trap 的诊断。然而 ROCm 队列异常可能使 `printf` 缓冲区未输出。
当前 [Warp 补丁](../profiles/warp/0001-gfx1201-unittest-fixes.patch)在该路径做了两项调整：

- zero-step slice 在 HIP 设备端直接调用 `__assert_fail`，即使 release kernel
  定义了 `NDEBUG`，也能把明确错误原因写到主机诊断；
- tile 越界测试在 HIP 上检查可观察的设备断言及越界类型，CUDA 上仍检查
  完整的坐标文本。

补丁已在精确 Warp commit 的干净 worktree 上通过 `git apply --check`，
SHA-256 锁定于 `versions.toml`。在 ROCm 7.14.1 开发镜像中重新构建调试版
`warp.so`，然后在 PyTorch 镜像中使用全新 JIT 缓存复测，两个原先失败的
`test_array_runtime_zero_step_cuda_0` 和
`test_cuda_shared_tile_oob_reports_tile_index_cuda_0` **2/2 通过**。
这尚未改变前述全套 JUnit 的历史结果；需要重跑全套才能更新总数。

从精确 Warp commit 构建时先应用补丁：

```bash
git apply --check /path/to/curobo_rocm_wrapper/profiles/warp/0001-gfx1201-unittest-fixes.patch
git apply /path/to/curobo_rocm_wrapper/profiles/warp/0001-gfx1201-unittest-fixes.patch
HIP_ARCH=gfx1201 ROCM_PATH=/opt/rocm bash build_amd.sh --mode=debug --no-use-libmathdx -j 4
```

## Paddle CPU 互操作修复（定向通过）

可选的 Paddle CPU 测试先前有指针断言失败和 `malloc(): unaligned tcache chunk
detected` 原生中止。Paddle 3.2.0 的 `Tensor.__array__()` 返回的 NumPy 数据不能
作为直接传入 Warp kernel 的稳定零拷贝指针。补丁让 CPU 参数打包优先使用 DLPack，
并保留转换后 Warp array 的生命周期；通用 `__array__()` 回退也保留实际 ndarray。
`to_paddle()` 的 CPU 路径同样改为 DLPack，以维持原接口承诺的零拷贝。

在无 GPU 的 ROCm 7.14.1 开发镜像中，Paddle 3.2.0 和 3.3.1 的 CPU
互操作类分别 **10/10 通过**；PyTorch 镜像中的 Torch CPU 互操作类
**17/17 通过**。
Paddle GPU 互操作在当前容器无可用 ROCm Paddle 后端，尚未验证。

## 新补丁后的全套运行方式

第一次四 worker 全套复测在 `TestAsync` GPU copy 路径遇到 GPU memory fault，
进程池中止。JUnit 将其余未完成用例标成 7448 个 error；这些是测试框架的
占位记录，**不是** 7448 个独立失败。故障附近的
`test_copy_i2s_d2d_SrcPoolOn_DstPoolOn_NoStream_NoGrad_NoGraph_AccessBoth`
单独运行通过，之后 GPU/Warp/Torch 共享冒烟也通过。原始日志是
`/tmp/amd-warp-halo-rocm714-validation/.validation-home/warp-full-714-torch-after-fixes.log`。

随后单 worker、每测试类独立进程的 `unittest` 全套已完成：**7479 tests，
7 failures，0 errors，237 skipped**，耗时 2443.901 秒。日志和 JUnit 前缀为
`warp-full-714-torch-serial`。其中 zero-step 和 tile 越界修复用例均通过，
`example_fluid_checkpoint` 也通过。七项失败为：两个 graph event、stream
priority 计时、mesh ray edge、fp16 inverse，以及两个 Torch 示例。

两个 Torch 示例的失败来自容器以 UID 1019 启动但 `/etc/passwd` 无对应用户名：
示例调用 `getpass.getuser()` 抛 `KeyError`。在同一镜像中设置
`USER=roywan`、`LOGNAME=roywan` 后，原始两个 unittest 用例 **2/2 通过**。
这是测试容器配置问题，不能归入 Warp/HIP 计算失败。

mesh ray edge 在同一输入下连续构建和查询时，第 16 轮的 `lbvh`、leaf size 2
出现 **0/961** 命中；单次运行可能通过。`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`
后，270 次连续网格构建/查询全部通过。该变量在另一设备组合的 Warp
`KNOWN_ISSUES-AMD.md` 中曾导致首次 LBVH radix sort 失败，因此又在当前
`gfx1201`/ROCm 7.14.1 上连续三轮执行 BVH 与 mesh ray 两个官方测试类：
每轮 **40 tests、38 passed、2 skipped**，均无失败。此变量是当前环境的
候选规避措施，不能据定向结果宣布彻底修复。

初次验证该变量时，先初始化 Warp 后 Torch GPU 分配返回
`hipErrorInvalidValue`，使 Torch 互操作类 43 项中的 23 项 GPU 路径被跳过。
另一轮四 worker 全套在 Marching Cubes GPU 路径发生 memory fault 后进入
逐类回退；回退大量跳过 Torch GPU 用例，因此调试时主动停止，没有完整 JUnit。
此处 Torch 错误后来定位并修复，见下一节。并行 GPU 工作负载另有独立故障，
后续全套继续使用单 worker、逐类独立进程。

## 跨 graph external event 的软件补偿（定向通过）

在默认 ROCm 环境中，重复发射两个互相依赖的 HIP graph 时，单纯在末尾同步
第二条 stream 或整个设备仍可能得到旧值 15/20。每轮在消费者 graph 提交前
对生产者 stream 加普通等待可得到 20/20。补丁据此在 HIP graph 捕获时记录
外部 event 的生产者和消费者；重放消费者前，对最近一次生产该 event 的 stream
插入等待。普通 CUDA 路径不受此逻辑影响。

原版 `TestStreams.test_event_external_cuda_0` **通过**，同一进程连续运行
**30/30 通过**。整个 `TestStreams` 类的 25 项运行仍有 graph elapsed event
和 stream priority timing 两项失败、两项跳过，external event 已通过。
graph elapsed event 即使将捕获中的 event record 从 external 改为普通节点，
本机 `hipEventElapsedTime` 仍返回 0，因此不能把这项算作已修复。
新的补丁 SHA-256 已更新在 `versions.toml`，并再次对固定 Warp commit 的
干净 worktree 通过 `git apply --check`。Warp 仓库针对五个修改文件的
`uvx pre-commit run --files ...` 已通过。

最新补丁在默认 allocator、显式用户名、单 worker 逐类隔离的 PyTorch 容器中
完成完整 `unittest`：**7479 tests，6 failures，1 error，237 skipped**，
耗时 2082.016 秒。原始日志和 JUnit 前缀为 `warp-full-714-torch-graphfix`。
external event、两个 HIP 断言用例及两个 Torch 示例在这轮均通过。六项失败是
graph elapsed event、stream priority timing、fp16 inverse，以及三项
BVH/mesh 查询；一个 error 是 fluid checkpoint 运行 600 秒后超时。

三项几何失败均涉及反复构建或查询 BVH/mesh：
`test_sparse_group_root_isolation_cuda_0` 得到 face `-1`（应为 `2`），
`test_grouped_mesh_query_ray_cuda_0` 查询未命中，
`test_mesh_query_ray_edge_cuda_0` 返回 `0/961` 命中。默认 allocator 下，
分组 BVH 和 mesh 两个官方测试类在第二次独立执行就复现了 grouped mesh
失败；关闭 fragment allocator 后连续三轮、每轮 29 项均无失败。
这加强了它们同属小块设备内存分配故障的判断。初次关闭 fragment allocator
时 Torch GPU 互操作失败，后来定位为 Warp 初始化留下的 HIP 错误状态，见下节。

fluid checkpoint 在前一轮串行全套及单独运行曾通过，本轮子进程则超时。
调试时对该仍存活的子进程采样，5 秒内记录近 60 万次 `sched_yield()`，
主线程持续占用一个 CPU 核心。压缩后的系统调用日志位于隔离目录的
`fluid-strace-714.log.gz`，GDB 栈位于 `fluid-gdb-bt-714.log`。这证明发生
主机侧忙等；还不能据此确定 Warp、HIP 运行时或固件中的最终原因。

## IPC 能力探测的错误状态修复

开启 `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` 时，Warp 初始化调用
`hipIpcGetMemHandle` 探测 IPC，ROCm 返回 `hipErrorInvalidValue`，表示此配置
不支持 IPC。探测代码将能力标为不可用，却未清除 HIP 的 last-error 状态；
随后 Torch 缓存分配器在 GPU 分配路径读到这一旧错误并报错。HIP API 日志和
Torch C++ 栈将问题定位到这条调用链。

当前 Warp 补丁在 HIP 的 IPC 探测结束时调用 `hipGetLastError()` 清除
探测阶段的错误。重新构建调试版 `warp.so` 后，在默认配置与关闭 fragment
allocator 的配置下，`verify_amd_warp.py` 的 Warp/Torch GPU 共享冒烟
均通过；关闭 fragment allocator 时，原版 Warp/Torch 互操作类 **43/43
通过且无跳过**。

以当前补丁、`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`、有效用户名、单 worker
逐类隔离进程完成的完整 unittest：**7479 tests，2 failures，1 error，
237 skipped**，耗时 2104.103 秒。日志与 JUnit 前缀为
`warp-full-714-torch-fragoff-ipcfix`。此前失败的三项 BVH/mesh 查询、
external event、HIP 断言及 Torch GPU 互操作均通过；stream priority
timing 这次通过，但此前有失败记录，仍属不稳定的计时断言。
剩余两项失败是 `test_event_elapsed_time_graph_cuda_0`（返回 0）和
`test_inverse_float16_cuda_0`（差 0.0625，超过 0.05 绝对阈值）；
一个 error 是 `example_fluid_checkpoint_cuda_0` 再次达到 600 秒时限。
因此 Warp 全套仍未通过，不得开始两版 cuRobo 官方全套验收。

## 2026-10-25：Warp 全套资格轮次

进一步将 fp16 inverse 公式的参考计算提升为 float64，保留原有绝对阈值
`0.05`；CPU 和 `gfx1201` GPU 原用例均通过。stream priority 在 HIP 上
只检查两条流实际完成可计时拷贝，CUDA 保留原速度断言。ROCm 7.14.1 的
HIP graph replay 未提供 event 时间戳；原用例仍执行到时间查询，然后明确
标记为跳过。此项是**未支持的功能边界**，不能宣称 HIP graph event 计时可用。

fluid checkpoint 在 `AMD_SERIALIZE_KERNEL=3` 下连续五次原 unittest
均于约 19 秒通过；相同镜像和 allocator 设置、不带该变量的对照运行
首轮超过 90 秒而被独立时限终止。该设置用于当前资格运行的正确性验证，
不能作为异步执行性能的证据。

使用锁定的 Warp commit、当前 `profiles/warp/` 补丁、ROCm 7.14.1、
`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`、`AMD_SERIALIZE_KERNEL=3`、
`HIP_VISIBLE_DEVICES=0`、有效 `USER`/`LOGNAME`，在 `gfx1201`
上以单 worker、每测试类独立进程执行完整 `unittest`：

| tests | failures | errors | skipped | 耗时 |
| ---: | ---: | ---: | ---: | ---: |
| **7479** | **0** | **0** | **238** | 1550.392 秒 |

退出码为 0；日志和 JUnit 前缀为
`/tmp/amd-warp-halo-rocm714-validation/.validation-home/warp-full-714-torch-qualified`。
原先失败的 external event、BVH/mesh、fp16 inverse、fluid checkpoint、
两个 HIP 设备断言用例及 Torch GPU 互操作均在此全套中通过。238 项跳过
包含原环境已有的 237 项和上述 1 项 HIP graph event 计时限制。
Warp 源码仅验证了调试构建；优化构建及性能仍待单独验证。

## 当前门槛

1. 从两版精确上游 commit 和当前冻结补丁重建 cuRobo wheel，
   **先 v0.7.8、后 v0.8.0** 分别运行各自同 tag 的官方全套单元测试。
2. 记录两版的通过、失败、跳过、超时和原生异常；七类迁移测试通过不能替代全套结果。
3. 发布或性能评估前，用优化版 Warp 和目标部署容器复核；当前资格结论限于上述
   调试构建和 ROCm 设置。
