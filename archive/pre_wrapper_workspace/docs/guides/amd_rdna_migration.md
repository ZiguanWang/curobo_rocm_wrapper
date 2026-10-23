# Migrating cuRobo to AMD RDNA GPUs

## Goal

Port the cuRobo capabilities required by RoboTwin/xpolicy evaluation to an AMD RDNA GPU through
ROCm, while preserving the CUDA backend and the user-facing motion-planning API. The first usable
release must run motion generation on the GPU, produce planning results comparable to the existing
CUDA implementation, and outperform the CPU-only MPLib fallback on the target evaluation workload.

This document is an engineering migration plan, not a statement that the current repository already
supports ROCm. The current implementation depends on NVIDIA-specific components and cannot be made
portable by changing device strings alone.

## Scope and success criteria

The migration has two delivery tracks:

1. **RoboTwin enablement:** implement only the execution paths exercised by xpolicy evaluation,
   including robot kinematics, collision checking, IK, trajectory optimization, graph planning, and
   trajectory interpolation as required by the captured workload.
2. **General ROCm backend:** complete the backend contract, remove NVIDIA-only assumptions from
   shared code, add packaging and CI, and progressively replace or port Warp-dependent features.

The first track is the release blocker. Perception, volumetric mapping, pose estimation, and other
Warp-heavy features must not delay RoboTwin enablement unless workload tracing proves that xpolicy
uses them.

The minimum release gates are:

| Area | Required result |
| --- | --- |
| Import and setup | cuRobo imports without CUDA packages on a supported ROCm installation |
| Device execution | The xpolicy cuRobo path keeps tensors and hot-loop operations on the AMD GPU |
| Functional coverage | The recorded RoboTwin workload completes without falling back to MPLib |
| Correctness | FK, gradients, collision distances, IK, and trajectories meet defined tolerances |
| Planning quality | Success rate is no worse than 2 percentage points below the CUDA reference |
| Performance | Median end-to-end planning latency is at least 2 times faster than CPU MPLib |
| Stability | At least 1,000 consecutive evaluation plans without a crash, hang, or growing memory use |
| Compatibility | Existing NVIDIA CUDA tests and benchmarks do not regress beyond agreed noise bounds |

The performance and quality thresholds are initial project gates. Freeze the final values after the
baseline phase, using identical scenes, seeds, robot models, planner settings, and time budgets.

## Supported platform contract

Do not describe support as generic "AMD GPU support." Publish and test an explicit matrix:

| Component | Project decision |
| --- | --- |
| Operating system | Linux only for the first release |
| GPU | One named RDNA SKU and its exact `gfx` target for bring-up |
| ROCm | One pinned major/minor release validated against that GPU |
| PyTorch | The matching official ROCm wheel or a documented source build |
| Python | One version already supported by cuRobo and available in the chosen PyTorch wheel |
| Compiler | The `hipcc`/Clang version shipped with the pinned ROCm release |
| Precision | `float32`; do not introduce `float64` into tests |
| Execution graph | Eager execution first, HIP graph replay after correctness is established |

RDNA support varies by GPU SKU, operating system, ROCm release, and PyTorch build. Before coding,
verify that the target machine passes all of the following checks:

```bash
rocminfo
rocm-smi
hipconfig --full
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
python -c "import torch; x=torch.ones(1024, device='cuda'); print((x+x).sum())"
```

PyTorch deliberately exposes ROCm devices through the `torch.cuda` Python namespace. Therefore,
`torch.cuda.is_available()` can be true on AMD and is not evidence that CUDA is installed. Backend
detection must use `torch.version.hip`, not namespace names.

Record the output of `rocminfo`, especially the `gfx` architecture. Do not use an unsupported
architecture override as a production solution. An override can be useful for a time-boxed
experiment, but a release needs either official support or a reproducible PyTorch/ROCm source build.

## Current portability assessment

### Existing backend boundary

The repository already has a useful kernel backend selector in
`curobo/_src/curobolib/backends/__init__.py`. Its current implementations are:

- `cuda_core`: runtime compilation and launch through NVIDIA `cuda.core`;
- `pybind`: ahead-of-time PyTorch C++/CUDA extensions.

The logical module contract covers kinematics, optimization, trajectory, geometry, dynamics, and
the parallel-banding distance transform. This is the correct boundary at which to add a HIP backend.
Do not fork the public motion planner or duplicate solver logic for AMD.

### Portability blockers

| Blocker | Examples in this repository | Migration consequence |
| --- | --- | --- |
| NVIDIA runtime compiler | `backends/cuda_core_backend/` imports `cuda.core` and CUDA runtime bindings | Keep it CUDA-only; add a separate HIP implementation |
| CUDA extension build | `setup.py` uses `CUDAExtension` and NVCC-only flags | Add a ROCm/HIP build path and architecture controls |
| CUDA headers and streams | `torch_cuda_utils.h`, CUDA stream types, C10 CUDA launch checks | Use HIP/PyTorch abstractions or a small compatibility header |
| Warp dependency | Collision, cost, mesh, mapping, and perception modules import `warp` | Replace required paths; defer non-RoboTwin features |
| 32-lane assumptions | `curobo_constants.h`, B-spline and line-search kernels | Make subgroup width explicit and test on the target `gfx` architecture |
| CUDA intrinsics | `__shfl_down_sync`, `__ballot_sync`, `__syncwarp`, CUDA atomics | Wrap portable subgroup operations and verify HIP semantics |
| CUDA graph utilities | graph, stream, event, and timer helpers use `torch.cuda` concepts | Rename by capability; validate HIP graph behavior independently |
| CUDA-only dependencies | `cuda-core` and NVIDIA runtime extras in `pyproject.toml` | Split optional dependencies by accelerator backend |
| CUDA terminology in config | `use_cuda_graph`, `cuda_graphs`, and CUDA-specific error messages | Preserve compatibility while introducing accelerator-neutral aliases |

### Warp decision

Treat NVIDIA Warp as unavailable on ROCm unless the exact pinned Warp release is independently
demonstrated to provide a supported HIP backend. CPU execution of Warp kernels is not an acceptable
hot-path fallback for this project because it recreates the performance problem that motivates the
migration.

The first release should replace only the Warp operations reached by RoboTwin. Likely candidates are
world collision costs and related autograd wrappers, but workload tracing, rather than filenames,
must determine the final list. Preserve Warp for CUDA so the AMD work does not destabilize the
existing backend.

## Target architecture

Use three layers:

```text
Public cuRobo APIs and solver logic
                |
Accelerator-neutral runtime capabilities
                |
    +-----------+-------------+
    |                         |
CUDA implementations     ROCm implementations
cuda.core / pybind       HIP PyTorch extension
Warp where supported     native HIP or ATen ops
```

The public APIs continue accepting PyTorch devices. Internally, introduce capabilities such as
`supports_graph_capture`, `supports_custom_kernels`, `supports_mesh_collision`, and
`subgroup_size`; avoid scattering checks for NVIDIA or AMD across planners and costs.

The recommended first ROCm kernel backend is an ahead-of-time HIP PyTorch extension. It has a lower
bring-up risk than building a new HIP runtime compiler and cache system, and PyTorch already owns the
tensor allocation and current stream. Runtime compilation can be considered later if it delivers a
measured deployment benefit.

Use a backend name such as `hip` or `rocm`, not `hip_pybind` in public configuration. The
implementation mechanism should remain private.

## Phase 0: Freeze the workload and baselines

**Duration:** 3-5 engineering days.

1. Pin the RoboTwin repository revision, xpolicy revision, task list, robot, scene assets, and cuRobo
   revision.
2. Run xpolicy once with the CUDA cuRobo path and once with CPU MPLib.
3. Capture every cuRobo entry point, configuration file, world representation, tensor shape, dtype,
   number of seeds, graph-planner setting, timeout, and interpolation mode used during evaluation.
4. Record whether scenes contain cuboids, meshes, voxels, ESDF grids, or attached objects.
5. Capture representative inputs and expected outputs into a redistributable replay corpus. Remove
   simulator timing and rendering from the microbenchmark.
6. Measure warm-up separately from steady-state latency. Report p50, p90, and p99 planning latency,
   GPU/CPU memory, success rate, path length, smoothness, and collision margin.
7. Profile the CUDA reference to rank kernels by total time and call count. This ranking determines
   port order.

Deliverables:

- a locked compatibility matrix;
- a RoboTwin replay corpus covering at least one success and one failure case per scene type;
- CUDA cuRobo and CPU MPLib baseline reports;
- an exact feature-reachability list for xpolicy;
- a list of quality metrics accepted by the evaluation owner.

Exit criterion: a developer can replay the planning workload without launching the full simulator,
and repeated baseline runs are stable enough to define regression limits.

## Phase 1: Establish the ROCm development environment

**Duration:** 2-4 engineering days after hardware support is confirmed.

1. Build a container or lock file from the pinned ROCm/PyTorch combination.
2. Pass PyTorch tensor, autograd, stream, event, and profiler smoke tests on the target GPU.
3. Compile and run a minimal PyTorch HIP extension that reads a tensor from the current stream,
   writes an output tensor, and supports backward.
4. Confirm the target architecture and subgroup size at compile time and runtime.
5. Verify AddressSanitizer on host code and ROCm debugging/profiling tools available on the target.
6. Store compiler, driver, firmware, kernel, and package versions in every benchmark artifact.

Run the repository preflight on the target machine after setting its real GFX architecture:

```bash
export PYTORCH_ROCM_ARCH=gfx1100
python -m curobo._src.curobolib.backends.hip_backend.runtime_preflight
```

Build the first kinematics slice independently before enabling the remaining modules:

```bash
export CUROBO_USE_HIP=1
export CUROBO_HIP_MODULES=kinematics
pip install -e ".[hip]" --no-build-isolation
```

Do not begin bulk kernel conversion until this phase passes. Many consumer RDNA failures originate
in an unsupported platform combination rather than in cuRobo code.

Exit criterion: a reproducible environment builds and runs a custom HIP PyTorch operator with a
correct gradient on the target machine.

## Phase 2: Refactor runtime and backend selection

**Duration:** 4-7 engineering days.

Implement the architecture seam before porting kernels:

1. Add accelerator detection based on PyTorch build metadata:
   `torch.version.hip` for ROCm and `torch.version.cuda` for CUDA.
2. Extend the backend selector to support `auto`, `cuda_core`, `pybind`, and `hip`.
3. Make `auto` select `hip` on a ROCm PyTorch build without first importing `cuda.core`.
4. Replace global CUDA booleans in internal code with a runtime capability object. Keep existing
   public flags as deprecated aliases where compatibility requires them.
5. Generalize event timers, streams, synchronization, graph executors, profiler labels, cache paths,
   and error messages. It is acceptable for their implementation to use `torch.cuda` on ROCm; their
   cuRobo names and behavior must be accelerator-neutral.
6. Make backend feature gaps explicit. A missing ROCm operation must raise a targeted error naming
   the feature and suggested configuration, never silently run a large CPU computation.
7. Add a backend reset hook for tests so selection is not permanently cached across parametrized
   cases.

Initial graph policy:

- keep eager execution enabled on ROCm;
- keep CUDA graphs enabled for normal NVIDIA runtime;
- disable graph capture only for ROCm bring-up and correctness tests;
- implement and enable HIP graph replay after the eager path is correct and profiled.

Exit criterion: imports, configuration construction, and backend selection work on CPU-only, CUDA,
and ROCm hosts without importing packages belonging to another accelerator.

## Phase 3: Add the HIP extension toolchain

**Duration:** 4-7 engineering days.

1. Add an explicit ROCm optional dependency group. Do not install `cuda-core` or NVIDIA runtime
   wheels in the ROCm environment.
2. Add a HIP build mode that is selected from the PyTorch build and an explicit environment override.
3. Compile a separate extension namespace so CUDA and ROCm artifacts cannot collide in editable
   installs or caches.
4. Pass the target `gfx` architecture explicitly through `PYTORCH_ROCM_ARCH` or the project build
   configuration. Avoid fat binaries until the first target is stable.
5. Translate NVCC-only optimization flags to reviewed HIP/Clang equivalents. Do not mechanically
   copy flags such as CUDA fast-math controls without numerical comparison.
6. Add a small compatibility layer for stream access, launch checking, runtime errors, dynamic shared
   memory attributes, vector types, infinities, and accelerator annotations.
7. Prefer PyTorch's extension APIs and `c10` abstractions where they are supported on both CUDA and
   ROCm. Use `hipify` as a conversion aid, not as the architecture of the port.
8. Generate build metadata that records cuRobo revision, PyTorch version, ROCm version, `gfx` target,
   and compiler flags.

Keep the mathematical kernel headers shared where practical. Separate launch adapters and
accelerator-specific primitives when conditional compilation would otherwise become difficult to
review.

Exit criterion: the package builds in a clean ROCm environment, loads the HIP extension, and reports
its compiled architecture and backend name.

## Phase 4: Port kernels in RoboTwin priority order

**Duration:** 4-8 weeks, strongly dependent on the traced workload and kernel correctness.

Port one vertical slice at a time. Each slice includes forward, backward, Python binding, focused
tests, replay-corpus validation, and a benchmark.

### 4.1 Kinematics

Port forward kinematics, Jacobians, sphere transforms, and backward kernels first. Most downstream
costs and solvers depend on them.

Validation:

- compare link poses and robot spheres against CUDA and a CPU reference;
- run finite-difference gradient checks on small robots;
- cover fixed, revolute, prismatic, mimic, locked, and high-DoF joint layouts used by RoboTwin;
- test non-contiguous rejection or handling according to the existing contract.

### 4.2 Primitive world and self-collision

Port sphere-cuboid and self-collision paths needed by the evaluation scenes. Preserve signed-distance,
activation-distance, sparsity, and gradient conventions exactly.

Pay special attention to:

- ballot result widths and lane masks;
- dynamic shared-memory limits;
- deterministic pair reduction;
- atomic ordering and equal-distance ties;
- empty collision-pair sets and padded robot spheres.

### 4.3 Mesh collision

If the trace contains mesh worlds, choose one implementation only after a short spike:

- port the required Warp mesh-query behavior to HIP;
- transform static meshes into a GPU-friendly BVH and implement sphere-mesh distance in HIP; or
- preprocess evaluation assets into a validated primitive/voxel representation if this is permitted
  by the evaluation contract and does not change collision results beyond tolerance.

Do not default to CPU mesh queries inside the optimizer. Host-device synchronization per iteration
will usually erase the benefit of GPU trajectory optimization.

### 4.4 Optimization kernels

Port line search and the L-BFGS step only after a PyTorch-op fallback is functioning on ROCm. This
provides a correctness oracle and lets end-to-end planning run before the tuned HIP kernels land.

For line search, replace the implicit 32-lane contract with a logical subgroup abstraction. The
algorithm's maximum number of line-search candidates should not be derived from a global hardware
warp constant.

### 4.5 Trajectory operations

Port finite differences, integration, B-spline forward/backward, boundary constraints, and
interpolation. Remove assumptions that interpolation steps fit a 32-lane CUDA warp. Derive launch
geometry from the logical tile width and test every spline degree and boundary mode used by xpolicy.

### 4.6 Graph planner support

The high-level graph search can remain in shared Python/PyTorch code. Verify that node sampling,
edge collision checking, batched seeds, and graph buffers remain on the GPU where intended. Port
custom operations reached by edge validation; do not port unrelated mapping kernels merely because
they share a directory.

### 4.7 Dynamics and PBA

Port inverse dynamics and the parallel-banding distance transform only if workload tracing requires
them. Otherwise schedule them for general ROCm completeness after the xpolicy milestone.

Exit criterion: the standalone motion planner completes the full replay corpus on RDNA without CPU
hot-path fallback.

## Phase 5: Make subgroup behavior portable

**Duration:** overlaps Phase 4; budget 1-2 focused weeks.

CUDA warp assumptions are correctness risks, not only performance concerns. Introduce reviewed
primitives for:

- subgroup ballot;
- shuffle-down and shuffle-by-index;
- subgroup synchronization;
- active-lane masks;
- first/last active lane;
- sum, maximum, and custom-structure reductions;
- logical tiles smaller than the hardware subgroup;
- block-wide reductions.

Compile-time assertions must document the supported subgroup modes. Runtime tests must query and log
the actual device value. Test partial subgroups and divergent branches; CUDA `_sync` intrinsics and
HIP equivalents do not justify assuming identical mask semantics.

For RDNA, tune and validate wave32 first if the selected toolchain and GPU support it. Do not bake
wave32 into shared algorithms. A future CDNA target may use a different native wavefront width.

Exit criterion: no algorithmically significant constant named `warpSize = 32` remains in shared
kernel logic, and subgroup unit tests pass under the target compilation mode.

## Phase 6: Restore graph replay and asynchronous execution

**Duration:** 1-2 weeks after eager correctness.

1. Verify whether the pinned PyTorch/ROCm release supports graph capture for every operation in the
   planning hot loop.
2. Add a minimal capture/replay test for a custom HIP operator on the current PyTorch stream.
3. Audit allocations, host reads, scalar extraction, logging, lazy compilation, and stream creation
   inside captured regions.
4. Warm up all kernels and caches before capture.
5. Maintain stable tensor shapes and storage addresses across solve calls, matching cuRobo's current
   graph-oriented buffer reuse.
6. Compare eager and replay outputs, then benchmark steady state and first-call latency separately.
7. If one operation cannot be captured, first isolate it outside the graph. Disable graph replay for
   the entire solver only as a documented temporary limitation.

Exit criterion: graph replay is enabled by default for the supported ROCm planning configuration, or
a benchmark demonstrates that eager execution still meets the agreed xpolicy latency gate and the
limitation is explicitly documented.

## Phase 7: Integrate RoboTwin/xpolicy

**Duration:** 1 week after standalone planning passes.

1. Add an accelerator selection option to the integration layer without changing planner semantics.
2. Remove NVIDIA-only environment checks, package pins, container bases, and `nvidia-smi` probes from
   the AMD launch path.
3. Keep device inputs as `torch.device("cuda")` where PyTorch ROCm requires that spelling, but report
   the accelerator as ROCm in logs and artifacts.
4. Ensure simulator tensors and cuRobo tensors are interoperable without staging through NumPy or
   host memory. If the simulator cannot expose ROCm tensors, measure and batch unavoidable copies.
5. Warm the planner before evaluation timing begins.
6. Run identical random seeds and planning budgets for CUDA cuRobo, ROCm cuRobo, and CPU MPLib.
7. Store per-episode failure reasons: IK failure, graph failure, optimization failure, timeout,
   collision, invalid trajectory, or integration error.

Exit criterion: xpolicy selects cuRobo on the AMD host, completes the target RoboTwin suite, and does
not silently route planning to MPLib.

## Test strategy

### Test layers

| Layer | Purpose | Examples |
| --- | --- | --- |
| Build smoke | Detect packaging and ABI failures | clean wheel build, extension import, architecture report |
| Kernel unit | Validate individual math and edge cases | FK, reductions, signed distance, spline gradients |
| Gradient | Validate optimization inputs | autograd comparison and finite differences |
| Backend parity | Compare implementations | CUDA versus ROCm from saved inputs; CPU oracle where available |
| Solver | Validate IK/trajopt/graph behavior | fixed seeds, known feasible and infeasible scenes |
| Replay corpus | Validate actual xpolicy call patterns | captured shapes, scene types, planner configs |
| End-to-end | Validate evaluation behavior | full RoboTwin task suite |
| Soak | Detect races and leaks | repeated plans, changing goals/worlds, graph reset |

### Numerical policy

Do not require bitwise equality across NVIDIA and AMD. Define tolerances per operation from their
effect on planning:

- pose and FK errors in translation and angle;
- absolute and relative error for costs and gradients;
- sign agreement near collision boundaries;
- cosine similarity for high-dimensional optimizer directions;
- trajectory feasibility, collision margin, smoothness, and terminal pose error.

Use `float32` in tests. Run multiple deterministic seeds because reduction and atomic order can differ
between architectures. Any tolerance increase must be justified by an end-to-end quality comparison,
not only by making a unit test pass.

### Required edge cases

Cover zero-sized optional inputs, one and many environments, batch size one, maximum xpolicy batch,
partial subgroups, high-DoF robots, disabled links, attached objects, goal sets, changing worlds,
infeasible goals, near-contact gradients, graph recapture, and out-of-memory reporting.

## Performance plan

Benchmark before optimizing. For every kernel and planning stage, capture:

- first-call compilation or loading time;
- warm eager latency;
- warm graph-replay latency;
- launch count and CPU launch overhead;
- achieved occupancy and limiting resource;
- memory bandwidth, cache behavior, and divergent branches;
- temporary allocation count and peak memory;
- host-device synchronizations.

Optimize in this order:

1. remove CPU fallbacks and synchronizations from the hot loop;
2. fuse operations only where profiling shows launch overhead dominates;
3. tune block size, wave mode, and shared memory for the selected RDNA GPU;
4. reduce unnecessary tensor materialization and memory traffic;
5. restore graph replay;
6. specialize common RoboTwin shapes only behind a general correct implementation.

Use the CUDA implementation as a quality and algorithm reference, not as a requirement for equal
kernel timing. Compare end-to-end planning latency against both CUDA cuRobo and CPU MPLib.

## Packaging and CI

Create separate install profiles:

- base Python dependencies;
- CUDA 12/13 runtime dependencies;
- ROCm dependencies without NVIDIA wheels;
- optional CUDA pybind build dependencies;
- development and documentation dependencies.

For ROCm wheels, encode compatibility in release notes and artifact metadata even if Python wheel
tags cannot fully express the ROCm and `gfx` matrix. Do not publish a single binary presented as
supporting untested RDNA architectures.

The minimum CI matrix is:

| Job | Frequency | Required checks |
| --- | --- | --- |
| CPU import/lint/docs | Every change | no accelerator import at module import time |
| NVIDIA CUDA | Every backend change | existing suite and focused parity tests |
| AMD RDNA ROCm | Every backend change | build, kernel, solver, and replay tests |
| RoboTwin end-to-end | Nightly or scheduled | quality, latency, and failure taxonomy |
| RDNA soak | Nightly | 1,000+ plans and memory trend |

Cache compiled artifacts by source hash, PyTorch ABI, ROCm version, compiler version, and `gfx`
target. Never reuse a CUDA cache entry for ROCm or one RDNA target for another without validation.

## Delivery milestones

| Milestone | Outcome | Indicative effort |
| --- | --- | --- |
| M0 | Platform and xpolicy workload proven, baselines frozen | 1-2 weeks |
| M1 | ROCm package builds; backend/runtime seam works | 1-2 weeks |
| M2 | FK, primitive collision, and IK run correctly on RDNA | 2-4 weeks |
| M3 | Trajopt and required graph-planning paths pass replay corpus | 3-5 weeks |
| M4 | xpolicy uses ROCm cuRobo end to end and beats CPU MPLib | 1-2 weeks |
| M5 | Graph replay, tuning, soak, packaging, and CI complete | 2-4 weeks |
| M6 | Deferred Warp/dynamics/perception features added as needed | separate roadmap |

These estimates assume one engineer familiar with GPU kernels plus support from someone who owns the
RoboTwin integration. Hardware/toolchain incompatibility, mesh collision requirements, and Warp
replacement are the largest sources of schedule variance.

## Risk register

| Risk | Early signal | Mitigation or decision |
| --- | --- | --- |
| Target RDNA GPU is unsupported by pinned ROCm/PyTorch | missing `gfx` target, runtime faults | change to a supported matrix or own a source-build toolchain before porting |
| Warp has no usable ROCm backend | import or device creation fails | replace only traced hot paths with HIP; defer perception |
| Mesh collision dominates work | trace shows mesh worlds on every solve | prioritize BVH spike in Phase 0 and choose one representation contract |
| 32-lane assumptions produce silent errors | parity fails only at some shapes | subgroup abstraction plus partial-wave tests and sanitizer runs |
| HIP graph capture is incomplete | capture errors or replay drift | ship eager milestone, isolate unsupported ops, restore capture incrementally |
| Numerical drift hurts success rate | kernels pass loose tests but plans fail | compare gradients and intermediate costs, then retune tolerances or solver parameters with evidence |
| PyTorch fallback is too slow | excessive launch count in profiler | use fallback only as oracle, port measured optimizer/collision hotspots |
| Extension ABI is fragile | load failures across PyTorch patches | pin supported builds, test clean wheels, minimize private PyTorch APIs |
| Simulator cannot share ROCm tensors | repeated device-host copies | add a measured interop layer or batch transfer outside optimization loop |
| CUDA backend regresses | NVIDIA CI or benchmark failure | retain separate implementations behind one contract and gate every backend change |

## Work breakdown and ownership

Use the following parallel workstreams after M0:

1. **Platform/build:** container, PyTorch HIP extension, packaging, CI, artifact metadata.
2. **Runtime/backend:** selection, capabilities, stream/event/graph abstraction, diagnostics.
3. **Kernel port:** compatibility primitives, kinematics, collision, optimization, trajectory.
4. **Validation/performance:** replay corpus, parity oracle, profiler reports, soak tests.
5. **RoboTwin integration:** xpolicy adapter, simulator interop, evaluation and failure analysis.

Every kernel task is complete only when its tests and benchmark land with it. Maintain a migration
dashboard with the columns: operation, xpolicy-required, CUDA implementation, ROCm implementation,
forward parity, backward parity, graph-safe, benchmark status, owner, and blocker.

## Definition of done

The RoboTwin migration is complete when all of the following are true:

- the supported RDNA/ROCm/PyTorch matrix is documented and reproducible;
- installing the ROCm profile does not install CUDA runtime packages;
- the `hip` backend implements every operation reached by the frozen xpolicy workload;
- no required planning operation silently falls back to CPU or MPLib;
- focused numerical, gradient, solver, replay, and soak tests pass;
- xpolicy planning quality meets the agreed CUDA-reference threshold;
- end-to-end planning latency beats CPU MPLib by the agreed margin;
- CUDA behavior and normal CUDA graph execution remain intact;
- unsupported features fail early with actionable messages;
- CI protects both the CUDA and RDNA paths;
- benchmark artifacts identify the exact software and hardware matrix.

General AMD support is a later milestone. It requires testing additional `gfx` targets and completing
the deferred Warp-dependent perception, mapping, mesh, dynamics, and distance-transform paths rather
than extrapolating from one successful RDNA machine.
