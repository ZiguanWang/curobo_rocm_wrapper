# Upstream modifications

The wrapper pins two upstream cuRobo releases and one shared AMD Warp revision as separate Git
submodules. Each cuRobo profile applies only to its selected release. Warp is patched in an
isolated build worktree so its shared submodule stays at the pinned commit.

## Shared AMD Warp

The `profiles/warp/0001-gfx1201-unittest-fixes.patch` patch adapts Torch tensor
sharing during HIP graph capture, clears the HIP error left by an unsupported
IPC probe, handles cross-graph external-event dependencies, and includes the
gfx1201 fixes used to qualify Warp's own unittest suite. Both cuRobo profiles
use the same built Warp package in separate Python environments.

## v0.8.0

The `profiles/v0.8.0/0001-rocm-hip-backend.patch` patch:

- adds an isolated HIP backend under `curobo/_src/curobolib/backends/hip_backend`;
- adds explicit ROCm runtime detection and fail-closed backend selection;
- adds build-time hipify for four extensions;
- adapts kinematics, collision, costs, optimization, trajectory, and planner paths;
- matches the Python kinematics dispatch to the backported HIP extension bindings;
- accumulates partial-block Warp mesh SDF alignment results without tile reductions;
- routes active mesh and voxel obstacles through cuRobo's existing Warp kernels
  on ROCm, while retaining the dedicated HIP cuboid path;
- uses Warp's public Torch device mapper in block-sparse mesh extraction and
  aligns ROCm pose results with preallocated output buffer shapes;
- adds runtime-compiled HIP RNEA forward and backward kernels using the pinned
  cuRobo dynamics templates, including external-force gradients;
- uses a batched linear solve for ROCm Levenberg-Marquardt updates and handles
  missing CUDA version metadata when checking graph reset support;
- preserves CUDA dispatch and CUDA Graph defaults;
- keeps the general GraphExecutor default tied to the public graph setting while
  allowing ROCm solver graph capture to remain disabled independently;
- retries failed ROCm single-pose IK with scene-free candidates, validates them
  against the active scene, and prioritizes candidates near the current joints
  before trajectory optimization;
- adds the seven retained HIP integration tests.

## v0.7.8

The `profiles/v0.7.8/0001-rocm-legacy-hip-backend.patch` patch:

- enables the existing PyTorch extension build under ROCm/HIP;
- requires ROCm PyTorch and an explicit `PYTORCH_ROCM_ARCH`;
- adds a fail-closed legacy extension preflight;
- uses the public Warp/Torch device mapping API for mesh collision initialization;
- rounds HIP self-collision launches to complete wave32 groups and keeps padded
  line-search threads in block-wide barriers with zero contribution;
- uses the existing PyTorch L-BFGS path on ROCm where the legacy fused kernel
  failed during planner graph execution;
- adds seven legacy build/import/binding migration tests.

The v0.7.8 profile is a compatibility profile, not the recommended production base. It remains
subject to the upstream NVIDIA License and its non-commercial use limitation.
