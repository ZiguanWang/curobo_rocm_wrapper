# Upstream modifications

The wrapper pins the two upstream cuRobo releases as separate Git submodules. Applying a profile
modifies only that version's submodule worktree; the other version remains separate.

## v0.8.0

The `profiles/v0.8.0/0001-rocm-hip-backend.patch` patch:

- adds an isolated HIP backend under `curobo/_src/curobolib/backends/hip_backend`;
- adds explicit ROCm runtime detection and fail-closed backend selection;
- adds build-time hipify for four extensions;
- adapts kinematics, collision, costs, optimization, trajectory, and planner paths;
- matches the Python kinematics dispatch to the backported HIP extension bindings;
- preserves CUDA dispatch and CUDA Graph defaults;
- adds the seven retained HIP integration tests.

## v0.7.8

The `profiles/v0.7.8/0001-rocm-legacy-hip-backend.patch` patch:

- enables the existing PyTorch extension build under ROCm/HIP;
- requires ROCm PyTorch and an explicit `PYTORCH_ROCM_ARCH`;
- adds a fail-closed legacy extension preflight;
- adds seven legacy build/import/binding migration tests.

The v0.7.8 profile is a compatibility profile, not the recommended production base. It remains
subject to the upstream NVIDIA License and its non-commercial use limitation.
