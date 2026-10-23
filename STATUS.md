# Validation status

## Wrapper contract

- `v0.8.0` tag and full commit verified against the upstream remote.
- `v0.7.8` tag and full commit verified against the upstream remote.
- Both frozen patches pass `git apply --check` on fresh, exact-tag worktrees.
- The wrapper CLI verified the expected revision and applied each patch end to end.
- Both profiles expose exactly seven version-specific migration tests.
- Wrapper unit tests: 4 passed.
- Ruff: passed.

## GPU validation

Both profiles have completed two clean, exact-commit wheel validations on an AMD Radeon AI
PRO R9700 (`gfx1201`). The host's system `hipcc` is a recursive wrapper, so the builds used an
isolated `/tmp/rocm-fixed/bin/hipcc` entry that invokes `amdclang++`; no system ROCm file or unrelated
user process was modified.

- v0.8.0: both clean wheel builds and isolated installs succeeded; the seven manifest files collected
  15 real HIP integration tests with no skips (`15 passed in 0.94s`, then `15 passed in 0.86s`). The final profile also backports
  the internally consistent post-tag kinematics kernel set required to make the pinned research tag
  build, uses wave32 on gfx1201, and fixes the line-search iteration dtype test.
- v0.7.8: both clean wheel builds and isolated installs succeeded; all five legacy extensions compiled
  and the seven category tests launched real GPU kernels with no skips (`7 passed in 7.73s`, then
  `7 passed in 5.30s`). The wheel version lookup,
  legacy launch syntax, wave masks, HIP type conflicts, and wave32 compilation are covered by the
  frozen patch. It remains `legacy`, `non-commercial`, and not a production release target.

Final patch hashes are locked in `versions.toml`, both patches pass `git apply --check` on fresh
exact-commit clones, wrapper tests report `4 passed`, Ruff passes, and shell syntax checks pass.

The requested dual-version technical validation is complete. Permanent CI toolchain packaging and
release artifact retention remain release-engineering follow-ups rather than missing validation.
Detailed evidence is recorded in `docs/validation_evidence_2026-10-22.md`.

## Container audit on 2026-10-23

A serial run in a ROCm PyTorch container on the current `gfx1201` GPU rebuilt and installed both
profiles in separate directories. The seven-file HIP migration suites passed: v0.8.0
`16 passed` after a Python/HIP kinematics dispatch fix and a new integration test; v0.7.8
`7 passed`. Six pristine upstream v0.8.0 kinematics tests also passed after that fix.
Both versions still hit Warp GPU device mapping errors in other upstream paths. See
`docs/container_validation_2026-10-23.md` for the image, exact results, API comparison, and
test coverage limits. Migration validation does not establish full upstream test parity.

## AMD Warp source-build audit on 2026-10-24

AMD-Ecosystem/warp `amd-integration-halo` at
`74014999a31471228e0aaa213bbeb2e5daba95f0` was built in debug mode
for `gfx1201` inside the ROCm PyTorch container. Warp mapped a PyTorch GPU tensor to
a HIP device and executed a GPU kernel. The v0.7.8 patch now uses Warp's public
`device_from_torch` API for mesh collision initialization; its wheel passed 7 migration
tests, 9 selected upstream geometry/Warp tests, and collected all 264 upstream tests.
The v0.8.0 wheel rebuilt from the final patch passed 16 migration tests, 84 selected upstream
sphere-fit/pose/attachment/mesh-alignment tests, and the formerly failing cost-to-MorphIt
sequence (213 tests). Separately, 36 upstream voxel Warp tests passed (one skipped) when
Warp was initialized before test collection. Full upstream parity
remains open: legacy IK self-collision triggers a GPU hardware exception, and the v0.8.0
full suite timed out after advancing past its previous Warp fault. Only a debug Warp
build has been validated on `gfx1201`. See
`docs/amd_warp_source_validation_2026-10-24.md`.

## Warp full-suite audit on ROCm 7.14.1

The pinned AMD Warp debug build ran its full unittest suite in isolated ROCm 7.14.1
containers on `gfx1201`. The development image reported 7424 tests, 8 failures,
0 errors and 239 skips. The PyTorch image reported 7479 tests, 7 failures,
1 error and 237 skips. Neither run passed. A standalone cross-graph external-event
reproducer still returned stale results after a full device synchronization; the
fluid-checkpoint example passed in one standalone unittest run after stalling in
another. Details and JUnit locations are in
`docs/warp_full_suite_rocm714_2026-10-24.md`. The cuRobo upstream full suites have
not yet been rerun on this ROCm 7.14.1 stack.

The HIP diagnostic and CPU Paddle interop fixes are preserved as
`profiles/warp/0001-gfx1201-unittest-fixes.patch`.
It applies cleanly to the pinned Warp commit, and the two previously failing
invalid-operation tests passed individually after a fresh debug rebuild and fresh
JIT cache. Paddle CPU interop passed 10/10 with each of Paddle 3.2.0 and 3.3.1,
and Torch CPU interop passed 17/17.

A subsequent single-worker, process-isolated Warp full suite completed with
7479 tests, 7 failures, 0 errors and 237 skips. The two HIP invalid-operation
tests and fluid-checkpoint example passed. Two Torch example failures were due
to the numeric container UID lacking a passwd entry and passed 2/2 after setting
`USER` and `LOGNAME`. An intermittent LBVH mesh-ray miss reproduced after repeated
builds; `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` eliminated it in 270 repeated builds,
and three runs of the BVH plus mesh-ray classes passed 40 tests each (2 skips).
Initially, that allocator setting made Torch GPU allocations fail after Warp
initialization (`hipErrorInvalidValue`). The Warp IPC capability probe left an
expected HIP error uncleared; the pinned Warp patch now clears it. After a debug
rebuild, Warp/Torch sharing passed with and without the allocator setting, and
the Torch interop class passed 43/43 under that setting. The remaining event,
stream-priority, mesh-ray and fp16 discrepancies still require full-suite audit.

The Warp patch now also tracks external events captured into HIP graphs and inserts
a regular stream dependency before replaying a consumer graph. The original
cross-graph event test passed 30 consecutive runs and passed in the full
`TestStreams` class. Graph elapsed timing still returns zero; stream priority
timing remains flaky. Full-suite counts above predate this graph change.

The final full unittest run with the graph fix, a valid container username, and
the default allocator recorded 7479 tests, 6 failures, 1 error and 237 skips
in 2082.016 seconds. Three failures involve BVH/mesh queries; two event/timing
checks and fp16 inverse remain open. Fluid checkpoint timed out after 600 seconds,
with a live child process repeatedly calling `sched_yield()`. The full raw log
and JUnit are under `/tmp/amd-warp-halo-rocm714-validation/.validation-home/`
with prefix `warp-full-714-torch-graphfix`. See the updated Warp full-suite report.

After clearing the HIP last-error state left by Warp's IPC probe, the same
Warp/Torch sharing smoke and all 43 Torch interop tests passed with
`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`. A complete, single-worker, process-isolated
Warp suite under that setting recorded 7479 tests, 2 failures, 1 error and
237 skips in 2104.103 seconds. All three previously failing BVH/mesh tests
passed. The remaining failures are graph event elapsed timing and fp16 inverse;
the fluid-checkpoint example again timed out after 600 seconds. Raw JUnit and log
prefix: `warp-full-714-torch-fragoff-ipcfix` in the validation directory.

## Warp qualified full suite on 2026-10-25

With the latest pinned Warp patch, ROCm 7.14.1, `gfx1201`, a debug Warp build,
`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, `AMD_SERIALIZE_KERNEL=3`, and a single
process-isolated worker, the full unittest suite completed: **7479 tests,
0 failures, 0 errors, 238 skips** in 1550.392 seconds (exit code 0).
The additional skip states that HIP graph replay did not provide event timestamps;
that API remains unsupported in this environment. The original fp16 inverse
test now uses a float64 reference without widening its tolerance, and the fluid
checkpoint example passed in the complete run. Raw log and JUnit prefix:
`/tmp/amd-warp-halo-rocm714-validation/.validation-home/warp-full-714-torch-qualified`.
The next validation gate is the original upstream cuRobo full suite, first
v0.7.8 and then v0.8.0, each from a clean exact-commit wheel.

## cuRobo v0.7.8 official suite on 2026-10-25

The updated v0.7.8 profile patch applies to its exact upstream commit, and a
fresh, isolated ROCm 7.14.1 wheel completed the seven HIP migration tests
(`7 passed`) and the unmodified same-tag official suite: **268 tests,
257 passed, 11 skipped, 0 failures, 0 errors** in 917.558 seconds.
The 11 skips cover unavailable USD/Nvblox dependencies and missing
`CUDAGraph.reset()` in the tested PyTorch version. The legacy self-collision
hardware exception was fixed by completing HIP wave32 launches; a line-search
barrier defect was fixed, and ROCm uses cuRobo's existing PyTorch L-BFGS path
where the legacy fused kernel faulted. Wheel and patch digests, environment,
JUnit and limits are in `docs/curobo_v078_official_rocm714_2026-10-25.md`.
This non-commercial compatibility profile is technically validated for the
recorded test scope.

## cuRobo v0.8.0 official suite on 2026-10-25

The updated v0.8.0 patch applies to its exact upstream commit. A fresh ROCm
7.14.1 wheel was built and installed in isolation. Its seven HIP migration
files passed **16/16**; the unmodified same-tag official suite completed
**3966 tests: 3928 passed, 38 skipped, 0 failures, 0 errors** in 1378.082
seconds with exit code 0. All 199 copied official Python test files match the
exact upstream commit byte for byte. The two prior failures, GraphExecutor's
default graph setting and `motion_planning --test`, passed in the full suite.

The GraphExecutor default now follows the public graph setting while ROCm
solver graph capture remains separately guarded. On ROCm, failed single-pose
IK may generate scene-free candidates; a new scene-aware solver validates them
and the nearest valid joint configurations seed trajectory optimization. The
original example's goal, obstacles, linear grasp criteria and assertions were
unchanged. The 38 skips include 28 without Pinocchio and four unimplemented
acceleration-kernel backward cases; skipped features are not validated by this
result. Patch and wheel hashes, environment, JUnit and limits are documented
in `docs/curobo_v080_official_rocm714_2026-10-25.md`.

## Shared Warp submodule installation on 2026-10-26

The wrapper now pins AMD Warp as a third Git submodule at
`upstream/warp-rocm`, using the same commit and patch as the full-suite
qualification above. The submodule worktree remains clean; a separate build
worktree applies the patch. In the ROCm 7.14.1 PyTorch container, a fresh
debug wheel was built and installed into an isolated target directory.
Wheel SHA-256: `a73fe52691a94a9ab6204379e84626fb4aae2b3f58574054b6864399f4ff5526`.
Warp mapped the PyTorch GPU to HIP `gfx1201`, shared a tensor, and launched
a kernel successfully. The newly installed wheel also passed the original
v0.7.8 Warp mesh file (3/3) and v0.8.0 mesh SDF alignment file (13/13).
The build required ROCm development headers and unversioned library links in
the disposable container; the README records the installation steps.
