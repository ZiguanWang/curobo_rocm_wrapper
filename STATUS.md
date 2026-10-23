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
