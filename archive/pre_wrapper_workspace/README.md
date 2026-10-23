# Pre-wrapper workspace archive

This directory preserves the untracked ROCm, Robotwin, CI, container, benchmark, test, and
documentation files that existed in the cuRobo working tree before the migration was extracted
into this wrapper.

The paths below this directory mirror their former paths relative to the cuRobo repository root.
They are retained for traceability and future selective extraction only. They are not applied by
the wrapper CLI and are not part of the seven-test migration contract.

The authoritative version adaptations are:

- `profiles/v0.8.0/0001-rocm-hip-backend.patch`
- `profiles/v0.7.8/0001-rocm-legacy-hip-backend.patch`

The original cuRobo working tree was restored after this archive was created, so the parent
repository remains unchanged apart from the untracked `curobo_rocm_wrapper/` directory.
