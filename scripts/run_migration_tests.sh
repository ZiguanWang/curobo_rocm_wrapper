#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

profile="${1:?usage: run_migration_tests.sh v0_8_0|v0_7_8}"
wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUROBO_USE_HIP=1
export CUROBO_KERNEL_BACKEND=hip
export CUROBO_REQUIRE_ROCM=1
: "${PYTORCH_ROCM_ARCH:?PYTORCH_ROCM_ARCH must be set, for example gfx1201}"

exec python -m curobo_rocm_wrapper.cli --root "${wrapper_root}" test --profile "${profile}"
