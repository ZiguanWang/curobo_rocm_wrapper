#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

acceptance_output_dir="${CUROBO_ACCEPTANCE_OUTPUT_DIR:-}"
runtime_args=()
pytest_args=()
if [[ -n "${acceptance_output_dir}" ]]; then
  mkdir -p "${acceptance_output_dir}"
  runtime_args+=(--output "${acceptance_output_dir}/container_runtime.json")
  pytest_args+=(--junitxml="${acceptance_output_dir}/container_tests.xml")
fi

python -m curobo._src.curobolib.backends.hip_backend.runtime_preflight "${runtime_args[@]}"

package_root="$(python - <<'PY'
from pathlib import Path

import curobo

print(Path(curobo.__file__).resolve().parent)
PY
)"

motion_tests=()
for test_file in "${package_root}"/tests/_src/motion/test_*.py; do
  if [[ "$(basename "${test_file}")" != "test_motion_planner_esdf.py" ]]; then
    motion_tests+=("${test_file}")
  fi
done

python -m pytest -q \
  "${pytest_args[@]}" \
  --rootdir=/tmp \
  --import-mode=importlib \
  "${package_root}/tests/_src/curobolib/test_hip_kinematics_integration.py" \
  "${package_root}/tests/_src/curobolib/test_hip_geometry_integration.py" \
  "${package_root}/tests/_src/curobolib/test_hip_collision_sphere_cuboid.py" \
  "${package_root}/tests/_src/curobolib/test_hip_tool_pose_integration.py" \
  "${package_root}/tests/_src/curobolib/test_hip_cspace_cost_integration.py" \
  "${package_root}/tests/_src/curobolib/test_hip_optimization_integration.py" \
  "${package_root}/tests/_src/curobolib/test_hip_trajectory_integration.py" \
  "${motion_tests[@]}"
