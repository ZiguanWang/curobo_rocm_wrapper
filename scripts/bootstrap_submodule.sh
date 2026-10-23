#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

profile="${1:-v0_8_0}"
wrapper_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${wrapper_root}"

configuration="$(python3 - "${profile}" <<'PY'
import sys
import tomllib

with open("versions.toml", "rb") as config_file:
    config = tomllib.load(config_file)
profiles = config["profiles"]
profile = sys.argv[1]
if profile not in profiles:
    raise SystemExit(f"unsupported profile: {profile}")
print(profiles[profile]["submodule_path"])
print(profiles[profile]["commit"])
print(config["wrapper"]["upstream_url"])
PY
)"
mapfile -t fields <<< "${configuration}"
submodule_path="${fields[0]}"
revision="${fields[1]}"
upstream_url="${fields[2]}"

if [[ -e "${submodule_path}/.git" ]] && [[ -n "$(git -C "${submodule_path}" status --porcelain)" ]]; then
  echo "${submodule_path} has uncommitted changes; restore it before repinning" >&2
  exit 1
fi

if [[ ! -f .gitmodules ]] || ! git config -f .gitmodules --get "submodule.${submodule_path}.path" >/dev/null; then
  mkdir -p upstream
  git submodule add "${upstream_url}" "${submodule_path}"
fi

git submodule update --init --recursive "${submodule_path}"
git -C "${submodule_path}" checkout --detach "${revision}"
git add .gitmodules "${submodule_path}"

echo "Pinned ${submodule_path} to ${revision} for ${profile}."
echo "Commit the .gitmodules and ${submodule_path} gitlink in the wrapper repository."
