# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Load and validate version-specific cuRobo adaptation profiles."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    """Describe one immutable upstream cuRobo adaptation target."""

    name: str
    submodule_path: Path
    tag: str
    commit: str
    license: str
    commercial_distribution: bool
    layout: str
    patch_series: Path
    patch_sha256: str
    test_manifest: Path

    def validate(self, wrapper_root: Path) -> None:
        """Validate immutable revision, licensing, and referenced files."""
        if self.submodule_path.is_absolute() or ".." in self.submodule_path.parts:
            raise ValueError(f"{self.name}: submodule path must stay inside the wrapper")
        if len(self.commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.commit
        ):
            raise ValueError(f"{self.name}: commit must be a full lowercase Git SHA")
        if self.name == "v0_7_8" and self.commercial_distribution:
            raise ValueError("v0.7.8 must remain non-commercial under its upstream license")
        for relative_path in (self.patch_series, self.test_manifest):
            if not (wrapper_root / relative_path).is_file():
                raise FileNotFoundError(wrapper_root / relative_path)
        patch_names = [
            line.strip()
            for line in (wrapper_root / self.patch_series).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(patch_names) != 1:
            raise ValueError(f"{self.name}: exactly one frozen patch is required")
        patch_path = wrapper_root / self.patch_series.parent / patch_names[0]
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        if digest != self.patch_sha256:
            raise ValueError(f"{self.name}: patch SHA-256 mismatch")


def _profile(name: str, data: dict[str, Any]) -> Profile:
    return Profile(
        name=name,
        submodule_path=Path(data["submodule_path"]),
        tag=str(data["tag"]),
        commit=str(data["commit"]),
        license=str(data["license"]),
        commercial_distribution=bool(data["commercial_distribution"]),
        layout=str(data["layout"]),
        patch_series=Path(data["patch_series"]),
        patch_sha256=str(data["patch_sha256"]),
        test_manifest=Path(data["test_manifest"]),
    )


def load_profiles(wrapper_root: Path) -> dict[str, Profile]:
    """Load all profiles from ``versions.toml`` and validate their contracts."""
    wrapper_root = wrapper_root.resolve()
    data = tomllib.loads((wrapper_root / "versions.toml").read_text(encoding="utf-8"))
    profiles = {
        name: _profile(name, profile_data)
        for name, profile_data in data["profiles"].items()
    }
    for profile in profiles.values():
        profile.validate(wrapper_root)
    paths = [profile.submodule_path for profile in profiles.values()]
    if len(paths) != len(set(paths)):
        raise ValueError("profiles must use distinct submodule paths")
    return profiles
