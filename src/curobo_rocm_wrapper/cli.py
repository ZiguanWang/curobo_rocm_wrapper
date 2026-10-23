# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Manage a pinned cuRobo submodule and apply ROCm adaptation profiles."""

from __future__ import annotations

import argparse
import os
import subprocess
import tomllib
from pathlib import Path

from curobo_rocm_wrapper.profiles import Profile, load_profiles


def _run(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def _wrapper_root() -> Path:
    configured = os.environ.get("CUROBO_ROCM_WRAPPER_ROOT")
    if configured:
        return Path(configured).resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "versions.toml").is_file():
            return candidate.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "versions.toml").is_file():
        return source_root
    raise RuntimeError(
        "cannot locate versions.toml; run inside the wrapper repository, pass --root, "
        "or set CUROBO_ROCM_WRAPPER_ROOT"
    )


def _submodule_root(wrapper_root: Path, profile: Profile) -> Path:
    return wrapper_root / profile.submodule_path


def _require_clean(repository: Path) -> None:
    status = _run(["git", "status", "--porcelain"], repository).stdout.strip()
    if status:
        raise RuntimeError(f"upstream submodule has uncommitted changes:\n{status}")


def _verify_revision(repository: Path, profile: Profile) -> None:
    revision = _run(["git", "rev-parse", "HEAD"], repository).stdout.strip()
    if revision != profile.commit:
        raise RuntimeError(
            f"{profile.name} requires {profile.commit}, but submodule is at {revision}"
        )


def _patches(wrapper_root: Path, profile: Profile) -> list[Path]:
    series_path = wrapper_root / profile.patch_series
    names = [
        line.strip()
        for line in series_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return [series_path.parent / name for name in names]


def apply_profile(wrapper_root: Path, profile: Profile) -> None:
    """Apply one profile to a clean, correctly pinned upstream submodule."""
    repository = _submodule_root(wrapper_root, profile)
    _require_clean(repository)
    _verify_revision(repository, profile)
    for patch in _patches(wrapper_root, profile):
        _run(["git", "apply", "--check", str(patch)], repository)
        _run(["git", "apply", str(patch)], repository)


def test_command(wrapper_root: Path, profile: Profile) -> list[str]:
    """Return the seven-test command for an applied profile."""
    manifest = tomllib.loads(
        (wrapper_root / profile.test_manifest).read_text(encoding="utf-8")
    )
    tests = [str(path) for path in manifest["migration"]["tests"]]
    return [
        "python",
        "-m",
        "pytest",
        "-n0",
        "-q",
        "--override-ini",
        "python_files=test_*.py",
        *tests,
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_wrapper_root())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "apply", "test", "build"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--profile", choices=("v0_7_8", "v0_8_0"), required=True)
    return parser


def main() -> None:
    """Run the wrapper command-line interface."""
    args = _parser().parse_args()
    wrapper_root = args.root.resolve()
    profile = load_profiles(wrapper_root)[args.profile]
    repository = _submodule_root(wrapper_root, profile)
    if args.command == "status":
        _verify_revision(repository, profile)
        print(f"{profile.name}: {profile.commit} ({profile.license})")
    elif args.command == "apply":
        apply_profile(wrapper_root, profile)
    elif args.command == "test":
        environment = os.environ.copy()
        environment.update(
            {
                "CUROBO_USE_HIP": "1",
                "CUROBO_KERNEL_BACKEND": "hip",
                "CUROBO_REQUIRE_ROCM": "1",
            }
        )
        subprocess.run(
            test_command(wrapper_root, profile),
            cwd=repository,
            env=environment,
            check=True,
        )
    elif args.command == "build":
        environment = os.environ.copy()
        environment.update({"CUROBO_USE_HIP": "1", "CUROBO_KERNEL_BACKEND": "hip"})
        subprocess.run(
            ["python", "-m", "pip", "wheel", ".[hip]", "--no-build-isolation", "--no-deps"],
            cwd=repository,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
