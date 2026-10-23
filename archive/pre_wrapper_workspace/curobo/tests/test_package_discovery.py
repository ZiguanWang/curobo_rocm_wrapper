# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python package discovery boundary."""

from pathlib import Path

from setuptools import find_namespace_packages


def test_package_discovery_excludes_build_artifacts() -> None:
    """Only the cuRobo namespace is included in wheels."""
    repository_root = Path(__file__).parents[2]
    packages = find_namespace_packages(
        where=repository_root,
        include=["curobo*"],
        exclude=["tests"],
    )

    assert packages
    assert all(package == "curobo" or package.startswith("curobo.") for package in packages)
    assert not any(package == "build" or package.startswith("build.") for package in packages)
