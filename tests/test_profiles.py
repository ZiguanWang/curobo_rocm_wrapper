# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Test immutable version and licensing profile contracts."""

from __future__ import annotations

from pathlib import Path

from curobo_rocm_wrapper.profiles import load_profiles


def test_profiles_pin_both_requested_tags() -> None:
    """Both requested upstream releases are pinned by full commit SHA."""
    root = Path(__file__).parents[1]
    profiles = load_profiles(root)

    assert profiles["v0_7_8"].tag == "v0.7.8"
    assert profiles["v0_8_0"].tag == "v0.8.0"
    assert profiles["v0_7_8"].commit == "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
    assert profiles["v0_8_0"].commit == "4ea77366ca48ee453e7df139e39fa6532af49f3b"
    assert profiles["v0_7_8"].submodule_path == Path("upstream/curobo-v0.7.8")
    assert profiles["v0_8_0"].submodule_path == Path("upstream/curobo-v0.8.0")


def test_legacy_profile_cannot_claim_commercial_distribution() -> None:
    """The NVIDIA-licensed v0.7.8 profile remains explicitly non-commercial."""
    profiles = load_profiles(Path(__file__).parents[1])

    assert not profiles["v0_7_8"].commercial_distribution
    assert profiles["v0_8_0"].commercial_distribution


def test_profile_patches_match_frozen_checksums() -> None:
    """Loading profiles verifies the committed patch SHA-256 values."""
    profiles = load_profiles(Path(__file__).parents[1])

    assert len(profiles["v0_7_8"].patch_sha256) == 64
    assert len(profiles["v0_8_0"].patch_sha256) == 64
