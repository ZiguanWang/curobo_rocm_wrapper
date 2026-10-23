# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the real XPolicyLab evaluation readiness gate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).parents[2] / "benchmark" / "robotwin_xpolicy_readiness.py"
    spec = importlib.util.spec_from_file_location("robotwin_xpolicy_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path: Path, module: ModuleType, *, policy: str = "real_policy") -> Path:
    root = tmp_path / "RoboTwin"
    xpolicy = root / "XPolicyLab"
    policy_root = xpolicy / "policy" / policy
    policy_root.mkdir(parents=True)
    (policy_root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
    (policy_root / "deploy.yml").write_text(f"policy_name: {policy}\n", encoding="utf-8")
    (policy_root / "eval.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    for repository in (xpolicy, root):
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repository, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    module.ROBOTWIN_REVISION = module._revision(root)
    module.XPOLICY_REVISION = module._revision(xpolicy)
    return root


def test_readiness_accepts_frozen_real_policy_assets(tmp_path: Path) -> None:
    """Accept a real importable policy with frozen revisions and checkpoint evidence."""
    module = _load_module()
    root = _checkout(tmp_path, module)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model-weights")

    result = module.evaluate(root, "real_policy", checkpoint, Path("python"))

    assert result["ready_to_launch"]
    assert all(result["checks"].values())
    assert result["checkpoint_manifest"][0]["path"] == "model.safetensors"


def test_readiness_rejects_placeholder_policy(tmp_path: Path) -> None:
    """Do not count the zero-action demo adapter as a real policy evaluation."""
    module = _load_module()
    root = _checkout(tmp_path, module, policy="demo_policy")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"placeholder": True}), encoding="utf-8")

    result = module.evaluate(root, "demo_policy", checkpoint, Path(sys.executable))

    assert not result["ready_to_launch"]
    assert not result["checks"]["real_policy"]


def test_readiness_reports_missing_policy_runtime(tmp_path: Path) -> None:
    """Fail closed when a policy adapter cannot import its runtime dependency."""
    module = _load_module()
    root = _checkout(tmp_path, module)
    model = root / "XPolicyLab" / "policy" / "real_policy" / "model.py"
    model.write_text(
        "import dependency_that_does_not_exist\nclass Model: pass\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model-weights")

    result = module.evaluate(root, "real_policy", checkpoint, Path(sys.executable))

    assert not result["ready_to_launch"]
    assert not result["checks"]["policy_model_import"]
    assert "dependency_that_does_not_exist" in result["policy_model_import_diagnostic"]


def test_readiness_preserves_virtual_environment_python_symlink(tmp_path: Path) -> None:
    """Invoke the requested environment entrypoint instead of its base interpreter."""
    module = _load_module()
    root = _checkout(tmp_path, module)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model-weights")
    environment_python = tmp_path / "environment-python"
    environment_python.symlink_to(sys.executable)

    result = module.evaluate(root, "real_policy", checkpoint, environment_python)

    assert result["ready_to_launch"]
    assert result["policy_python"] == str(environment_python.absolute())
