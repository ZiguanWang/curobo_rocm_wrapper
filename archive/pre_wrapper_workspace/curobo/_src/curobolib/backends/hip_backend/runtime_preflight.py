# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the ROCm runtime before building or loading cuRobo HIP extensions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from curobo._src.util.logging import log_and_raise, log_info, setup_logger


@dataclass(frozen=True)
class HipPreflightResult:
    """ROCm runtime information relevant to cuRobo HIP extension compatibility."""

    #: Installed PyTorch version.
    torch_version: str
    #: ROCm version used to build PyTorch.
    rocm_version: str
    #: User-facing GPU name.
    device_name: str
    #: GPU ISA target such as ``gfx1100``.
    gfx_arch: str
    #: Architectures requested for extension compilation.
    build_arches: tuple[str, ...]
    #: Whether PyTorch reports a visible GPU.
    gpu_available: bool


def _parse_build_arches(value: str) -> tuple[str, ...]:
    """Parse the semicolon- or comma-separated PyTorch ROCm architecture setting."""
    normalized = value.replace(",", ";")
    return tuple(item.strip() for item in normalized.split(";") if item.strip())


def _get_gfx_arch(device_properties: Any) -> str:
    """Extract the base GFX architecture from PyTorch device properties."""
    architecture = str(getattr(device_properties, "gcnArchName", ""))
    return architecture.split(":", maxsplit=1)[0]


def inspect_runtime(
    torch_module: Optional[Any] = None,
    environ: Optional[dict[str, str]] = None,
) -> HipPreflightResult:
    """Inspect PyTorch and the visible GPU without loading cuRobo extensions.

    Args:
        torch_module: Optional torch-compatible module used by isolated tests.
        environ: Optional environment mapping used by isolated tests.

    Returns:
        Detected ROCm runtime and architecture metadata.
    """
    if torch_module is None:
        import torch as torch_module
    if environ is None:
        environ = os.environ

    rocm_version = getattr(getattr(torch_module, "version", None), "hip", None)
    gpu_available = bool(torch_module.cuda.is_available())
    device_name = ""
    gfx_arch = ""
    if gpu_available:
        device_name = str(torch_module.cuda.get_device_name(0))
        gfx_arch = _get_gfx_arch(torch_module.cuda.get_device_properties(0))

    return HipPreflightResult(
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        rocm_version="" if rocm_version is None else str(rocm_version),
        device_name=device_name,
        gfx_arch=gfx_arch,
        build_arches=_parse_build_arches(environ.get("PYTORCH_ROCM_ARCH", "")),
        gpu_available=gpu_available,
    )


def validate_runtime(result: HipPreflightResult, require_build_arch: bool = True) -> None:
    """Validate that runtime metadata is sufficient for a supported HIP build.

    Args:
        result: Runtime metadata returned by :func:`inspect_runtime`.
        require_build_arch: Require an explicit ``PYTORCH_ROCM_ARCH`` target.
    """
    if not result.rocm_version:
        log_and_raise(
            "The installed PyTorch build does not provide ROCm support "
            "(torch.version.hip is empty)",
            exception_type=RuntimeError,
        )
    if not result.gpu_available:
        log_and_raise(
            "ROCm PyTorch is installed, but no GPU is visible to torch.cuda",
            exception_type=RuntimeError,
        )
    if not result.gfx_arch:
        log_and_raise(
            "PyTorch did not report a gcnArchName for the active GPU",
            exception_type=RuntimeError,
        )
    if require_build_arch and not result.build_arches:
        log_and_raise(
            "Set PYTORCH_ROCM_ARCH to the target gfx architecture before building cuRobo",
            exception_type=RuntimeError,
        )
    if result.build_arches and result.gfx_arch not in result.build_arches:
        log_and_raise(
            f"Active GPU architecture {result.gfx_arch} is not present in "
            f"PYTORCH_ROCM_ARCH={';'.join(result.build_arches)}",
            exception_type=RuntimeError,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def result_to_dict(result: HipPreflightResult) -> dict[str, Any]:
    """Convert preflight metadata to a JSON-compatible mapping."""
    data = asdict(result)
    data["build_arches"] = list(result.build_arches)
    data["accelerator"] = "rocm"
    data["backend"] = "hip"
    return data


def main() -> int:
    """Run the ROCm preflight validation as a command-line entry point."""
    args = _parse_args()
    setup_logger("info")
    result = inspect_runtime()
    validate_runtime(result)
    log_info(
        "ROCm preflight passed: "
        f"torch={result.torch_version}, rocm={result.rocm_version}, "
        f"device={result.device_name}, gfx={result.gfx_arch}, "
        f"build_arches={';'.join(result.build_arches)}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result_to_dict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
