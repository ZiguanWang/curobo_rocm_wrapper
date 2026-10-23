# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare isolated HIPified sources for PyTorch extension builds."""

from __future__ import annotations

import shutil
from pathlib import Path

from torch.utils.hipify import hipify_python


def prepare_hip_sources(
    source_root: Path,
    output_root: Path,
    extension_sources: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """HIPify extension sources into a generated build-only directory.

    Args:
        source_root: Root containing the shared cuRoboLib backends and kernel headers.
        output_root: Generated directory dedicated to HIP build artifacts.
        extension_sources: Relative sources grouped by extension module name.

    Returns:
        HIP-specific source paths and generated kernel include directories.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)

    device_sources = [
        output_root / relative_path
        for sources in extension_sources.values()
        for relative_path in sources
        if Path(relative_path).suffix == ".cu"
    ]
    hipify_result = hipify_python.hipify(
        project_directory=str(source_root),
        output_directory=str(output_root),
        header_include_dirs=["kernels"],
        includes=["*"],
        extra_files=[str(path) for path in device_sources],
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    prepared_sources: dict[str, list[str]] = {}
    for module_name, sources in extension_sources.items():
        prepared_sources[module_name] = []
        for relative_path in sources:
            generated_path = output_root / relative_path
            if generated_path.suffix == ".cu":
                result = hipify_result[str(generated_path)]
                if result.hipified_path is None:
                    raise RuntimeError(f"HIPify did not generate a source for {relative_path}")
                generated_path = Path(result.hipified_path)
            prepared_sources[module_name].append(str(generated_path))

    kernels_root = output_root / "kernels"
    include_dirs = [str(kernels_root)] + [
        str(path) for path in kernels_root.rglob("*") if path.is_dir()
    ]
    return prepared_sources, include_dirs
