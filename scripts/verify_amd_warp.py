#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Check that a source-built Warp package runs a kernel on a ROCm PyTorch GPU."""

from __future__ import annotations

import os

import torch
import warp as wp


@wp.kernel
def _increment(values: wp.array(dtype=wp.float32)):
    """Increment one GPU array element."""
    index = wp.tid()
    values[index] = values[index] + 1.0


def main() -> None:
    """Verify device mapping, tensor sharing, and an actual HIP kernel launch."""
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("A ROCm PyTorch build and visible AMD GPU are required")

    wp.init()
    torch_device = torch.device("cuda:0")
    try:
        warp_device = wp.device_from_torch(torch_device)
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError("Warp has no HIP device corresponding to the PyTorch GPU") from exc
    arch = torch.cuda.get_device_properties(0).gcnArchName
    requested_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    if not warp_device.is_hip or warp_device.arch_str != arch:
        raise RuntimeError(f"Warp mapped {torch_device} to {warp_device}, not HIP {arch}")
    if requested_arch and requested_arch != arch:
        raise RuntimeError(f"PYTORCH_ROCM_ARCH={requested_arch}, but GPU is {arch}")

    values = torch.arange(4, device=torch_device, dtype=torch.float32)
    wp.launch(_increment, dim=values.numel(), inputs=[wp.from_torch(values)], device=warp_device)
    torch.cuda.synchronize(torch_device)
    expected = torch.arange(1, 5, device=torch_device, dtype=torch.float32)
    torch.testing.assert_close(values, expected)
    print(f"Warp {wp.__version__}: {warp_device} ({arch}); HIP kernel and Torch sharing passed")


if __name__ == "__main__":
    main()
