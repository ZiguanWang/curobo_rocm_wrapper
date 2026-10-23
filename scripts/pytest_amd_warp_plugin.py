# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Initialize source-built AMD Warp before collecting upstream cuRobo tests."""

import warp as wp


def pytest_configure(config: object) -> None:
    """Make direct Warp/Torch calls in upstream tests use an active runtime."""
    wp.init()
