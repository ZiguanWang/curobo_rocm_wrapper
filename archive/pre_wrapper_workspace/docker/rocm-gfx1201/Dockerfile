# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

ARG ROCM_VERSION=7.14.1
ARG PYTORCH_VERSION=2.10.0

FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full AS rocm-devel

RUN mkdir -p /rocm-headers \
    && cp -aL /opt/rocm/include/thrust /rocm-headers/thrust \
    && cp -aL /opt/rocm/include/rocprim /rocm-headers/rocprim \
    && cp -aL /opt/rocm/include/hipcub /rocm-headers/hipcub

FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu24.04_py3.12_pytorch_release_${PYTORCH_VERSION}

ARG ROCM_VERSION
ARG PYTORCH_VERSION

# The PyTorch runtime image uses modular ROCm Python packages but omits the
# header-only development packages needed by torch C++ extensions. Keep the
# runtime image small by copying only the three required header trees from the
# matching ROCm development image.
COPY --from=rocm-devel /rocm-headers/thrust /opt/venv/include/thrust
COPY --from=rocm-devel /rocm-headers/rocprim /opt/venv/include/rocprim
COPY --from=rocm-devel /rocm-headers/hipcub /opt/venv/include/hipcub

RUN ln -s \
      /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7 \
      /opt/venv/lib/libamdhip64.so

ENV CUROBO_USE_HIP=1 \
    CUROBO_KERNEL_BACKEND=hip \
    PYTORCH_ROCM_ARCH=gfx1201 \
    LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

WORKDIR /workspace
COPY . /workspace

RUN python -m pip wheel . \
      --no-build-isolation \
      --no-deps \
      --wheel-dir /wheelhouse \
    && python -m pip install \
      /wheelhouse/*.whl

WORKDIR /tmp

CMD ["python", "-m", "curobo._src.curobolib.backends.hip_backend.runtime_preflight"]
