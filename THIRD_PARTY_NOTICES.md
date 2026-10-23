# Third-party notices

This wrapper does not contain copies of cuRobo, Warp, or their assets. It pins their source
repositories as Git submodules and keeps the ROCm adaptations as separate patches.

## cuRobo v0.8.0

- Upstream: <https://github.com/NVlabs/curobo>
- Commit: `4ea77366ca48ee453e7df139e39fa6532af49f3b`
- License: Apache License 2.0
- Redistribution must preserve the upstream license, copyright, patent, trademark, and attribution
  notices. Modified upstream files must be identified as modified.

## cuRobo v0.7.8

- Upstream: <https://github.com/NVlabs/curobo>
- Commit: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- License: NVIDIA License included in that tag
- License copy for this source profile: `licenses/NVIDIA-cuRobo-v0.7.8-LICENSE.txt`
- Use limitation: cuRobo v0.7.8 and derivative works may only be used non-commercially for
  research or evaluation, unless separate permission is obtained from NVIDIA.
- Redistribution must use the upstream NVIDIA License, include a complete copy of it, and retain
  upstream copyright, patent, trademark, and attribution notices without modification.

The wrapper's Apache-2.0 license does not replace either upstream license. The v0.7.8 adaptation
profile must not be included in a commercial binary release without separate legal authorization.
Robot and example assets remain governed by each upstream tag's `LICENSE_ASSETS` file.

## AMD-Ecosystem Warp source build

- Upstream: <https://github.com/AMD-Ecosystem/warp>
- Current tested commit: `74014999a31471228e0aaa213bbeb2e5daba95f0`
- License: Apache License 2.0 in that repository
- Source is pinned by the `upstream/warp-rocm` gitlink. ROCm changes are in `profiles/warp/`;
  compiled Warp wheels are not included.
