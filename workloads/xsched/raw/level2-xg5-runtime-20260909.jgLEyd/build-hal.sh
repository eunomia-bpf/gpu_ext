#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date -Is
cmake --build workloads/xsched/level2-build/.output/hal-tool-build-20260908 --parallel 2
cmake --install workloads/xsched/level2-build/.output/hal-tool-build-20260908
date -Is
