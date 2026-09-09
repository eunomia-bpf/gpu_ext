#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
xg_raw=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-callback-delivery-20260909.Y7qcwZ
xg_cuda=/usr/local/cuda-12.9
xg_nvbit=workloads/llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64/core
date -Is
"$xg_cuda/bin/nvcc" -c -arch=sm_120 -O3 -std=c++14 -Xcompiler -fPIC -I"$xg_nvbit" \
    "$xg_raw/source/xsched_guard_tool_callback.cu" -o "$xg_raw/callback.o"
"$xg_cuda/bin/nvcc" -arch=sm_120 -O3 "$xg_raw/callback.o" \
    workloads/xsched/raw/level2-xg5-build-20260909.yiBj3W/xg_guardian_carrier.o \
    -L"$xg_nvbit" -lnvbit -L"$xg_cuda/targets/x86_64-linux/lib" -lcuda -lcudart_static \
    -shared -o "$xg_raw/xsched_guard_tool.so"
date -Is
