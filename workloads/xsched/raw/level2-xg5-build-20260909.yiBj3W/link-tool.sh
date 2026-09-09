#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
xg_raw=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-xg5-build-20260909.yiBj3W
xg_cuda=/usr/local/cuda-12.9
xg_nvbit=workloads/llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64/core
date -Is
"$xg_cuda/bin/fatbinary" -64 --embedded-fatbin "$xg_raw/xg_guardian.fatbin.c" --image3=kind=elf,sm=120,file="$xg_raw/xg_guardian.cubin"
g++ -x c++ -fPIC -O2 -I"$xg_cuda/targets/x86_64-linux/include" -c "$xg_raw/xg_guardian.fatbin.c" -o "$xg_raw/xg_guardian_carrier.o"
"$xg_cuda/bin/nvcc" -arch=sm_120 -O3 workloads/xsched/level2-build/.output/xsched_guard_tool.o "$xg_raw/xg_guardian_carrier.o" -L"$xg_nvbit" -lnvbit -L"$xg_cuda/targets/x86_64-linux/lib" -lcuda -lcudart_static -shared -o "$xg_raw/xsched_guard_tool.so"
date -Is

