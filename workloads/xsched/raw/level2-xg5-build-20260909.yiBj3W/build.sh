#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
xg_raw=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-xg5-build-20260909.yiBj3W
mkdir -p "$xg_raw/source/tramp"
cp workloads/xsched/level2/tramp/xsched_guard_tramp.cu "$xg_raw/source/tramp/"
cp workloads/xsched/level2/xsched_guardian_abi.h "$xg_raw/source/"
cp workloads/xsched/level2-build/.output/ptx/xsched_guardian.ptx "$xg_raw/xsched_guardian.ptx"
date -Is
/usr/local/cuda-12.9/bin/nvcc -ptx -rdc=true -arch=sm_120 -O3 -std=c++14 --keep-device-functions "$xg_raw/source/tramp/xsched_guard_tramp.cu" -o "$xg_raw/tramp.ptx"
bash workloads/xsched/level2/tool/merge_ptx.sh "$xg_raw/xsched_guardian.ptx" "$xg_raw/tramp.ptx" "$xg_raw/merged.ptx"
/usr/local/cuda-12.9/bin/ptxas -arch=sm_120 -astoolspatch "$xg_raw/merged.ptx" -o "$xg_raw/xg_guardian.cubin"
date -Is

