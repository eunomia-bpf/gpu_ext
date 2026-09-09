#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date --iso-8601=seconds
cmake -S workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/source \
  -B workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build \
  -DXG_SM120_GENERATED_HEADER=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/native-retabs-20260909.20tGK7/xg_sm120_guardian_arrays.h
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build \
  --target install -j2
date --iso-8601=seconds
