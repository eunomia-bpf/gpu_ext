#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date --iso-8601=seconds
g++ -O2 -std=c++17 workloads/xsched/raw/level2-native-retabs-20260909.oJDCbU/ldc_patcher.cpp \
  -o workloads/xsched/level2-build/.output/native-retabs-20260909.20tGK7/xg_ldc_patcher
workloads/xsched/level2-build/.output/native-retabs-20260909.20tGK7/xg_ldc_patcher \
  workloads/xsched/level2-build/.output/native-padding-20260908.s03ZL6/native/check_preempt_port.cubin \
  workloads/xsched/level2-build/.output/native-padding-20260908.s03ZL6/native/restore_exec_port.cubin \
  /usr/local/cuda-12.9/bin/nvdisasm \
  workloads/xsched/level2-build/.output/native-retabs-20260909.20tGK7/xg_sm120_guardian_arrays.h
date --iso-8601=seconds
