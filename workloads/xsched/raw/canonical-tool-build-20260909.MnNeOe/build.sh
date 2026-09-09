#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
xg_build_record=workloads/xsched/raw/canonical-tool-build-20260909.MnNeOe
xg_carrier=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-xg5-build-20260909.yiBj3W/xg_guardian_carrier.o
date -Is
make -C workloads/xsched/level2-build guard-tool \
  SRC=/home/yunwei37/workspace/gpu/gpu_ext/$xg_build_record/source \
  BUILD=/home/yunwei37/workspace/gpu/gpu_ext/$xg_build_record/build \
  CARRIER_O="$xg_carrier" -o "$xg_carrier"
date -Is
