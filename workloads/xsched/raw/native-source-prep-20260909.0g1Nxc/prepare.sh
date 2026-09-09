#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date --iso-8601=seconds
bash workloads/xsched/level2/native/prepare_native_source.sh \
  workloads/xsched/deps/xsched f49289f0220931df78de948ed841ecbaf960a919 \
  workloads/xsched/level2-build/.output/native-source-prep-20260909.0Oy86H \
  workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/source
date --iso-8601=seconds
