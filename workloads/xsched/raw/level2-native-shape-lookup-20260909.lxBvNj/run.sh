#!/usr/bin/env bash
# Root execution of the local model's frozen lookup candidate.
# Invoke under both shared leases, after the LMCache performance campaign.
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build --target install -j2
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL XG_NATIVE_META_EXTEND=1 XG_NATIVE_META_KPARAM=1 \
  python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-native-shape-lookup-20260909.lxBvNj/cells
