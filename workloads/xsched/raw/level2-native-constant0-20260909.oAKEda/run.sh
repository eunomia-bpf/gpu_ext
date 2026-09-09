#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date --iso-8601=seconds
export UV_CACHE_DIR
UV_CACHE_DIR=$(mktemp -d /tmp/gpubpf-xsched-uv.XXXXXX)
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL XG_NATIVE_META_EXTEND=1 XG_NATIVE_META_KPARAM=1 \
  uv run --no-project --no-sync --python /usr/bin/python3 python -B -u \
  workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-constant0-20260909.oAKEda/cells
date --iso-8601=seconds
