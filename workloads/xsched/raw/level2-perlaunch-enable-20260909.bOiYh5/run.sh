#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
xg_raw=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5
xg_tool=workloads/xsched/level2-build/.output/xsched_guard_tool.so
cp "$xg_tool" "$xg_raw/previous-xsched_guard_tool.so"
trap 'cp "$xg_raw/previous-xsched_guard_tool.so" "$xg_tool"' EXIT
cp workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5/xsched_guard_tool.so "$xg_tool"
date -Is
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
    --no-initial --repetitions 1 --configs native_port \
    --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
    --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/blocks-done-20260909.ykPiLQ/priority_workload \
    --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
    --output "$xg_raw/cells"
date -Is
