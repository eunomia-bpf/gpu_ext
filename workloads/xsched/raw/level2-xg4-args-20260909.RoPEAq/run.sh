#!/usr/bin/env bash
# Run under both shared experiment leases; no driver changes.
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
cmake --build workloads/xsched/level2-build/.output/hal-tool-build-20260908 --parallel 2
cmake --install workloads/xsched/level2-build/.output/hal-tool-build-20260908
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/blocks-done-20260909.ykPiLQ/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-xg4-args-20260909.RoPEAq/cells
