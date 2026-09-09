#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
xg_pair=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns
xg_tool=workloads/xsched/level2-build/.output/xsched_guard_tool.so
cp "$xg_tool" "$xg_pair/previous-xsched_guard_tool.so"
trap 'cp "$xg_pair/previous-xsched_guard_tool.so" "$xg_tool"' EXIT
cp workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5/xsched_guard_tool.so "$xg_tool"
xg_args=(run --no-initial --tasks 50 --reps 9511106 --blocks 340 --threads 256
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/blocks-done-20260909.ykPiLQ/priority_workload
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy)
date -Is
# Block 0 native already exists in bOiYh5; only its two missing arms run here.
python3 -B -u workloads/xsched/level2/run_tool_pair.py "${xg_args[@]}" \
  --repetitions 1 --configs bpf_port,baseline --output "$xg_pair/block0-missing"
# Four additional rotated blocks; map their local indices 1..4 to global 1..4.
python3 -B -u workloads/xsched/level2/run_tool_pair.py "${xg_args[@]}" \
  --repetitions 4 --configs bpf_port,baseline,native_port --output "$xg_pair/remaining"
date -Is
