# First mismatched output in the tool-route retry, 2026-09-08

This retries only the previously failed native-C tool-actuator configuration
with the two-line first-mismatch diagnostic in worker `16a9aece`. No
successful baseline or BPF measurement was repeated. This is execution
failure diagnosis, not a new performance result or added acceptance rule.

The existing tool binary and `hal-tool-install-20260908` are unchanged
from `../level2-cached-entry-20260908.Mtt9XM/`; that report retains the
concurrent source/build caveat. The current uncommitted tool edits were
not compiled for this retry. The newly compiled worker is
`level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload`.

From the repository root, under both experiment locks:

```sh
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-first-mismatch-20260908.xlUdZS/cells
```

Runner returns 1; BE1 returns 2 before its service result. The actual error is:

```text
sink mismatch index=87040 value=0x0p+0 expected=0x1.45ef1cp+4
XG done functions=1 launches=200
```

There are `340 * 256 = 87040` output elements per task. Therefore the first
failure is stream 0 / task 1 / block 0 / thread 0; all task-0 elements were
checked before it. Zero at the next task boundary points toward skipped or
unreplayed work, not a small arithmetic discrepancy. The 200 guarded-launch
count matches the submitted commands and shows no additional counted replay
launches in this process. It does not alone identify the faulty replay step.

All workers and xserver are cleaned up; raw per-process results and the
runner exception are retained. Immediately afterward NVML reported 100%
utilization / 1 MiB / P0 without compute processes. A single uninstrumented
one-CTA, 32-thread, one-iteration invocation of the same worker returned 0
and all 32 outputs (`uninstrumented-after-failure.log`). This is a
post-failure device-use diagnostic, not a repeated performance baseline.
No driver reload, reset, reboot, or local-model termination was performed.
