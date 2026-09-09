# Tool-route reactivation is observed, 2026-09-08

This retries only the failed native-C tool-actuator configuration, with
bounded host-side logs in the existing HAL: Deactivate, Reactivate and its
returned index, plus the first 32 launches and context publications per
process. It adds no per-thread device instrumentation. Logging may change
timing; this is failure localization, not a performance comparison.

The first build fails because the new log strings add `%` before
`FMT_64U` / `FMT_64D`, which already include it. Root removes those duplicate
characters in the isolated source and matching saved patch; rebuild/install
returns zero. Both logs are retained. The full saved patch reverse-applies
in a dry run against the tested isolated source. The tool binary and
first-mismatch worker remain unchanged from the preceding attempt.

Under both shared experiment locks, from the repository root:

```sh
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-replay-diagnostic-20260908.d9Gydm/cells
```

Runner returns 1. BE1 and BE3 return zero; BE2 and BE4 return 2. All records
are retained; this is not a completed six-process cell.

- BE2 observes Deactivate and Reactivate on all four queues. Stream 0
  returns preempt index 1, followed by command 1 / launch type 2 / level 2
  and publication of its context slot. Its first missing output remains
  index 87040 (stream 0 / task 1 / block 0 / thread 0).
- BE4 also observes all four reactivations. Stream 1 returns index 1 and
  logs a type-2 launch. Its first missing output is index 61440, which is
  stream 0 / task 0 / block 240 / thread 0, within the first task.
- These observations contradict the conjecture that restoration was never
  called or that the queue necessarily fell below Level 2. They do not
  yet identify the offending device restore operation or argument state.

The tool's final launch counter still prints 200, but its increment is
non-atomic across launching threads. More importantly, the new HAL logs
directly show resume launches. The old count must not be used to infer
that no replay occurred. The earlier raw counts remain unchanged.

No completed baseline or BPF cell was repeated. After normal cleanup the
GPU reports 0% / 1 MiB / P8. No reset, driver reload or reboot was needed.
