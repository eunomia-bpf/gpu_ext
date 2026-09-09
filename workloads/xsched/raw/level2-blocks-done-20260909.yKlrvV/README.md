# Level-2 missing-output observation: block completion counts

2026-09-09 PDT, RTX 5090 / 575.57.08. This is a failed-path diagnostic,
not a new performance comparison. The previous native-port tool candidate
is unchanged. Root adds only stderr reporting on the existing sink failure:
the first five tasks' existing `blocks_done` counters for each stream.
There is no new device instrumentation or changed acceptance condition.

The one six-process cell exits 1; all four BE workers exit 2, while both LC
workers exit 0. Every BE first reports a missing value at index 87040
(stream 0, command 2). Command IDs are one-based; task indices are zero-based.

| Worker | Stream 0 command 1 blocks | Stream 1 command 2 blocks | Stream 1 command 3 blocks |
| --- | ---: | ---: | ---: |
| BE1 | 340 | 3 | 0 |
| BE2 | 680 | 33 | 0 |
| BE3 | 384 | 0 | 0 |
| BE4 | 680 | 0 | 0 |

Expected completion is 340 blocks per task. In every BE, stream 0 commands
2–5 have zero completed blocks. Counts greater than 340 for command 1 also
show that some task bodies execute more than once. The counter increments
after the output store, system fence, and block synchronization in the
original workload. These observations distinguish missing execution from
an otherwise complete task whose numerical recurrence is wrong, but do not
by themselves identify the actuator defect.

Importantly, this run differs from the earlier in-flight-window hypothesis:
BE1 stream 1 command 2 is first submitted **after** Reactivate returns
`preempt_idx=0`. Thus merely replaying commands retained at suspend cannot
explain or repair all the newly observed missing work. Stream 0 resumes
command 1, yet subsequent fresh guardian commands also lose output. The
per-command argument delivery and guardian/return path remain suspects;
neither cause is established by this observation. Do not infer an improvement
from a changed missing index or call this a completed Level-2 reproduction.

Both shared leases cover the build and, separately, the cell. Build exits 0.
The workload is built into the new `blocks-done-20260909.ykPiLQ` directory;
older binaries and all previous results remain unchanged. No driver reload
or recovery was needed; post-run GPU is idle at 0%, 1 MiB.

Commands, from repository root:

```sh
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_120,code=sm_120 \
  -I/usr/local/cuda-12.9/targets/x86_64-linux/include \
  workloads/xsched/priority_workload.cu \
  -L/usr/local/cuda-12.9/targets/x86_64-linux/lib \
  -Xlinker=-rpath -Xlinker=/usr/local/cuda-12.9/targets/x86_64-linux/lib \
  -lcupti -o workloads/xsched/level2-build/.output/blocks-done-20260909.ykPiLQ/priority_workload
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/blocks-done-20260909.ykPiLQ/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-blocks-done-20260909.yKlrvV/cells
```

The local tool session completed its requested diagnostic handoff naturally.
While root gathered this observation, its free execution slot was assigned
to the already queued full-record grouped-SoA implementation. LMCache and
native-blob XSched development remain live: three active local sessions,
not four. No silent session was cancelled and no manuscript was edited.
