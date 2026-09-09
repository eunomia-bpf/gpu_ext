# Context-scoped replay: callbacks restored, output still incomplete

2026-09-08, RTX 5090 / driver 575.57.08 / CUDA 12.9. One retry of the
failed `native_port` configuration ran from 22:38:27 to 22:38:58 PDT.
No completed baseline, control, or performance batch was repeated.

The local Qwen implementation pushes the queue's CUDA context around
`CudaQueueLv2::Reactivate`, then pops it after replay, including when the
resume index is zero. Ordinary launch workers already establish their
context in `OnXQueueCreate`; the scheduler-thread replay did not do so.
`reactivate-context.patch` captures the executed change relative to the
prior tool HAL. The tool itself remains the direct-prepare version from
`c6153b3b`; the runner includes footer compatibility repair `b2957657`.

The isolated HAL build/install exits zero. `build.log` records recompiling
`hal/src/level2/cuda_queue.cpp`; the usual existing-shim-symlink messages
are nonfatal. Root checked reverse application of the saved context patch
against the source. The real run then exits one:

| Worker | Exit | Prepare / filtered callback count | Result |
|---|---:|---:|---|
| BE1 | 2 | 201 / 201 | First missing output index 174080 |
| BE2 | 2 | 201 / 201 | First missing output index 87040 |
| BE3 | -2 | No final footer | Runner cleanup after another worker's failure |
| BE4 | 2 | 201 / 201 | First missing output index 87040 |
| LC1 | 0 | 200 / 200 | Completed |
| LC2 | 0 | 200 / 200 | Completed |

All three complete BE footers report zero disarms. Each of those workers
has 201 captured successful CUDA submissions and 201 filtered callbacks.
This closes the observed callback-count discrepancy from the previous
201-prepare/200-callback attempt. It does **not** fix the whole resume path:
missing output is still zero versus expected `0x1.45ef1cp+4`. BE3's cleanup
signal is not a separate demonstrated output failure. There is no complete
Level-2 performance sample or BPF/native comparison from this retry.

The context change is therefore retained, and the next investigation moves
to guardian/resume execution and per-command state. Another unchanged
callback-preparation retry is not justified by these results. The original
workload's output failure is retained; no check was weakened to obtain a
performance number.

## Executed commands

Both commands held the existing GPU and struct-ops locks:

```sh
cmake --build workloads/xsched/level2-build/.output/hal-tool-build-20260908 --target install -j2
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-context-replay-20260908.CgSRR6/cells
```

`cells/protocol.json`, six worker JSON files, `xserver.json`, `failure.json`,
and `runner.log` retain the configuration, process commands, environment,
stdout/stderr, and exits. No original raw record is overwritten.

## Post-run cleanup

After all workers exited, the GPU again reported 100% utilization, 1 MiB,
P0, and about 109 W, with no listed compute applications. Read-only device
holder inspection found only `nvidia-persistenced`. Under both locks, root
restarted only that service at 22:40 PDT. `persistence-recovery.log` records
the transition to 2% / 52.84 W; the subsequent query reached 0% / P8 /
12.17 W. Both GDM and persistence were active.

No module reload, explicit GPU reset, reboot, or OpenCode termination was
performed. Whether the post-run busy state reflects residual execution or
driver accounting remains unproven. The two storage policy loaders and all
earlier completed experiments were left intact; both locks are released.
