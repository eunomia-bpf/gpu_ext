# Ordinary-launch argument candidate: failure remains

2026-09-08, RTX 5090 / NVIDIA 575.57.08 / CUDA 12.9. This retries only the
failed native-C tool-actuator configuration, not a completed baseline or BPF
cell. The root applies one bounded source change: ordinary kernel launches
use `nvbit_set_at_launch(ctx, f, published_ctx)` without the stream argument.
This follows the ordinary-launch examples in the SDK selected by
`level2-build/Makefile`; graph-node examples use stream and launch handle.
The declaration alone did not prove that the prior call caused the failure.

`make -C workloads/xsched/level2-build -j2 guard-tool` succeeds. Its only
new warning is the now-unused `launch_stream` helper. The existing carrier,
HAL with bounded XG2 logs, and first-mismatch worker are unchanged. The tool
source has only the one-line call change relative to main `1e3b2ff8`.
Build and run both hold `/tmp/gpubpf-revision-gpu0.lock` and
`/tmp/gpubpf-revision-struct-ops.lock`.

```sh
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-ordinary-launch-value-20260908.Cp8bpS/cells
```

Runner exits 1. BE1 and BE2 exit zero; BE3 and BE4 exit 2. Both failed
workers report missing output at index 87040, with zero instead of
`0x1.45ef1cp+4`. BE3 stream 0 resumes command 1 (type 2, context slot 1);
BE4 stream 2 resumes command 1 (type 2, slot 3). Reactivation and resume
launches are still observed. The ordinary-call change alone therefore does
not close the missing-output bug. Different worker failures between runs
are not evidence of a reliability improvement.

All raw worker/server records and build/run logs are retained. Diagnostic
logging and unsuccessful completion mean this is not a performance result.
No baseline or BPF cell was repeated, no driver reload/reset/reboot occurred,
and normal cleanup left the GPU at 0% utilization / 1 MiB / P8 with both
locks released. Local Qwen receives these results and continues the existing
repair session; no artificial timeout or additional model session is used.
