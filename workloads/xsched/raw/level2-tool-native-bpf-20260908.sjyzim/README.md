# First Level-2 tool-route campaign — partial, integration failure retained

Five rotating blocks compare plain CUDA, native host HPF/device C decisions,
and host/device BPF decisions. Both policy ports use the same built NVBit
actuator and isolated Level-2 HAL. This is not the unfinished upstream
cuXtra/raw-SASS artifact path. Runner `c5f6e4ea`, worker source `e9529745`.

The existing full-workload shape is reused: 2 LC + 4 BE processes, 4 streams
each, 50 kernels per stream, 9511106 recurrence repetitions, 340 blocks and
256 threads. Metrics are GPU service span, LC/BE host elapsed and BE kernel
rate. These are not the old Level-1 arrival-queue-p99 metric. No new clock
calibration, separate initial batch or completed historical cell is run.

Invocation from repository root, under both shared leases:

```sh
python3 -u workloads/xsched/level2/run_tool_pair.py run --no-initial --repetitions 5 --tasks 50 --reps 9511106 --blocks 340 --threads 256 --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-output-20260908.66ppYl/priority_workload --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy --output /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-tool-native-bpf-20260908.sjyzim/cells
```

`run.log` retains runner output, `cells/` per-process records and results.
Extra tool launches caused by replay are diagnostics, not timing rejection
criteria. The runner exited 1 after one completed baseline and one failed
native-port start. BPF and the remaining blocks have not run. Do not treat
this prefix as all fifteen cells or a completed Level-2 reproduction.

## Observed prefix and next repair

Block-1 baseline completed 400 LC and 800 BE kernels. LC service p99 is
2049175.392 us; LC host elapsed is 77.9756558 s; BE throughput is
10.189760755 kernels/s. There is no policy-arm performance comparison yet,
and this service span is not the old Level-1 queue-p99 metric.

The native-port process reached the first launch, then HAL ToolPublish
could not resolve `xg_host_publish` through RTLD_DEFAULT. The tool was
preloaded and emitted its loaded line; nm/readelf show the function as an
exported GLOBAL DEFAULT symbol. Two scheduler-initialization messages and
duplicate per-PID IPC queues, together with NVBit's dlmopen import, suggest
a linker-namespace integration issue. That is a diagnosis to test, not a
confirmed root cause. The raw failure and all process outputs are retained.

All owned workers and xserver terminated; GPU returned to idle / 1 MiB and
both leases released. No driver module was changed. Local Qwen now owns
the publication-path repair. Resume failed/unstarted cells after repair;
do not rerun this completed baseline merely to create a new directory.
