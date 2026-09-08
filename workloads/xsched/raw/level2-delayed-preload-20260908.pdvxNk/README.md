# Level-2 continuation after preload ordering repair

Runner `42049461` defers LD_PRELOAD until the real workload, after taskset.
The earlier loader trace showed the tool loaded only in the taskset image,
not in the subsequent workload image. Baseline launch behavior is unchanged.
The original block-1 baseline in `../level2-tool-native-bpf-20260908.sjyzim/`
is retained; this invocation runs only the failed native_port and unstarted
bpf_port cells, not that completed baseline or a new initial test batch.

The actual workload remains 2 LC / 4 BE processes, 4 streams each, 50 tasks
per stream, 9511106 repetitions, 340 blocks and 256 threads. Both policy
arms retain the same built NVBit actuator/HAL and their native/BPF decisions.
This is not the still-unfinished raw cuXtra artifact path.

Under both shared leases, the invocation from repository root is:

```sh
python3 -u workloads/xsched/level2/run_tool_pair.py run --no-initial --repetitions 1 --configs native_port,bpf_port --tasks 50 --reps 9511106 --blocks 340 --threads 256 --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-output-20260908.66ppYl/priority_workload --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy --output /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-delayed-preload-20260908.pdvxNk/cells
```

`run.log` and `cells/` retain the actual outcome. No completed five-block
comparison is claimed merely from starting this continuation.

The native cell exits -11 at first launch before its running event; BPF
is unstarted. The old publication assertion is gone, and the tool loaded
message is present in the real workload. The subsequent
[stack diagnostic](../level2-launch-stack-20260908.7gAYEJ/README.md) localizes
the new crash to the tool's post-enabling instruction-index logging.
All owned workers/server ended and no performance sample is claimed here.
