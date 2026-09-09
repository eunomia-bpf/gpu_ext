# Level-2 callback/submit diagnostic, 2026-09-08

This retries only the failed `native_port` tool-actuator configuration.
It is **not a performance comparison**. The previous complete host trace
showed submissions without proving CUDA launch success. The new diagnostic
logs `LaunchWrapper` return values and thread/context pairs at publication
and at the filtered NVBit launch callback. Policy logic is unchanged.

Tool and isolated HAL builds exited zero. The six-worker run then exited
one because three background workers report missing output:

| Worker | Launch submissions | Filtered callbacks | Nonzero CUDA returns | Worker exit |
|---|---:|---:|---:|---:|
| BE1 | 200 | 200 | 0 | 0 |
| BE2 | 201 | 200 | 0 | 2 |
| BE3 | 204 | 200 | 0 | 2 |
| BE4 | 201 | 200 | 0 | 2 |
| LC1 | 200 | 200 | 0 | 0 |
| LC2 | 200 | 200 | 0 | 0 |

BE2's first mismatch is element 174080; BE3's is 8878080; BE4's is
87040. In each, observed value is zero rather than `0x1.45ef1cp+4`.
All diagnostic counters are below their 512-record logging limits.
The publication counter includes clearing the context, hence twice the
number of launch submissions. The six processes produced 1206 successful
CUDA submit returns but only 1200 filtered callbacks.

The missing callback pairs occur on the reactivation/scheduler thread.
For example, BE2 initially publishes context `0x7b640c601440` on worker
thread 135670326616064 and receives the callback. Its resume of command 1
(`type=2`) publishes the same context on thread 135679308722176, returns
CUDA success, and clears the context without the corresponding callback.
BE3 has four such unmatched pairs; BE4 has one. BE1 and both LC workers
have none.

This rules out a nonzero `LaunchWrapper` return for these missing outputs.
It does not distinguish callback suppression from the callback's earlier
API/function/symbol filters, nor prove correct device execution from CUDA
submit success. The scheduler-thread handoff is the next repair target;
no unchanged policy retry or completed baseline repeat is needed.

## Invocation

Both experiment locks were held. Existing modules and runtime were unchanged.

```sh
make -C workloads/xsched/level2-build -j2 guard-tool
cmake --build workloads/xsched/level2-build/.output/hal-tool-build-20260908 --target install -j2
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-callback-submit-20260908.ABq1yl/cells
```

`cells/protocol.json` records paths and configuration; each worker JSON
retains its command, environment, stdout, stderr and exit code. Original
logs and all earlier adverse cells remain intact. The source trace changes
are diagnostic overhead, not a proposed performance optimization.
