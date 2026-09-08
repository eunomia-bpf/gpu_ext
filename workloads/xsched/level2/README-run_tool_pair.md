# Level-2 tool-actuator pair runner (`run_tool_pair.py`)

Current execution: the first baseline completed, but native-port startup
failed at HAL-to-tool symbol publication. The failure and completed baseline
are retained in `../raw/level2-tool-native-bpf-20260908.sjyzim/README.md`.
The publication repair is ongoing; this is not a completed three-arm result.

Brings up the matched three-arm sm_120 Level-2 tool-actuator port on the
prebuilt components. `native_port` and `bpf_port` share the same trusted
NVBit actuator but differ in **both** the host HPF implementation (native
xserver vs bpftime-ubpf HPF xserver) and the device decision arm (compiled
native C vs compiled eBPF). This is a port bring-up, not an upstream cuXtra
Level-3 reproduction; the native cuXtra binary-prefix arm is separate and
unused here.

## Components (prebuilt; the runner builds nothing)

| component | path |
| --- | --- |
| service-output workload | `workloads/xsched/level2-build/.output/service-output-20260908.66ppYl/priority_workload` (source `e9529745`; old binary untouched) |
| legacy workload (unchanged) | `workloads/xsched/build/priority_workload` |
| native HPF host | `workloads/xsched/deps/xsched/output/bin/xserver HPF 50000` |
| BPF HPF host | `workloads/xsched/build/xserver-bpftime HPF 50000` + `GPUBPF_HPF_CODE=workloads/xsched/build/bpftime_hpf.bin` |
| NVBit tool actuator | `workloads/xsched/level2-build/.output/xsched_guard_tool.so` |
| isolated HAL install | `workloads/xsched/level2-build/.output/hal-tool-install-20260908/lib` (`libshimcuda.so`, `libhalcuda.so`, `libpreempt.so`, `libcuda.so.1` symlink; not the old Level-1 libs) |

## Arms

- `baseline` — plain CUDA, no XSched, no tool.
- `native_port` — native HPF xserver, Level-2 XQueues, tool actuator,
  `XG_DECISION=native`.
- `bpf_port` — same worker stack with the bpftime-HPF xserver and
  `XG_DECISION=bpf`.

Port-arm worker env: `XSCHED_SCHEDULER=GLB`, `XSCHED_AUTO_XQUEUE=ON`,
`XSCHED_AUTO_XQUEUE_LEVEL=2`, priority LC=1/BE=0, threshold LC=16/BE=4,
batch LC=8/BE=2, `XSCHED_CUDA_LV2_PORT_120=1`,
`XSCHED_LEVEL2_TOOL_ACTUATOR=1`, `XG_DECISION`, `XG_TARGET_SYMBOL`,
`LD_PRELOAD=<guard tool>`, `LD_LIBRARY_PATH=<isolated HAL lib>`.
All arms set `XG_SERVICE_ONLY=1`.

## Service-only metric scope

`XG_SERVICE_ONLY=1` selects the narrow output branch: per-kernel GPU
`exit_ns - entry_ns` (`service_ns`, device globaltimer only) plus host
`start_host_ns -> completion_host_ns` elapsed on one host clock;
`metric_scope=gpu_service_and_host_elapsed`. No submit/queue/cross-clock
metric is read or emitted; `CLOCK_OFFSET_NS` is passed as 0 and ignored in
this branch. Default legacy workload behavior is unchanged.

## Target symbol

`--target-symbol` is the exact mangled **or** demangled `compute_task`
symbol the tool matches. Verified read-only on the 66ppYl service-output
binary via `cuobjdump --dump-elf-symbols`:

- mangled: `_ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy`
- demangled: `compute_task(TaskStamp*, float*, int, int, unsigned long long)`

The mangled anonymous-namespace suffix is build-specific: re-resolve it for
each new service-output binary with the same `cuobjdump` command. If an
engagement diagnostic says the tool did not report an instrumented entry,
the symbol likely does not match.

## Usage (three-arm comparison, no extra initial batch)

```sh
python3 -B workloads/xsched/level2/run_tool_pair.py run \
    --no-initial --repetitions 5 \
    --workload workloads/xsched/level2-build/.output/service-output-20260908.66ppYl/priority_workload \
    --target-symbol '<exact symbol for that binary>' \
    --reps <reps> \
    --output workloads/xsched/raw/<new-dir>
```

Defaults: 5 kernels/stream, 340 blocks, 256 threads; `--repetitions 5`
paired blocks; the arm start position rotates per block. `--reps` sets the
kernel recurrence directly (no calibration). `--configs` selects a subset,
`--no-initial` skips the single initial run.

## Outputs

`protocol.json` (shape, env, paths); per cell
`block-NN-<config>/{result.json, <role><pid>.json, xserver.json}` where the
process logs keep raw stdout/stderr lines, argv, runtime env, and return
code; `failure.json` on abort; `summary.json` (pair-block medians and the
bpf-vs-native LC service-p99 delta). `result.json` carries
`lc_service`/`be_service` (kernels, mean/p50/p95/p99 in us), host
elapsed ns, `be_kernels_per_s`, and engagement fields.

## Diagnostics, not gates

Sample, XG-engagement, and priority-log checks record diagnostics only
(`sample_diagnostics`, `engagement.*.diagnostic`,
`priority_log_diagnostic`); they do not reject timing rows. Under replay,
guard/resume launches add on top of `streams*tasks`, so exact launch counts
are observations, not gates; raw XG counters stay in the worker logs.

## Boundedness and ownership

The tool actuator assigns one context slot per instrumented command from an
8192-slot pool that never retires; per-process original commands are bounded by
`streams*tasks` (default 20) and the runner refuses shapes at/above the
pool. The runner spawns and stops only its own child processes (xserver +
six workers in owned process groups); it takes no GPU lock, runs no
preflight/audit framework, and writes nothing outside its output directory.
Root wraps the GPU lock around the run.
