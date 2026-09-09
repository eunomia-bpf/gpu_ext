# XSched Level-2 host-submission mutex candidate: output failure remains

2026-09-08, RTX 5090, existing 575 driver and shared runtime unchanged.
This is a failed repair attempt, not a performance sample or a completed
native/BPF Level-2 comparison. Previous measurements and failed attempts remain.

## Candidate and build

Local OpenCode added the existing `launch_mtx_` around
`ToolPublish -> LaunchWrapper -> ToolPublish(0)` in both tool-actuator
branches of the isolated HAL. The tool also protects its instrumented-function
set and counter updates with `instrumented_mtx`. The saved
`level2/xsched-level2-sm120.patch` contains the HAL change. Native and BPF
tool modes use the same code; this attempt runs only native mode.

Root built `make -C workloads/xsched/level2-build -j2 guard-tool` and
`cmake --build workloads/xsched/level2-build/.output/hal-tool-build-20260908
--target install -j2`; the combined build exited zero. `build.log` retains
the build output, including the nonfatal existing-symlink messages. Both
revision GPU/struct-ops locks were held for build and run, separately.

## Run and observed result

The exact paths and configuration are in `cells/protocol.json`; invocation:

```sh
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-host-submit-lock-20260908.htUBAR/cells
```

The runner exited 1 after `be2` exited 2 without its service result.
Saved worker records show:

| Worker | Exit | First output mismatch index |
|---|---:|---:|
| be1 | 0 | none reported |
| be2 | 2 | 87040 |
| be3 | 2 | 87040 |
| be4 | 2 | 4439040 |
| lc1, lc2 | 0 | none reported |

Each mismatch reports zero instead of `0x1.45ef1cp+4`. Each worker's tool
reports one instrumented function and 200 launches. Those counts do not
establish complete command execution or absence of replay. `be2` explicitly
records reactivation with `preempt_idx=1` and a type-2 launch for command 1.
The first-32 host diagnostic limit still censors later launches; the first
output mismatch does not describe all affected outputs.

The host locking candidate did not resolve the existing output-loss failure.
This does not isolate its root cause or prove that other races are absent.
No BPF cell or already-completed baseline was run. Child cleanup completed;
the GPU returned to 0% utilization, 1 MiB used, P8, without module reload or
GPU reset. Root returned the result to the same live OpenCode session for
further repair. Raw files contain text/JSON only; no binaries are committed.
