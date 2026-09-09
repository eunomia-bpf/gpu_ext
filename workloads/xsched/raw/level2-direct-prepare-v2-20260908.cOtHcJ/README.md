# Direct at-launch preparation retry — still unsuccessful

2026-09-08. This attempts to remove callback-dependent argument delivery
for the existing Level-2 tool actuator. The HAL passes the actual CUDA
context, function and per-command context address to `xg_host_prepare`
before `LaunchWrapper`. The helper calls `nvbit_set_at_launch` directly,
including the first launch. A thread-local marker lets a normal callback
distinguish queued launches from unrelated target launches. Policy code
and workload parameters remain unchanged.

Tool and isolated HAL builds exit zero. The failed `native_port` cell
is then retried once under both experiment locks, without repeating the
completed baseline. It **still reports missing output**, not performance:

| Worker | Exit | Prepare count | Filtered callback count | Result |
|---|---:|---:|---:|---|
| BE1 | 2 | 201 | 200 | First missing element 87040 |
| BE2 | 2 | 201 | 200 | First missing element 87040 |
| BE3 | 2 | 201 | 200 | First missing element 87040 |
| BE4 | -2 | No final counter | No final counter | Runner cleanup after another worker's error |
| LC1 | 0 | 200 | 200 | Completed |
| LC2 | 0 | 200 | 200 | Completed |

Each reported mismatch is zero versus expected `0x1.45ef1cp+4`.
All captured `XG3 submit` returns are zero; the completed counter lines
report zero disarms. The extra preparation on each failed BE process
does not restore the missing resume callback or fix the output. Direct
SDK value-setting alone is therefore insufficient in this attempt.
CUDA success and preparation counts must not be treated as proof of
correct resumed execution. The precise reason for the filtered callback
absence remains open.

The runner exits one and releases both locks. BE4's interruption is the
runner's cleanup of its own worker after a real error, not termination of
an OpenCode session. All local model sessions remain available. No driver
change or GPU reset occurs in this XSched retry. Earlier failed candidates
and completed controls remain intact.

## Executed commands

```sh
make -C workloads/xsched/level2-build -j2 guard-tool
cmake --build workloads/xsched/level2-build/.output/hal-tool-build-20260908 --target install -j2
python3 -B -u workloads/xsched/level2/run_tool_pair.py run \
  --no-initial --repetitions 1 --configs native_port \
  --tasks 50 --reps 9511106 --blocks 340 --threads 256 \
  --workload /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
  --target-symbol _ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy \
  --output workloads/xsched/raw/level2-direct-prepare-v2-20260908.cOtHcJ/cells
```

`cells/protocol.json`, worker JSON files and `runner.log` retain paths,
configuration, commands, environment, stdout/stderr and exit codes.
No Level-2 performance claim or equivalence claim follows from this run.

## Post-run busy-state observation and recovery

At 21:57--21:59 PDT, after all workload processes had exited, the driver
continued reporting GPU utilization 100%, memory 1 MiB, P0 and about
106 W. There were no listed compute applications; only persistence held
the GPU device nodes. Root restarted `nvidia-persistenced` under both
experiment locks. The recorded query immediately fell to 2% / 51 W, and
a subsequent query reached 0% / P8 / 11.45 W with 1 MiB retained. Both
GDM and persistence services were active. No module reload, explicit GPU
reset or host reboot was used; no OpenCode session was stopped.

`persistence-recovery.log` records this operation. The observation does
not identify whether the earlier busy state was residual execution or
stale driver accounting, but it must not be reported as a clean idle
return immediately after the failed retry. The completed Hummingbird
campaign predates this XSched attempt and is not rerun.
