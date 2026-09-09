# Native window-argument relay: launch rejected

2026-09-08, RTX 5090 / NVIDIA 575.57.08 / CUDA 12.9. This is one retry of
the failed native redirected-entry configuration, not a performance sample.
No completed baseline or successful original-entry control was repeated.

Local GLM implemented
`../../level2/native/xsched-native-window-args-relay.patch` on the isolated
native HAL. It preserves original parameters in a launch `extra` buffer,
adds the 28-byte guardian arguments at buffer-relative offset `0x1500`
(absolute constant-bank window `0x1880`, sm_120 parameter base `0x380`),
and restores the original `params`/`extra` pointers after submission. The
buffer is rounded to 16 bytes. Guardian/resume code and workload are unchanged.
Earlier original-entry-control and resume-allocation-padding patches remain;
the control environment variable is explicitly unset in this run.

Root reviewed and returned the initial missing argument copy and offset
handling to GLM before building. Only the assembled corrected candidate
was built and run. The saved patch captures that candidate. The isolated
build/install succeeds (exit 0); existing shim-symlink "File exists"
messages are nonfatal. Both build and run hold the GPU and struct-ops locks.

```sh
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build --target install -j2
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL python3 -B -u \
  workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-window-relay-20260908.UEtnu1/cells
```

All six workers reach `ready`. Each of the four BE processes submits four
initial guardian launches; all **16 return 701** immediately. CUDA 12.9's
local `cuda.h` defines this as `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, including
possible excessive/incorrect arguments or register/thread resource limits.
No `window relay unsupported:` line is present. No BE `running` event or
service result is produced; neither LC process receives GO. The attempted
large argument buffer is therefore not evidence of device delivery, and the
new error does not diagnose or resolve the earlier CUDA 700 fault.

The current `InstrumentManager::Launch` ignores the returned launch error.
Root's debugger observations show BE1's main thread spinning on `started=0`
and all four launch-worker threads blocked in `CommandBuffer::Dequeue`.
The GPU is idle (0%, P8). Source inspection shows main cannot enqueue the
remaining tasks until `started` changes, explaining why this failed cell
cannot progress on its own. This is an error-handling defect, not slow GPU
computation or a reason to introduce a timing cutoff.

After those observations, root sent SIGINT to this runner only. Its existing
`finally` path saved every worker/server log and cleaned up. Runner exit is
130; worker return codes -2 record cleanup signals, not natural workload
completion. This was not an artificial timeout. The debugger logs describe
this unsuccessful run and are not performance measurements. The attempted
Python `py-bt` inspection was unavailable and supplies no Python stack evidence.

Cleanup leaves GPU 0% / 1 MiB / P8 and both locks released. No driver reload,
reset, reboot, or local OpenCode termination occurred. The existing GLM
session resumed with the launch-701 evidence: propagate real launch errors
and repair the native argument/launch path before another failed-cell retry.
All earlier raw data and adverse outcomes remain unchanged.
