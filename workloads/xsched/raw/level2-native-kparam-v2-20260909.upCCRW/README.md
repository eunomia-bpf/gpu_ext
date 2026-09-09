# Native KPARAM-V2 candidate: build succeeds, host constructor crashes

2026-09-09 PDT, RTX 5090 / CUDA 12.9 / 575.57.08. The local GLM candidate
converts parameter metadata to V2, retains original argument-copy sizes,
and pads owned host argument storage to the driver's grown extent. Plain
passthrough launch entry points also take an owned copy for grown kernels.
The isolated native source was frozen at handoff. Root reconfigured the
existing CMake build and installed it with `--target install -j2`; exit 0.
Existing-link messages during install are nonfatal and retained in build.log.

The next native six-process cell, with `XG_NATIVE_META_EXTEND=1` and
`XG_NATIVE_META_KPARAM=1`, fails before any performance result. The BE logs
show both metadata rewrites and the owned fatbin-wrapper copy, then SIGBUS.
This is not the old cuLaunchKernel 701 result, and is not a Level-2
performance measurement. `cells/protocol.json` and worker files retain
the exact paths, environments, commands, signals, and partial events.

Root's initial log-only description placed the failure during module load.
The subsequent single-worker GDB backtrace corrects that: the fault is in
`CudaKernelCommand::CudaKernelCommand`, reached through `XLaunchKernelImpl`,
not in the driver module loader. At the fault it indexes the local argument
layout immediately before calloc. Source inspection finds the missing
fallback initialization: when `GetXgOriginalLayout` returns false, `layout`
is copied into `alloc_layout` without first calling `GetLoadedLayout`.
That explains an uninitialized layout read on this branch; the reason for
the original-layout lookup miss remains separate and unresolved. The next
change restores that one initialization before testing the candidate again.

`load-backtrace.gdb`, `run-backtrace.sh`, `backtrace.log`, and
`debug-lifecycle.log` retain the diagnostic. GDB stops on the actual SIGBUS;
no application timing is inferred from this run. Both shared leases cover
build, the failed cell, and the separate backtrace run. No driver reload
or recovery was needed; GPU is idle at 0%, 1 MiB afterward. No completed
performance cell was repeated and no manuscript was edited.

Executed build:

```sh
cmake -S workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/source \
  -B workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build \
  --target install -j2
```

Executed cell:

```sh
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL XG_NATIVE_META_EXTEND=1 XG_NATIVE_META_KPARAM=1 \
  python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-kparam-v2-20260909.upCCRW/cells
```
