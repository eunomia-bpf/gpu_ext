# Native layout initialization/export repair: startup advances, launch still fails

2026-09-09 PDT. Root makes two bounded repairs after the preceding
`upCCRW` SIGBUS backtrace:

1. Initialize the loaded layout on the original-layout lookup's fallback
   branch, before copying it into the allocation layout.
2. Give `XgGetRelayOriginalParams` default symbol visibility. The actual
   shared-library build uses `-fvisibility=hidden`; before the repair this
   name was a local `t` symbol and absent from the dynamic symbol table,
   making the HAL's dlsym route unavailable.

The cumulative source patch is updated, including its hunk offsets; reverse
application checking against the live five source files succeeds. The build
and install exit zero. `nm -D` now exposes `XgGetRelayOriginalParams` as a
global `T` symbol. No baseline, policy algorithm, or original result changes.

The next real six-process attempt at 02:23 PDT passes the earlier constructor
fault and reaches launch. It still fails: BE1's first guardian launch returns
701, the next returns 4, and the process exits on SIGSEGV. Its log also says
the relay sees parameter extent 0x1520 beyond the window start 0x1500, so the
original-layout path is still not working everywhere. Export availability
alone does not prove the per-function lookup succeeded. The dynamic lookup
handle (`libcuda.so.1` versus the shim), function-name match, and remaining
large-parameter launch metadata must be resolved before claiming completion.
These are open causes, not conclusions from this run.

This is a failed implementation attempt, not a performance comparison or
evidence of BPF overhead. The raw worker files and protocol preserve the
exact environment and errors. Both shared leases cover build and execution.
GPU is idle afterward at 0%, 1 MiB; no driver reload or recovery was needed.
No completed cell was repeated and no manuscript was edited.

Build: `cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build --target install -j2`.

Run, from repository root:

```sh
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL XG_NATIVE_META_EXTEND=1 XG_NATIVE_META_KPARAM=1 \
  python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-layout-init-20260909.eluxHm/cells
```
