# XSched Level-2 native cuXtra sm_120 runtime

Navigation for running the original-actuator native cuXtra blob route on
sm_120: the portable multi-arm runner, its component paths, the completed
bring-up cell and the matched route-comparison campaign, and the
build-snapshot versus portable-runner distinction. Companion to
[table1-runtime.md](table1-runtime.md) and the
[Level-2 runner instructions](../../workloads/xsched/level2/README.md).
This document makes no new GPU run, measurement, or performance claim.

## What runs

[run_native_blob.py](../../workloads/xsched/level2/native/run_native_blob.py)
runs the original-actuator native cuXtra blob route on sm_120 in a single
portable invocation. It now runs up to four matched arms against one
workload shape (2 LC + 4 BE processes, 4 streams each, 340 blocks, 256
threads, service-only output branch): `l2_cuxtra` (captured SASS guardian
and resume blobs, cuXtra binary surgery, native xserver HPF),
`l2_bpfhost` (same actuator, host HPF decisions by the BPF xserver),
`l1_native` (same HAL, upstream Level-1 queue actuation), and `baseline`
(no XSched: no shim, no server). Isolation comes from
`LD_LIBRARY_PATH` resolving `libcuda.so.1` to the shim inside the
isolated native HAL install; `XSCHED_LEVEL2_TOOL_ACTUATOR=0` selects the
cuXtra binary-surgery path on the Level-2 arms. No NVBit tool, no
`LD_PRELOAD`; the runner takes no GPU lock (root wraps the GPU lease
around the invocation). Presence-based opt-in forwarding:
`XG_NATIVE_ORIGINAL_ENTRY_CONTROL`, `XG_NATIVE_META_EXTEND`,
`XG_NATIVE_META_KPARAM`.

## Component paths (portable runner)

The runner takes explicit component paths that retain the original
hardcoded locations as defaults, resolves them once per invocation, and
records them in `protocol.json`:

| Option | Meaning | Default (original location) |
| --- | --- | --- |
| `--hal-install-dir` | Isolated native HAL install directory; worker `LD_LIBRARY_PATH` is `<dir>/lib` | `workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install` |
| `--xserver-native` | Native xserver binary for the HPF server | the selected install's `bin/xserver`, else `workloads/xsched/deps/xsched/output/bin/xserver` |
| `--xserver-bpftime` | BPF HPF xserver for `l2_bpfhost` | `workloads/xsched/build/xserver-bpftime` |
| `--hpf-bin` | BPF HPF program binary for `l2_bpfhost` | `workloads/xsched/build/bpftime_hpf.bin` |
| `--workload` | Service-only `priority_workload` binary | `workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload` |
| `--configs` | Comma-separated arm subset in run order | all four, `baseline,l1_native,l2_cuxtra,l2_bpfhost` |
The preflight check respects `--configs`: the workload, xserver, and four
HAL libraries are required whenever any arm other than `baseline` is
enabled, and `xserver-bpftime` plus the HPF binary are required only for
`l2_bpfhost`. It lists missing components and exits 2 when any is absent.

## Example (placeholders)

```bash
cd <repo>
export UV_CACHE_DIR=$(mktemp -d /tmp/gpubpf-xsched-uv.XXXXXX)
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL \
    XG_NATIVE_META_EXTEND=1 XG_NATIVE_META_KPARAM=1 \
python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --hal-install-dir <level2-build/.output/hal-native-<stamp>.install-dir> \
  --workload <level2-build/.output/<workload-build>/priority_workload> \
  --configs l2_cuxtra \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output <workloads/xsched/raw/level2-native-blob-<timestamp>>
```

Build the HAL install and workload first
([Level-2 build instructions](../../workloads/xsched/level2-build/README.md));
the runner consumes finished components and does not build them. A recorded
absolute-path command in a raw report documents that execution; it is not
automatically a portable installer.

## Evidence status

The bring-up cell remains the single-cell historical record at
[raw/level2-native-retabs-20260909.oJDCbU](../../workloads/xsched/raw/level2-native-retabs-20260909.oJDCbU/README.md):
all six worker processes exited 0, with 400 LC and 800 BE kernel service
records, LC service p99 960.855 ms, BE 10.1635 kernels/s, one type-2
resume launch in each BE log, and no recurrence of the earlier CUDA 700
resume failure.

The matched multi-arm campaign it left open is complete at
[raw/level2-native-route-20260910-082400](../../workloads/xsched/raw/level2-native-route-20260910-082400/README.md):
five randomized paired blocks over the four arms (20 cells, all engagement
gates passing), run with this runner on the reproducible nine-patch source
install. The report carries the per-arm service medians and within-block
paired statistics; the figures are GPU-service and host-elapsed
measurements, not arrival-to-completion or queueing latency, and the
campaign is not the separately measured shared-actuator policy-port
comparison.

## Build snapshot vs portable runner
- **Build snapshot**: a raw report is a dated record of one specific
  build. The bring-up install
  (`hal-native-20260908.i8na51/install`) was configured at build time
  against the captured SASS array header; its build inputs, logs, and the
  run record are retained in that report directory only. The matched
  campaign ran on the reproducible nine-patch source install
  (`level2-build/.output/native-repro-supervisor-20260910.wEUkbn/install-gcc`,
  committed as the staged source-preparation recipe at `013427f5`); that
  install's `libcuda.so.1` is shim-linked to `libshimcuda.so`. Rerunning a
  report's recorded paths reproduces the recorded cells, not a fresh
  bring-up.
- **Portable runner**: `run_native_blob.py` is the reusable entrypoint.
  Given any compatible isolated HAL install and native xserver via
  `--hal-install-dir` / `--xserver-native`, it launches the requested
  arms without mutating module state; every run records the resolved
  component paths in its `protocol.json`, so each output directory states
  exactly which components it used.
