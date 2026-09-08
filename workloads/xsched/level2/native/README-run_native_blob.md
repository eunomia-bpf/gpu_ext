# Level-2 native-cuXtra-blob runner (`run_native_blob.py`)

First execution of the native (non-NVBit) Level-2 route on sm_120 with the
real captured SASS arrays. This is a bring-up run, not a performance
comparison and not a completed reproduction of the original Level-2 system.
The runner is owned by `level2/native/` and reuses the shared tool-pair
runner's process machinery (`run_tool_pair.py`) unchanged for spawn, log
capture, affinity, the ready/GO/running bar, and process cleanup; the
shared runner file itself is not modified.

Current status: the [first native run](../../raw/level2-native-blob-20260908.NhiJna/)
and [targeted launch diagnostic](../../raw/level2-native-diag700-20260908.lGURrN/)
both fail with CUDA 700 before a service result. Instruction-copy sync,
host parameter readback, and launch submission succeed in the latter;
GPU execution does not. No native Level-2 performance result is available.

## What differs from the tool-actuator route

- The worker uses the isolated **native** HAL install
  `level2-build/.output/hal-native-20260908.i8na51/install/lib`
  (`libhalcuda.so`, `libshimcuda.so`, `libpreempt.so`, `libcuda.so.1`
  shim links). That HAL was configured with
  `XG_SM120_GENERATED_HEADER` pointing at
  `.output/native-extraction-20260908.fnVM6q/xg_sm120_guardian_arrays.h`,
  so `GuardianSM120::GetGuardianInstructions` /
  `GetResumeInstructions` return the real captured stub streams
  (guardian 0x280 B, resume 0x150 B).
- `XSCHED_LEVEL2_TOOL_ACTUATOR=0` (explicitly off). `InstrumentContext`
  takes the cuXtra path: `InstrMemAllocator`, `cuXtraInstrMemcpyHtoD`
  binary surgery (guardian prefix + original kernel text), 28-byte
  debugger argument block via `cuXtraSetDebuggerParams`, entry-point
  surgery via `cuXtraSetEntryPoint`.
- No NVBit tool is preloaded and no `LD_PRELOAD` is set; Level-2
  isolation comes only from `LD_LIBRARY_PATH` resolving
  `libcuda.so.1` in the native install directory.
- **NVBit-entry accounting is not applicable.** No
  `XG tool loaded / instrumented entry / done` lines can exist because no
  tool is loaded; the runner records
  `engagement.nvbit_entry_accounting = "not_applicable..."` instead of
  expecting or parsing tool logs. Observable engagement evidence is
  limited to the xserver priority-assignment log (as a diagnostic) and
  per-sample validity checks.
- Only the native-blob arm runs (`native_blob`). The completed block-1
  baseline of `raw/level2-tool-native-bpf-20260908.sjyzim` is **not**
  rerun, its historical baseline data is not replaced, and no BPF
  raw-blob arm is invented. The separate tool-route publication repair
  and its BE replay remain with the shared runner's owner.

## Components (prebuilt; the runner builds nothing)

| component | path |
| --- | --- |
| workload (default) | `workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload` (adds first-sink-index and actual/expected hex-float diagnostics to the existing failure print; same pass/fail/compute/timing logic, source `16a9aece`) |
| workload (original measured, preserved) | `workloads/xsched/level2-build/.output/service-output-20260908.66ppYl/priority_workload` |
| native HPF host | `workloads/xsched/deps/xsched/output/bin/xserver HPF 50000` |
| native HAL install | `workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install/lib` (`libshimcuda.so`, `libhalcuda.so`, `libpreempt.so`, `libcuda.so.1`; not the tool-route or Level-1 libs) |
| NVBit tool actuator | none on this route |

## Native-blob worker environment

`XSCHED_SCHEDULER=GLB`, `XSCHED_AUTO_XQUEUE=ON`,
`XSCHED_AUTO_XQUEUE_LEVEL=2`, priority LC=1/BE=0, threshold LC=16/BE=4,
batch LC=8/BE=2, `XSCHED_CUDA_LV2_PORT_120=1`,
`XSCHED_LEVEL2_TOOL_ACTUATOR=0`, `LD_LIBRARY_PATH=<native HAL lib dir>`,
`XG_SERVICE_ONLY=1`. There is intentionally no `XG_DECISION`, no
`XG_TARGET_SYMBOL`, and no `LD_PRELOAD` on this route.

## Usage (native-blob arm only)

The campaign shape is the default shape: 6 processes (2 LC + 4 BE), 4
streams each (24 XQueues), 50 kernels per stream, 340 blocks, 256 threads,
recurrence reps set directly, 5 paired blocks, no initial run.
The inherited `paired` label in the wrapper is only a repetition label:
this native-only runner has no paired baseline or BPF configuration.
While repairing the current failure, request one attempt explicitly:

```sh
python3 -u workloads/xsched/level2/native/run_native_blob.py run \
    --reps 9511106 --repetitions 1 \
    --output workloads/xsched/raw/<new-dir>
```

Root wraps the shared GPU lease around the invocation; the runner takes no
GPU lock itself. Outputs: `protocol.json`; per block
`block-NN-native_blob/{result.json, be1..be4.json, lc1..lc2.json,
xserver.json}` with raw stdout/stderr, argv, runtime env and return codes;
`failure.json` on abort; `summary.json` (native-blob medians only).

## What failures look like (actionable by design)

- HAL init on the first Level-2 queue construction per context:
  `no Level-2 blob guardian for this architecture...` would mean the
  header macro or arch factory is missing in that install (build wiring,
  not workload).
- First launch under an active preempt level performs the binary surgery;
  cuXtra failures there surface as CUDA/XASSERT errors in the worker
  stderr captured in `be*.json`/`lc*.json`.
- A workload result mismatch with the default workload prints the first
  sink index and actual/expected hex floats (that print is the only
  difference from the preserved original worker), so a wrong guardian
  prefix can be located from the worker JSON.

Nothing here reruns completed historical cells, adds gates, or replaces
root-recorded evidence; root records extraction and build evidence
separately.
