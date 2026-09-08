# Native sm_120 prefix extraction, 2026-09-08

The GLM-authored extractor at main commit `0f810b7a` compiled and ran
successfully against the already-built CUDA 12.9 cubins. No GPU cell or
cubin compilation was repeated. Both shared experiment locks were held.

From the repository root:

```sh
g++ -O2 -std=c++17 workloads/xsched/level2/native/ldc_patcher.cpp \
  -o workloads/xsched/level2-build/.output/native-extraction-20260908.fnVM6q/xg_ldc_patcher
workloads/xsched/level2-build/.output/native-extraction-20260908.fnVM6q/xg_ldc_patcher \
  workloads/xsched/level2-build/.output/native-exit-retention-20260908.NAeniq/native/check_preempt_port.cubin \
  workloads/xsched/level2-build/.output/native-padding-20260908.s03ZL6/native/restore_exec_port.cubin \
  /usr/local/cuda-12.9/bin/nvdisasm \
  workloads/xsched/level2-build/.output/native-extraction-20260908.fnVM6q/xg_sm120_guardian_arrays.h
```

Compilation and extraction returned 0. `build.log` is empty (no compiler
diagnostics); `extract.log` preserves the actual output. The generated
4,387-byte header is retained here; the 51,888-byte executable stays only
in ignored `.output/`.

- Guardian: 640 bytes (40 instructions), through conditional EXIT at 0x270;
  excludes the build-only marker and final EXIT.
- Resume: 336 bytes (21 instructions), through register-target call at
  0x140; excludes the final EXIT at 0x150.

This is a completed host extraction step, **not native HAL execution or a
performance result**. The generated arrays still need the existing
GuardianSM120 integration built and exercised on the workload. The separate
NVBit tool-route BE replay failure remains unresolved in
`../../raw/level2-cached-entry-20260908.Mtt9XM/`; no failed or completed
performance cells were rerun by this extraction.
