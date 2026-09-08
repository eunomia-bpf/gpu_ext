# First native cuXtra Level-2 workload attempt, 2026-09-08

The native sm_120 HAL builds, but its first actual workload attempt fails
with CUDA error 700 (illegal memory access) before BE1 reports `running`.
There is **no native performance result** from this attempt.

Runner: main `1f3f334e`, `level2/native/run_native_blob.py`. Native HAL:
`level2-build/.output/hal-native-20260908.i8na51/install/lib`, whose build
record is `../../level2-build/hal-native-20260908.i8na51/README.md`.
Generated guardian/resume prefixes are 640/336 bytes. Worker:
main `16a9aece`, `.output/service-mismatch-20260908.sHSYtE/priority_workload`.

Executed from the repository root, under both existing experiment locks:

```sh
python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-blob-20260908.NhiJna/cells
```

This retains the existing two-LC/four-BE, four-stream-per-process shape.
The native environment explicitly sets `XSCHED_LEVEL2_TOOL_ACTUATOR=0`,
uses the isolated shim through `LD_LIBRARY_PATH`, and has no NVBit
`LD_PRELOAD`. The historical successful baseline was not repeated. No BPF
raw-blob arm was run. The runner's inherited `pair_blocks`/`paired` labels
mean a native-only attempt here, not a completed paired comparison.

The runner exits 1. `be1.json` records readiness followed by return code
-11 and two CUDA 700 messages from `cuda_command.cpp:31`; it has no
`running` or service-result event. `failure.json` and `run.log` preserve
the corresponding exception. All six worker logs and the xserver log are
retained, including processes stopped during cleanup after this failure.

Post-cleanup: no owned worker or xserver remains; GPU is 0% utilization,
1 MiB, P8. No reset, reboot, driver reload, or local-model cancellation
was needed. The next task is native execution repair, separate from the
still-unresolved NVBit tool-route BE replay/output failure.
