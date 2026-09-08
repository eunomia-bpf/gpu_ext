# Native Level-2 illegal-access localization, 2026-09-08

One retry of the failed native-cuXtra configuration applies
`level2/native/xsched-native-diag700-instrument.patch` to the isolated
`level2-build/.output/hal-native-20260908.i8na51/source` tree. Build/install
returned zero. The patch adds an instruction-copy synchronization and
launch/parameter/resource logging; it is diagnostic, not a policy change.
The installed driver is 575.57.08 (references to 610 in the patch's initial
comments are a documentation error, not a different test environment).

Under both shared experiment locks, from the repository root:

```sh
python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-diag700-20260908.lGURrN/cells
```

The runner returns 1; BE1 exits -11 before its `running` event, again with
CUDA 700 at the event synchronization. There is no service-latency or
performance result. No NVBit tool is preloaded on this native path, and no
completed baseline or BPF cell was repeated. The wrapper's inherited
`paired` label does not make this a paired comparison.

BE1's diagnostic output narrows the failure:

- The 336-byte resume instruction copy synchronizes successfully.
- The copied kernel is 1,408 bytes and guardian is 640 bytes. Register count
  reads back as 16 -> 32; barrier count is 1 -> 1.
- All four logged guardian launches have type 1 and index 1. Their host
  debugger-parameter readback matches, and launch calls return zero.
- GPU execution subsequently reports an illegal memory access.

Thus a synchronous copy error or rejected launch was not observed. A host
parameter roundtrip does not prove the device reads the intended constant
window, and these logs do not establish the offending SASS instruction.
The `entry=0x...` log has a decimal formatter despite its prefix; use the
correctly hexadecimal `guardian`/`ep_inst` fields when interpreting addresses.

Raw records for all six workers and the server, the runner exception, and
build log are retained. Cleanup leaves the GPU at 0% / 1 MiB / P8. No reset,
driver reload, reboot, or local-model termination was performed.
