# Native allocation-padding candidate does not resolve CUDA 700

2026-09-08, RTX 5090 / NVIDIA 575.57.08 / CUDA 12.9. The preceding
[original-entry control](../level2-native-original-entry-20260908.pn3UDn/README.md)
completed all six workers when entry replacement was omitted. This new
candidate restores ordinary redirected execution and changes only the resume
allocation reservation from 336 bytes to `ROUND_UP(resume_size, 256)` (512
bytes). The copy still contains the same 336 resume bytes; the guardian
prefix and original body remain 640 and 1408 bytes respectively. The
allocator advances by the requested size without additional alignment.

This tests an allocation-layout hypothesis, **not a documented requirement
that sm_120 entry points must be 256-byte aligned**. The saved additive patch
is `../../level2/native/xsched-native-resume-allocation-padding.patch`, applied
to the isolated native HAL source at `hal-native-20260908.i8na51` after the
optional original-entry control patch. That diagnostic option is explicitly
unset for this run; its successful control is not repeated.

Native HAL build/install succeeds. The actual first guardian address now ends
in `0x0200`, rather than `0x0150`. Resume copy synchronization succeeds,
host parameter roundtrips equal their inputs, and launches return zero.
The first BE worker still exits -11 before its running event, with CUDA 700
reported at event synchronization. Runner exits 1. Padding alone therefore
does not repair redirected execution. It neither proves nor disproves any
general architectural alignment requirement. This is not performance data.

Under both shared GPU and struct-ops locks:

```sh
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build --target install -j2
env -u XG_NATIVE_ORIGINAL_ENTRY_CONTROL python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-entry-alignment-20260908.jOzEYp/cells
```

The source reservation change remains applied in the isolated native build,
and the failed candidate patch, build log, run log and all worker/server
records are retained. No completed baseline, BPF cell or successful
original-entry control was repeated. No driver reset/reload/reboot occurred.
Local GLM receives the result and continues repair; no session was stopped
for silence and no additional model session was created.
