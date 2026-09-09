# Native original-entry control completes; redirected path unresolved

2026-09-08, RTX 5090 / NVIDIA 575.57.08 / CUDA 12.9. The original native
guardian attempt failed with CUDA 700. This diagnostic keeps its native HAL,
instruction-memory preparation, register/barrier settings, argument values,
`cuXtraSetDebuggerParams`, workload, queues and host scheduler, but skips the
pre-launch `cuXtraSetEntryPoint(..., entry_point)` call. The existing
post-launch original-entry restoration is unchanged. Thus it runs the
original workload entry, not the guardian or resume instruction prefix.

The additive source patch is
`../../level2/native/xsched-native-original-entry-control.patch`, applied to
`level2-build/.output/hal-native-20260908.i8na51/source`. It adds `<cstdlib>`
and makes the pre-launch entry replacement conditional on the absence of
`XG_NATIVE_ORIGINAL_ENTRY_CONTROL`. The control is **default off**; any value
present enables it. Root's bounded runner glue forwards this one option
after the shared environment cleanup. Tool-route sources are not changed.

## Attempts, retained separately

- `build.log`: native HAL build/install succeeds. Existing shim-link
  creation reports links already present; installed paths remain unchanged.
- `cells/`, `run.log`: first attempt does not forward the option because
  `clean_env()` removes all `XG_` variables. This runs the redirected path
  and still fails with CUDA 700. It is **not** original-entry control evidence.
- `run-forwarded.log`: the first forwarding edit lacks `import os` and
  fails in protocol construction, before launching GPU workers. This setup
  error is preserved, not a GPU sample.
- `cells-forwarded-fixed/`, `run-forwarded-fixed.log`: the forwarding/import
  repair enables the control in both role environments, recorded explicitly
  in `protocol.json`. Runner exits zero, all six workers exit zero, and
  400 LC / 800 BE kernels complete with no sample diagnostics. The same
  first-mismatch worker is used as in the earlier failing native attempts.

The successful control supports focusing the next repair on redirected
execution (entry redirection, copied code, guardian instructions or their
device-side argument consumption). It does not identify which of those is
faulty; host debugger-parameter roundtrip is still not proof that a redirected
GPU prefix reads the expected arguments. Preparation and ordinary workload
execution can complete together in this control.

**This is not Level-2 completion or a performance comparison.** The unchanged
runner's legacy `native_blob`, `native-cuxtra-blob`, and `paired` labels in
the raw JSON must be read with the explicit control environment: the prefix
is bypassed and only one diagnostic configuration was run. Timings remain
in the raw logs but must not be promoted to native guardian or BPF results.

## Executed commands

Both build and GPU run hold `/tmp/gpubpf-revision-gpu0.lock` and
`/tmp/gpubpf-revision-struct-ops.lock` as the ordinary workspace user.

```sh
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build --target install -j2
XG_NATIVE_ORIGINAL_ENTRY_CONTROL=1 python3 -B -u workloads/xsched/level2/native/run_native_blob.py run \
  --repetitions 1 --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --output workloads/xsched/raw/level2-native-original-entry-20260908.pn3UDn/cells-forwarded-fixed
```

The saved additive patch reverse-applies in a dry run against the tested
source. No completed baseline, BPF cell or successful control was repeated.
No driver reload/reset/reboot occurred. Normal cleanup left the GPU at
0% utilization / 1 MiB, with both locks released. All raw records, setup
failures and logs remain; compiled outputs are not committed. Local GLM
receives the completed control and continues the existing native repair.
