# Initial preempt-buffer clearing candidate, 2026-09-08

The candidate explicitly zeroes the initial `ResizableBuffer` allocation in
`InstrumentManager` and synchronizes its operation stream before first use.
`ExpandTo` already clears newly mapped tail regions, but the constructor did
not clear its initial allocation. This is a protocol-initialization change;
the bytes before initialization were not measured.

The isolated tool HAL in `level2-build/.output/hal-tool-source-20260908.f492JH`
was rebuilt and installed, and the current tool was freshly rebuilt after
removing its temporary instrumentation mutex. Both commands returned zero;
their logs are retained here. Build and execution held both shared experiment
locks. The preceding committed main revision was `46d12f32`; the candidate
was an uncommitted local-model edit at execution time.

The matching constructor hunk is now retained in
`level2/xsched-level2-sm120.patch`, alongside the freshly built tool source.
The local model's first save command used `set -e` around `diff -u` and
stopped on the normal difference exit status before updating the saved patch.
Root transferred the already-generated hunk with `apply_patch`; a reverse
dry-run of the complete saved patch against the tested isolated source
returns zero. This bookkeeping correction does not alter the tested binary.

Only the previously failed `native_port` configuration was retried, using
the first-mismatch worker `service-mismatch-20260908.sHSYtE/priority_workload`.
The command and shape match `../level2-first-mismatch-20260908.xlUdZS/README.md`,
changing only the output directory to this directory's `cells/`.
No completed baseline or BPF measurement was repeated.

The runner returned 1; BE1 returned 2 with the same failure:

```text
sink mismatch index=87040 value=0x0p+0 expected=0x1.45ef1cp+4
XG done functions=1 launches=200
```

That index is still stream 0 / task 1 / block 0 / thread 0. Initial buffer
clearing did not repair the missing output and is not an established cause
of the earlier failure. There is no new successful performance comparison.
The raw process records, including adverse outcomes, are retained.

After worker/server cleanup, NVML again showed 100% utilization and 1 MiB
without workload processes. An uninstrumented invocation created its CUDA
context and reached `ready`, but received no GO command and exited with
`orchestrator closed command pipe`; it did not execute a kernel. Its log is
retained separately. The subsequent NVML query showed 0% / 1 MiB / P8.
No driver reload,
reset, reboot, or local-model termination was performed.
