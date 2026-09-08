# Automatic-warp enabled-arm retry after PTX fix

Date: 2026-09-08, approximately 09:31 UTC. Source `bpftime` branch
`revision/automatic-warp-execution` at pushed commit `49d71e5`; isolated
build `build-auto-warp-575`, all four requested targets built successfully.
This retry fixes the missing `.b32` in emitted `activemask` instructions.

The original `cuda__shared` BPF object and loader/benchmark are unchanged.
Settings match the [initial attempt](../results-automatic-warp-initial-20260908.lyHajw/README.md):
128 threads, eight warmups, 128 measured launches, shared GPU array map,
strict verifier mode, and `BPFTIME_GPU_AUTO_WARP_EXECUTION=1`.
The new run ID is 90804. Neither baseline nor the completed disabled arm
was rerun, and the initial failed enabled arm remains recorded separately.

Observed elapsed time across all 128 launches: **0.517823994 ms**.
Application PID 2529173 and loader PID 2528912 both exited zero. The runtime
accepted the same 14-instruction probe for automatic warp execution,
compiled and loaded the patched PTX module, and the loader read back
key 0 = 6291613622346973184 with exactly one populated map key. The
application reports 128 checked outputs and zero errors. These are direct
execution records, not a separate preflight or clock-accuracy study.

This establishes an executed eligible-hook path through the new runtime,
not a measured reduction in scalar call count or a performance improvement.
The earlier disabled-arm value is 0.329535991 ms and baseline 0.260127991 ms;
this enabled sample is higher. There is only one sample per setting and
the retry is about four minutes later, so these do not constitute an
interleaved paired campaign. No confidence interval or stable-overhead
claim is made. Full same-object repeated measurement, original block/work
sweeps, and multiplicity-preserving batching for per-lane Table 1 events
remain unfinished.

Root held the GPU lock and released it after both processes completed.
No driver, service or original storage-policy loader was changed.
The private 256 MiB transport file
`/dev/shm/fig15_auto_retry_20260908_2528256` was deleted after confirming the
exact path and completed processes. Logs and the small map readback remain;
no large binary/cache payload is retained. Runtime-internal cache-key text
is omitted from the published log; all timing, error and output records
are unchanged.
