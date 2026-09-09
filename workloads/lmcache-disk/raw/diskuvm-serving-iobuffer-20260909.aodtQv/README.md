# Disk/UVM serving with valid descriptor and ioctl handling

Final status: 13 cells saved; the runner stopped between cells after actual
OOM failures. Original UVM restoration completed at 04:05:47 PDT. See
[RESULTS.md](RESULTS.md) for all observations and the corrected interpretation.
The live-run statements below are retained as the earlier execution record.

## Correction: interrupted requests in the first block

The earlier version of this report incorrectly described all eight requests
in each arm as complete. Server exit zero did not imply EngineCore success.
The original numbers and raw records below remain preserved, but the native
and BPF rates are partial-output rates from failed serving, not formal paired
throughput measurements. This correction supersedes the earlier interpretation.

Started 2026-09-09 03:36:41 PDT. Sources 8a352587 and a26c57a5 fix
the descriptor mode/flag and ctypes ioctl readback, and permit vLLM
worker drain on shutdown. This is a separate treatment from the
earlier fallback-only runs; none of their completed cells is reused.

| Block 0 arm | Warm token/s | Prepared | Restored | Restore errors / fallback |
| --- | ---: | ---: | ---: | ---: |
| Stock reclaim policy | 69.150960 | 48 | 16 | 52 / 52 |
| Native reclaim policy | 62.769598 | 48 | 5 | 15 / 15 |
| BPF reclaim policy | 61.733256 | 48 | 8 | 12 / 12 |

Native and BPF each record two HTTP 500 failures. Native reports only 2048
completed output tokens and 2167 observed streamed tokens across eight attempts;
several HTTP 200 streams also end early. Native server.log records an EngineCore
CUDA out-of-memory exception at line 853: an additional 20 MiB allocation fails
with 8.38 MiB free. Thus the previously reported -1.6510% ratio must not be used
as a BPF mechanism-overhead estimate. The same failure also occurs in later
cells. All raw observations remain, including unsuccessful runs.

For the first time these serving cells report successful backing
preparation and completed UVM restorations, rather than only a fallback
path. However, substantial restore errors still invoke the original
GDS read. Thus this is a **hybrid UVM-restoration / GDS-fallback
execution observation**, not a completed performance comparison or a claim
that every read uses UVM.
The exact restore exception is currently swallowed by the pre-existing
read wrapper; it is the next implementation issue to expose and repair.
Counters are existing observations, not a gate used to discard timings.

All three arms opt into the same disk/UVM transport. Stock disables
KV reclaim decisions; native and BPF run the same reclaim policy via
their respective implementation paths. This comparison does not
isolate UVM transport versus an untouched GDS transport. Warm-burst
throughput excludes startup, cold population, barriers and shutdown;
it is not the earlier disk-read p99 or pure GPU prefill rate.

run-serving.sh contains the exact five-block command. It reuses the
fixed workload, 62502 ns/token calibration, original fault helper and
previously built disk/UVM driver. Both shared leases cover the entire
campaign. Active GDS/KV loaders are 482025/482026; the existing EXIT trap
will restore the saved original UVM when the run completes.
Restoration remains pending during this live run.

The first-block raw JSON, responses and per-PID counter dumps are
retained under cells/block-00. Later blocks are still running and will
be appended, not treated as completed by this report. Old Table 1,
trampoline, LMCache and failed bring-up records remain unchanged.
No new paper baseline or manuscript edit is introduced.
