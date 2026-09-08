# Automatic warp: original shared-map probe, ten paired blocks

Status: completed, 30 fresh application processes, 20 attached loaders.
All applications and attached loaders exit zero. This is a negative result
for this small shared-map workload, not a Table 1 throughput measurement.
Date: 2026-09-08, approximately 09:46 UTC. RTX 5090, driver 575.57.08.

## Result

The metric is CUDA-event elapsed milliseconds over **128 measured launches**,
after eight warmups, with one 128-thread block per launch. It does not
include compilation, initial attachment, or final map readback.

| Configuration | Samples | Median elapsed ms | Minimum–maximum ms |
| --- | ---: | ---: | ---: |
| Original application, no injection | 10 | 0.2362399995 | 0.196480006–0.261920005 |
| Same BPF probe, automatic warp off | 10 | 0.3289919945 | 0.327935994–0.332608014 |
| Same BPF probe, automatic warp on | 10 | 0.5204000175 | 0.508415997–0.525183976 |

Within each block, `100 * (on / off - 1)` has mean **+57.2860%**,
median +57.2083%, and range +54.6277% to +60.1483%. Its 95% bootstrap
interval for the paired mean is **[+56.3079%, +58.2831%]**. All ten pairs
are slower with automatic warp enabled. Do not replace this with the
unexecuted-hook time from the earlier failed PTX attempt.

Relative to each block's native application, off/native paired mean change
is +41.8082% (95% interval +34.1446% to +50.5317%), and on/native is
+123.0335% (+111.0372% to +136.8618%). Native time varies more than either
instrumented arm. Arm medians, paired means and paired medians are distinct
statistics; they are not ratios of the numbers in the median column.

## Implementation and execution

Runtime source: `bpftime` branch `revision/automatic-warp-execution`, commit
`49d71e5`. Root uses its existing isolated `build-auto-warp-575` libraries,
built with CUDA 12.9, LLVM 15, verifier enabled, RelWithDebInfo, and
build-only `-include cstdint`. No library was rebuilt during the campaign.

The unchanged original files are `warp_map_bench.cu`, `warp_map_probe.bpf.c`
and `warp_map_loader.c`, with their existing `.output` binaries/object.
Both instrumented arms select `shared_update` / `cuda__shared`, the same
14-instruction BPF program updating key 0 of shared GPU array map type 1503.
They differ only in `BPFTIME_GPU_AUTO_WARP_EXECUTION=0` versus `1`.
There is no new manual lane guard or tool-specific probe.

All ten on-arm application logs report automatic eligibility and a compiled,
loaded patched module. Both attached arms report key 0 equal to
6291613622346973184 in all ten loader readbacks. These records show that the
new path ran and produced the expected shared-map output. They do **not**
measure the dynamic number of scalar BPF executions; the requested execution
counter and block/work sweeps remain separate unfinished work.

Root reused the original executable commands directly while the local model
continues the reusable runner extension. Every cell uses
`warp-map-bench --threads 128 --warmup 8 --launches 128 --run-id N`.
Run IDs and cell order are in `cells.csv`. Block orders rotate:

1. `native, off, on`
2. `off, on, native`
3. `on, native, off`

This three-block cycle repeats through block 10. It is deterministic rotation,
not a claim of randomized ordering. All cells are fresh processes. The native
application has no injected library. Each attached arm uses a fresh private
256 MiB bpftime segment, 128 GPU thread slots, 1024 maximum FDs, `sm_120`,
and strict verifier mode. The loader preloads the new syscall-server library;
the application preloads its matching agent with deferred PTX extraction
and targeted late bootstrap enabled. Final loader readback follows normal
SIGINT shutdown after the application finishes. No timeout expired.

Root held `/tmp/gpubpf-revision-gpu0.lock` throughout and released it after
the final cell. No driver, service or original storage loader was changed.
All 20 task-owned temporary shared-memory segments were removed after their
respective processes completed; no large buffer or binary is archived.
The baseline `execution.log` loader field is a shell placeholder; no loader
runs in that arm, represented as null/empty in the analysis JSON/CSV.

## Analysis and next step

`cells.csv` retains all 30 observations and points to the original per-cell
application/loader/agent/exit logs. `analysis.json` includes individual paired
changes and the arm statistics. Intervals use 100000 whole-block paired
bootstrap resamples, seed 1797, and linearly interpolated percentile bounds.
The analysis records the deterministic PRNG and comparison order. No cell
was excluded. Runtime-internal cache-key text is omitted from published logs;
no timing, error or output value was changed.

This result confirms that the first automatic-leader implementation can
cost more than it saves on this probe. It does not attribute the slowdown
to a single instruction or generalize to per-lane event logging. Root found
an unconditional `vote.ballot.sync(true)` following `activemask` in the
emitted preamble and asked the local model to remove that redundant work
while keeping the predicated path. This is a candidate improvement, not
an established explanation or measured fix. Multiplicity-preserving event
batching, repeated Table 1 measurements, and the original block/work sweeps
remain unfinished. Initial single-run and failed-compilation records remain
in their original sibling directories and are not pooled into these blocks.
