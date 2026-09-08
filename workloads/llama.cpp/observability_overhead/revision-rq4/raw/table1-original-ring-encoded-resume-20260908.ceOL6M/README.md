# Resume failed original-ring transport cells

Original campaign: `../table1-original-ring-encoded-20260908.iVdbES/`.
Its ten baseline cells completed, while all attached loaders failed before
timing because of the 10 GiB runtime segment ceiling. Runtime `241872b`
raises the explicit ceiling to 16 GiB; the default segment size and probe
policy/event payload/capacity remain unchanged. Both runtime libraries were
rebuilt; `build-runtime.log` is retained.

This resume reuses all ten completed baseline records and the original
compiled kernelretsnoop probe object. It runs only the failed off/on cells,
using the existing `run_table1_perf.py` measurement helpers and the original
within-block arm order. `reused_from` identifies original baseline records
in the combined result. Baselines therefore precede the repaired attached
cells; this is not a new fully interleaved 30-cell campaign. Interpret the
paired off/on comparison separately from baseline-normalized overhead.

Command: `python3 resume-failed-arms.py` under both shared GPU/struct-ops
leases. Diagnostic counting is disabled. Every completed/failed attempt is
retained. Do not treat the presence of 30 rows as proof that all 30 timed
measurements completed; inspect actual throughput/exit fields.

Completed: all twenty attached benchmarks return zero and numeric throughput;
the ten retained baseline measurements also return zero. Mean off/on throughput
is 34.514/194.059 token/s, with paired ratio median 5.7702x. Every loader
teardown records -9; final collector reports are unavailable and the failure
is retained, not hidden. See the complete scope and comparison discussion in
`../../results-original-ring-encoded-20260908.md`. `analyze_pairs.py` derives
`paired-analysis.json` without executing a benchmark. Existing baseline
records remain in their original directory and are not rerun.
