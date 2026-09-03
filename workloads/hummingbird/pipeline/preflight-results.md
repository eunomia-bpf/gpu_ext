# Fixed-bound pipeline preflight — 2026-09-03

All **8/8 real GPU preflight cells** pass the independent raw audit: native C
and actual host-uBPF, each at outstanding-event bounds 1 and 2, under both
frozen arrival patterns. Each cell uses the original ten-second preflight
window, exact-output checks, complete request accounting, CTA coverage,
continuous telemetry and owned cleanup. The original profile and SLO are
unchanged; no outcome-based retuning was performed.

Every bound-2 cell reaches an observed peak of 2; bound-1 cells remain at 1.
All 3,775,992 issued LP events are retired, with zero final outstanding events.
The four BPF cells execute 123,067,318 actual JIT decisions; native cells
report zero JIT calls. These are host-issued/unretired completion events,
not measured hardware queue occupancy or a preemption-latency bound.

Raw evidence: [analysis.json](raw/preflight-575-01/analysis.json), the full
61-file campaign below that directory, and the
[coordinator/service log](../../../docs/experiment/revision-safety/hummingbird-pipeline-preflight-575-01/coordinator.log).
The audit reports `complete=true`, `pipeline_exercised=true`, but
`formal_complete=false` and `causal_interpretation_ready=false`.

GDM and persistenced were stopped for this GPU window and restored on exit;
GPU and struct-ops leases were held throughout. This was a functional
preflight: part overlapped the separately pinned, CPU-only Expert Buffering
offloader build. Do not interpret these short-run values as formal performance
or pool them with the old 50-cell campaign. Only these newly generated,
non-cache raw files were reassigned from root to the repository user for
publication; their content and runtime inventories were unchanged.

Next: after all heavy preparation stops, run the frozen **40-cell** randomized
five-block comparison with this exact runtime/profile and audited preflight.
The old conservative port's 19–20% gap remains unexplained by formal ablation
until that full comparison finishes; preflight engagement is not a speedup claim.
