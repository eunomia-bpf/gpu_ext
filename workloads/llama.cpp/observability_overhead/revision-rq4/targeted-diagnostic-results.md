# Table 1 targeted repair checks — 2026-09-03

## Histogram diagnostic 01: counts agree; exact stdout still fails

The fresh three-cell run at 13:33:05–13:35:02 UTC used the rebuilt private
575 runtime and predicate-corrected NVBit adapter. The coordinator retained
normal CPU affinity, clients used CPUs 8–15, telemetry used CPU 16, and both
revision leases were held. GDM/persistenced remained active; no service or
driver change was made for this untimed diagnostic. The 400 W state and all
three before/after safety/owned-cleanup checks pass.

| Actual path | Normalized stdout bytes | Exit events | Nonzero histogram slots | Outcome |
| --- | ---: | ---: | ---: | --- |
| No probe | 47 | — | — | Exact reference output |
| Predicate-corrected NVBit | 47 | 720,896 | 22,528 | Output and engagement pass; 220 selected launches |
| BPF | 150 | 720,896 | 22,528 | Engagement/full readback pass; exact output fails |

The BPF array readback covers all 1,048,576 entries / 8,388,608 bytes. Its
private loader and segment were removed after owned client exit. No new Xid,
abnormal kernel record, surviving CUDA client or thermal-throttle failure was
observed. The runner exits **2**, preserving `diagnostic_passed=false`; it does
not reach the final comparison step after the failed output check.

Independent inspection of the retained per-cell probe fields shows the two
new aggregate counts agree. The NVBit guard repair removes the earlier
901,120-versus-720,896 discrepancy on this deterministic workload; the old
count difference is not evidence of BPF loss. This does not prove complete
per-launch/coordinate equality or fix the distinct exit-event channel limits.

The initial logger repair moved extraction messages off stdout and removed
the duplicate-registration warnings. One separately loaded PTX-pass logger
still emits `kretprobe: matched=1, in=375367, out=376814` to stdout before the
unchanged generated sentence. We do **not** strip it or weaken exact-output
checking. Repair that diagnostic source, rebuild its actual plugin, and use a
fresh output directory. Launch correlation and clocks remain unresolved.

Full original evidence: [diagnostic.json](raw/diagnostic-histogram-575-01/diagnostic.json),
per-cell application/probe logs, command/exit records, GPU telemetry and safety
records beneath that directory. **Zero performance cells ran.** This is not a
successful seven-arm preflight or a completed Table 1 comparison.
