# Table 1 targeted repair checks — 2026-09-03

## Histogram diagnostic 02: all three selected checks pass

After the [PTX-plugin repair](ptxpass-output-repair.md), the fresh run at
13:45:25–13:47:21 UTC passes, exit **0**. All three paths preserve exactly the
same 47-byte normalized application output. NVBit and BPF each report
**720,896 exit events and 22,528 nonzero slots**; NVBit observes 220 selected
launches and BPF reads all 1,048,576 entries / 8,388,608 bytes. All three
safety checks pass, and the owned BPF client/loader/segment cleanup completes.

[diagnostic.json](raw/diagnostic-histogram-575-02/diagnostic.json) and all original
streams/telemetry are retained. This run used normal coordinator affinity,
clients on CPUs 8–15 and telemetry on CPU 16, with the existing services active.
The separate EB CPU build overlapped part of this **untimed correctness** run;
none of its durations are performance evidence.

This establishes repaired exact stdout and matched aggregate histogram counts
on the selected real workload. It does not establish per-launch/coordinate
identity, lossless exit-event transport, valid clock correlation, or completion
of the seven-arm Table 1 comparison. **No timing cells ran.** Diagnostic 01 and
the older failed preflights below remain unchanged.

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
