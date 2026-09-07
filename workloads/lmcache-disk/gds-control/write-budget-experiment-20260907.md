# LMCache live write-budget comparison

The admission diagnostic (`37299d27`) measures tens of microseconds at median
admission versus hundreds of milliseconds for read completion. Historical
live/event-driven records show most writes exhausting a 10 ms delay budget.
The next experiment tests whether matching the deferral window more closely
to outstanding demand reads reduces storage interference.

Five rotated blocks each contain FIFO, native 10 ms, BPF 10 ms, native
200 ms and BPF 200 ms, for 25 fresh-process cells. Each arm occupies every
position once. Old 10 ms numbers remain; contemporaneous 10 ms controls are
necessary for this new parameter comparison because the SSD varied strongly
within prior campaigns. This is not a repeat of an already answered question.

Only the live provider's cumulative write-delay budget changes. The same
stateless native/BPF policy is called at each admission; the existing shared
event-driven executor waits for demand drain or cumulative expiry, then asks
the decider again. The scalar BPF defer request still has its existing clamp;
200 ms describes the executor's cumulative budget, not a 200 ms BPF action.
Demand reads always submit immediately; no synthetic HBM pressure is used.
Default 10 ms, fixed-delay and polling paths remain available unchanged.

Use 24 MiB objects, 64 reads and 96 writes, 2/4 ms scheduled arrivals,
4096 MiB GPU staging pool, real LMCache cuFile compatibility/direct I/O.
Both admission timing and GIL-retention ablation are off. Run the existing
`--single-cell` route with absolute `--cell-dir` paths, the rotated `--block`
and `--position`, `--config`, `--policy-variant live-event-driven`, and
`--write-delay-budget-ms 10` or `200`.

Primary comparisons: BPF 200/10 and native 200/10 scheduled-arrival read p99,
BPF/native at each budget, and each policy/FIFO. Always report paired values,
medians and ranges, read p50, write throughput and total bandwidth. A longer
budget may lower read tail at the cost of write completion; that is a policy
tradeoff, not free mechanism improvement. Mixed/negative pairs remain visible.
No per-cell retries, extra gate, or additional correctness/clock study.

Record every result, request, feedback, process return code and full log.
Each successful cell's generated synthetic `cache/` can be removed after its
process exits and result is recorded, retaining all measurement files. Failed
cell caches remain for diagnosis. This prevents approximately 94 GiB of
regenerable payloads from exhausting local storage. The cleanup is outside
request timing and is applied identically to all arms; it can still influence
subsequent SSD behavior, which the balanced order mitigates but cannot remove.

Raw target: `raw/gds-write-budget-575-20260907-five-block`. No cell from a
completed campaign is overwritten or used as a substituted contemporary pair.
