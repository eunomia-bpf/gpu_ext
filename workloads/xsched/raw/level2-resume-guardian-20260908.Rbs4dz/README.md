# Resume-to-guardian / volatile-access retry: output loss remains

2026-09-08 PDT, RTX 5090 / driver 575.57.08 / CUDA 12.9.
Local Qwen changes the shared native/BPF trampoline in
`../../level2/tramp/xsched_guard_tramp.cu`: successful resume clears the
restore flag and continues into an explicitly tagged GUARDIAN snapshot,
matching the original restore-to-guardian control transfer. Mutable argument
and preemption words use volatile accesses. The 48-byte decision context and
real BPF decision function remain unchanged. Volatile is not a universal
coherence or freshness guarantee, and this combined candidate does not
separate the effects of the two changes.

`make -C workloads/xsched/level2-build -j2 guard-tool` exits 0, rebuilding
trampoline PTX, merged native/BPF PTX, cubin, carrier and tool (`build.log`).
The existing HAL with the context-scoped replay repair is unchanged.
Root runs only the previously failed `native_port` configuration with
`--no-initial --repetitions 1 --tasks 50 --reps 9511106 --blocks 340
--threads 256`, the existing service-mismatch workload and target symbol.
`cells/protocol.json` records resolved paths and environments. No completed
baseline or successful control is repeated. Both shared leases cover build
and execution, separately from the native-blob retry.

| Worker | Exit | Preparations / callbacks | First missing sink index |
| --- | ---: | ---: | ---: |
| BE1 | 2 | 204 / 204 | 4526080 |
| BE2 | 2 | 201 / 201 | 174080 |
| BE3 | 2 | 201 / 201 | 87040 |
| BE4 | 2 | 201 / 201 | 49152 |
| LC1 | 0 | 200 / 200 | none reported |
| LC2 | 0 | 200 / 200 | none reported |

All four BE workers report zero where the per-lane recurrence expects
`0x1.45ef1cp+4`; all report disarms=0. The runner exits 1. A different first
missing index is not evidence of an improvement: this remains an incomplete
cell, not a performance comparison. Command IDs are one-based; task/sink
indices are zero-based, and 87040 elements form one task's sink region.

The source/protocol candidate and adverse records are retained. GPU is
immediately idle at 0% / 1 MiB / P8 after cleanup, unlike some earlier
attempts; no driver change or persistence recovery occurs. This observation
does not establish that the candidate repairs the earlier post-run busy
condition. The same Qwen session continues from these failures; no unchanged
retry or completed-cell repeat is requested.
