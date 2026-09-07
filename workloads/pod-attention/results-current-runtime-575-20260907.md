# POD current-runtime startup and operator performance

All nine fresh-process measurements completed with exit zero: three rotated
blocks of inline POD, the CUDA adapter, and the BPF selector. This is the
[planned current-runtime follow-up](current-runtime-cold-start-plan-20260907.md),
not a replacement of the original shape or phase results.

## Results

Medians of three process-level observations per arm. Each operator observation
is the mean of 100 CUDA-event samples after ten warmups, at the unchanged
Llama-3-8B / decode-batch-32 shape.

| Metric | POD inline | CUDA adapter | BPF selector |
| --- | ---: | ---: | ---: |
| CUDA-event operator time, ms | 3.249527 | 3.262904 | 3.311722 |
| Client spawn to first Python statement, s | 0.019688 | 0.021567 | 234.967227 |
| Client spawn to observed exit, s | 2.602718 | 2.602236 | 239.166103 |

The within-block BPF/CUDA operator increases are **1.1754%, 2.6016%,
and 1.5239%**, with median **1.5239%**. These three pairs describe this
shape/runtime; they do not establish equivalence or a universal overhead bound.
The operator comparison is not the complete process-duration comparison.

| Block | BPF pre-Python, s | BPF client wall, s | BPF operator, ms |
| --- | ---: | ---: | ---: |
| 1 | 234.967227 | 239.166103 | 3.306004 |
| 2 | 241.117315 | 245.557055 | 3.347792 |
| 3 | 234.098142 | 238.364397 | 3.311722 |

Cold startup remains a substantial cost. BPF loader readiness was observed
after 0.201184, 0.100691 and 0.100881 seconds, separately from client startup;
post-client cleanup took 0.185456, 0.172742 and 0.197240 seconds. The launcher
observes client exit at 0.2-second polling intervals and loader readiness at
0.1-second intervals, so those wall/readiness observations include polling
delay. These are not precise loader-service measurements.

## What changed, and what did not

The selected build is
`bpftime-table1-hostfix-plt/build-table1-575-warp` (worktree revision `89a1244`).
Main `6b8d66fd` lets the runner select it and resolves the POD adapter's stale
PTX-pass dependency. Main `60fb1d63` adds the performance-only launcher and
`35575243` adds the offline analyzer. CPU affinity remains 8–15; all cells
use the same original selector, adapter, existing six PTX packets and
`bench.py --phase-study`. No driver changes, preflight campaign or new
correctness work was added. The unchanged benchmark's outputs are retained.

The [historical phase study](results-phase-full-575-01.md) reported a
271.225-second median pre-Python interval and a 1.78% paired operator cost.
Its old `bpftime/build-cuda-pr503` binary is absent at its recorded location.
The new and old experiments are separate campaigns, not interleaved builds
or an isolated source-change ablation. The lower new startup observation
does not prove that the path override or ingest-once implementation caused
an improvement. All old measurements remain unchanged.

Source inspection shows that the selected runtime already compiles PTX
packets in parallel and ingests the external directory once during bootstrap.
The current timestamps do not separate PTX rewriting, compilation and module
loading. An opt-in stage-timing patch is in local-model development to locate
the next optimization; it is not yet applied or measured.

## Data and reproduction

The [raw directory](raw/current-runtime-575-20260907-01/) contains exact
commands/environments, loader/client logs, phase timestamps, child exits,
operator samples and retained benchmark output. Its [summary](raw/current-runtime-575-20260907-01/summary.json)
is generated solely from these nine cells:

```sh
python3 -B workloads/pod-attention/analyze_current_runtime.py \
  workloads/pod-attention/raw/current-runtime-575-20260907-01
```

No failed or partial cell is pooled into this completed campaign, and no
completed cell was repeated. This coordinating session did not edit paper files.
