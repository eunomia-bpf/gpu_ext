# Stage 3 Trace Overhead

Status: `MEASURED_BORDERLINE_UNATTACHED_OVERHEAD`.

The 256 MiB custom no-policy demand-kernel comparison used ten independent processes per condition.

| Condition | n | Mean (ms) | Median (ms) | Stddev (ms) |
|---|---:|---:|---:|---:|
| Stage 2 custom no-policy reference | 10 | 240.731 | 240.516 | 2.889 |
| Enhanced module, decision trace not attached | 10 | 243.496 | 244.159 | 1.983 |
| Enhanced module, decision trace attached | 10 | 285.711 | 286.255 | 2.445 |

The empty probe call increased the mean by 1.149% relative to the Stage 2 reference, slightly above the approximately 1% target. The two approximate 95% mean intervals only marginally overlap, so this is retained as measured overhead rather than dismissed as noise. The hot path was audited: when unattached it performs one noinline empty wrapper call plus a compiler barrier, with no allocation, lock, string handling, counter, or logging.

Attaching the BPF decision trace added 17.337% over the enhanced untraced module. This is observation overhead and is why timing and trace runs remain separate. These figures apply to the sequential 256 MiB vector-add only.
