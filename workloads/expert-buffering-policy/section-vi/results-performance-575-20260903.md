# Expert Buffering Section VI: complete matched-policy performance

The replacement RTX 5090 campaign completed all **15/15 cells in five
randomized FIFO/native/BPF blocks**. Every cell passed exact-token, policy
engagement, runtime-inventory, telemetry, cleanup and kernel-safety gates.
No cell was retried, replaced or excluded. This is the same Qwen/FineMoE
executor and fixed per-layer K=16 cache validated by
[preflight 02](correctness-results-575-02.md).

## Results

| Median across five blocks | FIFO baseline | Section VI native | Section VI BPF/JIT |
| --- | ---: | ---: | ---: |
| Throughput (token/s, higher better) | 5.508 | 5.663 | 5.621 |
| Request-median TTFT (ms, lower better) | 754.09 | 726.48 | 721.61 |
| Request-median TPOT (ms, lower better) | 130.63 | 128.89 | 130.07 |
| Timed CPU seconds (lower better) | 134.86 | 131.02 | 131.83 |

Whole-block paired bootstrap (10,000 draws, pointwise 95% intervals) gives:

| Contrast | Throughput change | Interpretation |
| --- | ---: | --- |
| Native / FIFO | **+2.55%** [+2.10%, +3.09%] | Section VI policy improves this same-K baseline |
| BPF / FIFO | **+1.79%** [+0.79%, +2.81%] | BPF retains a positive policy benefit |
| BPF / native | **−0.74%** [−1.40%, −0.20%] | measurable mechanism-path overhead |

Native/FIFO TTFT is 4.43% lower [3.20%, 6.22% lower] and TPOT is 1.75%
lower [1.32%, 2.25% lower]. BPF/FIFO TTFT is 3.68% lower [1.80%, 5.49%
lower], while its TPOT interval includes zero change. BPF/native TTFT also
includes zero change; BPF/native TPOT is 1.74% higher [0.30%, 3.88% higher].
These are secondary, pointwise intervals over only five blocks and are not
multiplicity-corrected or equivalence tests.

## Why the policy helps, and what BPF costs

Across each evaluation cohort, FIFO performs 13,304 demand copies/evictions;
native and BPF perform the same 11,595. Thus the Section VI inactive-first/LIFO
policy reduces completed logical whole-expert copy payload by **12.85%**:
1,150,896,046,080 bytes over five FIFO cells versus 1,003,054,694,400 bytes
over five policy cells. These are logical payload bytes, not measured PCIe
traffic. The smaller transfer reduction becomes a 2.55% native throughput gain
because expert execution and other inference work remain common.

Native and BPF make identical final decisions, admissions, evictions,
residency and layer choices in every block. BPF executes 17,959 actual JIT
decisions per evaluation cell (89,795 total); the native arm uses the same
algorithm without that JIT call path. The observed −0.74% throughput and
+0.67% CPU-time point effects therefore quantify a modest cost of this BPF
mechanism path under this workload, rather than a policy-quality difference.
It is **not** valid to claim equivalence or zero overhead.

## Evidence and scope

The [analysis](raw/575-section-vi-full-02/analysis.json) reaudits the full
five-block matrix, raw timing arithmetic, current 94-file runtime, model and
original reference, and the exact numerical preflight before computing paired
statistics. Root's [post audit](raw/575-section-vi-full-02/root-post-audit.json)
independently finds 15 passed results and no cleanup errors, residual compute
applications, struct-ops maps/links, Xids or recorded kernel abnormalities.
The complete raw campaign contains **78 files / 28,715,176 bytes** including
those audits. GDM and persistenced were restored to active.

The [affinity guard](../../../docs/experiment/revision-safety/eb-section-vi-full-02/affinity-guard.json)
records 1,015 checks while all 34 identified OpenCode threads remained on CPU
17; it then restored all 34 to CPUs 0–23 with no error. The child and its owned
group exited 0/empty. This addresses the CPU interference that invalidated
[attempt 01](full-01-abandoned.md), but does not prove total machine or memory-
bandwidth isolation.

This is a faithful Section VI **policy port** on a different single-GPU Qwen
workload, not the original paper's distributed system, hardware, model or
end-to-end throughput reproduction. FIFO is its matched same-K cache baseline,
not UVM or the earlier MoE three-way baseline. The BPF arm is the actual
host-uBPF JIT selector using the shared private executor; it is not a GPU-SIMT
verifier experiment or proof that every original system component was ported.
