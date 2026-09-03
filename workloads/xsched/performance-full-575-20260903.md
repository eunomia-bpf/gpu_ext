# Full XSched / BPF policy comparison on 575 and sm_120

**The full ten-block protocol completed on 2026-09-03 UTC:** forty mixed cells
and six isolated controls passed, followed by an independent raw audit of all
46 cells. The campaign was not restarted, shortened or filtered by outcome.
Earlier five-kernel pilots remain separate historical experiments.

The same-frontend BPF implementation successfully exercises the original HPF
policy on this workload. That correctness/expressibility result is separate
from whether its measured implementation outperforms original XSched.

## Scope and fixed workload

This is upstream **XSched Level-1 on RTX 5090 / sm_120**, not the paper's
Level-3 mechanism or its original hardware speedups. All arms use the same
custom 575.57.08 driver (`849ea75d`), Linux 6.15.11 and fixed 400 W:

| Arm | Decision and execution path |
| --- | --- |
| `native` | Native CUDA, no attached scheduling policy, same custom driver. |
| `xsched` | Original HPF scheduler and original XSched Level-1 actuator. |
| `bpftime_hpf` | Actual ubpf-JIT implementation of the same HPF decision rule, original frontend and Level-1 actuator; bounded to 64 queues. |
| `gpubpf` | A **different** driver policy: LC/BE timeslices 1,000,000/200 us, persistent control callback, and LC-triggered GR-target preemption with 100 us cooldown. |

The frozen workload has two LC and four BE processes, four streams per process,
50 kernels per stream, 340 blocks and 256 threads per kernel. An isolated
calibration froze 9,511,106 repetitions at 79.968544 ms; the repetitions are not
retuned per arm. Seed 1797 fixes all ten four-arm orders. Three isolated LC and
three isolated BE controls are included. The mixed cells each contain 400 LC
samples and 800 BE kernels; p99 is no longer the short pilot's sample maximum.

GPU and struct-ops leases covered the uninterrupted run. The workers use the
same saved CPU masks and runtime settings; no compilation or other GPU
experiment ran concurrently. Small CPU-8-only closed-data Git checkpoints
retained earlier completed blocks while measurement continued. Those files
explicitly record their historical incomplete status, not separate attempts.

## Metrics and results

LC p99 is **submission to first CTA entry** for a burst of 50 tasks per stream:
`priority_workload.cu` takes the minimum CTA entry timestamp. These seconds-long
values describe queueing, **not one hardware preemption operation**. BE rate is
800 kernels divided by common BE release to the last BE process's recorded
stream-completion time, before that process's D2H output validation.

The table reports medians of ten per-cell measurements:

| Arm | LC entry p99 (s) | BE throughput (kernels/s) |
| --- | ---: | ---: |
| Native | 76.878036 | 10.237739 |
| Original XSched HPF | 26.978372 | 10.149674 |
| Same-policy BPF HPF | 27.250205 | 10.161646 |
| Driver BPF policy | 22.169128 | 6.137262 |

The predeclared analysis bootstraps complete paired blocks, with 10,000 draws.
The following are intervals for **mean paired differences/relative changes**,
not confidence intervals around the table's medians:

| Comparison against original XSched | Paired mean | 95% interval |
| --- | ---: | --- |
| Same-policy BPF HPF LC p99 difference | +0.291953 s | [-0.109087, +0.642367] s |
| Same-policy BPF HPF BE relative rate | +0.090314% | [+0.029818%, +0.147607%] |
| Driver BPF LC p99 difference | -4.763111 s | [-5.039399, -4.494553] s |
| Driver BPF BE relative rate | -39.545174% | [-39.623051%, -39.465667%] |

The BPF HPF comparison does not establish a latency improvement or regression;
its paired interval crosses zero. It satisfies the predeclared 5% BE
noninferiority margin, but no LC equivalence test was prespecified and no
equivalence claim is made. The difference includes scheduler implementation
and wrapper behavior, so it is not isolated pure JIT cost.

The separate driver policy trades substantially lower LC delay for much lower
BE throughput. The runner consequently classifies its performance as `mixed`,
and same-policy BPF HPF's advantage as `inconclusive`. These are performance
categories, **not failures to reproduce or execute the original HPF policy**.

## Correctness, engagement and independent audit

All 49,200 kernels and 4,282,368,000 expected output values passed numerical
checks. Every mixed cell retained all six worker logs, clock calibration and
pre/post safety. Each XSched frontend had 24 unique queues and observed
suspend/resume for all sixteen BE queues in every cell. Actual BPF HPF executed
1,066 JIT calls and 14,185 queue decisions. The driver policy executed 60
persistent control overrides and 356 successful preemptions, with zero setter
or preemption errors.

The independent audit rechecked all 46 closed cells using `audit_saved_cell`
in the original CPU-visible environment, without mocking affinity. It checked
the exact frozen workload/order, executable argv and runtime environment,
raw sample clocks, output counts, primary metrics, policy engagement and
pre/post safety, then reproduced the aggregate analysis exactly. The secondary
mixed-cell `lc_completion_p99_us` was not independently recomputed and is not
reported here. Final live safety had GPU 2 MiB/0%, zero UVM references, empty
struct-ops state, no compute clients and no Xid/abnormality. Both leases were
released before the coordinator's separate service restoration.

Evidence:

- [Frozen protocol](raw/full-persistent-575-20260903/protocol.json).
- [All per-cell records and historical checkpoints](raw/full-persistent-575-20260903/).
- [Independent 46-cell raw audit](raw/full-persistent-575-20260903/independent-raw-audit.json).
- [Recomputed results and paired intervals](raw/full-persistent-575-20260903/summary.json).
- [Execution plan and post-reboot canaries](full-runtime-plan-575-20260903.md).

`run_xsched_rq3.py analyze --output DIR` recomputes the aggregate statistics;
that command alone does not replace the separate per-cell raw audit.
