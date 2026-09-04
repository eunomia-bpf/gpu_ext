# Cross-layer raw-record result: RTX 5090 / 575.57.08

## Outcome

The formal `raw/full-575-02` campaign passed all 15 fresh-process cells in five
seeded, randomized complete blocks. Every native and instrumented CUDA truth
array matched the frozen launch geometry. Every device-BPF aggregate matched
that truth.

| Arm | Cells | CUDA truth per path | Raw tuples read back | Full drops | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| small: 256 threads, 3 launches | 5 | 3,840 | 3,840 | 0 | accepted complete stream |
| large: 2,048 threads, 3 launches | 5 | 30,720 | 30,720 | 0 | accepted complete stream |
| overflow: 256 threads, 6 launches | 5 | 7,680 | 5,120 | 2,560 | rejected incomplete stream |

Across the positive cells, the host recovered 34,560 distinct bounded raw
tuples exactly. Across the deliberately overflowing cells, all 2,560 omitted
records were reported as full-capacity drops; none was accepted as complete
evidence. Every cell removed its private shared segment and ended with no owned
process-group survivor, UVM reference, struct-ops link/map, Xid, or recorded
kernel/service abnormality.

Reproduce the offline audit with:

```bash
python3 analyze_raw_map.py raw/full-575-02
```

## What this answers

This is direct evidence for Reviewer A's question about non-composable state:
the current device-to-host map ABI can carry individual coordinate/sequence
records, not only a reducible counter or minimum. The separate per-thread
aggregate is a control, not the source from which the raw tuples are inferred.

## Boundary

This is an expressibility and exact-readback result, not a performance result.
It does not measure map latency or bandwidth, exercise the on-chip/shared-memory
shard, prove automatic placement, run the strict device verifier, or imply
support for arbitrary or unbounded structures. The bounded ring can overflow;
the demonstrated property is explicit detection and fail-closed rejection.

## Retained failed attempt

`raw/full-575-01` is an earlier failed campaign. Seven cells passed before the
eighth probe failed while opening its BPF object; its old diagnostic did not
retain enough detail to identify a unique cause. Those seven cells are not
selected into this result. The failure remains committed, the runner now logs
the libbpf and `errno` details and proves live-child ownership before recording
a cleanup identity, and `full-575-02` started from a new preflight and a new
output directory with no within-run retries.
