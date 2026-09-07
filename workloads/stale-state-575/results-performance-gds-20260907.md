# Stale-state live performance: 21 completed cells

The new GDS-compatible driver campaign completed all 21 planned cells:
three blocks of UVM default, matched native/BPF fresh state, and native/BPF
100 ms and 1000 ms delayed state. No execution failure is recorded.
Afterward the runtime restored the saved original GDS module; its full log
ends with `RESTORED_SAVED_GDS_MODULE`. The same storage BPF policy was
reattached, and the subsequent 15-cell LMCache diagnostic completed.

Source: main performance runner `e88e1265`, driver `a2b40efd`.
RTX 5090, kernel 6.15.11-061511-generic, NVIDIA 575.57.08.
The runtime is the existing real 40 GiB UVM workload, with 1.2 s bootstrap
and six measured 2 s dense/sparse phases. The two policy implementations
consume driver-owned timestamped snapshots and use common diagnostics.
UVM default is a contextual control without that policy-observer overhead.

## Measured performance

Median checked-word throughput across three cells, words/s. This is this
synthetic alternating-access UVM workload, not application token/s.

| State | Native | BPF |
| --- | ---: | ---: |
| Fresh | 272832.196 | 271397.413 |
| 100 ms delayed | 267195.743 | 267604.088 |
| 1000 ms delayed | 208904.486 | 209832.744 |

UVM default median: 201173.969 words/s.

| BPF/native | Block 1 | Block 2 | Block 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Fresh | -1.117% | -0.526% | +0.794% | -0.526% |
| 100 ms | +0.299% | +1.240% | +0.079% | +0.299% |
| 1000 ms | +2.345% | -0.145% | +0.344% | +0.344% |

Paired 1000 ms/fresh throughput change has median
-23.192% for native and
-22.684% for BPF.
Thus delayed state materially erodes this policy's benefit on both
implementations. These three-block results do not establish universal
stale-state tolerance or a tight equivalence margin. Small observed
native/BPF differences must not be presented as a novel policy gain.
All workload result files record zero mismatches; no new correctness gate
was used to admit these performance measurements.

Phase-aligned snapshot age, wrong-phase decisions and UVM-event interpretation
are still being computed by a separate local-model offline analyzer; this
report does not yet attribute the slowdown to a specific migration/thrashing
counter. The measured end-to-end performance itself is complete.

## Execution, restoration and raw records

The runtime held the existing GPU and struct-ops ownership locks, temporarily
loaded the compatible module and ran:

```sh
python3 -B workloads/stale-state-575/live_runner.py execute-performance \
  --output workloads/stale-state-575/raw/stale-state-575-performance-gds-20260907-01 \
  --inherited-lease-fds 11 12
```

Run through the recorded root lifecycle wrapper, not on an arbitrary module.
No historical preflight or post-run promotion validator was required.
The campaign's `validated: false` records this intentionally skipped old
validator path; completion and ordinary execution errors are separate fields.
The wrapper restores the exact saved module path and retained module
parameters. The new interfaces were only loaded for this campaign.

Two earlier startup attempts are retained: the first module unload met a
transient in-use reference after detaching the owned GDS loader; the second
could not open a user-owned lock file for root truncation. Neither ran a cell.
The third used read-only lock descriptors and completed the whole matrix.
No completed performance cell was rerun.

[Raw campaign](raw/stale-state-575-performance-gds-20260907-01/) includes every
workload result, execution status, diagnostics, telemetry, campaign and all
three lifecycle logs. Large decision and observer JSONL files are committed
as complete `.jsonl.gz` copies, with uncompressed originals also retained
locally. Use `gzip -dk FILE.jsonl.gz` in a fresh checkout to expand them.
No traces were sampled or trimmed to reduce archive size.
