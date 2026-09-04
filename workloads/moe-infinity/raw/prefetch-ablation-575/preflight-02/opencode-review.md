# OpenCode review of predictive-prefetch preflight 02

Date: 2026-09-03

Review session: `ses_f94ebee90ffeHhGwjAwHVGehBn`

Reviewer: `opencode/ling-3.0-flash-fin-free`, with snapshots and sharing
disabled, every permission denied, and write, edit, shell, network-fetch, and
subtask tools disabled. The first response was followed in the same session by
a factual-correction request; the corrected response is the verdict recorded
below.

## Read-only re-audit

The repository validator reread and accepted the preflight. The focused
CPU-only test suite passed all 12 tests. A separate read-only parser then
checked the raw files without trusting the top-level `passed` fields:

- the root result contains exactly the four scheduled arms, and every embedded
  cell equals its per-arm stored result;
- the launch records select the exact native/BPF and prefetch-off/on
  combinations under the same runtime inventory, 25,251,987,456-byte cache
  budget, CPU affinity, model, and request configuration;
- each raw SSE contains 64 JSON completion events and one `DONE` marker, and
  reconstructs the retained prior same-frontend golden exactly;
- both prefetch-off arms have zero in every forbidden speculative-work and
  residual-state field; and
- all four cells have zero temporary-slot use, zero eviction mismatch, a clean
  server exit, an empty cleanup-error list, unchanged RM warnings, and clean
  post-run policy state.

The measured-window mechanism counters are:

| Arm | Prefetch copies | First-use hits | BPF rank calls | BPF match calls | BPF demand evictions | BPF prefetch evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPF, prefetch on | 2,012 | 1,195 | 2,304 | 2,304 | 1,768 | 2,012 |
| BPF, prefetch off | 0 | 0 | 2,304 | 2,304 | 2,675 | 0 |
| Native, prefetch off | 0 | 0 | 0 | 0 | 0 | 0 |
| Native, prefetch on | 1,964 | 1,163 | 0 | 0 | 0 | 0 |

The BPF logs independently show all three `ubpf-jit` programs becoming ready
and terminating with zero policy errors; native logs contain no such load.
For BPF-on, completed prefetches conserve as
`2,012 = 1,195 hit + 773 wasted + 44 resident at drain`, and bytes conserve as
`26,651,885,568 = 15,829,524,480 + 10,239,516,672 + 582,844,416`. For
native-on, the corresponding equations are
`1,964 = 1,163 + 768 + 33` and
`26,016,055,296 = 15,405,637,632 + 10,173,284,352 + 437,133,312`.
Demand prefill/decode hit-plus-miss counts, demand/prefetch eviction counts,
and BPF demand/prefetch call counts also conserve in every applicable arm.

Each raw telemetry file has 31 data samples. Recomputing the sample count,
clock range, peak memory and temperature, and mean power reproduces the stored
summary. No hardware or thermal slowdown reason is active and every cell is
recorded as unthrottled. The BPF-off cell has two allowed software power-cap
samples under the common fixed 400 W cap; this is recorded rather than hidden.
Pre- and post-run records contain no XID, kernel/journal anomaly, active compute
process, UVM reference, or lingering struct-ops map/link.

## Independent verdict and limits

After correction, the OpenCode reviewer returned `VERDICT: READY`. It found
the raw evidence credible enough to admit the preregistered full campaign. Its
important limitations were that the retained correctness oracle covers only
one prompt here, the producer and checked-in auditor share some code, telemetry
is an environment/consistency gate rather than causal attribution, and the
host wait counters are not GPU-kernel stall measurements. The separate raw
reparse above reduces, but does not erase, the shared-code limitation.

Run status: **PREFLIGHT PASSED**; dependency and path correctness only.

Tested hypothesis: **The performance hypothesis was not tested**; this run only
shows that the four causal arms execute with the intended mechanisms and gates.

Research value: **Readiness evidence** for the full factorial campaign.

Paper impact: **No comparative-performance claim** follows from this run;
single-request timing fields must not be used as throughput, latency, speedup,
or equivalence evidence.

Next decision: **Proceed to the fixed five-block run**, then require all 20
cells, 120 exact-checked requests, 7,680 verified output tokens, raw review,
engagement gates, and paired analysis before making a result claim.

VERDICT: READY
