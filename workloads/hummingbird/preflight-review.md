# Hummingbird preflight: independent qualification review

2026-09-03 UTC. **Technical qualification passes: 10/10 closed cells.** The
unchanged [50-cell plan](plan.md) may proceed when the coordinator assigns an
exclusive GPU window. This is not a performance win or completed reproduction;
the single-round performance remains unfavorable to the BPF idle scheduler
against fixed GPreempt. Performance direction was not a qualification gate.

## Evidence checked

Source: [raw/preflight-575-01](raw/preflight-575-01). Independently decoded each
`config.json` and `client.log` with `analyze_study.parse_client`, without calling
the producer's measurement function. Checked the seeded five-arm/two-scenario
order, 10-second common windows, FIFO arrival prefixes, original service sample
counts, full-output numerical counts, saved request/arrival reports, per-cell
results and completed-cell inventory. No cell was excluded or rerun.

All 5,460 offered LC requests completed and were numerically verified inside
their windows; no LC backlog or conditional p99. Every BE cell retained its one
verified late completion, excluded from window goodput. All recorded maximum
absolute output errors were zero. The fixed SLO was 1,811,879 ns; attainment
uses all offered LC requests and excludes deadline-late completions.

Reused the existing read-only engagement/safety validators, rather than claiming
an independently implemented CUDA validator. Actual C/BPF mode, profile paths,
JIT counts, complete-model CTA/copy counts, owned cleanup, raw GPU telemetry and
unchanged binary size/mtime inventories passed. Native retained its observed
stream priorities; fixed retained its protected host-mapped GPreempt path.
Both idle arms and the unsplit/ungated control had two distinct owned contexts,
1,000,000 us timeslices and priority-zero streams. No runtime source was changed.

The shared selected profile enables output bubbles only. Actual input-small
launches were zero, as required by that disabled setting. Both C and BPF really
executed splits and output-small launches:

| Arrival prefix | C splits / output-small | BPF splits / output-small | Actual BPF JIT decisions |
| --- | ---: | ---: | ---: |
| Periodic | 672 / 439 | 481 / 323 | 32,304,292 |
| BurstGPT | 44 / 30 | 58 / 36 | 30,319,590 |

C reported zero JIT decisions; BPF's JIT count equaled its total decisions.
All four idle cells observed the configured one-launch in-flight bound.

## Single-round observations, not formal estimates

| Arrival prefix | Arm | LC response p99 (ms) | LC SLO attainment | BE window goodput (req/s) |
| --- | --- | ---: | ---: | ---: |
| Periodic | Native | 1.973 | 75.5% | 180.2 |
| Periodic | Fixed GPreempt | 1.780 | 99.2% | 163.6 |
| Periodic | Equal-timeslice control | 6.573 | 7.4% | 172.9 |
| Periodic | Idle C | 1.847 | 98.0% | 133.3 |
| Periodic | Idle BPF | 1.939 | 97.0% | 133.1 |
| BurstGPT | Native | 3.296 | 84.8% | 189.4 |
| BurstGPT | Fixed GPreempt | 3.286 | 96.7% | 191.3 |
| BurstGPT | Equal-timeslice control | 9.838 | 6.5% | 190.7 |
| BurstGPT | Idle C | 3.264 | 95.7% | 151.3 |
| BurstGPT | Idle BPF | 4.587 | 91.3% | 150.9 |

Each periodic cell offered 1,000 LC requests. The unscaled first ten seconds of
the fixed BurstGPT trace offered only 92, so its nearest-rank p99 is its maximum
observed response. This prefix is not the full 6,000-request/60-second workload;
do not extrapolate its performance or attach five-block confidence intervals.

BPF BE goodput was 0.15%/0.26% below C but 18.64%/21.12% below fixed GPreempt
(periodic/BurstGPT). This points toward costs shared by the idle execution path,
not a large BPF-specific BE penalty, but does not establish their cause. The
single BPF LC p99 was worse than C in both prefixes; no equivalence claim follows.

The common executor recorded 25.7–29.3 million yields while an LP event was
not ready, 1.72–1.98 seconds inside event queries, and 3.16–3.54 seconds in those
yields per cell; policy calls accumulated 0.362–0.388 seconds. These host-time
counters overlap GPU execution. An event-not-ready yield may also coincide with
HP protection or an unexpired tick. They cannot be added to latency or apportioned
causally to the one-launch fence. The fence, polling, tick pacing, launch path
and protection are shared candidates, not isolated treatments in this preflight.

The correct label remains **a partial, paper-described scheduling-component
port**: real splitting, output-bubble admission and consolidation are engaged;
input-bubble filling did not qualify, and the stronger completion fence and
other exclusions in the plan remain. Only the full paired study can evaluate
the predefined throughput/protection criteria with uncertainty.
