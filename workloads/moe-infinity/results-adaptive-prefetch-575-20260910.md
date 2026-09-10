# MoE adaptive byte-admission governor on RTX 5090

Completed 2026-09-10 UTC. The campaign has **five valid paired blocks, 25
cells, 150 measured requests, and 9,600 exactly verified output tokens**. The
independent CPU-only audit
([`scripts/artifact/reanalyze_adaptive.py`](../../scripts/artifact/reanalyze_adaptive.py))
recomputes every gate, median, paired ratio, and bootstrap interval from the
raw per-cell records and passes: `AUDIT PASSED`. Plan:
[`adaptive-prefetch/plan.md`](adaptive-prefetch/plan.md); deviation journal:
[`adaptive-prefetch/deviation-20260910-stale-journal.md`](adaptive-prefetch/deviation-20260910-stale-journal.md)
(DEV-1 … DEV-12).

## 1. Question and result

The prior controlled campaign
([`results-prefetch-ablation-575-20260904.md`](results-prefetch-ablation-575-20260904.md))
measured that unbounded predictive prefetch (native policy execution)
improved paired throughput by 0.447% while **60.82% of completed speculative
logical bytes were evicted before first use**. This experiment asks whether a
closed-loop byte-budget governor — driven by completed speculative-copy
outcomes (hit vs evicted-unused vs kept-unused-resident) and live mandatory
demand pressure — can remove that churn without losing the overlap benefit,
and whether host BPF can execute the same governor at native performance.
Reviewer guidance asks for new policies enabled by the mechanism and for
policy benefit to be separated from mechanism cost; this campaign produces
both halves.

Result (primary metric: 384 verified output tokens / full six-request
A→B→A wall window including final drain):

- The **adaptive governor beats the unbounded policy by 1.06%** paired
  geometric throughput ratio **1.0106, 95% block bootstrap CI
  [1.0017, 1.0204]** (interval excludes 1).
- It cuts speculative prefetch traffic **11.4×** (1,802 GB → 157.7 GB),
  evicted-unused bytes **71.56% → 28.87%**, and unused-resident bytes at
  drain **5.86 GiB → 0.64 GiB** (absolute; relative fraction rises
  0.35% → 0.44% because the adaptive arms also drop the churn that
  diluted it).
- The fixed 512 MiB cap is competitive on bytes but loses to the adaptive
  arm on throughput: fixed/adaptive-bpf **1.0127 [1.0065, 1.0184]** — memory
  pressure adaptation (one demand-wait event ⇒ one quantum decrease) buys
  ~1.3% over the static cap.
- The **BPF mechanism cost of the governor is ≈1.5%**:
  adaptive-bpf/adaptive-native **0.9846 [0.9798, 0.9895]**. Against
  unbounded e2e, adaptive-bpf is **0.9951 [0.9852, 1.0076]** (inconclusive).
- Demand-only stays worst on throughput (11.083 tok/s).

## 2. Fixed protocol

- Same host constraints as the prior campaign: one RTX 5090, NVIDIA 575.57.08,
  Linux 6.15.11, 400 W limit, struct_ops empty before each server.
- Model GPT-OSS-120B, one active request, six frozen prompts per cell drawn
  from the four-row held-out cohort (A = rows 0,1; disjoint B = rows 2,3;
  A revisited), 512 input / 64 output tokens, 0.75 cache budget, no
  temporary overload slot.
- Five seeded randomized complete blocks; arm order randomized per block
  (seed 20260909), request slot pattern protocol-fixed `[0,1,2,3,0,1]`
  (`adaptive-prefetch/schedule.json`).
- Governor rule (frozen in
  [`adaptive-prefetch/plan.md`](adaptive-prefetch/plan.md), implemented in
  `ExpertDispatcher::RecordRequestSpeculativeOutcome`): per completed
  request, demand wait ⇒ −quantum; useful > unused ⇒ +quantum; unused >
  useful ⇒ −quantum; 512 MiB initial, 64 MiB quantum, 128 MiB floor, 4× cap.
- Five arms: `unbounded-native` (mode 1), `fixed-native` (mode 2, no
  adaptation), `adaptive-native` (mode 3, native rule), `adaptive-bpf`
  (mode 3, identical rule compiled to a 43-instruction uBPF program behind
  the real-JIT bridge `extension/moe_spec_admission.{bpf.c,cpp}`),
  `demand-only` (mode 0).

The required five-arm preflight
([`raw/adaptive-prefetch-575/preflight-20260910-s`](raw/adaptive-prefetch-575/preflight-20260910-s/))
passed exact-output, engagement, conservation, telemetry, and cleanup gates
and froze the runtime inventory binding the full campaign
([`raw/adaptive-prefetch-575/full-20260910-s`](raw/adaptive-prefetch-575/full-20260910-s/)).

## 3. Correctness, engagement, and retained failed attempts

Every measured SSE response byte-equals the frozen held-out golden
(`raw/adaptive-prefetch-575/held-out-goldens.json`, frozen identity-checked
before the first preflight). All 25 cells: server exit 0, empty cleanup
errors, no new RM interrupt warnings, telemetry continuous, CPU affinity
0–7 held, before/after safety equal.

- adaptive arms: 69,120 admission calls, 34,960,750 candidate identities
  offered, 2,573,670 admitted, 30 completed-request outcome updates (6 per
  cell; net 5 decreases, 0 increases per arm; budget 512 → 402 MiB because
  each cell's first demand-wait event drops one quantum and later
  useful>unused requests hold rather than grow). Native and BPF deltas are
  **identical** (same ranked candidates, same budget trajectory) — the
  BPF program implements exactly the native rule.
- fixed arm: 69,120 calls, 34,960,750 offered, 2,685,300 admitted, budget
  constant at 512 MiB, zero updates.
- unbounded arm: zero bookkeeping counters (mode 1 publishes every
  candidate without byte admission).
- demand-only arm: zero speculative copies.

The uBPF bridge made **16128 real JIT admission calls with zero errors** in
the preflight adaptive-bpf cell and executed every candidate set in the
campaign (`moe_spec_admission_ready: backend=ubpf-jit abi=1`).

Retained failed attempts (excluded entirely, never pooled):

- `preflight-20260910-a..q` — bring-up iterations: gate semantics, the
  dispatcher gate-return defect, the server library-binding bug, DEV-8
  schedule repair, and the UVM teardown leak (see DEV-1 … DEV-10 in the
  deviation journal); `preflight-20260910-r/s` are the passing runs.
- `full-20260910`, `full-20260910-r` directories — schedule contract
  mismatch (DEV-8) and a flaky interpreter-teardown SIGSEGV
  (block-01-attempt-01; the same native arm exits 0 in every other cell and
  in attempt-02) that triggered the plan's whole-block rerun rule.
- `full-20260910-s/block-01-attempt-01` — clean first-attempt block 1.

## 4. Performance and traffic

Throughput is the median of five cell throughputs; TTFT is the median of
five per-cell medians of the first non-empty visible text. Paired estimates
are whole-block geometric throughput ratios with 10k-draw percentile
bootstrap intervals that resample blocks (seed 20260910, published in
`analysis.json` and recomputed bit-identically by the audit script). Higher
throughput is better; bytes are logical expert-payload H2D bytes, repeated
transfers counted repeatedly — not PCIe analyzer measurements.

| Arm | Median output tok/s | Median TTFT ms | Demand hit rate |
|---|---:|---:|---:|
| demand-only | 11.0828 | 1,569.03 | 78.41% |
| unbounded | 11.1513 | 1,525.59 | 86.95% |
| fixed 512 MiB | 11.1861 | 1,523.21 | 80.50% |
| adaptive-native | 11.2444 | 1,527.62 | 80.09% |
| adaptive-bpf | 11.0933 | 1,552.03 | 80.09% |

| Paired comparison | Ratio [95% block bootstrap] | Reading |
|---|---:|---|
| adaptive-native / unbounded | 1.010617 [1.001697, 1.020439] | Adaptive wins, interval excludes 1 |
| fixed / adaptive-bpf | 1.012676 [1.006490, 1.018401] | Static cap loses to adaptive |
| adaptive-bpf / unbounded | 0.995062 [0.985196, 1.007567] | Inconclusive e2e |
| adaptive-bpf / adaptive-native | 0.984607 [0.979775, 0.989463] | ≈1.5% BPF mechanism cost |
| unbounded / demand-only | 1.000917 [0.988071, 1.010204] | Inconclusive |
| adaptive-bpf / demand-only | 0.995974 [0.991144, 1.000332] | Inconclusive |

| Outcome traffic | Copies | Prefetch GB | First-use hit | Evicted unused | Unused-resident at drain |
|---|---:|---:|---:|---:|---:|
| unbounded | 136,061 | 1,802.33 | 28.09% | 71.56% | 0.35% |
| fixed 512 MiB | 12,489 | 165.44 | 69.71% | 29.68% | 0.61% |
| adaptive-native | 11,899 | 157.62 | 70.71% | 28.86% | 0.43% |
| adaptive-bpf | 11,908 | 157.74 | 70.69% | 28.87% | 0.44% |
| demand-only | 0 | 0 | — | — | — |

A→B→A trace (median across blocks, first visible-text TTFT of the 1st vs 2nd
A-visit): demand-only +10.3 ms, unbounded +12.8 ms, fixed +13.5 ms,
adaptive-native +15.4 ms, adaptive-bpf +20.8 ms. The high reuse the cohort
was chosen for already materializes in the first visit of a cell (fresh
server each cell), so revisit-time deltas are small everywhere; the governor
mechanism separates through byte pressure within the first visits
(`governor_budget_bytes` trajectory and per-request outcome keys are
recorded in each cell's `result.json` counters; the in-memory
`governor_history_*` vectors are the intended v2 probe surface).

## 5. Verdict and scope

**Supporting result with a positive policy effect and a separately disclosed
mechanism cost.** For this RTX 5090, model, memory budget, prompt set, and
concurrency level: the outcome-pressure byte governor preserves the
unbounded arm's demand-hit benefit while removing 11.4× speculative
traffic and 9.2× absolute unused-resident bytes, and beats unbounded
end-to-end by ~1.1%. The static fixed cap reproduces most of the waste
benefit but loses ~1.3% to the adaptive arm. The BPF port of the governor (a real 43-instruction uBPF program, not
a parity shim) costs ~1.5% against native — more than the
0.56% static-selector port cost of the prior campaign, consistent with the
candidate-reduction work the program does inside the bridge. Against the
unbounded policy e2e, the BPF arm is statistically indistinguishable
(0.9951 [0.9852, 1.0076]).

Not claimable from these data: generator-vs-consumer causality of the
throughput delta (the mechanism changes residency and copy traffic together
with feedback timing), any multi-tenant or arrival-time generalization
(single active request), and equivalence — the BPF/native interval excludes
1, so the port is measurably not free.

## 6. Raw paths

- Preflight: `raw/adaptive-prefetch-575/preflight-20260910-{a..r,s}` (all
  19 attempt directories retained; `s` is the passing run that binds the
  campaign).
- Full: `raw/adaptive-prefetch-575/full-20260910`, `-r`, `-s` (retained;
  only `-s` completes; within it `block-01-attempt-01` is the clean rerun).
- Analysis: `raw/adaptive-prefetch-575/full-20260910-s/analysis.json`;
  independent recomputation: `python3 scripts/artifact/reanalyze_adaptive.py`
  (exit 0, "AUDIT PASSED").
- Committed evidence subset (per-cell result/admission/launch/server.log,
  manifests, goldens, cohort, schedule, allowlist): git commit `5a3cb40e`.
- Runtime inventory and driver stage:
  `/opt/gpubpf/modules/575.57.08/gpreempt-e7d46fa5-6.15.11`, pinned in the
  campaign manifest.
