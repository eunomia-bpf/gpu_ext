# Independent Plan Review

## Timing result-readiness review

Date: 2026-09-01

Verdict: **APPROVE**. The reviewer authorized all five frozen paired timing
blocks after the proposal-6 correctness result passed.

Conditions retained in the timing runner are the frozen hot set, action table,
prompt and configuration orders, per-block snapshot formula, API/token/UTF-8/
server-error gates, observation-only zero-reorder engagement, protection-mode
positive hot/cold/access engagement with zero cold-head and setter failures,
complete 32-layer context-route coverage, ownership cleanup, and thermal
telemetry. UVM Tools events are omitted from timing validity and policy
inference because the correctness run established only a zero-event diagnostic.
Final analysis must report observation/plain mechanism overhead separately from
protection/observation policy effect. The correctness-stage 0.00384% activation
difference is a null result, not evidence of a hot-residency benefit.

## Proposal 6 framework-route review

Date: 2026-09-01

Initial verdict: **BLOCK**. Although the context gate required 32 routed
layers, it did not reject partial per-graph coverage.

Final verdict after repair: **APPROVE**.

The shared runner validator now requires exactly 32 routed layers, zero
incomplete graphs, and identical per-layer and total graph counts. The saved
`--n-cpu-moe 32` trace passes with 216 source layouts, 641,958 routes, 1,105
graphs, 1,105 graphs for every routed layer, and zero dropped events. Layers
0--31 are the CPU-streamed layers; layers 32--35 remain device-resident and are
not claimed to emit the streaming marker. The separate `--n-cpu-moe 36`
calibration retains all-layer coverage.

## Proposal 5 observability repair review

Date: 2026-08-31

Initial verdict: **BLOCK**. The reviewer required an implemented per-layout-
block activation map and complete semantic snapshots, synchronization of the
runner with the revised nondeterminism and zero-event gates, and removal of
unobservable transfer/eviction/residency measurement promises.

Final verdict after repair: **APPROVE**.

The approved repair uses a global atomic `u64` activation counter for every
admitted layout index. On request, the loader emits snapshot metadata and all
hot indices including zero values; the runner validates base, block count,
class, index completeness, and monotonicity before computing full and repeated
activation bytes. The runner records output disagreement without claiming
bitwise equivalence, retains zero UVM Tools events only as diagnostics, and no
longer requires a positive completed-eviction count. The plan limits measured
claims to the page-activation proxy and throughput. Build, parser tests, and a
live zero-value snapshot smoke passed. No hash-based evidence is used.

## Proposal 4 repair review

Date: 2026-08-31

Verdict: **APPROVE**.

The narrow independent re-review covered the repaired plan after the exact
120B cold-head failure and the successful `protect` smoke. It approved
`gpubpf_observe` versus `plain_uvm` as the mechanism-overhead contrast and
`gpubpf_profile_protect` versus `gpubpf_observe` as the policy-effect contrast.
It found the correctness and engagement gates falsifiable, the claim boundary
appropriately limited to a page-granular Expert Buffering analogue, no
unsupported safety or performance claim, and no hash-based evidence gate.

The approval authorizes the four-cell correctness/lifecycle run and leaves full
timing gated on that result.

## Round 1 — Proposal 1

Verdict: **BLOCK**

Fatal findings:

1. The UVM path has no host-visible router IDs, and `gpu_block_access` does
   not fire for resident reuse on the production 610 path. The proposed
   hotness therefore learned misses only, resident selections never entered
   the denominator, and the positive access gate was unreachable. Any cache
   hit/miss claim would require an all-cell real-router selection observation
   plus independently validated residency evidence; otherwise outcomes must
   be described as transfers or no-transfers rather than hits.
2. Activate-only typed reordering cannot refresh an expert selected while it
   is already resident. The proposal was miss-time insertion ordering rather
   than full hot residency. A full claim would require a safe typed actuation
   point for every selection or bounded eviction candidate, including a
   resident-reuse case with no activate event.
3. The proposed `2 of preceding 4` class was not Huang et al.'s policy and
   could protect an inactive historical expert ahead of an expert active in
   the current batch. The plan must freeze inactive-before-active priority and
   a fixed budget, or narrow the claim to an eviction-order analogue. Any
   history-based adaptation must be a separate ablation.

The reviewer found the paper value, matched-control intent, framework control,
correctness, ownership, schedule/retry discipline, paper freeze, and no-content-
hash rule otherwise acceptable.

Disposition: proposal 1 is not authorized for implementation or execution.
Proposal 2 must remove the impossible access/hit gates and either provide the
missing router/residency/actuation path or explicitly narrow the claim.

## Round 2 — Proposal 2

Verdict: **BLOCK**

Round 1's semantic defects are closed: the claim is narrowed to a
profile-guided page analogue; access/hit/current-active and `2-of-4` claims are
gone; actuation is activate-only; and the matched policies differ only in the
frozen hot-set lookup. Two execution details remain fatal:

1. `--n-cpu-moe 32` exposes host router IDs for only the first 32 offloaded
   MoE layers and cannot produce the required 36-layer calibration. Use a
   separate setup-only `--n-cpu-moe 36` calibration, retain 32 for the timed
   framework context, and freeze graph/layer identity derivation.
2. “Activated more than once after warm-up” does not define whether the first
   measured activation is included. Freeze the post-warm snapshot and formula.
   The recommended compulsory-allocation-excluding metric is
   `2 MiB * sum(max(0, N_x - 1))` over frozen-hot blocks, where `N_x` is the
   post-snapshot activate count; retain full activation bytes separately.

No other must-fix defect was found.

Disposition: proposal 2 is not authorized. Proposal 3 must repair these two
items and receive a final review before implementation.

## Round 3 — Proposal 3

Verdict: **APPROVE**

- Setup-only `--n-cpu-moe 36` forces all 36 MoE layers through the existing
  host-ID path while the timed framework context remains at 32 layers.
- The exported graph-compute uprobe, post-get/synchronize per-used-ID marker,
  and tensor-base-to-layout join provide an executable complete-layer
  `(graph, layer, expert)` calibration path.
- The warm-up/idle snapshot and
  `2 MiB * sum(max(0, N_x - 1))` formula remove metric ambiguity; full
  post-snapshot activation bytes remain secondary.
- The analogue-only claim, activate-only actuation, matched P/E decision path,
  layout/boundary handling, protection cap, four cells, preflight, schedule,
  ownership, paper freeze, and no-content-hash rule are coherent.

Disposition: proposal 3 is approved for implementation and bounded preflight.
