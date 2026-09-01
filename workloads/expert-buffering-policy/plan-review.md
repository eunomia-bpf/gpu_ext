# Independent Plan Review

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
