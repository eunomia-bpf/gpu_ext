# Expert Buffering Section VI: whole-expert policy port

Status (2026-09-03): independent integer selector built; nine CPU tests passed
with 2,131 native and 2,131 actual host-uBPF JIT decisions. The subsequent
[fresh real three-arm GPU preflight](correctness-results-575-02.md) now passes,
including all 27 full logits arrays and 20,182 live-input JIT/native checks.
The 15-cell Section VI performance campaign remains pending. Full attempt 01
was stopped for CPU interference and is [retained but excluded](full-01-abandoned.md);
preflight 02 validates the repaired cleanup helper before a fresh full attempt.
The older 20-cell page-profile experiment is an analogue,
not evidence that this algorithm has been reproduced.

The subsequent [adapter state interface](adapter-step-a.md) now passes 1,432
CPU checks, including 650 additional native/actual-JIT paired decisions. It
provides epoch/cohort/residency and status validation for the C++ worker.
[Step B](adapter-step-b.md) now prepares the private source wiring and passes
61 fake-device control checks plus eight source/cleanup tests. Actual node
locking, copy completion and eviction are wired in the private patch. The
[19-unit offloader build and exact-path import](adapter-build-01.md) now pass;
the adapter now also passes the separate real CUDA preflight linked above.
The same-worker [Step C correctness/performance source](correctness-plan.md)
is now prepared for three exact-logit preflight arms and a separately gated
15-cell timed matrix. Its shadow build, six bridge tests and eight controller
tests pass ([logs](step-c-cpu-01/execution.md)); the three real correctness
cells now pass, but performance is not complete. [OpenCode's completed read-only review](opencode-review.md) reports
no blockers; it does not replace the required GPU execution.

Scope: this independent policy, real host-uBPF path, CPU oracle and private
adapter source; do not change the old page-profile policy or the frozen FineMoE runtime.
The smallest acceptable change implements victim decisions and their tests;
it does not add a new offloader or an experiment framework.

## Paper semantics and explicit port choices

Source: [local complete paper](../../../docs/paper/asplos-27-rebuttal/ref/moe-offloading-2303.06182.pdf),
Section VI.B, PDF p. 8: actual positive token counts after gating define the
current batch's active experts. A miss copies the **whole expert** from CPU
storage to GPU computation. Full caches first evict inactive experts, then
use LIFO. Experts execute serially in increasing expert ID. Its E=4, K=2,
active={1,2,3} example evicts expert 2 when admitting 3, preserving 1.

Section VI.C, PDF p. 9 explicitly says cache capacity is per device and
evaluates miss rates on different MoE layers; it does **not** specify a
cross-layer cache-sharing contract. This single-GPU port chooses a separate
fixed-K cohort per `(device, MoE layer)`, not a global pool that incorrectly
treats another layer's unknown routing as inactive. Each invocation of that
layer's actual gating begins a new epoch; residency persists across invocations.
This is an explicit interpretation, not a claim to recover the original
implementation. Prefill's tokens form one invocation; each decode step forms
another. LIFO means successful insertion order, never last access order.
The text does not specify ordering among several inactive residents: this
port uses LIFO within that class, then lower expert ID on equal insertion
serials. Real successful insertions must have unique monotonic serials.

The original distributed all-to-all/copy overlap and original MT/LM model
performance are outside this RTX 5090/Qwen policy-port claim.

## Implemented CPU boundary

`policy.c` is compiled unchanged to native C and eBPF bytecode. The 1008-byte
ABI contains at most 60 indexed experts, actual token counts, resident/eligible
flags and 64-bit insertion serials. Qwen's local `config.json` at snapshot
`ec052fda178e241c7c443468d2fa1db6618996be`, FineMoE's
`configuration_qwen2_moe.py:138`, and `finemoe_policy.h:6` all specify 60 experts.
The output is hit, admit, evict(index), invalid, or blocked: no silent fallback.
Expert indices are unique by construction; the future host adapter must reject
duplicate tensor/node mappings. Inactive current input and malformed snapshots
are rejected. A fully ineligible full cache returns blocked, not a retry loop.

The bridge loads and JIT-compiles the actual raw BPF program through the
existing CUDA-free uBPF library, counts real calls and checks return/output
consistency and input immutability. This is host-uBPF, not kernel UVM or device
SIMT verification. `test_policy.py` independently orders candidates, replays
batch/cache sequences, and compares **both** native and actual JIT decisions
on identical snapshots. Its epoch/cohort/eligibility commit checks are an
oracle for the future host adapter, not a currently implemented live safeguard.

Build/test (bounded, CPU 17, no CUDA imports or dependency rebuild):

```sh
timeout 30s taskset -c 17 make -C workloads/expert-buffering-policy/section-vi -j1 policy
timeout 30s taskset -c 17 make -C workloads/expert-buffering-policy/section-vi -j1 test
```

Both commands exited 0 on CPU 17; unittest reported 0.095 s. Original logs are
[`cpu-01/build.log`](cpu-01/build.log) and [`cpu-01/test.log`](cpu-01/test.log).
The resulting raw BPF is 744 bytes (93 instructions); `libeb_policy.so` is
58,768 bytes and its dynamic dependencies are only libstdc++, libgcc and libc.
No GPU, torch import, offloader build or bpftime rebuild occurred. The 2,131
same-snapshot comparisons include 1,200 fixed-seed snapshots, serial batch
replays, the paper example, malformed inputs and blocked victims. Epoch/cohort
commit rejection is tested only in the independent host oracle, not a live
driver/adapter. The repository ignores `*.log`; preserve these two raw logs
explicitly when committing, while leaving generated build outputs untracked.

## Minimal remaining real adapter

Reuse `workloads/finemoe/deps/FineMoE-EuroSys26/` only through a separately
reviewed adapter/build; preserve the recorded FineMoE experiment.

1. `finemoe/models/modeling_qwen/modeling_qwen2_moe.py:898`: after actual top-k,
   count `expert_mask` assignments and notify the current layer before the
   existing increasing-ID expert loop. Preserve its zero-token skip and math.
   Disable trajectory prediction equally in all new arms; do not substitute
   predicted probabilities or old `replace_cache_candidates` for activity.
2. `finemoe/runtime/model_offload.py:674` already maps `(layer, expert)` to a
   representative tensor ID. Resolve it to the complete topology node; do not
   assume model layer IDs equal topology stage IDs. Add batch notification to
   the offloader's Python/C++ boundary and cohort/entry-capacity state.
3. `core/prefetch/task_scheduler.cpp:258` is the real eviction decision point;
   its current all-layer `prob * incache_visit_count` order is not Section VI.
   Pass eligible complete-node snapshots to native or JIT selection. Host code
   owns locks, epoch/residency revalidation and resource-specific failure, then
   performs the chosen eviction; BPF must choose the victim, not just approve
   a native preselection. Both entry K and the existing strict byte pool apply.
4. `core/model/model_topology.cpp:61` already synchronizes whole-node copies.
   Update insertion order only after successful admission and residency
   publication. Preserve `AcquireTensor`/`ReleaseTensor` and compute-stream
   synchronization in `core/prefetch/archer_prefetch_handle.cpp:85` so locked or
   executing experts cannot be evicted. Admission serial exhaustion must fail,
   not wrap. A live stale commit must re-snapshot boundedly or fail explicitly.

The [private source patch](adapter-source.patch) now wires these boundaries;
its offloader-extension build now passes, but real GPU correctness gates remain.
The standalone selector and CPU tests alone do not establish that live path.

## Future protocol: not executed or performance-frozen

Proposed matched arms: same-K FIFO baseline (the comparator in Fig. 12),
Section VI native, and identical Section VI host-BPF, all sharing the same
actual-gating notification, sequential executor, byte budget and correctness
gates. FIFO is a new cache baseline, not the old UVM/page-profile result.
Use the existing Qwen 24-layer, top-4, BF16 model and FineMoE's eight held-out
prompts plus one warmup, at most 16 input and exactly 16 generated tokens.
No prediction-history generation is needed for this current-batch policy.

Before timing, rerun all three real arms against the retained original-HF
token/logit oracle at its unchanged tolerances. Check full expert copy/residency
ledgers, K and byte limits, actual miss/eviction engagement, exact native/BPF
decisions on retained snapshots, and clean owned teardown. These GPU checks
remain missing; CPU selection agreement cannot replace them.

Candidate K=16/layer stores 384 sparse experts: from the local Qwen dimensions,
3 * 2048 * 1408 * 2 = 17,301,504 bytes/expert, or 6,643,777,536 sparse bytes
before other allocations. Actual topology/pool occupancy must pass an untimed
admission/engagement check before K, order seed and runtime are frozen. This is
a planning estimate, not a measured resident-memory claim.

Then freeze five randomized paired blocks x three arms = **15 new cells**;
report paired native/BPF overhead and FIFO policy effect separately. None have
run. The old FineMoE 20 cells occupied 27m25s including reloads, suggesting
roughly 20–25 minutes for 15 similar cells, plus correctness/setup; this does
not include unknown adapter implementation/build time or guarantee completion.
Do not mix prior page-profile or FineMoE cells into this campaign.
