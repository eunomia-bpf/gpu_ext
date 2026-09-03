# MoE-Infinity activation-aware policy: explicit paper-v3 port

Status: **earlier three-mode enhanced real canaries passed** on driver stage `849ea75d`,
including finite numerics, two full nonstream requests and one full SSE request
per mode, exact same-frontend outputs, and actual policy engagement. See the
[three-mode preflight record](raw/paper-v3-575/preflight-849ea75d/README.md).
The subsequent exact-ABI bulk-packing optimization also passed its own
[enhanced BPF canary](raw/paper-v3-575/canary-bpf-packed-849ea75d-01/README.md).
The first repeated comparison was interrupted by a reboot and has **zero valid
paired blocks**; [the recovery audit](recovery-20260903-013741.md) rejects the
damaged native raw responses. Prediction-set protection and stale-task epochs
were subsequently added to the common executor as described below. That update
was rebuilt and passed fresh [three-mode enhanced real canaries](raw/paper-v3-575/preflight-protected-849ea75d/README.md)
on 2026-09-03: finite numerics, exact nonstream/SSE outputs, actual prediction-set
protection, stale-task rejection, three JIT selectors and clean teardown. Their
retained raw responses, telemetry and final JIT logs also passed independent
CPU-only audit. A fresh five-block three-mode timed campaign is now running;
the earlier interrupted results are not reused.
There is **no paired performance or completed-reproduction claim yet**.
This replaces the stopped generic-policy
campaign, whose diagnostic data remain in [results-575.md](results-575.md).

The target is [arXiv:2401.14361v3](https://arxiv.org/abs/2401.14361), dated
2025-03-12, specifically sections 4.3–4.7, Algorithm 1, and appendix B.1.
The downloaded source is
`docs/paper-material/policy-expressibility-papers/20-2024-moe-infinity.pdf`.
GPT-OSS and this serving integration are not the authors' original evaluated
hardware/model combination, so this is an algorithm reimplementation and a
same-frontend deployment comparison, not reproduction of their original numbers.

## Three distinct configurations

| Configuration | Activation algorithm | Decision executor |
|---|---|---|
| `native-off` | Current pinned GPT-OSS default, prediction disabled | Original dispatcher visit-count eviction |
| `paper-native` | Explicit paper-v3 port described below | Native Python/C++ selectors |
| `paper-bpf` | Identical port and float64 feature inputs | Real userspace BPF JIT selectors |

All three use the same API, model weights, routing, Triton kernels, repaired
expert row chunking, deterministic accumulation, and lossless SSE transport.
`paper-bpf` is userspace BPF through the bpftime/ubpf JIT substrate, not the
old kernel-UVM stride/LFU program. It must not be labelled that program.

The pinned author source is `b766f8f1f6379fac6cd23594713ba6f4c7650ad9`.
Its OpenAI initializer supplies no prefetch flags; `ArcherConfig` defaults
`prefetch`, `speculative_prefetch`, and overlap to false. The GPT-OSS router
calls `DistributedExpertExecutor.dispatch_local`, which dispatches the actual
routed expert union. Predictor/tracer objects are constructed but no
`ExpertPredictor.predict` call is reached in this serving path. The newer
router-logit mean/top-2 and DFlash route-ahead paths are different algorithms.

Simply enabling the dormant tracer is not a faithful paper-v3 reproduction:
its `find_most_similar` averages per-layer cosine and masks the observed prefix,
while section 4.5 specifies flattened-vector cosine. Its full-collection
replacement uses access counts, while appendix B.1 replaces the **closest**
old matrix to preserve diversity. Its predictor also uses an `L+1` decay
denominator rather than the paper's `L`. These differences are retained as
source evidence, not silently reinterpreted as the paper algorithm.

## Algorithm and explicitly fixed conventions

The original paper omits some numerical and scheduling details. The following
conventions are identical in native and BPF modes and are not claimed to be
recovered author implementation choices:

- An iEAM is reset for each real model forward. Router counts update exactly
  one row in ascending zero-based layer order. Request-level matrices sum
  completed iEAMs separately for prefill and decode. Both nonempty phase matrices
  enter one capacity-1,000 EAMC only on successful request completion; aborted
  requests do not train it. GPT-OSS top-k routing counts every selected expert,
  so row totals are `k × tokens`, unlike the paper's single-route illustration.
- Matching uses cosine of full flattened vectors, with unobserved iEAM rows
  zero. Shared float64 tensor math produces cosine features; the selector
  returns every exact maximum-cosine tie in original collection order. Matched
  raw matrices are summed then normalized per row. Zero-norm cosine is zero.
- At full capacity, the first closest old entry is replaced with the completed
  incoming rEAM (appendix B.1), not the least-used entry.
- Reuse scores implement Algorithm 1 with epsilon `1e-8` and global layer
  factor `1 - layer/L`; zero-probability rows have score zero. Prefetch scores
  use the future rows only and factor `1 - (future-current)/L` (section 4.5).
- An empty EAMC supplies a neutral uniform reuse prior and submits no learned
  prefetch. This avoids inventing activation evidence for the first request.
  Equal eviction scores retain the first safe entry in the existing unordered
  cache traversal. Positive prefetch scores are stably ordered descending in
  original layer-major order. No top-k or stride/LFU substitution is used.
- The initial implementation explicitly supports a single active sequence and
  non-speculative execution. It refuses batched/mixed/speculative requests
  instead of mixing traces. The benchmark retains 512-input/64-output requests,
  eight measured prompts, and a separate excluded warm-up.

## What executes in BPF

Shared native code does routing/count extraction, cosine and probability
arithmetic, matrix accumulation, physical transfers, CUDA events and kernels.
In `paper-bpf`, the BPF programs actually choose:

1. All nearest EAMC matches and the first diversity-replacement entry from
   unsorted float64 cosine features.
2. Positive expert-prefetch candidates and their stable descending order from
   unsorted float64 scores, including the filtering step.
3. The minimum-score eviction victim from the original cache order, including
   node-present, CUDA-resident, no-pending-demand and execution-idle checks.

The bridge never sorts, prefilters, runs a native fallback or replaces returned
decisions. `MOE_REVISION_VERIFY=1` additionally runs a same-snapshot native
oracle for correctness runs and fails on any mismatch; performance runs will
disable this redundant oracle and report actual BPF invocation counters.

## Same-owner prefetch execution

The current old prefetch scheduler and GPT-OSS dispatcher maintain separate
cache bookkeeping. The port therefore uses one dispatcher fetch worker and one
cache membership/byte ledger. Its demand queue has priority over the replaceable
background prediction queue. Only unstarted prefetch work is discarded on a
new prediction; an in-flight transfer completes before its node becomes idle.
No already-routed demand is cancelled. Prefetch never waits for an unavailable
safe victim, and cannot evict a pending/executing expert. Scores affect cache
replacement in both prefill and decode; the upstream prefill-only overload
shortcut remains only in `native-off`.

The shared executor additionally mirrors the pinned upstream prefetcher's
prediction-set protection: `memory/expert_prefetcher.py` registers every positive
predicted candidate before enqueueing, and `task_scheduler.cpp` excludes those
residents from speculative eviction. Our native and BPF arms now apply that same
restriction before building the scored selector snapshot. Demand does **not**
apply this set restriction: an oversized prediction cannot prevent a required
expert from being fetched. If all safe residents are protected, the speculative
candidate is skipped. There is no new candidate-score versus victim-score
threshold, top-k budget, or changes to the three BPF selection programs.

This protection is a common **executor eligibility constraint**, not a fourth
BPF policy or a claim that BPF performs every runtime decision. The selector
bridge itself still neither sorts nor filters its input. The paper's Algorithm 1
defines demand eviction; page 6 explicitly omits detailed prefetch implementation.
Accordingly this scheduling detail is attributed to the pinned source, not to an
unstated paper requirement. The previous implementation lacked this protection,
allowing a later low-ranked speculative candidate to evict an earlier, unused
prefetch from the same prediction; its retained observations are not mixed with
new measurements.

Prediction updates and drains invalidate the previous epoch. Each background
item carries its publishing epoch. The worker checks it before claiming a node,
again when committing a selected victim (also rechecking the protection set),
and immediately before physically issuing its copy. The final check and issue
share the publisher's mutex, closing the check-to-issue race. An already-issued
copy completes and releases its node even after the epoch changes; its event wait
does not hold the publisher mutex. A stale task may have already evicted an
unprotected victim while its epoch was current, but cannot start a new copy after
invalidation. Demand never carries a speculative epoch.

Prefetch candidates are enqueued in the selected order, not issued through the
dormant prefetch engine. H2D uses the same native `SetDevice`, stream and event
pool; event completion precedes publishing IDLE. The kernel math is unchanged.
Counters expose submitted candidates, completed transfers, first-use hits,
unused-prefetch evictions and bytes, unused residents, actual scored selections,
BPF calls and mismatches. Additional counters expose protection-set resident
skips, stale discarded items, unavailable victims, copy starts, selected-victim
recheck rejections, current epoch and protected-candidate count. An unused
resident at shutdown is not a completed wasted eviction. These are necessary
attribution evidence, not performance
results by themselves.

The canary requires all seven new protection/epoch/copy counters from the real
loaded store. Native-off must keep them all zero. Both paper arms must show a
positive prediction epoch and protected-resident skip count, no remaining
protected candidates after drain, and equal issued/completed copy counts. An old
store binary without these fields is rejected even when its numerical outputs
match. Per-snapshot protected-resident skips are accumulated locally and then
added once to their counter; no per-resident atomic operation is introduced.
Formal timing additionally requires positive protection/epoch increments during
the measured requests, so excluded warmup cannot supply the engagement evidence.
The protection counter is workload coverage, not a universal correctness rule:
it records traversal of a protected resident, not proof that an otherwise-safe
victim would have been evicted. Selector calls also differ from actual successful
evictions. Transfer completion is checked after the real drain endpoint returns,
not inferred from equal counters in an arbitrary live snapshot.

The protection helper is exercised separately by
`test_revision_prediction_set.cpp`: protection versus demand progress, stale
claim, changed protection after victim selection, stale-before-copy rejection,
completion after copy issue, queue replacement, drain, duplicate candidates and
full 64-bit identities. `test_revision_prefetch_source.py` checks the live-source
wiring without compiling or loading CUDA. Passing its static checks is not
evidence that the new store binary has been built or run.

Known non-measured accessor limitation: the upstream `GetCacheOccupancyBytes()`
iterates cache membership without a shared writer lock. It must not be used for
concurrent live occupancy observations during this port's background work. The
revision endpoint instead uses atomic `get_cache_counts`, and the comparison's
state snapshots occur after drain; neither depends on that unsafe accessor.
This focused change does not restructure the upstream/native-off cache pipeline.

The first `paper-bpf` canary on the coordinator's `e7d46fa5` driver stage passed
at 2026-09-03 01:10 UTC. Four expert row sizes and four accumulation arrival
orders had zero numerical error. Both 512-input/64-output responses exactly
matched the retained same-frontend MoE goldens. BPF executed 8,919 scored
evictions, 2,304 EAMC matches, and 4,608 ranks; same-snapshot verification had
zero mismatches in every selector. The worker completed 2,021 prefetches
(26,771,103,744 bytes), with 1,100 first-use hits, 890 unused-prefetch evictions
(11,789,352,960 bytes), and 31 unused residents. Request latencies were 9.255
and 7.675 seconds **with shadow verification enabled**, not performance samples.
The server exited normally with code zero, and post-cleanup reported GPU 2 MiB,
0% utilization, UVM refcount zero, and empty struct_ops. Full evidence is in
`raw/paper-v3-575/canary-bpf-01/`; this is not sufficient to claim all three
configurations or five repeated blocks completed.

Completed-request and explicit drain paths discard queued speculation and wait
for already-started H2D to finish. The queue tracks in-flight work under the
same mutex as Pop, so a popped-but-not-started transfer cannot escape the drain.
Performance accounting must include the final drain tail in whole-cell elapsed
time, alongside separately labelled per-request token latency.

## CPU verification and build

```bash
CUDA_VISIBLE_DEVICES='' taskset -c 8-15 .venv/bin/python -m unittest -v \
  test_paper_policy.py test_paper_server.py
/usr/bin/g++-13 -std=c++17 -pthread -Wall -Wextra -Werror \
  test_revision_fetch_queue.cpp -o /tmp/moe-paper-fetch-queue-test
/tmp/moe-paper-fetch-queue-test
taskset -c 8-15 .venv/bin/python build_paper_store.py
```

The extension build uses the existing CUDA-12.9/sm_120 environment and adds
`CPLUS_INCLUDE_PATH=<gpu_ext>/extension` for the explicit selector ABI header.
GPU runs remain serialized by the experiment coordinator; CPU selector tests
are not a substitute for finite numerical canaries, real prefetch engagement,
exact same-frontend output checks or repeated paired performance measurements.

The canary command, only after the coordinator gives the GPU slot, is:

```bash
.venv/bin/python run_paper_policy.py canary --mode paper-bpf \
  --output raw/paper-v3-575/canary-bpf-01
```

The current canary revision runs the finite expert numerical check, then two
full nonstream 512+64 requests and one additional full SSE parity request with
same-snapshot native/BPF verification enabled. All outputs must match the
retained same-frontend MoE goldens; the stream must contain 65 frames and both
engine and metrics counters must increase by exactly 64 tokens. The retained
first canary above predates this extra SSE request and is not relabelled as
having run it. All three modes subsequently passed the enhanced canary on
`849ea75d`; that diagnostic is still explicitly not a paired performance result.
