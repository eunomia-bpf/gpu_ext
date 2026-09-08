# vLLM preemption victim-selection seam (2026-09-07)

Artifact: `vllm-preemption-seam.patch` (this directory).

## Target

- vLLM 0.27.1, file `vllm/v1/core/sched/scheduler.py`
  (`workloads/lmcache-disk/current-venv/lib/python3.12/site-packages/`).
- Preemption victim selection in `Scheduler.schedule()`, the
  `kv_cache_manager.allocate_slots` failure loop (stock lines 588-621:
  PRIORITY `max(self.running, key=lambda r: (r.priority, r.arrival_time))`
  at 591, FCFS `self.running.pop()` at 615, `_preempt_request` at 617).

## What the patch adds

1. `Scheduler.preempt_victim_callback` instance attribute, default `None`
   (set in `__init__`, no new constructor arg, no new config/ABI).
2. `Scheduler._select_preempt_victim(default_victim)` helper called at the
   victim-selection point, before the victim is removed from
   `self.running`.
3. Stock selection is first computed as `default_victim` exactly as
   upstream (PRIORITY: `max(...)` over `(priority, arrival_time)`; FCFS:
   last running request), then optionally overridden.

Not touched: `_preempt_request` (still performs the actual block free,
encoder-cache free, status/KV state reset, waiting-queue reinsertion),
the PRIORITY key expression, `Request` fields, and the
`reset_prefix_cache` forced-drain loop (a cache-reset mechanism, not a
selection point). No separate scheduler copy, no simulator.

## Callback contract

```python
def pick_victim(running: list[Request], default_victim: Request) -> Request | None:
    ...
```

- Invoked once per preemption attempt inside the `allocate_slots` failure
  loop, with the live `self.running` list and the stock victim choice.
- May return a different *existing member* of `self.running` (checked by
  identity) to preempt instead, or `None` to keep `default_victim`.
- Return values that are not current members of `self.running` are
  ignored (logged warning) and `default_victim` is used.
- Under `SchedulingPolicy.PRIORITY`, an override must share
  `default_victim.priority` (the stock worst-priority class); the
  arrival-time tie-break may change. A different priority class is
  rejected with a warning and `default_victim` is used.
- The callback must not mutate `self.running`; the scheduler performs the
  list removal, cursor adjustment, and rollback around the returned
  victim. A raising callback is not caught (fail-fast).

## Baseline preservation (callback `None`)

Stock victim-selection semantics are retained: PRIORITY uses `list.remove(victim)` and
FCFS uses `list.pop()`; the same-step rollback
(`scheduled_running_reqs`, `token_budget`, `req_to_new_blocks`,
`scheduled_spec_decode_tokens`, `scheduled_encoder_inputs` + encoder
compute budget, `req_index`) runs only when the victim was already
scheduled this step, exactly as upstream.

## Cursor (`req_index`) handling for callback-selected victims

`scheduled_running_reqs` is only a subset of the running requests before
the cursor: requests may have been *skipped* this step (async
placeholders, PP decode cadence, deferred prefill,
`num_new_tokens == 0`), so a victim before `req_index` need not be in
`scheduled_running_reqs`. For a callback-selected victim the patch
records its position before removal and applies exactly one adjustment:

- `victim_index < req_index` -> `req_index -= 1` at removal (the current
  request shifted down one slot); state rollback still happens
  independently iff the victim is in `scheduled_running_reqs`.
- A scheduled callback victim therefore gets the cursor adjustment once
  (at removal) plus the state rollback, never a double decrement.
- The default-victim path keeps the stock cursor semantics untouched
  (including the latent stock quirk where a stock PRIORITY victim that
  was skipped this step and sits before the cursor does not adjust
  `req_index`; intentionally out of scope here).

Freeing/recompute state always uses upstream `_preempt_request`, which
requires the victim to already be removed from `self.running` — the seam
hooks selection, before removal, not `_preempt_request` after the fact.

## Future invocation

Set the attribute on the `Scheduler` instance owned by
`vllm.v1.engine.core.EngineCore` (`engine_core.scheduler`, created at
`core.py:160`), from the engine-core process, e.g. once the disk-backed
ownership/cost provider exists:

```python
from vllm.v1.request import Request

def disk_aware_victim(
    running: list[Request], default_victim: Request
) -> Request | None:
    # Pick the running request whose KV is cheapest to reclaim via the
    # LMCache disk-backed ownership/cost provider; return None to keep
    # the stock choice.
    return cheapest  # or None

engine_core.scheduler.preempt_victim_callback = disk_aware_victim
```

With the attribute left `None` (the default), the stock victim choice is
retained; the added helper call is not a claim of identical runtime cost.
Exact wiring into the LMCache transport is
root's integration task; this artifact contains no driver ABI or policy
code.

## Application

Patch labels are `a/vllm/...` / `b/vllm/...` (plain unified diff, no
index/hash lines). From the site-packages root of a vLLM 0.27.1 install:

```
patch -p1 < vllm-preemption-seam.patch
```

Not applied to the installed vendor copy or to repo code yet.

## Verification done

- `patch -p1 --dry-run` and real application onto a scratch copy of the
  0.27.1 `scheduler.py` file, then `py_compile` with
  `current-venv/bin/python` — both clean.
- Root also ran `patch -p1 --dry-run -i` with the published artifact against
  the current installed site-packages directory; it exited zero without
  modifying that installation.
- No GPU runs, no behavior tests; correctness of the seam rests on the
  source review above. Existence of the seam does not make disk-aware
  KV reclaim or automatic offload complete; actual behavior is measured
  by root after integrating the real LMCache disk-backed
  ownership/cost provider.

## Installed into the experiment environment, September 7, 19:55 UTC

After repaired async campaign 03 completed and the GPU had no compute
processes, root mechanically applied this committed patch to the installed
vLLM 0.27.1 scheduler in `current-venv`. Ordinary `py_compile` passed, and
the callback attribute and helper are present. This supersedes the earlier
"not applied" status above; the patch artifact itself is unchanged.

The original 137,118-byte scheduler is preserved at
`/var/tmp/vllm-kv-reclaim-seam-20260907.9YpmaZ/scheduler.py`. No original source
was discarded. The installed environment now differs from the unpatched
environment used by async campaigns 01-03.

The callback still defaults to `None`. Disk-backed registry, policy binding,
and actual recovery-route integration remain local-model implementation work;
installation and compilation do not establish policy execution or performance.

## Overlap / deferred-free compatibility fix (incremental, 2026-09-07, this section only)

### Applicability correction (root-confirmed)

Every async arm in this campaign (stock, native, BPF positions) ran
`kv_role=kv_both` with `LMCacheConnectorV1` and the same GDS backing, so
`kv_transfer_config.is_kv_consumer` is True in all of them. With async
scheduling auto-enabled (`vllm/config/vllm.py:1095-1143`),
`max_concurrent_batches=2` for the V1 runner at PP1
(`vllm/config/vllm.py:539-552`), and so `defer_block_free=True` in every
async arm (`vllm/v1/core/sched/scheduler.py:145-158`). An external
assumption that the native arm had no kv_transfer_config (and therefore
no fence) is wrong and is not used here.

Scope label correction (root): the earlier async native/BPF cells are
requested configurations, not executed policy comparisons - their
sitecustomize bootstrap failed with a CudaIPCWrapper ImportError before
any Scheduler/connector hook installed (adapter import fix owned
elsewhere). The stock-async vs stock-no-async control is the meaningful
comparison; the fix below is common runtime and independent of which
victim callback is later attached.

### Verified mechanism chain (all read directly, 2026-09-07)

1. `allocate_slots` failure retry loop (scheduler.py:590-675): a running
   request retries allocation after each victim preemption until it fits
   or until the failing request itself is preempted (`if preempted_req ==
   request: break`). Victims come from `_select_preempt_victim`
   (seam); `_preempt_request` is called with
   `drop_stale_output=self.requires_kv_delivery` (661-665).
2. With `defer_block_free=True`, `_free_request_blocks` (2437-2451) pops
   the preempted victim's blocks and parks them in `deferred_frees`
   fenced at the current `sched_step_seq` whenever
   `request.last_sched_seq > processed_step_seq` (an in-flight GPU step
   may still write them). The same-step allocation retry then sees a
   pool view that is transiently short by exactly those fenced blocks
   and can preempt a further victim. That a deferred-free victim cascade
   is the operative mechanism of the async collapse is a hypothesis
   supported by consistent runtime observations - rollback pairs printed
   at the same millisecond, preemption ~12.5/s, ~0.03 generated tokens
   per preemption, solo remainder at 39.6 TPS vs pairwise 1.64 TPS
   (async stock), and post-bootstrap-repair async arms logging ~26k
   restore batches/rollback warnings vs 15/7 in the non-overlapped
   controls (root analysis 6828f598; the logged hundreds-of-MiB-per-cycle
   restore payloads are repeated backend volume, not physical SSD
   traffic) - not a per-step executed trace pinning the causal line.
3. Fencelifetime: `processed_step_seq` and `_drain_deferred_frees`
   advance only for token-carrying steps (update_from_output,
   ~1781-1784, both inside the `total_num_scheduled_tokens > 0` guard);
   CoW retentions fence at `sched_step_seq + 1` (1230-1233).
4. Re-admission after preemption admits each waiting request
   independently (`get_num_new_matched_tokens` at 822, load allocation at
   1017) with no joint-fit check, so both loads could be re-admitted in
   the same step into the drained pool and the cycle restart. Re-admission
   pacing is policy-level and is explicitly NOT changed by this fix.
5. Control evidence: stock `--no-async-scheduling` (fence disabled via
   `max_concurrent_batches=1`) ran 8/8 warm requests at 71.787 TPS while
   stock async ran 2/8 at 1.6416 TPS
   (`raw/gds-kv-reclaim-scheduling-ablation-575-20260907-01/`). Sync
   reference points: 67.224 / 68.530 / 69.622 TPS. This establishes
   overlap/fence dependence; it does not prove a specific faulty line -
   the patch below is the proposed repair, to be judged by root's
   applied measurement. Post-bootstrap-repair default-async arms
   (root raw `raw/gds-kv-reclaim-bootstrap-repaired-575-20260908-01/`,
   adapter import fix e46cdd07) ended with warm 1/8 (native) and 0/8
   (BPF, natural end, 1238.8s, zero completed goodput under the old
   runner): the thrash is not a bootstrap artifact alone. Those runs
   do not establish victim-callback effects on performance. The changing
   kv_reclaim_snap cookies were observed in the repaired **bpf-sync**
   cell, not the bpf-async cell. The failed async outcomes do not isolate
   per-arm policy quality or prove the exact scheduler cause.

### Fix spec v2 (minimal, fence-preserving; v1 draft superseded)

Three small insertions in scheduler.py (see the diff below):

1. `Scheduler.__init__`: `self._kv_grace_used: set[str] = set()` next to
   the `deferred_frees` deque - request ids granted one preempt-grace
   yield since the last drain.
2. The `allocate_slots` failure retry loop (immediately after the
   `new_blocks is not None` break, before default-victim selection):
   yield instead of preempting further victims when ALL of
   - `self.defer_block_free` (same condition that arms the fences),
   - `self.sched_step_seq > self.processed_step_seq` (submitted
     token-carrying work is still outstanding - both counters advance
     only inside the `total_num_scheduled_tokens > 0` paths at
     scheduler.py 1292-1294 and 1781-1785),
   - a pending fence lies in `(processed_step_seq, sched_step_seq]`
     (owned by submitted-but-unprocessed work),
   - the failing request has not already yielded since the last drain
     (`request.request_id not in self._kv_grace_used`).
   Fences for future not-yet-submitted steps - CoW retention parks
   `fence = sched_step_seq + 1` (1230-1233) before any non-empty step
   may exist, and `_drain_deferred_frees` stops at the first pending
   fence - can NOT own a completion guarantee, so they never satisfy
   the gate; once all submitted work drains, normal allocation and
   stock preemption resume even if such a future fence remains.
3. `_drain_deferred_frees()`: clear `self._kv_grace_used` at entry, so
   the yield re-arms at a drain invocation. A pending first fence can
   stop that invocation without any blocks actually returning.

Semantics and costs, stated precisely:

- The yield is a `break` inside the retry while-loop with `new_blocks
  = None`; the existing `if new_blocks is None: break` below the loop
  then exits the OUTER running-request loop for this step, so later
  runnable running requests are skipped this step too. This is the same
  outer-loop shape stock already produces when a victim cascade cannot
  fit (the failing request becomes the victim). Skipped requests merely
  retry next step; nothing is reordered, freed, or reset.
- Source-derived (not hypothesis): fences in
  `(processed_step_seq, sched_step_seq]` belong to steps that were
  scheduled with tokens and whose outputs will be processed, advancing
  `processed_step_seq` and running the drain; both counters advance
  only on token-carrying steps, so the range gate cannot wait on a
  phantom step. Stale (already-drainable) fences are excluded by the
  strict inequality; future CoW fences by the upper bound.
- Fallback guarantee (source-derived): the per-request marker means a
  request that fails again before any drain takes the stock preempt
  path unchanged. A yield therefore can suppress at most the
  chain of further same-step victim preemptions; it can never replace
  stock behavior across repeated attempts.
- Not proven here (explicit hypothesis to be confirmed by root's
  applied measurement): the exact per-step preemption-count reduction,
  the resulting throughput recovery, and the transient latency of one
  skipped scheduling step per collision. No zero-token-livelock-free
  proof is claimed beyond the marker fallback above.
- Preserved exactly: fence lifetime (no early drain), stock victim
  semantics and the seam `preempt_victim_callback` (during a yield no
  victim selection occurs at all; stock and callback arms behave
  identically and reselect after the drain), async scheduling, and
  re-admission policy.

### Incremental patch (applies on top of the installed seam scheduler)

Apply from the site-packages root as with the seam patch:
`patch -p1 < <this-diff>`; labels follow the seam artifact convention
(`a/vllm/...`, `b/vllm/...`). The installed file must already contain the
seam (it does). Base artifact `vllm-preemption-seam.patch` is unchanged
and the victim callback is retained.

```diff
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -343,6 +343,9 @@
         # FIFO of (fence_seq, blocks): blocks become safe to free once
         # processed_step_seq >= fence_seq.
         self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()
+        # Request ids granted one preempt-grace yield since the last
+        # deferred-free drain (see the allocate_slots failure loop).
+        self._kv_grace_used: set[str] = set()
 
         self.perf_metrics: ModelMetrics | None = None
         if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
@@ -599,6 +602,40 @@
                         # The request can be scheduled.
                         break
 
+                    # Overlapping batches: blocks freed by a prior preempt
+                    # may be fenced behind a submitted token-carrying step
+                    # that is still executing (_free_request_blocks), so
+                    # the pool reads transiently short and stock victim
+                    # chaining here would destroy KV that is about to
+                    # return. Yield once per request per drain cycle:
+                    # defer only while submitted-but-unprocessed work can
+                    # own those fences (sched_step_seq >
+                    # processed_step_seq and a pending fence lies in
+                    # (processed_step_seq, sched_step_seq]); fences for
+                    # future not-yet-submitted steps (CoW retention,
+                    # sched_step_seq + 1) never qualify. new_blocks stays
+                    # None, so the existing `if new_blocks is None: break`
+                    # below exits the running loop for this step and later
+                    # runnable requests are skipped this step - the same
+                    # outer-loop semantics as when a stock victim cascade
+                    # cannot fit - and they retry after the drain. The
+                    # per-request marker, cleared at each drain in
+                    # _drain_deferred_frees, re-arms stock preemption as
+                    # the fallback whenever a yield does not resolve.
+                    if (
+                        self.defer_block_free
+                        and self.sched_step_seq > self.processed_step_seq
+                        and any(
+                            self.processed_step_seq
+                            < fence
+                            <= self.sched_step_seq
+                            for fence, _ in self.deferred_frees
+                        )
+                        and request.request_id not in self._kv_grace_used
+                    ):
+                        self._kv_grace_used.add(request.request_id)
+                        break
+
                     # The request cannot be scheduled.
                     # Preempt the lowest-priority request.
                     if self.policy == SchedulingPolicy.PRIORITY:
@@ -2467,6 +2504,8 @@
         can lead request-free fences by one step), so stop at the first
         pending one; any satisfied entry behind it is merely freed later.
         """
+        # Re-arm the preempt-grace yield at every real drain.
+        self._kv_grace_used.clear()
         while self.deferred_frees:
             fence, _ = self.deferred_frees[0]
             if fence > self.processed_step_seq:
```

### Verification done (installed runtime untouched)

- Built only on scratch copies under /tmp: installed scheduler copied,
  v2 edits applied to the copy, `diff -u` generated (3 hunks).
- `py_compile` clean on the edited scratch copy and on a freshly patched
  sandbox copy, using `current-venv/bin/python`.
- `patch -p1 --dry-run` clean against a pristine copy of the current
  installed scheduler; sandbox application is byte-identical to the
  edited scratch copy.
- No perf claim is attached; root applies and measures once the GPU
  queue is clear.

### Explicit non-goals (this section)

- No removal or weakening of deferred frees / write-ordering fences.
- No re-admission joint-fit gating (policy-level; the existing
  deferral primitive is `get_num_new_matched_tokens` returning `None`,
  scheduler.py:812-818).
- No `is_kv_producer`/`requires_kv_delivery` change (separate candidate,
  not bundled).
- No adapter changes (the bootstrap import issue is owned elsewhere).
- No new framework, simulator, or test scaffolding.
