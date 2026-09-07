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
