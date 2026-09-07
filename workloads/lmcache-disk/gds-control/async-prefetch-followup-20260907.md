# LMCache disk prefetch linked to gpubpf asynchronous decisions

Status: implementation follow-up, not a completed experiment or novelty claim.
User direction on 2026-09-07: prioritize real disk prefetch coordinated with
gpubpf; do not start more paper reproductions or edit paper files.

## Implemented decision interface; serving integration pending

Main `3dd808ac` adds matched native/BPF opt-in decisions. Qwen 27B generated
the BPF branch; root completed its mechanical Python mirror and the explicit
read-op condition. `make gds_policy.bpf.o` and Python compilation pass. The
new object has not replaced the live attached policy or supplied serving data.
`HINT_PREFETCH_JIT = 1 << 62` selects the new speculative-read branch: with a
positive transfer estimate and slack exceeding that estimate, defer for the
difference capped at the existing 10 ms scalar limit; otherwise submit now.
Demand reads and all no-hint paths retain their old behavior.

The two local-model assignments use this integration contract:

- GLM: opt-in async backend/bootstrap through
  `LMCACHE_GDS_ASYNC_PREFETCH=1`, with ordinary LMCache
  `LMCACHE_ENABLE_ASYNC_LOADING=True`; existing policy modes remain
  `fifo`, `native`, `bpf`.
- Qwen 27B: new `run_gds_async_prefetch.py`, with overlapping warm serving
  requests and four demand/eager/native/BPF arms, reusing existing assets.
- Optional `kv_transfer_params` entry `lmcache.prefetch_deadline_ns` travels
  through the existing request-config/lookup/CacheEngineKey path. This is an
  explicit application host-monotonic deadline, not a measured prediction of
  scheduler use time. All arms receive the same hints. A zero/default lead
  cannot establish benefit from adaptive timing.
- Native/BPF get completion-derived service estimates and fresh remaining
  slack; neither receives a preselected action. Backend/runner code and their
  end-to-end performance are still pending.

On 2026-09-07 the Qwen 27B call ended with a recorded APIError HTTP 524
(completed at 14:52 UTC), and its session was absent from the live status
endpoint. Root resumed the same session with local Qwen Next for the
unfinished runner, preserving all completed policy work. GLM's backend task
remained live. No session was stopped for silence or an artificial timeout;
the total live OpenCode concurrency remains at most three.

Existing serving data are further decomposed in
`serving-stage-analysis-20260907.md`: the GDS whole-response advantage is in
the post-first-token interval, while first-token latency is worse. This is
not a causal attribution to disk or BPF and does not replace the unfinished
async-prefetch experiment.

## Question and scope

RQ1 (Policy Expressibility and Benefits): Can AI agents safely express GPU
resource-management policies through gpubpf—including policies from published
state-of-the-art systems—and do those policies improve performance?

Specific question: can storage prefetch decisions made before GPU consumption
reduce request waiting while bounding staging-memory occupancy and unnecessary
disk reads? Compare the same decision algorithm in native and BPF execution.
This addresses Reviewer E's storage-offload discussion and the shepherd's
policy/mechanism distinction. It is not another completed-cell rerun.

The intended endpoint is real LMCache/vLLM serving with asynchronous disk
retrieval and completion-driven availability to the GPU consumer. A backend-only
transfer run is setup evidence, not a replacement for that endpoint. Hardware
NVMe-to-GPU P2P is not required or claimed: retain the working cuFile
compatibility-mode transport.

## Source findings that change the implementation route

Installed LMCache 0.5.4 source under
`../current-venv/lib/python3.12/site-packages/lmcache/` provides:

- `v1/storage_backend/storage_manager.py:async_lookup_and_prefetch`: actual
  lookup, asynchronous reads, completion events and scheduler notification.
- `v1/storage_backend/gds_backend.py:submit_prefetch_task`: currently returns
  `False`; it does not submit disk prefetch.
- GDS does not override `batched_async_contains` or
  `batched_get_non_blocking`; their abstract implementations raise
  `NotImplementedError`.
- Our `lmcache_gds_backend_adapter.py` currently wraps single-key nonblocking
  reads and blocking batch reads, but not those two framework async seams.
- GDS allocates a GPU MemoryObj before cuFile retrieval. Therefore the first
  extension must account for real staging allocations, not invent an independent
  CPU staging tier or claim UVM eviction control that is absent.
- `v1/cache_engine.py:_async_process_tokens_internal` takes the completed
  lookup's actual MemoryObjs; the normal `retrieve` path passes them to
  `gpu_connector.batched_to_gpu` and subsequently unpins/releases them. Read
  completion alone is therefore not the point to return the staging budget.
  Avoid a per-chunk admission deadlock in which several incomplete requests
  fill the pool while the scheduler waits for complete prefixes. Reserve enough
  space for a request's admitted prefix, or use the existing shorter-prefix
  result semantics; do not invent an unbounded staging allocation fallback.

Keep installed third-party sources unchanged. Add opt-in project adapters at
these existing seams and preserve all old default execution paths.

## Implementation assignment

1. Add an opt-in GDS async-prefetch adapter used by the existing storage
   manager. Implement ordered prefix lookup and asynchronous batch retrieval
   using real GDS allocation/read methods, without blocking its event loop.
2. Use the existing gpubpf storage decision ABI and native decider. Submit or
   defer speculative reads through the existing executor; demand should adopt
   an existing in-flight read instead of duplicating it. Completion must make
   the actual MemoryObj available to LMCache's normal consumer.
3. Bound outstanding staging allocations by bytes; count in-flight plus
   completed-but-unconsumed objects. Preserve reference ownership, ordinary
   cancellation and error propagation. These are implementation semantics,
   not extra measurement gates.
4. For an adaptive policy, use only information available before the decision:
   completion-derived read-cost estimates, current occupancy/in-flight work,
   and an actual scheduler-provided use-time estimate if available. Do not
   use future trace labels or hard-code measured test timings. BPF must make
   the scheduling decision, not receive an action already chosen in Python.
   Record any missing scheduler signal explicitly rather than pretending that
   a static hint is live telemetry.
5. Provide opt-in serving-runner wiring and exact commands using the existing
   environment/model. Root runs GPU experiments; local models implement code
   and may run ordinary CPU compilation/import checks. No driver reload,
   package replacement, new paper, new gate framework or GPU run by subagents.

## Comparison and interpretation

Use the existing Qwen3-30B-A3B-FP8 serving workload and real disk-populated KV.
The new comparison must include overlapping requests so that retrieval can
precede GPU consumption, rather than only serial cold-then-warm requests.
All arms receive the same request arrivals and available hints, storage
transport, staging budget and I/O concurrency.

Before any new performance run, source inspection corrects the initial
max-num-seqs=1 suggestion: installed vLLM's `v1/core/sched/scheduler.py`
breaks out of the waiting-request loop at lines 685-692 when the running
request limit is reached, before the connector lookup at line 778. Thus a
queued request cannot start its lookup while one request is already running
with that limit. The new four-arm experiment uses max-num-seqs=2 identically
in every arm, keeping the initial 768 MiB KV and 256 MiB GDS staging budgets.
This creates a scheduling opportunity for read/compute overlap; actual overlap
and a performance benefit are not yet measured. Old single-sequence results
and defaults remain unchanged. Native/BPF still receive identical inputs;
there is no BPF-only batching advantage.

- Original demand-driven disk retrieval: reference serving path.
- Eager prefetch on the same real lookup opportunity: strong simple policy.
- Adaptive native and adaptive BPF: identical information and algorithm.

Use five rotated paired blocks after the implementation is runnable; retain
all attempted outputs. Do not repeat old all-submit or write-budget cells.
Measure serving TTFT and output throughput, with disk bytes, unused prefetches
and peak staging bytes explaining tradeoffs. Storage request p99 is not TTFT.
Do not add clock-calibration or separate correctness/preflight campaigns.

A useful positive result improves serving performance over eager prefetch or
reduces its memory/I/O cost without hiding a latency regression. If eager
prefetch matches or beats adaptation, retain that result: asynchronous policy
expressibility may be demonstrated without establishing algorithmic benefit.
Do not call just-in-time prefetch novel without a closer prior-work comparison.

Raw results will use a new `raw/gds-async-prefetch-575-*` directory; the runner
command and final configuration must be recorded before its first performance
cell. No performance numbers exist for this follow-up yet.
