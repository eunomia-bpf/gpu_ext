# Safe-ABI policy routes

This directory contains two deliberately different policy families that use
the current transaction-validated GPU-memory actuation ABI. They are intended
as executable expressibility examples, not performance claims.

## What “safe ABI” means here

- Eviction code observes the callback-local PMM snapshot and requests only a
  `USED/HEAD` or `USED/TAIL` reorder with `bpf_gpu_request_reorder()`.
- Prefetch code requests only an absolute subregion of the callback's current
  `max_prefetch_region` with `bpf_gpu_set_prefetch_region()`.
- Neither policy writes driver lists or decision contexts directly, retains a
  decision pointer, invokes a raw VA-space migration helper, or migrates data
  across VA blocks.
- Loaders own and destroy only the links they create. They do not attempt to
  remove another process's struct-ops registration.

The delta/Markov policy also uses a read-only kprobe on
`uvm_perf_prefetch_get_hint_va_block` to associate the temporary bitmap-tree
callback context with a VA block. The current `gpu_page_prefetch` signature
does not contain a VA-block identity, so strict block-local learning cannot be
implemented from that struct-ops callback alone. This observation dependency
is version-sensitive, but all mutation still goes through the safe request
ABI.

## Approximate 2Q / segmented LRU

Files: `eviction_2q_approx.bpf.c`, `eviction_2q_approx.c`

The policy overlays two logical segments on the existing UVM used list:

- a newly observed chunk enters the probationary segment at `USED/HEAD`;
- after observations in two distinct list-generation episodes by default, it
  enters the protected segment at `USED/TAIL`;
- later observations refresh a protected chunk at `USED/TAIL`.

The admission threshold is configurable with `-p`. A PMM/root identity pair
comes from the callback snapshot. Metadata lives in 16,384 direct-mapped,
per-CPU slots so the hot path has bounded memory and no dynamic map allocation.
A same-generation activate/access callback pair counts once: UVM increments
`list_generation` only when the root's list state changes, not when it moves
within `USED`. A later generation (normally after a pin/unpin or UNUSED-to-USED
transition) supplies the next admission observation. A slot identity collision,
callback CPU migration, or an excessive generation jump causes conservative
re-admission at the head.

This is not exact 2Q. The current ABI cannot maintain a separately sized A1out
ghost queue, insert at an interior queue boundary, or demote an arbitrary Am
entry. Consequently the implementation is best described as two-hit
segmented LRU: the physical list endpoints encode probation versus protection,
while metadata loss reduces protection rather than creating an unsafe action.

The loader emits cumulative JSON metrics:

| Metric | Meaning |
|---|---|
| `activate_events`, `access_events` | Observations by callback type |
| `admissions` | Cold or recycled identities admitted to probation |
| `identity_resets` | Direct-map collisions that replaced metadata |
| `generation_resets` | Identity-preserving generation jumps treated as reuse |
| `same_episode_events` | Duplicate callbacks suppressed within one generation episode |
| `probation_head_requests` | Requests to keep one-hit entries evictable |
| `promotions` | Probation-to-protected transitions |
| `protected_tail_requests` | Refreshes of already protected entries |
| `reorder_errors` | Request recorder calls that did not accept the request |
| `eviction_prepares` | Observed memory-pressure preparation callbacks |

These are request/engagement counters. They do not claim that a later driver
validation committed every request.

## Block-local delta/Markov prefetch

Files: `prefetch_delta_markov.bpf.c`, `prefetch_delta_markov.c`

For each observed VA block, the policy forms page deltas and learns a bounded
first-order transition table, `delta[n] -> delta[n+1]`. Each predecessor keeps
one candidate successor with a saturating confidence counter. Matching
transitions raise confidence; mismatches first decay it and eventually replace
the candidate. Once confidence reaches `-c`, the predicted page anchors a
contiguous `-n`-page window, clamped to the callback's current maximum region.
`-m` rejects implausibly large deltas and breaks the current transition chain.

This is an approximate Markov predictor, not a full transition-probability
matrix. Block and transition maps are bounded LRU maps (4,096 blocks and
16,384 transitions), so pressure can discard learned state. The association
kprobe and prefetch callback are expected to execute in the same synchronous
UVM call path; a missing association safely produces an empty prefetch request.
Pointer reuse can also retain stale learning until LRU replacement. Prediction
is strictly intra-block.

The loader emits cumulative JSON metrics:

| Metric | Meaning |
|---|---|
| `context_captures`, `callbacks` | Observation-hook and policy-hook engagement |
| `blocks_initialized` | VA-block histories created |
| `deltas_observed`, `invalid_deltas` | Accepted versus rejected/zero deltas |
| `transitions_created` | New predecessor-to-successor entries |
| `transition_matches` | Observations reinforcing the current successor |
| `transition_decays` | Conflicting observations reducing confidence |
| `transition_replacements` | Low-confidence successors changed |
| `confident_predictions` | Learned transitions passing the confidence gate |
| `prefetch_requests`, `empty_requests` | Non-empty versus demand-only requests |
| `map_errors`, `request_errors` | Bounded-state or request-recorder failures |

As with the eviction policy, request counters measure policy engagement, not
post-callback commit confirmation.

## Offline verification

No GPU, CUDA process, module reload, or struct-ops attachment is needed for
these checks:

```bash
make -C extension test_safe_policy_models
make -C extension eviction_2q_approx prefetch_delta_markov
extension/eviction_2q_approx -h
extension/prefetch_delta_markov -h
```

`test_safe_policy_models.c` directly executes the same pure helper state
machines included by the BPF programs. It covers configurable 2Q promotion,
generation reset, Markov reinforcement and adaptation, alternating-delta
prediction, independent learned states, and forward/backward region bounds.

The corresponding experiment smoke has an explicitly read-only admission
mode and a separately named execution mode:

```bash
python3 -B workloads/uvm-policy-mechanism/test_safe_policy_smoke.py
python3 workloads/uvm-policy-mechanism/run_safe_policy_smoke.py admit
python3 workloads/uvm-policy-mechanism/run_safe_policy_smoke.py run \
  --policy prefetch_delta_markov --output /explicit/new/output/directory
```

`run` holds the same exclusive GPU and struct-ops leases as the reviewed MoE
harness, accepts only an idle/clean safety snapshot, checks loader-reported
map/link ownership, and gives the fixed 8 GiB / 64 KiB fault-stream workload a
hard 60-second limit. It then requires zero data mismatches, policy-specific
engagement counters, and a clean post-run Xid, kernel-log, UVM-refcount,
struct-ops, and GPU-idle snapshot. Cleanup signals only process groups created
by that invocation. Its summary records paths and sizes, the Git revision, and
observed counters; it does not inspect model/object-store contents.
