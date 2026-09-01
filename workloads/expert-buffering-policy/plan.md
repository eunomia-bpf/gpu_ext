# Experiment Plan: Profile-guided hot-expert residency analogue

Status: proposal 4, independently approved after the exact-model preflight
exposed unsafe application behavior from aggressive cold-head pressure. The
four-cell correctness run is authorized; full timing remains gated on it.

## Research Question And Hypothesis

- Paper RQ: **RQ1 (Single-Tenant Management): How much performance gain can
  gpubpf's programmable memory and scheduling policies provide on
  oversubscribed single-tenant workloads?**
- Specific question: on the exact GPT-OSS-120B MXFP4 workload, does a fixed,
  independently calibrated expert-hot set plus safe page-ordering reduce
  repeated HBM activations of hot-expert pages relative to matched native UVM
  ordering on the same gpubpf mechanism, and what throughput, transfer,
  eviction, and residency trade-offs result?
- Preregistered hypothesis: expert-semantic ordering lowers post-warm-up
  repeated activation bytes for calibration-hot expert pages relative to the
  attached observation-only control. Throughput direction is not preregistered because 2 MiB
  mixed-expert blocks can amplify conservative protection.

## Paper-Value Admission

- Planned role: **decisive** for the promised implementation of Expert
  Buffering's hot-expert residency idea and for the Shepherd's existing-policy
  expressibility question; **supporting** for RQ1 performance.
- The strongest credible story is narrower than proposal 1: gpubpf can express
  profile-guided hot-expert protection using safe PMM ordering, but the current
  hooks cannot implement Huang et al.'s complete
  current-batch, expert-atomic cache. The experiment measures the analogue and
  exposes that mechanism boundary rather than hiding it.
- The result may be positive, null, or negative. A null or negative outcome
  still answers whether the general mechanism adds cost when implementing this
  existing policy idea. Any benefit is attributed to the calibrated policy;
  the mechanism contribution is safe loading, bounded actuation, and rollback.
- A generic prefetch sweep would not answer the named Expert Buffering
  commitment or distinguish policy value from mechanism value.

## Published Policy, Artifact, And Claim Boundary

- Primary paper: Huang et al., “Toward Efficient Inference for Mixture of
  Experts,” NeurIPS 2024, Section 5:
  <https://openreview.net/attachment?id=stXtBqyTWX&name=pdf>.
- The published mechanism holds inactive experts in CPU memory, uses a fixed
  GPU expert buffer, obtains current-batch active experts from gating/all-to-all
  metadata, transfers missing active experts, evicts inactive-current-batch
  experts first, and then applies LIFO because experts execute in increasing-ID
  order.
- The authors' public repository contains a README but no source or benchmark
  and says release approval is pending:
  <https://github.com/hyhuang00/moe_inference>. No original-implementation
  timing is possible, and the paper's reported numbers are context only.
- Production source and live traces establish two hard limits. On this UVM
  path, router IDs stay device-resident, and PMM callbacks do not expose the
  current selected expert. `gpu_block_access` does fire on observed access
  paths and can refresh classified hot/shared pages, but it is not a complete
  resident-selection stream and missing callbacks are never interpreted as
  hits. The safe typed reorder request is callback-local.
- Therefore this experiment implements only a **profile-guided, page-granular
  hot-residency eviction-order analogue**. It does not claim current-batch
  inactive-first selection, an expert-atomic cache, transfer overlap, a cache
  hit rate, or reproduction of Huang et al. A missing activate event is never
  called a hit. The unavailable current-selection and resident-refresh
  actuation points are mechanism limits, not repaired with raw list mutation.

## Exact Workload And Independent Calibration

- Hardware: one RTX 5090 with 32 GiB HBM, the 125 GiB host, Samsung 9100 PRO
  NVMe, Linux 7.1.12, and Open Kernel Modules 610.43.02.
- Model: existing exact `gpt-oss-120b-MXFP4.gguf`, 63,387,346,208 bytes, from
  the frozen public GPT-OSS revision in the approved MoE plan.
- Runtime: pinned llama.cpp and the `llama-server` command from
  `workloads/moe-infinity/plan.md`; concurrency one, context 4096, eight
  P-cores, no prefix cache, no speculative decoding, and no request batching.
- Evaluation uses the same distinct warm-up and eight frozen 512-input,
  64-output prompt-token arrays, greedy decoding, and exact client boundary as
  the MoE plan. Prompt order is never selected after policy observation.
- Calibration uses eight disjoint 512-input, 64-output ShareGPT prompts selected
  before execution by the same eligibility/tokenizer-equivalence procedure
  with seed 1796. It is a separate setup-only run with `--n-cpu-moe 36`, so all
  36 MoE layers traverse llama's existing selected-expert streaming path. The
  scheduler already copies router IDs to host and synchronizes before grouped
  expert copies. The timed/context framework cell remains `--n-cpu-moe 32`.
- A uprobe on exported `ggml_backend_sched_graph_compute_async` assigns a
  monotonically increasing graph ordinal per process/thread. After the existing
  router-ID get and synchronize, a noinline marker is called once for every
  distinct set bit in the already-built `used_ids`, passing the source expert
  tensor base and expert ID. The loader joins that base to the layout marker to
  derive the layer, and records `(graph, layer, expert)` without another ID
  copy or synchronization. A graph is complete only if it has a valid record
  for every layer that executed; calibration must cover all 36 layers.
  Calibration is a setup artifact, not a performance result.
- For each layer, experts are ranked by distinct calibration-graph selections;
  ties use ascending expert ID. The top ten per layer form the frozen hot set,
  matching the published experiment's ten-expert-per-GPU cache point while
  avoiding tuning on evaluation outcomes. The tracked calibration report must
  contain all 36 layers, IDs in `[0,127]`, positive observations, and exactly
  ten selected IDs per layer before any policy run.

## Layout Registration And Safe Policy

- A second noinline marker records numeric `(layer, tensor_kind, base,
  total_bytes, per_expert_bytes, n_experts, is_bias)` after backend allocation.
  It changes no allocation, routing, copy, synchronization, contents, or order.
  Every configuration uses the same marker-enabled binary.
- Uprobe observations populate a map keyed by 2 MiB VA-block start. The exact
  GGUF has 36 layers x 3 expert weight tensors with 128 slices of 4,406,400
  bytes. Because slices are not block-aligned, a block can overlap adjacent
  experts. Bias blocks and blocks spanning tensor-registration boundaries are
  conservatively shared/protected. The exact union of hot-set blocks is
  recorded as the fixed page-level protection budget; it must be at most 8 GiB
  or the plan closes before timing.
- The exact-model `hot` preflight requested `USED/HEAD` on 528,597 cold
  activations and then failed with `cudaErrorIllegalAddress`; the matched plain
  custom-UVM control passed. The repaired `protect` mode passed the same
  shortest request by removing all cold-head requests. `page` and `hot` remain
  available as diagnostic modes, but neither is a measured configuration.
- `protect` requests reordering only through the typed PMM setter. The repaired
  action table is frozen:

| Callback and block class | Request | Meaning |
|---|---:|---|
| activate, shared/bias/boundary | `USED/TAIL` | conservative non-expert protection |
| activate, overlaps a frozen hot expert | `USED/TAIL` | hot-set residency priority |
| activate, mapped only to cold experts | none/default | preserve native UVM ordering |
| access, hot or shared | `USED/TAIL` | refresh protection when this callback is emitted |
| access, cold or outside registered ranges | none/default | preserve native UVM ordering |

- The observation-only control executes the same callbacks, layout lookups,
  class lookups, and counters but never calls the typed setter. It isolates the
  cost of the general attached mechanism from the effect of the hot-protection
  policy. This remains an eviction-order analogue rather than the full
  published cache.

## Configurations And Fairness

1. `plain_uvm`: exact UVM llama command, no struct_ops policy.
2. `gpubpf_observe`: same UVM command, attached policy object, layout and class
   lookups, and engagement counters as the policy cell, but no reorder request.
   This versus configuration 1 measures mechanism overhead.
3. `gpubpf_profile_protect`: byte-identical UVM command; hot/shared blocks
   request tail and cold blocks preserve native ordering. This versus
   configuration 2 is the primary matched policy-semantics contrast on the
   same mechanism.
4. `llama_ncmoe32`: exact framework selected-expert streaming command with
   `--n-cpu-moe 32`, no UVM override and no struct_ops policy. It is deployment
   context and supplies the separate calibration trace, not the original
   Expert Buffering artifact.

Every measured cell uses the same custom 610.43.02 UVM. Only idle
`nvidia_uvm` is temporarily replaced; matching core, modeset, DRM, and display
modules stay loaded. Distribution UVM is restored afterward.

## Correctness, Engagement, And Event Semantics

- Each cell runs one excluded warm-up and two untimed complete passes over all
  eight evaluation prompts. Every request must report 512 prompt tokens, 64
  completion tokens, length termination, valid UTF-8, and byte-identical output
  across passes and configurations.
- Layout admission requires exactly 108 weight and 108 bias registrations, 36
  layers, 128 experts per layer, positive mapped weight and mixed-expert block
  counts, and no inconsistent overlapping registration. Shared/boundary cases
  are counted separately and are not mislabeled conflicts.
- The protection policy requires positive mapped activates, hot-tail,
  cold-native, and hot-access-tail decisions after warm-up, zero cold-head
  decisions, zero typed-setter failures, and positive completed UVM evictions.
  The observation control requires positive mapped activation and access
  classification, zero reorder requests, and positive completed evictions.
  Plain UVM requires completed migration and eviction events. Framework context
  requires positive existing selected-expert copy bytes and complete route
  observations.
- Immediately after the excluded warm-up reaches idle, the runner takes one
  policy/event snapshot before the first measured request. For each frozen-hot
  registered 2 MiB block `x`, let `N_x` be its activate-count delta after that
  snapshot. The primary compulsory-allocation-excluding quantity is exactly
  `2 MiB * sum_x max(0, N_x - 1)`: the first post-snapshot allocation of each
  block is excluded and every later allocation contributes one block. Full
  post-snapshot hot activation bytes, `2 MiB * sum_x N_x`, are retained as a
  secondary metric. These are HBM allocations, not router misses. Completed
  UVM Tools eviction records retain address and size and are joined to the
  frozen layout. Dropped events, ambiguous timestamps, or inconsistent address
  classification invalidate the cell. No-transfer cases are not called hits.
- The loader owns exactly one struct_ops link, records its map/program IDs, and
  refuses foreign or ambiguous registrations. Shutdown detaches only that link.

## Metrics, Repetitions, And Analysis

- Primary policy metric: the frozen post-warm snapshot formula
  `2 MiB * sum_x max(0, N_x - 1)` for frozen-hot blocks,
  `gpubpf_profile_protect` versus `gpubpf_observe`.
- Primary application metric: aggregate verified output-token throughput, 512
  output tokens divided by the interval from first measured request start to
  eighth completion.
- Secondary metrics: repeated activation events for hot and cold strata;
  completed eviction bytes by hot/cold/shared stratum; total migration bytes;
  per-request TTFT and latency; GPU peak memory; process-tree CPU time and
  storage reads; head/tail decisions; mixed-block fraction; and protected-byte
  amplification relative to the ten-expert slice bytes.
- Five valid complete paired blocks are collected from at most eight attempts.
  With `U=plain_uvm`, `O=gpubpf_observe`, `E=gpubpf_profile_protect`, and
  `F=llama_ncmoe32`, fixed orders are `U,O,E,F`; `O,E,F,U`; `E,F,U,O`;
  `F,U,O,E`; and `F,E,O,U`. An invalid attempt retries the same slot/order.
  Prompt orders are frozen from seed 1798 before the first run.
- Report the geometric mean of paired throughput ratios and paired differences
  in repeated hot activation bytes with block-bootstrap 95% intervals from one
  frozen resample index matrix. Prompts are not independent replicates. All
  valid blocks remain, including null and negative outcomes.

## Execution Gates And Retry Bound

- Offline gates: both marker source audits; marker-disabled and marker-enabled
  builds; calibration parser and tie-break tests; GGUF layout tests; host-model
  tests for hot/cold/mixed/shared/default classes and LIFO order; verifier
  admission; BTF ABI checks; and ownership-safe loader tests. These are setup
  evidence, not paper results.
- The route/layout calibration passed. The original hot-LIFO exact-model
  attempt failed and its matched plain control passed. One repaired
  protection-mode retry then passed the same 512+1 request. After proposal 4 is
  independently re-approved, one four-cell 512+64 correctness/lifecycle run is
  the only remaining pre-timing execution; it is implemented through the same
  runner used for timing, not a separate preflight harness. Failure closes the
  protocol with no timing until the cause is understood and the plan reviewed.
- Startup timeout is 1,800 seconds, request timeout 600 seconds, and owned
  shutdown timeout 120 seconds. Admission requires an idle GPU, at most 256 MiB
  residual HBM, zero compute processes, UVM reference count zero before module
  replacement, enough filesystem space, and no thermal throttling.
- Full execution starts only after preflight and a fresh result-readiness
  review. A block is valid only if all four cells pass correctness, engagement,
  ownership, event, and thermal gates. Fewer than five valid blocks is an
  inconclusive performance result with all evidence retained.

## Interpretation And Deliverables

- Report `gpubpf_observe` versus `plain_uvm` separately as the measured cost of
  the attached general mechanism. Do not attribute this delta to the expert
  policy.
- If hot repeated activations fall and throughput improves, attribute the
  improvement to profile-guided expert classification; mechanism value is safe
  deployment and replacement.
- If activations fall without throughput improvement, quantify mixed-block
  amplification and ordering/transfer costs. If results match or regress,
  report that safe expressibility did not produce an advantage.
- Always report the unsupported parts of Huang et al.: current-batch router
  visibility, resident-selection refresh, expert-atomic capacity, and overlapped
  whole-expert copies. Never label this a reproduction or compare its timing to
  the unavailable original implementation.
- Raw path: `workloads/expert-buffering-policy/raw/`. Tracked outputs are the
  approved plan/review, minimal source patches, configs, tests, semantic raw
  summaries, analysis, and result review. No paper file is edited in this gate.

Never generate, refresh, compare, or record file/content hashes, checksums,
digests, or fingerprints. Use explicit versions, file inventories and sizes,
semantic checks, builds, correctness, and real engagement observations. Git
commit IDs and upstream revisions remain ordinary version bookkeeping.
