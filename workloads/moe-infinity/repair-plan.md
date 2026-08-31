# MoE-Infinity oversized-route repair experiment — proposal 3 revision 4

Status: **attempt 3 passed exact model execution but failed the O_DIRECT
metadata-classification gate; all three attempts are exhausted; timing is not
authorized**.

This proposal reopens the MoE-Infinity axis only after a disclosed source
repair. The failed proposal-2 preflight remains preserved and is not relabeled
as a sample. No paper text is changed while this experiment is in progress.

## 1. Research question and claim boundary

Can a minimally repaired public MoE-Infinity artifact complete the same
GPT-OSS-120B MXFP4, 512-input/64-output workload used by the approved
four-configuration comparison, and if so how does its deployment-level
performance compare with llama.cpp framework offload, plain UVM, and the
gpubpf host-policy ablation on one RTX 5090?

The primary estimand, directional hypothesis, system-level comparison
boundary, and prohibition on presenting the host-only gpubpf policy as the
submitted device-observed policy remain exactly those in `plan.md` proposal 2
revision 3. A valid result may match or lose to the research baseline. The
repair itself is not a gpubpf mechanism or policy contribution and must not be
used to attribute a performance improvement to gpubpf.

## 2. Normative inheritance

Sections 2 through 9 of `plan.md` remain normative without modification for:

- host, driver, CUDA, model, GGUF, tokenizer, prompt, and dataset identity;
- all four commands and controlled environments;
- smoke goldens, exact token accounting, and deterministic-output checks;
- request event semantics and aggregate output-token throughput;
- MoE offload, O_DIRECT, gpubpf hook, and completed-eviction engagement;
- exclusive ownership, cleanup, cooldown, telemetry, and failure safety;
- frozen schedule, five-valid-of-eight stopping rule, estimator, bootstrap
  indices, confidence interval, and mutually exclusive interpretation.

This proposal overrides only MoE-Infinity source identity, repaired-build
identity, preflight attempt accounting, and the execution authorization in
section 10. Any other change requires a new reviewed revision.

Revision 3 leaves the approved source repair and all scientific settings
unchanged. It moves the existing `taskset -c 0-7` prefix outside the `strace`
wrapper so both the owned tracer and Python server inherit CPU 0--7, and records
the actual wrapped command in `launch.json`. Revision-2 attempt 1 remains the
first of three attempts. Only a reviewed protocol-ID change permits the next
attempt after that deterministic failure; an unchanged deterministic failure
still blocks retry.

Revision 4 responds to attempt 2 without weakening its exact-output oracle.
The four upstream expert compute threads and their CUDA streams remain intact,
but each worker installs its external-stream guard before constructing its
mask/input, checks completion of its float32 output, and places it in a pending
list. `WaitHiddenStates()` propagates any worker failure instead of returning a
partial result, then reduces successful outputs on the caller stream in
expert-index order. This removes both the producer/consumer handoff race and
concurrent in-place writes, and makes the reduction order independent of
worker arrival order. Attempts 1 and 2 remain the first two of the fixed
three-attempt budget.

## 3. Disclosed source repair

The source base remains EfficientMoE/MoE-Infinity commit
`b766f8f1f6379fac6cd23594713ba6f4c7650ad9`. Three patch artifacts are applied:

1. `instrumentation.patch`, the already approved load-only cache-counter and
   stats-route patch; and
2. the tracked `row-chunking.patch` file; and
3. `deterministic-accumulation.patch`, which repairs the upstream dispatcher
   race exposed by attempt 2.

The row repair changes `core/parallel/expert_module.cpp` and its declaration in
`core/parallel/expert_module.h`. The repair-specific numerical test is exposed
through the existing `_store` Python binding in
`core/python/py_archer_prefetch.cpp`; that binding is carried by
`instrumentation.patch`.
`MoEMLP::forward()` retains the existing reusable 256-row workspace. Calls of
1 through 256 rows retain the upstream copy, compute, synchronize, clone, and
return fast path. Calls above 256 rows reuse the same resident and already
dequantized expert weights over stable consecutive chunks of at most 256 rows,
copy each chunk result into the corresponding rows of one full-sized output,
then restore all reusable buffer views to 256 rows.

The accumulation repair keeps four parallel expert compute workers. It binds
each worker's PyTorch mask/input construction and expert forward pass to that
worker's non-blocking external stream, checks output completion before
publication, and stores the output and mask without touching the shared
destination. The caller rejects any worker failure, sorts completed records by
expert ID, and performs the original float32 weighted addition on its stream.
It does not serialize expert computation or change the number of expert
threads.

The patch does not change routing, expert selection, routing weights, model
weights, quantization, activation, prompt length, output length, cache
capacity, device-memory ratio, expert-store layout, or the comparison cells.
They add no prefetch or caching policy. The baseline must be labeled
"MoE-Infinity public commit plus disclosed oversized-route and deterministic-
accumulation correctness repairs," not the unmodified public artifact.

## 4. BUILD gate

Before independent review, and again after any repair edit, all of the
following must pass:

1. `git apply --unidiff-zero --check --reverse row-chunking.patch` against the
   staged source; the zero-context form avoids embedding Git blob identifiers
   in the patch artifact;
   `deterministic-accumulation.patch` must pass the same reverse-application
   check;
2. source checks proving the old `batch_size > kMaxTokens` fatal is absent,
   the `<=256` fast path and `>256` stable-row chunk path are both present, and
   the full output is restored in original row order;
3. boundary checks for 1, 256, 257, and the observed 353 routed rows;
4. all offline workload/runner/ownership tests;
5. CUDA 12.9 rebuild with `MOE_ENABLE_SM120=1`, `MOE_ENABLE_SM90=0`,
   `NVTX_DISABLE=1`, and the frozen CUTLASS checkout;
6. `_store` contains sm_120 device code and was built from the admitted source
   tree with the frozen build flags;
7. the standalone GPU numerical gate executes `MoEMLP::forward()` for 1, 256,
   257, and 353 rows against the same-parameter reference evaluated in stable
   chunks of at most 256 rows, explicitly synchronizes both paths, and requires
   `rtol=1e-2` and `atol=1e-2`;
8. the same GPU gate sends four distinct arrival orders of four synthetic
   353-row expert outputs through the production reduction helper and requires
   exact equality with an expert-index-order reference after explicit
   synchronization; and
9. source checks require the external-stream guard before mask/input
   construction, a checked output-publication barrier, and propagation of
   worker exceptions through `WaitHiddenStates()`; and
10. read-only admission accepts the exact model, binaries, custom loaded-UVM
   BTF interface, idle GPU, NVMe filesystem, and empty struct_ops inventory.

Together these checks establish build identity, control-flow boundaries, and
the declared standalone GPU numerical comparison. They do not establish
full-model completion or performance.

## 5. Repaired-protocol correctness preflight

Only after independent approval may the existing ownership-safe runner launch
the GPU preflight. It uses a new raw directory and never overwrites
`raw/correctness-preflight-610-20260831-01`.

The first repaired-protocol attempt runs the unchanged four-configuration
correctness smoke. In particular, the identical frozen 512-token warm-up must
complete in `moe_infinity_075`; absence of the old fatal alone is insufficient.
The two smoke passes must then satisfy every proposal-2 gate, including exact
512+64 token accounting per request, deterministic within-configuration output,
1,024 post-warm-up generated MoE tokens, positive expert-cache activity,
positive direct-I/O evidence, and the gpubpf hook/eviction gates in its cell.

Raw server logs must retain any exception, CUDA error, fallback, and cleanup
event. A request failure, missing engagement, mismatched output, throttle,
foreign process, residual owned state, or timeout invalidates the attempt and
cannot become a timing sample.

This genuinely revised protocol has at most three real preflight attempts under
the fixed `raw/repaired-preflight/attempt-01` through `attempt-03` namespace.
The runner creates the next directory before the first GPU action, retains a
failed result, refuses overwrite or attempt four, and refuses an unchanged
retry after a deterministic `GateError`. The
old proposal-2 attempt remains permanently recorded but does not consume this
new budget because it ran a different, independently closed source protocol.
This is not a reset by renaming: the only admitted change directly repairs its
deterministic 256-row capacity failure. If a new failure is deterministic and
unchanged inputs imply repetition cannot add evidence, remaining attempts are
not spent repeating it; a further code/protocol change requires another review.

## 6. Timed execution and analysis

No timed block is authorized until one complete repaired-protocol correctness
preflight passes. After that gate, the runner follows the already frozen eight
attempt orders and stops at exactly five valid complete blocks. The proposal-2
analysis code and persisted 10,000-by-5 bootstrap matrix are reused unchanged.

The repaired MoE build used for preflight must remain in place for every timed
block: no rebuild or binary replacement is permitted between those stages.
Preflight records resolved runtime filenames, sizes, device/inode identity,
and modification/change timestamps; the runner requires exact equality before
the timed schedule and before every measured configuration launch.
Failed or partial blocks are retained. The final result bundle requires a fresh
independent result review before any number is promoted to revision evidence.

## 7. Required interpretation

Any eventual report must disclose all of the following alongside the result:

- the unmodified public artifact failed the same frozen warm-up at 353 rows;
- the published baseline therefore uses both exact disclosed correctness
  patches;
- the repair preserves the upstream fast path for routes at or below 256 rows,
  but no unsupported claim of zero binary-level overhead is made;
- this is a deployment comparison across different runtimes, not a causal
  page-granularity or policy-versus-mechanism experiment;
- the gpubpf cell is a host-policy ablation unless a separately reviewed full
  device-observed cell is added later.

Matching MoE-Infinity is an acceptable outcome. A regression is reported as a
regression; gates and interpretation thresholds are not weakened after results
are visible.

## 8. Auto-research gate state

- BUILD: stream handoff, checked publication, worker-error propagation, and
  deterministic accumulation are implemented in one disclosed patch. The
  fresh rebuild and both GPU gates pass, including exact equality across four
  arrival orders. Forty offline checks and read-only admission pass. Follow-up
  review approves the fixed attempt 3.
- EXPERIMENT: attempt 1 completed the original 512+64-token warm-up but failed
  the CPU-affinity gate because the tracing wrapper sat outside `taskset`; see
  `runtime-preflight.md`. Revision 3 fixed that launcher defect, and attempt 2
  completed both eight-prompt smoke passes but exposed nondeterministic
  cross-stream expert accumulation in the upstream dispatcher. Revision 4
  implements the bounded stream-handoff and deterministic-accumulation repair;
  follow-up review authorized the final fixed-namespace attempt. Attempt 3
  completed the warm-up and all 16 smoke requests with exact equality for all
  eight output pairs, then failed because the all-files O_DIRECT gate rejected
  ordinary `archer_index` metadata writes. No full MoE timing run is authorized
  after the fixed three-attempt budget.
- WRITE: closed by user instruction until experiments are complete.
- REVIEW: a fresh result review is required only after a complete result bundle
  exists.
