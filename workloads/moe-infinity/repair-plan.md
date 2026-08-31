# MoE-Infinity oversized-route repair experiment — proposal 3 revision 2

Status: **revision 2 attempt 1 failed after the repaired warm-up completed;
unchanged execution is not authorized**.

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

## 3. Disclosed source repair

The source base remains EfficientMoE/MoE-Infinity commit
`b766f8f1f6379fac6cd23594713ba6f4c7650ad9`. Two patch artifacts are applied:

1. `instrumentation.patch`, the already approved load-only cache-counter and
   stats-route patch; and
2. the tracked `row-chunking.patch` file.

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

The patch does not change routing, expert selection, routing weights, model
weights, quantization, activation, prompt length, output length, cache
capacity, device-memory ratio, expert-store layout, or the comparison cells.
It adds no prefetch or caching policy. The baseline must be labeled
"MoE-Infinity public commit plus disclosed oversized-route correctness repair,"
not the unmodified public artifact.

## 4. BUILD gate

Before independent review, and again after any repair edit, all of the
following must pass:

1. `git apply --unidiff-zero --check --reverse row-chunking.patch` against the
   staged source; the zero-context form avoids embedding Git blob identifiers
   in the patch artifact;
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
8. read-only admission accepts the exact model, binaries, custom loaded-UVM
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
- the published baseline therefore uses the exact disclosed correctness patch;
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

- BUILD: implementation, rebuild, offline checks, standalone GPU numerical
  comparison, read-only admission, and independent re-review complete.
- EXPERIMENT: attempt 1 completed the original 512+64-token warm-up but failed
  the CPU-affinity gate because the tracing wrapper sat outside `taskset`; see
  `runtime-preflight.md`. A minimal launcher-only revision and independent
  review are required before attempt 2.
- WRITE: closed by user instruction until experiments are complete.
- REVIEW: a fresh result review is required only after a complete result bundle
  exists.
