# Step C: three-arm correctness and matched performance path

2026-09-03: the new shadow build and CPU tests pass after the unrelated HB
timing window closed: six bridge tests (five actual host-uBPF decisions) and
eight controller/mode tests, including analyzer AST coverage. Original logs
are in [step-c-cpu-01](step-c-cpu-01/execution.md). The subsequent
[three-arm real GPU preflight](correctness-results-575-01.md) and independent
27-array audit now pass. **The 15-cell performance matrix remains pending.** The prior private
offloader build/import passed separately (`4438d62`, 59,238,064 bytes); this
step does not modify its source tree, `State`, or the accepted Step B patch.

The requested OpenCode independent review completed after the CPU tests and
closed HB timing window; its [actual final report](opencode-review-02/final.md)
has no blocking findings. Attempt 01 returned no visible text and remains
preserved as incomplete; attempt 02 supplied the full report with no tool-use
events, and its owned process group was empty after exit. See
[review outcome and caveat handling](opencode-review.md). Its complete
bounded [prompt](opencode-correctness-review-prompt.md) and exact seven-source
[context](opencode-correctness-review-context.md) are prepared. Default routing
now names an nvfp4 gateway; an external HTTPS endpoint alone does not establish
that its backend is independent of this host's GPU. The coordinator therefore
deferred the request during formal timing; no EB workload will run during
this review. No model override, service change or fabricated review verdict is
permitted. Both actual invocations used `snapshot:false`, all tools denied,
CPU 17 and a ten-minute deadline, retaining their complete requests/events/stderr.
The successful second invocation additionally retains its exact final report.

## Boundary and reuse

`correctness.py` adapts the frozen FineMoE controller's `run_stage`, leases,
telemetry, failure retention and pre/post safety checks through process-local
callbacks. It does not change that controller on disk. The existing owned-PGID
survivor cleanup from `build_adapter.py` replaces the older leader-only exit
check. Both target and telemetry are created with `start_new_session=True`;
the existing telemetry launcher establishes this in
`workloads/moe-infinity/run_moe_head_to_head.py`.

`inference_eb.py` directly reuses FineMoE's `validate_data`, `generate`,
`create_finemoe`, `decoding_configuration` and `retain_and_check_result`.
Only the package source and EB arm/K/library are selected here. No history
store is loaded, and the common private adapter disables prediction for all
three arms. Generation mathematics, eager BF16, seeds, native-DSL compatibility,
request order, .5 memory ratio, original checkpoint generation configuration,
and the existing 16,834,658,304-byte strict pool remain unchanged.

Each fresh process runs the original one warmup plus eight held-out MT-Bench
requests, with 16 generated tokens each. Raw results distinguish
`evaluation_generated_tokens=128` from `correctness_generated_tokens=144`.
In preflight, all nine complete `(16,1,151936)` float32 logits arrays are retained and checked
against original HF `raw/golden-v4/stage` at its unchanged **0.0** tolerance.
After worker exit, the controller independently reloads every original/current
pair and verifies exact values, shape, dtype and finiteness. Correctness cells
do not publish throughput or count as a paired performance block.

The gate also requires actual evictions, all 24 layers, whole-expert copy bytes,
matching decision/acquire/copy-completion counts, no speculative copies, zero
copy/compute errors, the original pool capacity, K/residency bounds and complete
owned cleanup. Native and BPF final counts must match. A mismatch fails the
campaign and preserves every attempt; missing eviction is not called success.

## Untimed-only actual-input shadow

Only BPF correctness selects the new `libeb_shadow.so`. Its ABI matches the
already built `State`, so neither offloader nor State needs rebuilding.
The bridge requires `EB_SECTION_VI_UNTIMED_SHADOW=1` and the exact absolute
real selector path. On each call it first executes the **actual uBPF JIT**,
then runs native on a copy of the same pre-decision context. It compares input,
output, return and one-call progression without replacing the BPF result or
preselecting a victim. Any discrepancy poisons the instance and fails closed.
The worker retains the bridge handle and verifies
`checks = jit_calls = total EB decisions > 0`, with zero mismatches.

This is a live-input check only after the real GPU workload executes. The
separate fake-ABI and real-JIT CPU tests do not establish live GPU evidence.
The same worker now supports timed execution: it uses `libeb_policy.so`
directly, omits shadow/capture and sets neither untimed guard. This path is
prepared, not executed. It calls the unchanged `generate(..., False)` and
`retain_and_check_result(..., output=None)` for warmup and evaluation alike:
all 144 tokens are still checked exactly, but `logits_checked=False`, with no
logits arrays, per-request preparation files, or invented numerical-error fields.

`correctness.py --mode full` requires a completed three-arm preflight from
the same private source, runtime/binaries, model files, original data/golden,
K and decoding configuration. Before launching it reaudits every saved
preflight launch, cleanup, telemetry, worker and all 27 retained logits arrays.
The worker, controller and analysis source are inventoried before preflight;
changing them or the selector/offloader afterwards invalidates that preflight.
The same `audit_engagement` checks real JIT, copy, residency and pool facts in
both modes without turning token-only timing into numerical verification.
The original golden has no decoding-configuration field; configuration is
checked against its actual checkpoint JSON and unchanged generation overrides.

The proposed full schedule takes five distinct permutations of
`(fifo, native, bpf)` shuffled with Python `Random(20260903)`: five complete
blocks, 15 fresh-process cells. Each block advances only after all three
arms and native/BPF accounting match. Failed attempts and partial blocks are
retained, never silently replaced or counted. K=16 and this schedule remain
candidates until actual preflight engagement and correctness pass.

The timed application window uses the copy ledger's real `steady_clock`,
includes eight evaluations (128 tokens), and excludes model load, warmup,
final drain and teardown. It includes the existing exact-token checks and
per-request progress log, as the FineMoE protocol does. The worker separately
records drain/inclusive cost and real CPU usage. TTFT is request begin to
first actual output; TPOT is the first-to-last output interval divided by 15.
`analyze_results.py` independently reconstructs these from raw timestamps,
requires exactly the 15 expected cells and a non-overlapping observed order,
and reuses FineMoE's 10,000 whole-block bootstrap draws. It reports arm medians,
native/FIFO and BPF/FIFO policy effects, and BPF/native implementation effects.
Its 95% intervals describe paired mean differences/geometric ratios, not
uncertainty around the displayed arm medians. No cells or zero metrics are
removed to improve a ratio.

## Coordinator execution after the timing window

The two bounded CPU checks below passed and their complete output is retained
in `step-c-cpu-01`. The first command builds only the tiny shadow library and
test fixture; it does not rebuild the offloader or original selector. Any later
rerun must use a fresh log attempt, never overwrite these records.

```sh
timeout 30s taskset -c 17 make -C workloads/expert-buffering-policy/section-vi -f shadow.mk -j1 test-shadow
timeout 30s taskset -c 17 /usr/bin/python3 -B workloads/expert-buffering-policy/section-vi/test_correctness.py
```

After code/test review, the sole GPU launcher runs the three cells serially:

```sh
taskset -c 8-17 workloads/finemoe/.venv/bin/python -B workloads/expert-buffering-policy/section-vi/correctness.py --mode preflight --source workloads/expert-buffering-policy/section-vi/build/stage-check-02 --capacity 16 --output workloads/expert-buffering-policy/section-vi/raw/575-section-vi-correctness-01
```

K=16 is still an untimed candidate, not a performance-frozen result. Each child
has the existing 8–11 CPU placement and a bounded 1,800-second deadline. The
controller refuses an existing output or mismatched reference/runtime inventory.
It acquires the same GPU/struct-ops leases and does not load BPF into the driver,
change modules, start services, or regenerate the original HF reference.
Only after all three cells and their independent raw-array audits pass should
the coordinator freeze and run the separate 15-cell performance protocol.
These are future commands, not execution records:

```sh
taskset -c 8-17 workloads/finemoe/.venv/bin/python -B workloads/expert-buffering-policy/section-vi/correctness.py --mode full --source workloads/expert-buffering-policy/section-vi/build/stage-check-02 --capacity 16 --preflight workloads/expert-buffering-policy/section-vi/raw/575-section-vi-correctness-01 --output workloads/expert-buffering-policy/section-vi/raw/575-section-vi-full-01
taskset -c 17 workloads/finemoe/.venv/bin/python -B workloads/expert-buffering-policy/section-vi/analyze_results.py workloads/expert-buffering-policy/section-vi/raw/575-section-vi-full-01
```

The second command is post-window, read-only JSON output. Runtime source,
selector library, the built stage02 extension and the original FineMoE remain
untouched by either entry. This is a Section VI policy port, not a claim to
recreate the original paper's distributed system or original-model throughput.
