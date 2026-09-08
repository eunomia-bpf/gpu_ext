# Automatic warp implementation continuation

The user transferred this implementation and two performance studies from the
paper task. Original worktree: `/home/yunwei37/workspace/gpu/bpftime-auto-warp`,
branch `revision/automatic-warp-execution`, starting at `eef8a51`.

## Source checkpoint at 08:30 UTC

Local GLM implementation is committed on the same branch as `e801d28`.
This is an incremental source checkpoint, not a completed runtime or a
performance result. It adds the opt-in `BPFTIME_GPU_AUTO_WARP_EXECUTION`
switch, verifier-informed transformation eligibility, per-entry mode storage,
JSON propagation to the existing PTX pass, and elected-leader kretprobe
calls at supported hook sites. Ineligible programs retain the per-thread
execution path. Per-lane append/atomic effects are not silently deduplicated.

Root reviewed and the local model corrected duplicate PTX predicates,
shifted kernel-body bounds, duplicated labels, a missing JSON flag field,
and a missing C++ include. Root ran `g++ -std=c++20 -fsyntax-only` on
`ptxpass_core/src/core.cpp`, `ptxpass_kretprobe/main.cpp`, and
`bpftime-verifier/src/gpu/warp_execution_eligibility.cpp`; each exited zero.
These checks do not establish full runtime compilation/linking, generated
PTX execution, matching output, or an observed speedup. Full builds and GPU
measurements remain pending while the original Fig.13 campaign runs.

The model's local dependency symlinks made Git reject tracked submodule
paths. Root removed only those three links and initialized real checkouts of
`vm/llvm-jit`, `third_party/spdlog`, and `bpftime-verifier/ebpf-verifier`
at their existing gitlink revisions, using the frozen tree as a read-only
clone reference with `--dissociate`. Git status works again. Frozen source
was not changed, submodule revisions were not updated, and no dependency
cache or build binary is part of this commit. Syntax checks were repeated
successfully against the resulting real dependency checkouts.

Next remains the full isolated build and same-object optimization-off/on
measurement using the existing prefill runner, followed by the queued
original device microbenchmark sweeps. Unsupported relevant probes or
silent fallback alone would not complete that requested experiment.

## Existing-runner integration gap

Source inspection at 08:42 UTC confirms `run_table1_perf.py` currently fixes
seven arms and builds all three observability tools. Its existing CLI does
not express the requested same-object baseline/optimization-off/optimization-on
comparison. The current original-runner OpenCode owner has this queued after
its scheduler/plot changes: add an opt-in three-arm mode to the existing
runner/helpers, retain the seven-arm default, record actual per-cell mode
settings and use ten rotated blocks. No second benchmark framework or
rerun of the completed Table 1 cells is intended. This runner extension is
not yet implemented; no new automatic-warp measurements exist.

The original Qwen Next session `ses_f80dc8da2ffev9TpuxcuajuRMa` ended with
HTTP 524 on its first assistant request, with no implementation patch. Root
confirmed the checkout is clean. The same session is now resumed using GLM;
this does not discard prior context or create a fourth local session.

The slot became free only after Hummingbird session
`ses_f80c496aeffeBh330rKhrRwPgX` ended with actual terminal HTTP 524 at
1788844635421 ms Unix time, and its CLI exited. Hummingbird produced no patch;
its original implementation task and context remain available for continuation.
Its failed Qwen Next request had followed GLM at a natural step boundary.

Current active implementation tasks are disk UVM (GLM), XSched (Qwen 27B),
and automatic warp execution (GLM fallback). All three local models have been
used; the preferred one-per-model arrangement is temporarily changed because
Qwen Next ended with actual API failures on two separate tasks. No session
was stopped for silence and no short timeout was imposed. Fig.13 remains
queued. Neither this dispatch nor the earlier source-reading work is a
completed implementation or a new measured speedup.

## Exact continuation prompt

### Hook-count follow-up location, clarified by the user

The hook-count studies must extend `microbench/fig15-device`, not substitute
the separate trampoline-scaling workload. The existing
`strict-warp-map-scaling/warp_map_bench.cu` already accepts thread count but
launches one CTA; its `fig15_warp_map_kernel` has one hook before the output
calculation. Its companion `run_strict_warp_map_scaling.py` provides existing
launch, attachment and timing entrypoints. The parent `map_bench.cu` and
`run_map_tier.py` supply the corresponding fixed-32-thread map-tier path.
Reuse these files and their output conventions, keeping old modes and results.

The required bounded follow-up adds grid/block-count control and arithmetic
work outside the unchanged hook, with output indexing/allocation matching the
grid. First vary block count at fixed work; then vary non-hook work with
geometry and hook count fixed. Compare the same BPF object and transport with
runtime optimization off/on plus uninstrumented control. Do not add a manual
lane guard or silently discard per-lane side effects to obtain a speedup.
Separate logical hook encounters from actual scalar handler executions and
warmup from measured launches. If dynamic execution counters add work, collect
them in explicitly labelled untimed companion launches rather than penalizing
only one timed arm; static launch arithmetic alone is not an observed count.

The old script's strict-admission/preflight campaigns are historical and are
not prerequisites for these requested performance studies. Record ordinary
execution errors and preserve all raw numbers. This downstream code task is
queued behind the active original Fig.13 runner work; no fourth local model
session was created. No new hook-count measurements exist yet.

```text
Continue this SAME automatic-warp task and existing worktree /home/yunwei37/workspace/gpu/bpftime-auto-warp, branch revision/automatic-warp-execution at eef8a51. The earlier Qwen Next request ended in terminal HTTP 524 before producing code. Root verified the worktree is clean and is resuming with GLM after a separate Qwen Next continuation also ended with terminal HTTP 524. Do not repeat a broad survey or create another worktree/session. Root coordinates builds, GPU runs, reports, commits and branch pushes; you implement source with apply_patch. Read workspace AGENTS and local CLAUDE/CONTRIBUTING. No hashes/digests, new papers, paper edits, own subagents, PR/CI/review campaigns, driver install/reload, GPU runs, or short timeouts. Max 3 local OpenCode sessions globally; two other sessions own gpu_ext/kernel-driver source. You own ONLY bpftime-auto-warp source and isolated build outputs; shared frozen bpftime-table1-hostfix-plt and bpftime-sass-existing-application are read-only.

Scope/minimality: implement reusable automatic warp optimization in the existing compiler/runtime path, with an on/off switch for the requested matched measurement; do not change the input probe, observation contract, map/output transport, or special-case tool names. No general backend rewrite, no replacement benchmark harness. Write an incremental source patch early and do a modest scoped CPU build, repairing ordinary compile errors; do not wait to write until an entire alternative design is narrated.

User handoffs: /tmp/warp-hook-experiments-continue-handoff.md and /tmp/opencode/automatic-warp-measurement-plan.md. Their old running status is superseded by the explicit terminal error above. Required end state: same input BPF object runs with optimization disabled/enabled; eligible uniform work uses genuinely automatic leader execution and broadcasts, preserving observable side effects and outputs. Per-lane events must not disappear. No manual lane guard added to the probe, no marker-only demonstration, no warning bypass used to claim verifier-informed uniformity. Existing per-lane semantics may require generic helper batching/aggregation instead of whole-program deduplication; preserve required records and report which transformation actually applies. Merely leaving every relevant probe unoptimized or silently falling back is not completion. Handle actual active masks and predicated hook sites using the existing pass/runtime mechanisms.

Source starting points: attach/nv_attach_impl/pass/ptxpass_core/{include/ptxpass/core.hpp,src/core.cpp}, trampoline/default_trampoline.cu, nv_attach_impl_patcher.cpp, ptx_compiler/, the existing SIMT verifier and nearby tests. Read only relevant surrounding functions, not the whole repository. Build using documented CMake/Make targets; the existing Makefile build-gpu and component CMake targets are the starting point. Keep builds isolated; do not overwrite frozen Table1 runtime artifacts. Do not create a root docs/ directory in bpftime.

After reusable implementation builds, root will reuse gpu_ext run_table1_perf.py and existing llama.cpp pp512/TinyLlama setup for ten matched repetitions with uninstrumented baseline, optimization disabled and enabled. Root also owns the requested original device-microbenchmark block-count/work sweeps; you may provide narrowly necessary runtime count reporting to distinguish hook encounters from scalar handler executions, with equal counting costs across compared arms. Do not modify gpu_ext benchmark scripts from this session. Do not run performance cells yet or claim a desired speedup. Deliver actual source, build command/output and exact on/off invocation plus remaining limitations. Keep old modes/results intact and leave code uncommitted for root review and publication.
```
