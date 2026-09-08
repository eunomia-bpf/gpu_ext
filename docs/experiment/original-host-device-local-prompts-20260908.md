# Original host/device implementation prompts

2026-09-08. Local OpenCode implementation tasks; no new GPU measurements yet.
Qwen Next's earlier FineMoE session ended in a terminal HTTP 524 without a
patch. These tasks implement existing paper-described logic, not new policies.

## Additional user-authorized queue — 04:42 UTC

The existing three sessions (disk UVM, XSched, Hummingbird) were confirmed busy
through the OpenCode API. Do not stop them or launch a fourth local session.
The user has also transferred continuation of the following work from the
paper task to this experiment task; manuscript changes remain prohibited.

1. Resume automatic warp execution in
   `/home/yunwei37/workspace/gpu/bpftime-auto-warp`, branch
   `revision/automatic-warp-execution`, at the next free local-model slot.
   The earlier Qwen Next session `ses_f80dc8da2ffev9TpuxcuajuRMa` has an actual
   terminal `APIError`, HTTP 524, completed at 1788840775838 ms Unix time.
   Its worktree is clean; there is no implementation patch to discard or
   duplicate. Continue through Qwen Next if the provider is available.
2. Finish reusable compiler/runtime automatic warp execution with the same
   input BPF object, output semantics and transport on both paths. No manual
   lane guard, probe-specific exception, or discarded per-lane observations
   may stand in for automatic optimization. Reuse `run_table1_perf.py` and
   the existing llama.cpp setup for ten paired repetitions of optimization
   disabled/enabled, with uninstrumented controls. Preserve old measurements.
3. Extend the original device microbenchmark for two complementary sweeps:
   vary block count at fixed per-thread work; then fix launch geometry and
   hook count while varying arithmetic work outside the hook. Record actual
   hook encounters separately from scalar BPF handler executions, elapsed
   time, controls and repetitions. Counting overhead must not selectively
   penalize one timed arm. Reuse the original kernel/timing/script instead
   of creating a replacement harness. These experiments await implementation;
   no new numbers or completion claim exist.
4. The [original Fig.13 runner follow-up](../eval/multi-tenant-memory/combined-followup-plan-20260908.md)
   remains queued, not cancelled. Coordinate GPU and struct-ops ownership;
   run performance cells serially, not alongside another GPU campaign.

Source handoffs received from the paper task are
`/tmp/warp-hook-experiments-continue-handoff.md`,
`/tmp/opencode/automatic-warp-measurement-plan.md`, and
`/tmp/fig13-reuse-original-scripts-handoff.md`. The old automatic-warp plan's
"running" status is superseded by the terminal API evidence above. The
worktree remains needed for this authorized continuation and is not a cleanup
target. No additional paper reproduction or download is requested.

## XSched — Qwen 27B

Session: `ses_f80c49c7cffebjDrdYLKy8X0ZC`.

```text
Workspace /home/yunwei37/workspace/gpu/gpu_ext. User requests faithful ORIGINAL host+device policy implementation, not a new algorithm. Root reviews/builds/runs/commits/pushes; local models implement code. No paper edits, new papers/downloads, own agents, new worktrees, hashes/digests, artificial short timeouts or correctness/preflight campaigns. Read appropriate AGENTS.md. Preserve all old data and modes; no GPU tests, driver installs/reloads, or loaded-BPF changes. CPU builds of your scoped implementation allowed, keep parallelism modest. Other tasks own driver worktree and a different workload directory. Use apply_patch for source edits. Provide actual code and ordinary build command, not another broad literature survey. Context200k: prioritize bounded source deliverable and concise checkpoint over repeated broad reads. Existing original paper PDFs and completed implementations are authoritative; never treat CUDA workload compute as device BPF.
You own only workloads/xsched/ source, preferably an isolated level2 subdirectory and a patch if the existing vendored XSched needs changes. Do not mutate frozen Level1 build/deps in place, SASS compiler shared files, bpftime worktrees, Hummingbird or LMCache.
Implement XSched's original Level2 guardian rule as a real device BPF function, paired with its original HPF rule on host BPF. Local PDF docs/paper-material/ref-paper/xsched_osdi25.pdf §6.2 Figs7–8 explicitly has host deactivation flag, device guardian abort/record, host clearing and command replay. Existing report workloads/xsched/README.md is only Level1 on sm120; do not relabel its numbers as Level2. Source:
- deps/xsched/platforms/cuda/hal/src/level2/instrument.cpp
- deps/xsched/platforms/cuda/hal/src/arch/sm86.cpp and arch.cpp
- deps/xsched/platforms/cuda/hal/src/level2/cuda_queue.cpp
- deps/xsched/platforms/cuda/hal/include/xsched/cuda/hal/level2/
- bpftime_hpf.* and priority_workload.cu
Preserve actual source protocol for queue identity, command index/start/abort replay; paper Fig7 is simplified, don't abort half an already executing command by naively rereading a mutable flag independently in every thread. Native and BPF variants must share the same protocol/actuator and workload; replace only bounded decisions in the BPF arm. Existing architecture list lacks sm120 guardian, so implement a faithful portable guardian adapter for sm120 and label it as a port if the original injector is unavailable. Do not substitute Level1, standalone return42, counter-only instrumentation, or arbitrary running-kernel cancellation and call that completion.
Reuse real BPF->PTX/SASS support read-only from workloads/sass-kretprobe, /home/yunwei37/workspace/gpu/bpftime-sass-existing-application and /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt. The current SASS EXIT tool already embeds device-callable BPF in NVBit; this task needs entry control affecting real command execution, not just observation. Trusted entry/abort/replay glue may execute device BPF result; BPF itself must be real compiled eBPF, no host fallback.
Deliver a runnable matched native/BPF Level2 port on the existing real priority workload and build/runner glue, retaining original HPF semantics. If a hook/interface genuinely cannot supply required control, report exact boundary and implement the smallest necessary in-scope adapter. No general SASS backend rewrite, no fabricated success or latency. Root will coordinate performance run after build.
```

## Hummingbird — GLM

Session: `ses_f80c496aeffeBh330rKhrRwPgX`.

```text
Workspace /home/yunwei37/workspace/gpu/gpu_ext. User requests faithful ORIGINAL host+device policy implementation, not a new algorithm. Root reviews/builds/runs/commits/pushes; local models implement code. No paper edits, new papers/downloads, own agents, new worktrees, hashes/digests, artificial short timeouts or correctness/preflight campaigns. Read appropriate AGENTS.md. Preserve all old data and modes; no GPU tests, driver installs/reloads, or loaded-BPF changes. CPU builds of your scoped implementation allowed, keep parallelism modest. Other tasks own driver worktree and a different workload directory. Use apply_patch for source edits. Provide actual code and ordinary build command, not another broad literature survey. Context200k: prioritize bounded source deliverable and concise checkpoint over repeated broad reads. Existing original paper PDFs and completed implementations are authoritative; never treat CUDA workload compute as device BPF.
You own only workloads/hummingbird/ source. Preserve existing idle and pipeline results/builds; implement opt-in new outputs in an isolated host-device subdirectory, using patch/preparation conventions rather than overwriting frozen GPreempt frontend or shared source. Do not edit XSched, LMCache, SASS shared compiler or bpftime worktrees.
Implement ORIGINAL Hummingbird host+device split with no new scheduling heuristic. Local PDF docs/reference/2026-hu-hummingbird-v2.pdf §4.2 PTX kernel transformation adds offset_x/y/z to original device blockIdx; §4.3 host splitting/consolidation/idle policy. Existing host idle selector is actual uBPF, but device coordinate remapping is not BPF. Keep the original splitting budgets, host scheduler and pipeline settings fixed. Replace consumed device logical-block coordinate mapping with genuine device BPF, preserving exactly the original mapping. Do not merely write a marker or unused coordinates.
Read split_grid.h, prepare/build scripts and pipeline/{prepare.py,Makefile,runner.patch,results-575-20260903.md}; find actual split cubin and transformations on the established real DNN workload. Reuse POD's real device selector machinery or bpftime PTX compiler through local adapter as needed. Read-only assets: workloads/pod-attention, workloads/sass-kretprobe, /home/yunwei37/workspace/gpu/bpftime-sass-existing-application, /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt. Do not edit shared frozen assets.
Native device mapping vs BPF device mapping must use same context and same host inputs. Provide modes sufficient to distinguish original native host+native device, BPF host+native device, and BPF host+BPF device; baseline continues to use existing unsplit native frontend. Avoid duplicating full framework or inventing new workloads; implementation plus existing runner integration is task. Preserve thread/CTA semantics: coordinates must actually drive original kernel memory accesses. A trusted wrapper may apply BPF outputs, but C fallback must never provide BPF decisions silently.
Deliver real source patch and ordinary build command, plus minimal invocation for current frontend and honest hook limitations if any. No GPU runs yet; root serializes measurement. Do not turn this into new feedback algorithm or separate observability benchmark.
```
