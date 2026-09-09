# Original host/device implementation prompts

## Runtime-directed continuation — 2026-09-08 23:28 PDT

The native XSched metadata candidate builds after refreshing CMake's source
inventory and bracing three disabled-debug conditional bodies. Actual launch
still fails; the recorded debugger diagnosis in `fefe8898` shows the shim's
`cuLibraryLoadData` does enter `XgMetaExtendImage`, but its argument is CUDA's
version-1 `__fatBinC_Wrapper_t` (`0x466243b1`), not the bare fatbin/ELF images
the candidate handles. Root asked the same GLM session to preserve this
wrapper and patch its nested image. Do not add new proc-address interception
based on the earlier absence of an owned-copy log. This is a concrete
implementation gap, not a successful Level-2 performance measurement.

The tool-path Qwen session's index inference also needed correction:
`preempt/include/xsched/preempt/xqueue/async_xqueue.h` initializes
`next_hw_cmd_idx_` to 1, and `async_xqueue.cpp` assigns it with `fetch_add`.
`priority_workload.cu` computes zero-based `flat = stream_idx *
tasks_per_stream + task_idx`. Consequently, `preempt_idx=1` is command 1 /
task 0; the missing sink element 87040 at the measured 340 x 256 geometry
belongs to task 1, not task 0. It is incorrect to equate the missing task
with the resumed command solely from those numbers. This correction was
sent to the same session before its next device-protocol change.

LMCache remains actual put/get integration work, not another primitive
campaign. Its Qwen session naturally reached a response-length limit and
was resumed on the same session; it was not cancelled. Root clarified that
allocation padding is only for VA alignment: register exactly the existing
KV span when its size is a supported 2 MiB multiple, otherwise retain the
stock path. Do not extend/truncate KV files, compare whole-buffer snapshots
in the timed path, or repeat preparation inside every demand read. Await
real asynchronous writeback completion without an arbitrary short timeout.
After a subsequent oversized write-call error, the still-live session was
asked to prefer small incremental edits if that write failed again.

All three local sessions remain allocated to these tasks; automatic
compaction is live work. Grouped full-record layout optimization remains
queued for a free slot. No manuscript, completed cell, or global driver
configuration is changed by these continuation instructions.

## Coordinator continuation notes — 2026-09-08

OpenCode permission queries and replies must use the same `directory` query
parameter as the active session. An unscoped `/permission` query returned
an empty list while the XSched tool session was actually waiting for access
to its task-specific temporary patch directory. Root found the request with
`/permission?directory=%2Fhome%2Fyunwei37%2Fworkspace%2Fgpu%2Fgpu_ext`,
checked the operation, and replied `once` to that request using the same
directory parameter. Do not interpret a pending tool call as model inference
without checking this scoped endpoint. Do not restart or terminate the
session to resolve a permission request. Buffered SSE events may describe
older requests; current scoped API state is authoritative.

Hummingbird integration must retain the existing inline native coordinate
mapping as an original-implementation control. The new callable native
wrapper is an additional adapter control, not the original inline path:
comparing only BPF against that wrapper would omit their shared wrapper
overhead. Keep the same workload, arrival trace, host rule and tile budgets.
The compiled components in `504e8fc9` and `c239abb7` do not yet establish
actual DNN kernel consumption or a host/device performance result.

## Provider recovery — 2026-09-08 17:58 PDT

The resumed Hummingbird Qwen Next request
`msg_0839582f5001K3LnUd2wCBRNt7` exhausted provider retries and ended with
`APIError`, HTTP 524, completion time `1788915359547` ms Unix. It delivered
no code. Root observed the session disappear from active status and the
terminal error before resuming the same Hummingbird session on the available
`spark-direct-qwen27/qwen3.8-27b-nvfp4` route. This is not a silence timeout,
a discarded implementation, or a fourth session. Original scope is unchanged.
Active work remains XSched tool Qwen 27B, XSched native GLM, and Hummingbird
Qwen 27B. Three distinct models were attempted; Qwen Next's service failure
necessitates this fallback. No new Hummingbird result is claimed.

## Live continuation — 2026-09-08 17:34 PDT

The SoA implementation/measurement and its queued read-only paper feedback
finished naturally (`finish=stop`) in session
`ses_f7d059dcdffeTZqUX0xZIV5cYc`. Root used this freed slot to resume the
existing Hummingbird session `ses_f80c496aeffeBh330rKhrRwPgX` on
`spark-gateway/qwen3.8-flash-next-nvfp4-220k`. Its earlier HTTP 524 was a
confirmed terminal API error, not a silence timeout imposed by root.
The current API reports Hummingbird busy. Qwen 27B continues XSched tool
repair (`ses_f7d8b1dbeffeMN1yhvQzVm7VsS`), and GLM continues native sm_120
repair (`ses_f7d4e2503ffesJKDZlvxdw0z0A`): three active sessions total.
No Hummingbird host/device completion or new performance result is claimed.
The completed host-only campaigns remain untouched.

Exact continuation prompt:

```text
Resume this existing Hummingbird task from its actual terminal HTTP524, preserving the original scope in docs/experiment/original-host-device-local-prompts-20260908.md (Hummingbird section). One slot is now free because SoA + its queued readonly review finished naturally; there will be only three sessions: Qwen27 XSched tool repair, GLM XSched native repair, and this Qwen Next Hummingbird task. No new papers, manuscript edits, hashes, worktrees, own subagents, GPU runs or driver changes. Read current source and implement the original consumed device logical-block mapping through genuine device BPF, paired with the existing host idle policy, native and host-only controls. Do not re-run completed 50-cell idle or pipeline batches. You own only workloads/hummingbird/ and may add opt-in host-device files; do not edit shared exporter/runtime/XSched. Root builds/runs/commits; provide source patch and ordinary build command. Reuse exporter read-only from workloads/xsched/level2/bpf/bpf_to_ptx_ctx48.patch or workloads/sass-kretprobe if helpful, preserving coordinate semantics in real DNN memory accesses; no marker-only substitute. Focus a bounded runnable implementation over another survey. No artificial timeout; continue until implementation handoff or actual error.
```

2026-09-08. Local OpenCode implementation tasks; no new GPU measurements yet.
Qwen Next's earlier FineMoE session ended in a terminal HTTP 524 without a
patch. These tasks implement existing paper-described logic, not new policies.

Later execution update: Hummingbird ended with terminal HTTP 524 at
1788844635421 ms Unix time, without a source patch. Root confirmed its CLI
exit and resumed the original automatic-warp session in the freed slot using
GLM; see the [exact continuation prompt](automatic-warp-continuation-20260908.md).
Subsequently XSched also ended with terminal HTTP 524 at 1788845412755 ms
Unix time, and its CLI exited without an implementation patch. Its freed
slot now runs the original Fig.13 runner task with Qwen Next, session
`ses_f8079f034ffexM3YIHNS6mzN8I`; the exact prompt is appended to the linked
Fig.13 follow-up plan. Disk UVM and automatic warp execution remain live.
XSched and Hummingbird are still unfinished and retain their session contexts.
Earlier running/model
descriptions below are historical snapshots, not a claim of current execution.

Execution update after runtime checkpoint `c4c83cd`: automatic-warp GLM
completed its source handoff naturally (`finish=stop`, CLI exit 0). Root
resumed the existing XSched session `ses_f80c49c7cffebjDrdYLKy8X0ZC` with
`spark-gateway/qwen3.8-flash-next-nvfp4-220k`; its API reports busy. The other
two active sessions remain Table1 and disk UVM on direct Qwen 27B. This is
three active sessions, not a fourth; no session was stopped for silence.
The original XSched scope below is unchanged, except heavy CPU builds also
wait for the current GPU experiment owner's timing window. This resumption
is not a completed Level2 implementation or a new performance result.

The resumed XSched request subsequently ended with terminal HTTP 524 at
1788869659115 ms Unix time after its provider retries; the CLI also exited.
Root resumed the same session with GLM, without aborting the failed request
or discarding context. Table1's existing session also now uses GLM after a
natural Qwen request boundary; disk UVM remains on direct Qwen 27B. These
are still three sessions. Qwen Next was attempted but did not produce a
source patch, so the preferred three-distinct-model arrangement is not
currently available through that route. No model was stopped for silence.

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

## Hummingbird — GLM, then Qwen Next in the same session

The user prefers one active session per local model. Root queued a model
selection message without interrupting the pending GLM request. The request
finished naturally at 1788843284841 ms Unix time, and the next assistant
request at 1788843284846 uses
`spark-gateway/qwen3.8-flash-next-nvfp4-220k`. This is a continuation with the
same session and source context, not a restarted task or a fourth session.
XSched remains on Qwen 27B and disk UVM on GLM. No implementation completion
or performance result is implied by the model change.

Session: `ses_f80c496aeffeBh330rKhrRwPgX`.

```text
Workspace /home/yunwei37/workspace/gpu/gpu_ext. User requests faithful ORIGINAL host+device policy implementation, not a new algorithm. Root reviews/builds/runs/commits/pushes; local models implement code. No paper edits, new papers/downloads, own agents, new worktrees, hashes/digests, artificial short timeouts or correctness/preflight campaigns. Read appropriate AGENTS.md. Preserve all old data and modes; no GPU tests, driver installs/reloads, or loaded-BPF changes. CPU builds of your scoped implementation allowed, keep parallelism modest. Other tasks own driver worktree and a different workload directory. Use apply_patch for source edits. Provide actual code and ordinary build command, not another broad literature survey. Context200k: prioritize bounded source deliverable and concise checkpoint over repeated broad reads. Existing original paper PDFs and completed implementations are authoritative; never treat CUDA workload compute as device BPF.
You own only workloads/hummingbird/ source. Preserve existing idle and pipeline results/builds; implement opt-in new outputs in an isolated host-device subdirectory, using patch/preparation conventions rather than overwriting frozen GPreempt frontend or shared source. Do not edit XSched, LMCache, SASS shared compiler or bpftime worktrees.
Implement ORIGINAL Hummingbird host+device split with no new scheduling heuristic. Local PDF docs/reference/2026-hu-hummingbird-v2.pdf §4.2 PTX kernel transformation adds offset_x/y/z to original device blockIdx; §4.3 host splitting/consolidation/idle policy. Existing host idle selector is actual uBPF, but device coordinate remapping is not BPF. Keep the original splitting budgets, host scheduler and pipeline settings fixed. Replace consumed device logical-block coordinate mapping with genuine device BPF, preserving exactly the original mapping. Do not merely write a marker or unused coordinates.
Read split_grid.h, prepare/build scripts and pipeline/{prepare.py,Makefile,runner.patch,results-575-20260903.md}; find actual split cubin and transformations on the established real DNN workload. Reuse POD's real device selector machinery or bpftime PTX compiler through local adapter as needed. Read-only assets: workloads/pod-attention, workloads/sass-kretprobe, /home/yunwei37/workspace/gpu/bpftime-sass-existing-application, /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt. Do not edit shared frozen assets.
Native device mapping vs BPF device mapping must use same context and same host inputs. Provide modes sufficient to distinguish original native host+native device, BPF host+native device, and BPF host+BPF device; baseline continues to use existing unsplit native frontend. Avoid duplicating full framework or inventing new workloads; implementation plus existing runner integration is task. Preserve thread/CTA semantics: coordinates must actually drive original kernel memory accesses. A trusted wrapper may apply BPF outputs, but C fallback must never provide BPF decisions silently.
Deliver real source patch and ordinary build command, plus minimal invocation for current frontend and honest hook limitations if any. No GPU runs yet; root serializes measurement. Do not turn this into new feedback algorithm or separate observability benchmark.
```

## Current continuation — 2026-09-08 22:20 PDT

The Hummingbird comparison is complete in `eb4574c5`: five blocks / twenty
cells. Do not repeat it. Its existing session
`ses_f80c496aeffeBh330rKhrRwPgX` now uses direct Qwen 27B and has consumed
the queued LMCache task. Root's scope message `msg_08491dab20018rwwf8DDgmq6F5`
specifies real serving/KV integration with driver `dea1fefc`, reusing the
measured disk-UVM primitive. The current read-only backing contract applies
to completed immutable chunks, not the mutable live vLLM KV pool. The
existing native/BPF policy comparison and ordinary serving runner should
be retained. Code integration and new serving performance remain unfinished.
The unused Hummingbird launcher is WIP, not a prerequisite for LMCache.

Two other sessions remain live; root has not stopped or replaced them:

- Tool actuator: direct Qwen 27B, `ses_f7d8b1dbeffeMN1yhvQzVm7VsS`.
  Root's source observation in `msg_08496b81f001Y8J1JJ9P5MjFDI` is that
  `LaunchWorker` calls `OnXQueueCreate` to set the current CUDA context,
  whereas scheduler-thread `AsyncXQueue::Resume` calls
  `CudaQueueLv2::Reactivate` without that setup. This is a candidate
  explanation for the difference between ordinary and replay submissions,
  not a demonstrated cause. Existing counters run after API/function
  filters; their absence does not prove that no NVBit callback occurred.
  The model is checking scoped context setup before a broader replay-worker
  rewrite. Only a changed failed configuration will be retried.
- Native actuator: GLM, `ses_f7d4e2503ffesJKDZlvxdw0z0A`.
  Its metadata-extension candidate remains in the isolated native source.
  Root's `msg_08499264e001nAFMjI2Eb7EQZV` identifies two concrete defects:
  the returned vector data pointer is incorrectly deleted as a vector
  object, and the inferred 10,856-byte ELF extent truncates the real
  11,136-byte input. Existing local crash output is consistent with the
  first defect; no new GPU run of this candidate has occurred. The model
  must also preserve non-target kernels and reconcile the parameter-size
  records with the constant section, rather than assume a single changed
  field implements compiler-equivalent metadata. Its launch-error
  propagation patch is already published separately in `2b04ecaf`.

At the scoped status check all three sessions were busy with no pending
permissions. GLM's automatic context compaction resumed in the same session;
historical SSE error/idle events are not its current state. GPU was idle and
root held neither experiment lock. Root reviews, builds, measures and
publishes; local models implement nontrivial code. No manuscript, new paper
reproduction, clock gate, or extra correctness campaign is part of this
continuation. These are implementation findings and task handoffs, not new
performance results.
