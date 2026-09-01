# ASPLOS 2027 revision experiment handoff — 2026-08-31

The active work is implementation and evaluation against
[the revision plan](paper/asplos-27-rebuttal/revision-plan.md). One complete,
independently reviewed GPU experiment quantifies same-policy mechanism cost, and
one reviewed CPU/source safety bundle establishes verifier evidence plus the
remaining transition and provenance gaps. Other build and CPU-test milestones
remain dependencies rather than results.

The [verbatim author response and shepherd follow-up](revision-shepherd-comment.md)
are first-class revision requirements. In particular, completion requires more
than bringing up the three named baselines: the revised paper must compare
faithful gpubpf policy reimplementations with runnable original monolithic
implementations, attribute headline gains to policy versus mechanism, disclose
mechanism overheads, identify genuinely new agent-discovered policies, group
cross-cutting evidence under common headings, and fix the two shepherded
typographic issues.

## Persisted work

| Work | Evidence and current status |
| --- | --- |
| NVIDIA 610 port | [port branch](https://github.com/eunomia-bpf/gpu_ext-kernel-modules/tree/port/nvidia-610.43.02), including [build and runtime notes](https://github.com/eunomia-bpf/gpu_ext-kernel-modules/blob/port/nvidia-610.43.02/GPUBPF-PORT.md). All five modules build for 6.15.11 and 7.1.12; BTF verifies both struct_ops and all nine kfuncs. No runtime replacement/attach test yet. |
| MoE/model-weight SOTA | MoE-Infinity's [approved plan](../workloads/moe-infinity/plan.md) exhausted three repaired preflights without a valid request; its [runtime evidence](../workloads/moe-infinity/runtime-preflight.md) preserves each failure. The reviewed DeepSpeed fallback was [closed before execution](../workloads/deepspeed-zero-inference/availability.md): pinned Transformers stores the dominant MXFP4 expert projections as non-Parameter Triton tensors, outside official ZeRO-3 parameter offload. PowerInfer has no GPT-OSS path. No correctness or performance sample exists for any of these research artifacts. |
| LMCache NVMe | [Plan](../workloads/lmcache-disk/plan.md) and [review](../workloads/lmcache-disk/plan-review.md) approved. All eight CPU tests pass. Three real preflight attempts failed before any request: trace-path launch error, DeepGEMM FP8 scale-layout failure after loading the checkpoint, then an invalid 0.99 vLLM startup-memory budget. The reviewed protocol is closed without O_DIRECT, correctness, or performance evidence; a fourth attempt needs a newly reviewed protocol. |
| Agent-study artifacts (R7) | [Public entry-point index](eval/agent/README.md) links historical analyses and benchmark sources. The metric extractor now accepts explicit corpus/output paths. Original study sessions are absent from the old local directory; prompts/logs remain unreleased pending archive recovery and privacy review. |
| XSched | [Review](../workloads/xsched/plan-review.md) closed the first proposal after three rounds. The CUPTI/globaltimer epoch proof and interpretation categories require repair in a new proposal. The CUDA workload builds; no GPU result. |
| RTX 5090/NVBit | [Review](../workloads/llama.cpp/observability_overhead/revision-rq4/plan-review.md) closed the first proposal. Final runner defects were repaired but not independently approved. The frozen NVBit comparison requires a supported 575.x stack; a 610 diagnostic cannot satisfy R6. |
| Same-policy mechanism cost | [Approved plan](../workloads/uvm-policy-mechanism/plan.md), [semantic preflight](../workloads/uvm-policy-mechanism/results/preflight-1.md), 15 paired raw blocks, [analysis](../workloads/uvm-policy-mechanism/results/analysis.md), and [independent review](../workloads/uvm-policy-mechanism/result-review.md) are complete. On the scoped CPU-resident, non-first-touch UVM fault path, gpubpf adds 3.219% overhead versus the built-in no-prefetch switch (paired bootstrap 95% CI 2.247--4.202%). This is mechanism-cost evidence, not a policy improvement or application result. |
| Safety/design depth | [Approved plan](experiment/revision-safety/plan.md), [verifier results](experiment/revision-safety/phase-a-results.md), [transition audit](experiment/revision-safety/phase-b-transition-audit.md), [50-event reconciliation](experiment/revision-safety/phase-c-event-sources.md), and [fresh review](experiment/revision-safety/result-review.md) are complete. The verifier phase passes; transition validation is a production `GAP`; event provenance is 46 `SUPPORTED` plus 4 `PARTIAL`, with raw sessions absent. Aggregate disposition is reviewed `PARTIAL`. |
| Transition validator | [Five-round approved plan](experiment/revision-transition-validator/plan-review.md), [Phase A](experiment/revision-transition-validator/phase-a-results.md), [scheduler](experiment/revision-transition-validator/phase-b-scheduler-results.md), [prefetch](experiment/revision-transition-validator/phase-b-prefetch-results.md), and [PMM](experiment/revision-transition-validator/phase-b-pmm-results.md) results. Scheduler, UVM prefetch, and PMM production integration are ported across 575 and 610; host tests pass at 12 cases/145 assertions and all five modules build on each line. The PMM live preflight passes: kernel-native ioctl `NV_OK`, typed setter admitted, hidden write rejected with `-EACCES`, exact PMM totals 2/1/1/2, and distribution UVM restored. Disposition remains `PARTIAL` only for the five scheduler fixture outcomes and runtime scheduler path, which require a display maintenance window. |

## Complete revision requirement matrix

The active queue is broader than the three new R1 artifacts. Requirements are
separated below so that a build, qualitative discussion, or historical number
cannot be mistaken for a completed experiment.

| Revision item | Required experiment or evidence | Acceptance boundary | Current disposition |
| --- | --- | --- | --- |
| R1: MoE research baseline | Same-model, same-workload head-to-head against a runnable research MoE offload system on the RTX 5090. | Correct outputs, actual baseline-offload and gpubpf-hook engagement, valid paired blocks; a host-only policy remains an ablation, not the submitted full device-observed policy. | No runnable exact-model research artifact remains: MoE-Infinity exhausted three preflights, DeepSpeed is source-confirmed unable to manage GPT-OSS's non-Parameter MXFP4 expert tensors through official ZeRO-3, and PowerInfer lacks a GPT-OSS path. Do not weaken the workload. Proceed with the separately promised Expert Buffering policy implementation and record the three named artifact reasons. |
| R1: scheduling research baseline | XSched on sm_120, explicitly labeled public Level-1 inter-kernel preemption. | Correct epoch/timestamp proof, matched multi-tenant workload, policy engagement, and a predeclared interpretation that permits regressions. | First proposal closed after three reviews; a repaired proposal is needed. Orion is the named fallback if XSched cannot yield an accepted run. |
| R1: storage-tier KV baseline | LMCache local-NVMe backend versus CPU retention and recomputation. | Exact cache hits, O_DIRECT proof, deterministic outputs, and valid paired blocks; no failed preflight may become a performance sample. | First protocol closed after three failed real preflights and no served request. A new reviewed protocol or an explicitly documented fallback is required. |
| R1: existing GPREEMPT-equivalent evidence | Re-establish the foreground launch-latency and best-effort throughput claims on a matched, interleaved schedule. | Measure the intended host-submit-to-kernel-entry event, prove timeslice and preemption engagement per run, and report robust run-level effects rather than an outlier-driven mean. | Historical evidence audit found that the submitted 96% aggregation is driven by one native run and that the launcher does not establish preemption engagement. Fresh evidence is required before retaining the strong numeric claim. |
| R3: expert-management granularity | Quantify page-granular partial-expert reuse and compute/transfer overlap against an expert-atomic design. | Treat the MoE-Infinity deployment comparison separately from any causal granularity comparison; report unsupported cases as unavailable, not reproduced. | No dedicated accepted protocol yet. MoE-Infinity can provide system-level context but cannot alone identify the granularity effect. |
| R4: mechanism versus policy | Re-run the memory-policy and scheduling-policy comparison underlying Fig. 13. | Matched repeated blocks, successful attach and live-hit evidence, correct concurrent completion-time estimand, and separately reported memory/scheduling effects. | A new 15-pair same-policy page-fault study passed independent review and measures a 3.219% general-mechanism cost (95% CI 2.247--4.202%) for one scoped memory path. This answers one Shepherd question but does not repair the historical Fig. 13 policy-benefit or scheduling evidence; the 55--92% and under-1% claims remain unsupported by fresh revision runs. |
| R5: safety and design depth | Executable verifier/transition rejection cases plus a source-backed re-count of the 50 agent safety events. | Demonstrate lane-varying, loop/bounds, overflow, stale, and conflicting-transition handling without claiming full-stack formal verification. | Reviewed `PARTIAL`: five full-pipeline verifier tests pass (28 targeted assertions; full suite 137/23), covering seven unsafe/control pairs. Scheduler, UVM prefetch, and PMM consume the shared validator on both 575 and 610; host tests pass at 12 cases/145 assertions and both five-module builds pass. The PMM kernel-native ioctl and two PMM admission fixtures pass live with exact 2/1/1/2 outcomes. Five scheduler fixture outcomes and live scheduler commit remain open. The 50-row inventory remains 46 `SUPPORTED` plus 4 `PARTIAL`; original sessions are absent. |
| R6: current-hardware overhead | RTX 5090 Table 1 comparison including NVBit, plus the gpubpf device-side tools. | Same benchmark and measurement boundary, supported NVBit stack, exact tool/source identities, correctness before timing, repeated runs. | Hard rebuttal commitment. Current 610 driver is outside NVBit's frozen supported stack; the first proposal is closed and a new 575 execution protocol is required. |
| R7: study artifacts | Publish prompts, interaction logs, metric extraction, and benchmark harnesses. | Redact secrets/private data, retain raw provenance and explicit public inventories, and make every reported aggregate reproducible without file/content hash gates. | Hard artifact commitment. Public index and path-parameterized extractor are pushed; the original study sessions are still absent from the local archive. |
| R8: portability/deployment evidence | Audit the SASS patching prototype, the reported 273 ms one-time ptrace attach, the LD_PRELOAD route, and the approximately 100-LOC open-module patch boundary. | Bind each statement to runnable code or retained raw evidence; otherwise narrow it to design discussion. | The 610 Open Kernel Modules port is built and pushed, but that port is not a substitute for the promised SASS/attach/deployment evidence. |
| R2, R9, and LOC corrections | Expressibility table, design/discussion text, and corrected policy LOC arithmetic. | Cite concrete in-tree programs and distinguish measured, inferred, and out-of-scope claims. No fabricated experiment. | Pending paper work. CXL/GDS implementation, multi-vendor campaigns, LithOS-scale experiments, and upstream GPREEMPT binary reproduction remain explicitly out of scope. |
| Shepherd: policy versus mechanism | For each existing policy, establish whether gpubpf implements it and, when the original artifact is runnable, compare against the original ad-hoc/unsafe/monolithic implementation. Audit headline claims in the abstract and introduction for policy/mechanism attribution and disclose any general-mechanism cost. | Matching SOTA remains a valid outcome; no result may imply that a policy improvement was caused by the mechanism. Novel-policy claims require concrete code and evidence, especially for agent-discovered policies. | The first fresh matched result is complete: the same no-prefetch outcome through gpubpf costs 3.219% on a scoped real UVM fault path, with independent review and explicit limitation. Existing-policy coverage, application/SOTA comparisons, novel-policy evidence, and the paper-wide claim audit remain pending. |
| Shepherd: exposition and typography | Regroup scattered safety and SOTA-reimplementation answers under common labeled headings; fix Section 5.3 paragraph headings with two trailing periods and audit bibliography curly-brace escaping. | Preserve technical meaning and bibliography semantics; compile the paper and inspect the affected output. | Pending after experiment evidence stabilizes; the verbatim source comment is retained in the repository. |

The two author-response commitments that cannot be dropped are R6's RTX 5090
Table 1/NVBit result and R7's public prompts and benchmark harnesses. The fuller
revision plan additionally names the three R1 research artifacts and R3/R4
quantification. Evidence defects discovered during revision are treated as
requirements to repair, even when the original plan described the old figure
as merely being foregrounded.

## Live state and next actions

The host changed from Linux `6.15.11-061511-generic` to
`7.1.12-070112-generic` during this work. The installed driver remains official
NVIDIA Open Kernel Modules 610.43.02 via DKMS. The 610 port was rebuilt for the
new kernel; GCC 14 matches the installed DKMS build, but compiler/objtool
warnings remain documented in the port notes. Its BTF was generated with the
7.1 native script and the running kernel's base BTF, without editing system
headers or installed modules.

The GPU is idle. The mechanism-cost study safely reloaded only the unused custom
610 `nvidia_uvm` module between cells; it never unloaded display-owned core
NVIDIA modules. The final state is prefetch enabled, UVM refcount zero, and no
attached policy. GDM/Xorg still holds the core NVIDIA module. A request for permission to
temporarily stop GDM for full scheduling-module validation has not been
answered; display ownership is therefore unchanged.

While the GPU remains idle, memory-hook validation may replace only the unused
`nvidia_uvm` module while retaining the matching official 610 core.
Full scheduling validation needs an idle GPU and an authorized display
maintenance window. Use temporary `insmod`, with system `modprobe` as recovery;
never install custom modules persistently.

MoE and LMCache now record the pre-run deviation to one uniform 610.43.02
stack per comparison, preserving the reviewed workloads, correctness checks,
and analysis. MoE pins the four custom 7.1.12 modules from port commit
`74a036fe7b7c8701914f0703d802eb17269a730f`; LMCache uses the installed official
610 stack and needs no custom hooks. The MoE monitor's wire layout passes
23 compile-only assertions against the 610 headers. The MoE and LMCache CPU
tests pass; no runtime attach/event-delivery result is implied.

The reboot also renumbered the workspace Samsung 9100 PRO from `nvme1n1p1`
to `nvme0n1p1`. Both runners now identify its ext4 filesystem by UUID, not by
the unstable device number. Their fresh read-only admissions pass artifact,
driver-version, and storage checks. LMCache admission subsequently passed on
the idle GPU, but its three preserved preflights produced no usable result. The
third failure was caused by the experiment's own 0.99 memory budget, not a
foreign process. In accordance with the preflight stopping rule, LMCache is
routed out of execution until a new protocol is reviewed. MoE-Infinity likewise
exhausted its three repaired preflights without a paper timing sample. Both are
closed until newly reviewed protocols; do not pool samples across stacks. The
R5 verifier/source bundle remains reviewed `PARTIAL`; its production
transition-validator plan passed after five review rounds. The shared header
passes at 12 cases/145 assertions and now has production scheduler consumers on
both driver lines. Scheduler, UVM prefetch, and PMM production paths are now
implemented and ported. The UVM-only PMM live preflight passes its kernel-native
ioctl and both PMM verifier fixtures, and restores the distribution UVM module.
The five scheduler verifier outcomes remain open because the display-owned core
module was not replaced.
The NVBit 575
requirement is separate and cannot be waived by this port.

Publication now happens after each validated scoped change, as requested.
Dependencies, virtual environments, compiled modules, and old July diagnostic
logs remain outside source commits. Unrelated paper-submodule, FAISS, and
PyTorch worktree changes are preserved and were not staged.
The five bpftime observability-example changes are preserved as an exact
[patch artifact](../workloads/llama.cpp/observability_overhead/revision-rq4/gpubpf-observability.patch)
instead of modifying the unrelated active bpftime PR branch. This is a remote
recovery copy, not an upstream-reviewed bpftime change.
