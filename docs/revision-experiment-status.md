# ASPLOS 2027 revision experiment handoff — 2026-08-31

The active work is implementation and evaluation against
[the revision plan](paper/asplos-27-rebuttal/revision-plan.md). No new complete
GPU experiment or paper-facing performance result is claimed by this handoff.
The driver port and CPU tests are dependencies, not research results.

## Persisted work

| Work | Evidence and current status |
| --- | --- |
| NVIDIA 610 port | [port branch](https://github.com/eunomia-bpf/gpu_ext-kernel-modules/tree/port/nvidia-610.43.02), including [build and runtime notes](https://github.com/eunomia-bpf/gpu_ext-kernel-modules/blob/port/nvidia-610.43.02/GPUBPF-PORT.md). All five modules build for 6.15.11 and 7.1.12; BTF verifies both struct_ops and all nine kfuncs. No runtime replacement/attach test yet. |
| MoE-Infinity | [Plan](../workloads/moe-infinity/plan.md) and [review](../workloads/moe-infinity/plan-review.md) approved; [runtime evidence](../workloads/moe-infinity/runtime-preflight.md) records that full admission passed but the first 512-token warm-up hit upstream's fixed 256-row expert buffer with 353 routed rows. No request completed. The frozen protocol is closed and the MoE axis is routed to DeepSpeed ZeRO-Inference or PowerInfer. |
| LMCache NVMe | [Plan](../workloads/lmcache-disk/plan.md) and [review](../workloads/lmcache-disk/plan-review.md) approved. All eight CPU tests pass. Three real preflight attempts failed before any request: trace-path launch error, DeepGEMM FP8 scale-layout failure after loading the checkpoint, then an invalid 0.99 vLLM startup-memory budget. The reviewed protocol is closed without O_DIRECT, correctness, or performance evidence; a fourth attempt needs a newly reviewed protocol. |
| Agent-study artifacts (R7) | [Public entry-point index](eval/agent/README.md) links historical analyses and benchmark sources. The metric extractor now accepts explicit corpus/output paths. Original study sessions are absent from the old local directory; prompts/logs remain unreleased pending archive recovery and privacy review. |
| XSched | [Review](../workloads/xsched/plan-review.md) closed the first proposal after three rounds. The CUPTI/globaltimer epoch proof and interpretation categories require repair in a new proposal. The CUDA workload builds; no GPU result. |
| RTX 5090/NVBit | [Review](../workloads/llama.cpp/observability_overhead/revision-rq4/plan-review.md) closed the first proposal. Final runner defects were repaired but not independently approved. The frozen NVBit comparison requires a supported 575.x stack; a 610 diagnostic cannot satisfy R6. |

## Complete revision requirement matrix

The active queue is broader than the three new R1 artifacts. Requirements are
separated below so that a build, qualitative discussion, or historical number
cannot be mistaken for a completed experiment.

| Revision item | Required experiment or evidence | Acceptance boundary | Current disposition |
| --- | --- | --- | --- |
| R1: MoE research baseline | Same-model, same-workload head-to-head against a runnable research MoE offload system on the RTX 5090. | Correct outputs, actual baseline-offload and gpubpf-hook engagement, valid paired blocks; a host-only policy remains an ablation, not the submitted full device-observed policy. | MoE-Infinity full admission passed, but its first real warm-up deterministically exceeded the pinned 256-row expert buffer. No result. Select and review the named DeepSpeed ZeRO-Inference or PowerInfer fallback instead of weakening the frozen workload. |
| R1: scheduling research baseline | XSched on sm_120, explicitly labeled public Level-1 inter-kernel preemption. | Correct epoch/timestamp proof, matched multi-tenant workload, policy engagement, and a predeclared interpretation that permits regressions. | First proposal closed after three reviews; a repaired proposal is needed. Orion is the named fallback if XSched cannot yield an accepted run. |
| R1: storage-tier KV baseline | LMCache local-NVMe backend versus CPU retention and recomputation. | Exact cache hits, O_DIRECT proof, deterministic outputs, and valid paired blocks; no failed preflight may become a performance sample. | First protocol closed after three failed real preflights and no served request. A new reviewed protocol or an explicitly documented fallback is required. |
| R1: existing GPREEMPT-equivalent evidence | Re-establish the foreground launch-latency and best-effort throughput claims on a matched, interleaved schedule. | Measure the intended host-submit-to-kernel-entry event, prove timeslice and preemption engagement per run, and report robust run-level effects rather than an outlier-driven mean. | Historical evidence audit found that the submitted 96% aggregation is driven by one native run and that the launcher does not establish preemption engagement. Fresh evidence is required before retaining the strong numeric claim. |
| R3: expert-management granularity | Quantify page-granular partial-expert reuse and compute/transfer overlap against an expert-atomic design. | Treat the MoE-Infinity deployment comparison separately from any causal granularity comparison; report unsupported cases as unavailable, not reproduced. | No dedicated accepted protocol yet. MoE-Infinity can provide system-level context but cannot alone identify the granularity effect. |
| R4: mechanism versus policy | Re-run the memory-policy and scheduling-policy comparison underlying Fig. 13. | Matched repeated blocks, successful attach and live-hit evidence, correct concurrent completion-time estimand, and separately reported memory/scheduling effects. | Historical audit found one round per memory pair, missing scheduling engagement evidence, and invalid sequential wait timing. The 55--92% and under-1% claims are not yet fresh revision evidence. |
| R5: safety and design depth | Executable verifier/transition rejection cases plus a source-backed re-count of the 50 agent safety events. | Demonstrate lane-varying, loop/bounds, overflow, stale, and conflicting-transition handling without claiming full-stack formal verification. | Writing exists in the submitted paper, but the revision-grade rejection bundle and raw-event reconciliation remain pending. |
| R6: current-hardware overhead | RTX 5090 Table 1 comparison including NVBit, plus the gpubpf device-side tools. | Same benchmark and measurement boundary, supported NVBit stack, exact tool/source identities, correctness before timing, repeated runs. | Hard rebuttal commitment. Current 610 driver is outside NVBit's frozen supported stack; the first proposal is closed and a new 575 execution protocol is required. |
| R7: study artifacts | Publish prompts, interaction logs, metric extraction, and benchmark harnesses. | Redact secrets/private data, retain raw provenance and public hashes, and make every reported aggregate reproducible. | Hard artifact commitment. Public index and path-parameterized extractor are pushed; the original study sessions are still absent from the local archive. |
| R8: portability/deployment evidence | Audit the SASS patching prototype, the reported 273 ms one-time ptrace attach, the LD_PRELOAD route, and the approximately 100-LOC open-module patch boundary. | Bind each statement to runnable code or retained raw evidence; otherwise narrow it to design discussion. | The 610 Open Kernel Modules port is built and pushed, but that port is not a substitute for the promised SASS/attach/deployment evidence. |
| R2, R9, and LOC corrections | Expressibility table, design/discussion text, and corrected policy LOC arithmetic. | Cite concrete in-tree programs and distinguish measured, inferred, and out-of-scope claims. No fabricated experiment. | Pending paper work. CXL/GDS implementation, multi-vendor campaigns, LithOS-scale experiments, and upstream GPREEMPT binary reproduction remain explicitly out of scope. |

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

The unrelated SGLang processes later exited without intervention, leaving no
compute process on the GPU. No process was killed and no module was unloaded.
GDM/Xorg still holds the core NVIDIA module. A request for permission to
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
routed out of execution until a new protocol is reviewed. The next admitted
experiment is MoE-Infinity, whose approved protocol has consumed zero real
preflight attempts. Do not pool samples across stacks. The NVBit 575
requirement is separate and cannot be waived by this port.

Publication now happens after each validated scoped change, as requested.
Dependencies, virtual environments, compiled modules, and old July diagnostic
logs remain outside source commits. Unrelated paper-submodule, FAISS, and
PyTorch worktree changes are preserved and were not staged.
The five bpftime observability-example changes are preserved as an exact
[patch artifact](../workloads/llama.cpp/observability_overhead/revision-rq4/gpubpf-observability.patch)
instead of modifying the unrelated active bpftime PR branch. This is a remote
recovery copy, not an upstream-reviewed bpftime change.
