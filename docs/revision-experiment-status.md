# ASPLOS 2027 revision experiment handoff

The [completion checklist](revision-completion-checklist.md) tracks the full
author/shepherd commitments separately from experiment completion. The
[verbatim comments](revision-shepherd-comment.md) remain the scope authority.

## User-confirmed objective — 2026-09-03

The user confirmed that **correctly implementing the existing policies in BPF
and recording their measured performance is sufficient**; outperforming the
original policy is not a completion requirement. Each comparison must still
show baseline, native policy, and actual BPF execution, with correctness,
engagement, raw results, and implementation/compatibility limits retained.
Reproducing an algorithm does not imply reproducing the paper's original
hardware, transport, or headline performance numbers.

### Active follow-on objective

The requested MoE-Infinity, XSched, GPreempt, Expert Buffering, FineMoE,
Hummingbird and POD-Attention comparisons are complete within the boundaries
below. Matching an existing policy is sufficient; none of the close native/BPF
results is promoted to formal equivalence. The canonical 48-paper
[expressibility ledger](experiment/policy/reference/RELATED_POLICY_EXPRESSIBILITY.md)
separates whole-system feasibility from the strongest local evidence.

RTX 5090 Table 1 now has a valid two-tool result: all five correctness cells
and 10/10 randomized pp512 blocks pass for matched exit records and exit-count
histograms. The outcome is mixed, and the cross-clock `launchlate` comparison
remains invalid. Its [public RM/PTIMER canaries and versioned endpoint fallback](../workloads/llama.cpp/observability_overhead/revision-rq4/launch-clock-recovery/rm-correlation-results.md)
show why: public xfer/direct brackets fail at 4.759/4.730 us, while the distinct
driver endpoint passes Phase 0 with 200/200 samples and a 0.759 us median
bracket. PTIMER/%globaltimer identity and the 220-launch gates are still open,
so this is not completion of the original three-tool campaign. The separate
[A1 admission study](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-a1/results-a1-575-02-20260904.md)
measures one-time STRICT checks for the two real callbacks, and the completed
[S0 campaign](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-s0/results-s0-575-02-20260904.md)
finds no detected steady-state STRICT/NO_VERIFY difference but predeclares no
equivalence margin. Neither result relabels the accepted verifier-off table.
The separate operation-matched
[device-map campaign](../microbench/fig15-device/results-map-tier-full-575-06-20260904.md)
completed 128/128 fresh processes: direct host mapping is 9.4307x the
device-resident update latency and 1.0904x the lookup latency, with both 97.5%
paired intervals above 1. This is a single-block scalar-runtime placement
result, not the old generic 6000x RPC/PCIe claim. A strict check on the exact
object accepts only its no-op callback: all six per-lane map callbacks violate
the current branch/key/value uniformity rules, so the placement result remains
explicitly verifier-off. The CPU-only
[device-verifier scaling campaign](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-scaling/results-verifier-scaling-575-01-20260904.md)
accepted 200/200 programs through 4,096 instructions but contradicted the
near-linear hypothesis for straight-line programs; it is a program-shape
admission boundary, not device execution or verifier soundness evidence.
Original agent transcripts also remain unavailable. LMCache local disk is
explicitly **paused by user
direction** after its cross-arm correctness failure and has no formal
performance result.

### Current baseline -> native policy -> BPF ledger

| Comparison | Primary measured sequence | Completion and boundary |
| --- | --- | --- |
| MoE-Infinity | **11.8964 -> 11.2233 -> 11.1900 token/s** | [Five blocks / 15 cells complete](../workloads/moe-infinity/results-paper-v3-protected-575.md). This is a same-frontend paper-algorithm port; the baseline alone has a prefill overload shortcut, so baseline/policy is not a pure policy contrast. |
| XSched HPF | LC p99 **76.8780 -> 26.9784 -> 27.2502 s**; BE **10.2377 -> 10.1497 -> 10.1616 kernels/s** | [Ten blocks plus controls / 46 cells complete](../workloads/xsched/performance-full-575-20260903.md). Original Level-1 executes suspend/resume; BPF supplies bounded HPF decisions. No Level-3 or formal equivalence claim. |
| GPreempt, continuous BE | LC response p99 **1.795937 -> 1.614817 -> 1.610008 ms**; BE **197.717 -> 179.967 -> 180.100 req/s** | [45-cell load study complete](../workloads/gpreempt/results-load-study-575-20260903.md); the separate [27-cell knee sweep](../workloads/gpreempt/results-lc-knee-575-20260903.md) exposes increasing BE starvation and conditional overload at 800 LC requests/s. Host-mapped compatibility is not original GDRCopy reproduction. |
| Expert Buffering Section VI | Same-K FIFO -> native -> BPF **5.508 -> 5.663 -> 5.621 token/s** | [Five blocks / 15 cells complete](../workloads/expert-buffering-policy/section-vi/results-performance-575-20260903.md). Native/BPF decisions match; BPF/native throughput is -0.74% [-1.40%, -0.20%]. This is not the original distributed system. |
| FineMoE dynamic set | Demand-only **5.1713**, all-positive **3.6228** -> native **4.4994** -> BPF **4.5144 token/s** | [Five blocks / 20 cells complete](../workloads/finemoe/results-performance.md). Dynamic selection cuts unused speculation versus all-positive but remains 12.62% below demand-only; BPF/native is inconclusive. |
| Hummingbird idle admission | Periodic BE native/fixed controls **179.2333 / 164.1500** -> C **133.2167** -> BPF **133.0500 req/s**; BurstGPT **179.1500 / 164.6167 -> 133.1667 -> 132.8167** | [Five-arm 50-cell study complete](../workloads/hummingbird/results-575-20260903.md), with a separate [40-cell bound-2 ablation](../workloads/hummingbird/pipeline/results-575-20260903.md). The conservative ports lose 19--20% to fixed GPreempt; bound 2 recovers about 15% over bound 1 without proving unchanged LC protection. |
| POD-Attention device selector | At the fixed phase shape, inline -> CUDA adapter -> BPF **3.4430 -> 3.4567 -> 3.5120 ms** CUDA-event latency | [250-cell shape study](../workloads/pod-attention/results-575-20260903.md) and [15-cell phase decomposition](../workloads/pod-attention/results-phase-full-575-01.md) complete. Matched BPF/CUDA operator cost is +1.78% [1.64%, 1.92%]; the current fresh-process pre-Python path is about 271 s and strict verification was off. |

These are the authoritative completed performance comparisons. Strict-device
engagement, raw-map readback and trampoline scaling are separate mechanism
studies, not extra policy-performance rows. GPU timing was exclusive; builds,
preflights, partial attempts and failed cells are not pooled into these values.

### Historical execution trace (superseded by the ledger above)

The next real-GPU dependency is now complete: Hummingbird's split-model/copy
calibration verified 23 ResNet152 and 102 VGG19 outputs, and an isolated
60-second LC reference verified all 6,000 offered requests. Its SLO is fixed
at 1,811,879 ns. All 20 small-bubble qualification cells passed independent raw
audit; output-copy admission qualified, while input admission did not actually
engage and remains disabled. The ten-cell, five-arm preflight also completed,
including 62,623,882 actual BPF JIT decisions and all 5,460 foreground requests
verified inside their windows. The full 50-cell performance matrix has now
passed independent raw audit, including 768,248 numerically verified timed
requests and 1,907,838,828 actual host-JIT decisions. It does not support
throughput recovery: C/fixed BE ratios are 0.81149 and 0.80160 for periodic
and BurstGPT, and BPF/fixed ratios are 0.80685 and 0.79693. BPF/C is about
0.994 in both; periodic SLO attainment also falls by 1.743 percentage points.
The BurstGPT equal-timeslice control misses 317 in-window LC completions;
its conditional p99 cannot support a headline protection speedup. These
results bound the conservative single-in-flight port, not the original full
Hummingbird system. The full raw bundle is committed and pushed in `2658e0e`. See the
[current results and raw calibration](../workloads/hummingbird/results-575-20260903.md).
FineMoE's native/actual-host-BPF selector passes independent arithmetic tests;
the full original Qwen model has now completed a normal 73-request reference
and nine exact repeated-output checks. All 18 saved original/repeated FP32
arrays passed independent numerical recomputation with zero error. Preparation
required a common allocator setting and disabling optional PyTorch DSL overrides, not the separate uBPF
policy JIT. Earlier OOM/SIGILL and diagnostic records remain retained; the
precise fatal-signal cause is not established. Loading the missing checkpoint
generation configuration resolved the observed history token failure: all 64
requests and the actual 1,000-entry history store now pass independent audit.
The first BPF numerical warmup then matched all 16 tokens, but 155 prefill
logits differed by up to 0.03125 from the original model; all 15 decode steps
were exact. The strict zero-tolerance gate stopped the run. Matching the original
model's LM-head input shape then removed the discrepancy: fresh history-v4 passed
all 64 requests, and four-arm preflight-v2 independently passed all 36 saved arrays
(87,515,136 finite FP32 values, zero difference). The complete C/BPF decision and
downstream-admission traces also matched. All 20 formal cells / five paired
blocks passed independent raw analysis, including 160 measured requests and
2,560 exact tokens. Throughput medians are demand-only 5.1713, all-positive
3.6228, native C 4.4994 and BPF 4.5144 token/s. Relative to all-positive, BPF
reduces completed evicted-unused payload by 60.41% and improves throughput by
24.64%; it remains 12.62% slower than demand-only. BPF/C paired throughput is
1.002107 [0.998266, 1.005226], not a superiority or equivalence result. See the
[formal report](../workloads/finemoe/results-performance.md) and retained
[preparation record](../workloads/finemoe/results-preparation.md).
The original complete LMSYS dataset requires
access approval, so the same 64/8/1 split uses public LMSYS MT-Bench first-turn
inputs instead, with the dataset difference explicitly recorded. POD's real
attention device-call ABI has been retained through compilation. All four
required numerical translation units now have CPU-validated adapter paths
within the existing transport limit; only unreferenced helpers are removed,
and oversized units are partitioned with all entry bodies retained and explicit
state-independence checks. The complete original fused/FlashAttention builds
and final linked six-packet extraction have passed. The first GPU attempt
stopped while checking original FlashAttention against its FP32 reference,
before running POD/BPF. A retained full-prefill diagnosis found one threshold
exceedance among 33,554,432 outputs. Its actual saved row agrees with an
independent numerical model of the original intermediate FP16 probability
conversion, not just final-output rounding. A disclosed protocol revision now
reports this additional full-FP32 characterization separately,
while preserving the official FA-to-fused output threshold and rerunning all
five arms. Both original failures remain failures. The third preflight completed
only the original-streams cell before the host reboot at 09:04 UTC interrupted
BPF initialization. Its cause is unknown; it is not a completed BPF experiment.
The launcher now avoids preloading the BPF agent into the CPU-affinity wrapper;
37 CPU tests pass. Fresh preflight 04 completed all five arms: actual device-BPF
engine 2 covered 3,328 CTA task choices, with exactly-once prefill/decode coverage,
14 verified bridge calls and zero output difference versus original FP16 FA.
All five cells passed cleanup; the runtime inventory remains unchanged.
The full five-block/ten-shape comparison started at 11:05 UTC in `full-575-01`.
Preflight raw evidence is committed and pushed in `2658e0e`. Strict device verification
remains a separate missing result; the performance runtime has it disabled.

The 09:04 UTC reboot restored stock 575 modules, as configured. The completed
FineMoE campaign ended at 08:59:10 UTC and was not interrupted. Hummingbird's
remaining four non-native arms require the separately staged custom context/
timeslice interface; a matching driver version alone does not establish readiness.
The same staged `849ea75d` runtime was restored at 09:50:34–35 UTC; both
required-RPC context canaries passed 2,048 exact outputs and 17 negative cases.
Hummingbird's unchanged full 50-cell batch completed at 10:53 UTC under exclusive leases.
See the [restoration and pending service cleanup](experiment/driver-575-linux-6.15-runtime.md#second-unexpected-reboot-and-follow-on-restoration).

At that historical checkpoint, the active paper draft included Q2
pseudocode/algorithm/TCB/rejection
examples, the policy-expressibility table, completed matched comparisons and
the scheduling figure, calibrated headline attribution, discussion topics and
typographic repairs. A fresh three-pass LaTeX/BibTeX build and two-figure
render check passed, without undefined references/citations. That draft was
18 total pages with conclusion on page 16, so condensation and a fresh final
build remain necessary before claiming paper integration complete. The checked
draft source is committed and pushed as paper revision `b5e2f25`.
The [source-backed safety audit](revision-safety-design.md) exposes a substantive
remaining implementation gap: the current device experiment runtime has GPU
verification disabled, and the separate strict verifier does not yet model
POD's pointer/leader-ticket ABI. These are not closed by prose edits.
LMCache's harness now supports explicit 575 selection and rejects smoke-sized
or mixed-driver formal comparisons. Its lifecycle now includes exclusive leases,
fixed worker placement, continuous monitors, owned cleanup and boot checks;
23 lightweight CPU tests passed independently and the repairs were pushed as
`bbc4d3f`, with no new 575 GPU-compatibility or disk-performance claim. The [remaining artifact
work](revision-remaining-artifacts.md) includes disk, Table 1 and unavailable
original agent transcripts.

All three original scoped comparisons are complete, including the unchanged
full XSched campaign. The subsequent literature search identifies useful,
interesting questions; it does not relabel proposed component ports as
reproduced systems. The user has since prioritized the GPreempt load study below.
The [follow-on survey](background-related-work.md) records seven additional
papers, their downloaded PDFs and artifact/hardware limits. Its priorities are
adaptive expert prefetch, idle-interval-aware scheduling and device-local task
selection; these are proposals, not new measured results.

### Completed follow-on: GPreempt contention study

Independent XSched runtime development is paused. The
[new fixed GPreempt plan](../workloads/gpreempt/load-study-plan.md) compares
native / original-C / actual BPF with LC 100 requests/s and BE 100, 200, or
closed-loop continuous supply. It adds a common-phase FIFO arrival schedule,
complete response timestamps, and explicit deadline/backlog accounting; the
old service-p99 experiment and its binaries remain unchanged.

**All 45/45 cells passed independent audit**, five paired blocks per scenario,
with no rejected, missing or unexpected attempts. Execution ran
04:53:52–05:50:03 UTC on 2026-09-03 under exclusive leases and a fixed 5-second
inter-cell cooldown. The three 10-second preflight cells are excluded.
All 270,000 foreground requests completed correctly inside the measurement
windows. No driver or service changes were made; final GPU/struct-ops cleanup
passed and the runtime stayed frozen throughout.

For continuous background supply, native / original-C / BPF medians are
**1.795937 / 1.614817 / 1.610008 ms** LC response p99 and
**197.717 / 179.967 / 180.100 req/s** BE goodput. BPF/native paired LC p99 is
**10.81% lower** at an **8.88% BE goodput cost**. BPF/C LC change is
**−1.32%**, 95% paired-block interval **[−4.19%, +0.51%]**; BE change is
**+0.08% [−0.36%, +0.51%]**. All three loads show the same broad policy
tradeoff, not formal BPF/C equivalence. BE 200 backlog is retained explicitly;
the 100 req/s result is rate-capped. This remains the disclosed host-mapped
transport variant, with host-JIT decisions and kernel BPF timeslice callbacks,
not original GDRCopy/hardware reproduction or a universal mechanism speedup.

See [all loads, paired effects and limits](../workloads/gpreempt/results-load-study-575-20260903.md),
[final raw audit](../workloads/gpreempt/raw/load-study-full-575-20260903/independent-audit-final.json)
and [four-panel XSched/GPreempt figure](../workloads/gpreempt/figures/scheduling-comparison-2x2.pdf).
The new response metric must not be compared directly with old service-stage
p99 or XSched queue-entry latency. The original three-way results below remain
valid for their separately documented workload; paper integration was not part
of that completed campaign and is now proceeding separately.

## Completed comparisons — 2026-09-03

This section supersedes the dated August 31 execution state below. The
user-requested GPreempt, MoE-Infinity, and XSched comparisons are complete.
LMCache was paused at this historical milestone and remains paused by explicit
user direction.
The host is Linux `6.15.11-061511-generic`, RTX 5090, and NVIDIA 575.57.08 with
the temporarily loaded `849ea75d` scheduling port. After an unexpected reboot
at 01:37:41 UTC, the verified modules were restored with ordinary unload/load;
the reboot cause is unknown. See the
[driver recovery record](experiment/driver-575-linux-6.15-runtime.md)
and [MoE raw-data recovery](../workloads/moe-infinity/recovery-20260903-013741.md).

| Experiment | Actual completion and interpretation |
| --- | --- |
| GPreempt three-way | **5 complete paired blocks, 15/15 cells passed**, each retaining the original 60-second config-A workload. Independent raw-request, numerical, engagement, telemetry and cleanup audit accepts all 15. LC service-stage p99 medians: native **1.414351 ms**, original C **1.415130 ms**, BPF **1.419397 ms**. BPF/original-C paired geometric ratio **1.002575**, 95% paired-block bootstrap CI **[1.001278, 1.003740]**: a small measured overhead, not superiority or an equivalence proof. BE throughput medians are 100 req/s, with BPF/original ratio **0.9998666**, CI **[0.9995999, 1]**. This is the disclosed **host-mapped flag compatibility variant**, not original GDRCopy reproduction; the rate-limited workload does not establish saturated throughput. [Raw final audit](../workloads/gpreempt/raw/575-host-mapped-three-way-01/audited-analysis-final.json). |
| MoE-Infinity same algorithm | **5 complete paired blocks, 15/15 cells independently raw-audited**, no failed, incomplete or rejected attempts. Throughput medians: baseline **11.8964**, paper-native **11.2233**, paper-BPF **11.1900 token/s**. BPF/native paired geometric ratio **0.996540**, 95% CI **[0.989239, 1.005508]**; no predeclared equivalence margin, so this is not formal equivalence. Relative to baseline, BPF throughput is **6.92% lower**, while the secondary first-visible-text TTFT is **22.48% lower** (paired ratio 0.775152, CI [0.757434, 0.795402]). All arms share the current MoE frontend and weights; the two paper arms share EAMC prediction, prefetch ranking, score-based eviction and the repaired prediction-protected executor. BPF selectors execute in the **host ubpf JIT**, not the old kernel-UVM program. This is an explicitly scoped algorithm port, not the authors' original hardware/model reproduction. [Final raw audit](../workloads/moe-infinity/raw/paper-v3-575/timing-849ea75d-02-postboot/audited-analysis-final.json), [algorithm scope](../workloads/moe-infinity/activation-aware-port.md), [why the old 1.63 token/s comparison was confounded](../workloads/moe-infinity/diagnosis-before-prediction-protection.md). |
| XSched | **10 complete four-arm blocks plus six isolated controls, 46/46 cells raw-audited**; the full 50-kernel/stream workload was not shortened. LC submission-to-first-CTA-entry p99 medians: native **76.8780 s**, original XSched **26.9784 s**, same-policy BPF HPF **27.2502 s**; BE medians **10.2377 / 10.1497 / 10.1616 kernels/s**. BPF HPF minus XSched paired mean LC difference **+0.291953 s**, 95% CI **[-0.109087, +0.642367]**; no clear latency difference or formal equivalence. BE paired mean change **+0.090314%**, CI **[+0.029818%, +0.147607%]**. The separate driver-BPF policy gives **22.1691 s / 6.1373 kernels/s**: lower LC delay but **39.545% lower BE throughput** than XSched. This verifies bounded HPF on the original **Level-1** frontend/actuator, not Level-3 or a pure JIT-cost experiment. [Full report](../workloads/xsched/performance-full-575-20260903.md), [46-cell raw audit](../workloads/xsched/raw/full-persistent-575-20260903/independent-raw-audit.json). |

GPU timing was exclusive, with no overlapping compilation or model hydration.
An additional independent calculation rebuilt all 49,200 XSched samples and
the paired intervals from the 246 worker records and found no discrepancy.
After final cleanup and released leases, GDM and nvidia-persistenced were
restored successfully at 03:50 UTC. The node labels already had their original
values and were not overwritten. Raw failures and
partial attempts are retained, never counted as complete pairs. Local scoped
changes and portable raw evidence are committed and pushed; unrelated worktree
changes are preserved. Completing these scoped comparisons does not complete
every revision requirement listed in the historical matrix.

## Historical handoff — 2026-08-31 (not the current machine or queue)

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
| RTX 5090/NVBit | The [accepted two-tool result](../workloads/llama.cpp/observability_overhead/revision-rq4/result-review.md) completes 10/10 blocks on driver 575.57.08: exit records slightly favor NVBit, while exit-count histograms favor gpubpf. For `launchlate`, both public RM paths failed precision, but the [versioned endpoint canary](../workloads/llama.cpp/observability_overhead/revision-rq4/launch-clock-recovery/rm-correlation-results.md) passes Phase 0 at 0.759 us over 200/200 samples; clock identity and 220-launch gates remain. Verifier-off execution and native-transport boundaries are explicit. The strict-device follow-up is complete for the two valid rows: [A1](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-a1/results-a1-575-02-20260904.md) measures one-time admission, while [S0](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-s0/results-s0-575-02-20260904.md) measures steady state without establishing equivalence. |
| Same-policy mechanism cost | [Approved plan](../workloads/uvm-policy-mechanism/plan.md), [semantic preflight](../workloads/uvm-policy-mechanism/results/preflight-1.md), 15 paired raw blocks, [analysis](../workloads/uvm-policy-mechanism/results/analysis.md), and [independent review](../workloads/uvm-policy-mechanism/result-review.md) are complete. On the scoped CPU-resident, non-first-touch UVM fault path, gpubpf adds 3.219% overhead versus the built-in no-prefetch switch (paired bootstrap 95% CI 2.247--4.202%). This is mechanism-cost evidence, not a policy improvement or application result. |
| Safety/design depth | [Approved plan](experiment/revision-safety/plan.md), [verifier results](experiment/revision-safety/phase-a-results.md), [transition audit](experiment/revision-safety/phase-b-transition-audit.md), [50-event reconciliation](experiment/revision-safety/phase-c-event-sources.md), and [fresh review](experiment/revision-safety/result-review.md) are complete. The verifier phase passes; transition validation is a production `GAP`; event provenance is 46 `SUPPORTED` plus 4 `PARTIAL`, with raw sessions absent. Aggregate disposition is reviewed `PARTIAL`. |
| Transition validator | [Five-round approved plan](experiment/revision-transition-validator/plan-review.md), [Phase A](experiment/revision-transition-validator/phase-a-results.md), [scheduler](experiment/revision-transition-validator/phase-b-scheduler-results.md), [prefetch](experiment/revision-transition-validator/phase-b-prefetch-results.md), and [PMM](experiment/revision-transition-validator/phase-b-pmm-results.md) results. Scheduler, UVM prefetch, and PMM production integration are ported across 575 and 610; host tests pass at 12 cases/145 assertions and all five modules build on each line. The PMM live preflight passes: kernel-native ioctl `NV_OK`, typed setter admitted, hidden write rejected with `-EACCES`, exact PMM totals 2/1/1/2, and distribution UVM restored. Disposition remains `PARTIAL` only for the five scheduler fixture outcomes and runtime scheduler path, which require a display maintenance window. |

## Complete revision requirement matrix

The active queue is broader than the three new R1 artifacts. Requirements are
separated below so that a build, qualitative discussion, or historical number
cannot be mistaken for a completed experiment.

| Revision item | Required experiment or evidence | Acceptance boundary | Current disposition |
| --- | --- | --- | --- |
| R1: MoE research baseline | Same-model, same-workload head-to-head against a runnable research MoE offload system on the RTX 5090. | Correct outputs, actual baseline-offload and gpubpf-hook engagement, valid paired blocks; a host-only policy remains an ablation, not the submitted full device-observed policy. | **Complete as a scoped paper-algorithm port, not an original binary/system reproduction.** [Five blocks / 15 cells](../workloads/moe-infinity/results-paper-v3-protected-575.md) report baseline/native/BPF 11.8964/11.2233/11.1900 token/s and actual host-uBPF engagement. The baseline's distinct prefill overload shortcut prevents a pure baseline-policy attribution. |
| R1: scheduling research baseline | XSched on sm_120, explicitly labeled public Level-1 inter-kernel preemption. | Correct epoch/timestamp proof, matched multi-tenant workload, policy engagement, and a predeclared interpretation that permits regressions. | **Complete within public Level-1 scope.** [All 46 cells](../workloads/xsched/performance-full-575-20260903.md) pass: native/original/BPF LC p99 is 76.8780/26.9784/27.2502 s and BE is 10.2377/10.1497/10.1616 kernels/s. This is neither Level-3 nor a formal BPF/original equivalence result. |
| R1: storage-tier KV baseline | LMCache local-NVMe backend versus CPU retention and recomputation. | Exact cache hits, O_DIRECT proof, deterministic outputs, and valid paired blocks; no failed preflight may become a performance sample. | **Paused by user direction, incomplete.** Storage engagement was observed, but warm CPU/disk outputs disagree with recompute. No formal cells or valid performance result exist; see the [retained status](revision-completion-checklist.md). |
| R1: existing GPREEMPT-equivalent evidence | Re-establish foreground latency and best-effort throughput on a matched, interleaved schedule without retaining the unsupported historical 96% proxy. | Complete response timestamps, policy engagement per run, request/backlog accounting, and robust run-level effects rather than an outlier-driven mean. | **Complete scoped compatibility evidence.** The [45-cell load study](../workloads/gpreempt/results-load-study-575-20260903.md) and [27-cell knee sweep](../workloads/gpreempt/results-lc-knee-575-20260903.md) show both original C and BPF protecting LC at substantial BE cost. The 800-rate tail is conditional, and host-mapped signaling is not original GDRCopy/hardware reproduction. |
| R3: expert-management granularity | Implement the promised Expert Buffering hot-residency policy and keep any page-granularity causal claim separate. | A matched baseline/native/BPF comparison may establish policy portability; it cannot by itself reproduce the distributed original system or isolate page-versus-expert granularity. | **Matched policy port complete:** [15/15 cells](../workloads/expert-buffering-policy/section-vi/results-performance-575-20260903.md) report FIFO/native/BPF 5.508/5.663/5.621 token/s, identical native/BPF decisions and 12.85% less logical copy payload. Original distributed-system reproduction and a causal page-granularity comparison remain unclaimed. |
| R4: mechanism versus policy | Separate policy benefit from the cost of executing the same decision through gpubpf. | Matched repeated blocks, successful engagement, and explicit execution/actuation boundaries. | **Supported by multiple completed ports.** The [48-paper evidence ledger](experiment/policy/reference/RELATED_POLICY_EXPRESSIBILITY.md) links MoE-Infinity, XSched, GPreempt, Expert Buffering, FineMoE, Hummingbird and POD results; the separate [UVM study](../workloads/uvm-policy-mechanism/results/analysis.md) measures +3.219% [2.247%, 4.202%] on one matched fault path. Historical Fig. 13's fresh engaged matrix remains unmeasured and its old 55--92%/under-1% causal wording remains unsupported. |
| R5: safety and design depth | Executable verifier/transition rejection cases plus a source-backed re-count of the 50 agent safety events. | Demonstrate scoped rejected programs and invalid transitions without claiming full-stack formal verification or universal fallback semantics. | **Substantially complete, with explicit limits.** [Two original strict device pairs](../workloads/bpftime-device-smoke/results-strict-575-20260903.md) and [two fresh pairs on the ported Table 1 runtime](../workloads/bpftime-device-smoke/results-table1-port-strict-575-20260904.md) each admit 32,768 callbacks and reject the lane-varying negative before policy-entry insertion and bootstrap; generic CUDA interception already exists. [Invalid-prefetch](experiment/revision-safety/prefetch-invalid-575-02/result-review.md) and [scheduler-init](experiment/revision-safety/sched-init-live-575-05/result.md) live controls pass; the loader study shows strict rejection while warning/default modes admit. The real Table 1 objects now have completed [A1 one-time admission](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-a1/results-a1-575-02-20260904.md) and [S0 steady-state STRICT/NO_VERIFY](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-s0/results-s0-575-02-20260904.md) measurements. These scoped controls do not retroactively verify the accepted verifier-off table or prove equivalence; original agent sessions also remain absent. |
| R6: current-hardware overhead | RTX 5090 Table 1 comparison including NVBit, plus the gpubpf device-side tools. | Same benchmark and measurement boundary, supported NVBit stack, exact tool/source identities, correctness before timing, repeated runs. | **Complete for the predeclared non-cross-clock subset; original three-tool campaign remains partial.** The [full result](../workloads/llama.cpp/observability_overhead/revision-rq4/result-review.md) accepts all five correctness cells and 10/10 blocks. Exit-record overhead is gpubpf/NVBit 99.663%/99.621%, while histogram overhead is 4.007%/10.301%. `launchlate` remains invalid: public RM xfer/direct missed the precision gate at 4.759/4.730 us, while the [versioned endpoint fallback](../workloads/llama.cpp/observability_overhead/revision-rq4/launch-clock-recovery/rm-correlation-results.md) passed Phase 0 with 200/200 samples and a 0.759 us median bracket; clock identity and 220-launch gates remain. The performance runtime was verifier-off, the NVBit arms are matched custom adapters using native transport, and histogram closure is aggregate only. |
| R7: study artifacts | Publish prompts, interaction logs, metric extraction, and benchmark harnesses. | Redact secrets/private data, retain raw provenance and explicit public inventories, and make every reported aggregate reproducible without file/content hash gates. | Hard artifact commitment. Public index and path-parameterized extractor are pushed; the original study sessions are still absent from the local archive. |
| R8: portability/deployment evidence | Audit the SASS patching prototype, the reported 273 ms one-time ptrace attach, the LD_PRELOAD route, and the approximately 100-LOC open-module patch boundary. | Bind each statement to runnable code or retained raw evidence; otherwise narrow it to design discussion. | The [deployment audit](experiment/revision-deployment-575/RESULTS.md) validates scoped CPU-only LD_PRELOAD and running-process lifecycle paths, but does not validate working SASS, the historical 273-ms number or an unqualified approximately-100-LOC claim. The 610 port remains separate build/portability evidence. |
| R2, R9, and LOC corrections | Expressibility table, design/discussion text, and corrected policy LOC arithmetic. | Cite concrete in-tree programs and distinguish measured, inferred, and out-of-scope claims. No fabricated experiment. | The [48-paper inventory](experiment/policy/reference/RELATED_POLICY_EXPRESSIBILITY.md) and grouped discussion are integrated in the latest paper source; [eval.tex](paper/tex/eval.tex) contains the corrected sequential-prefetch 573 LOC and composite 1334 LOC values. No surveyed whole system is classified `FULL`. CXL/GDS implementation, multi-vendor campaigns, LithOS-scale experiments and upstream GPREEMPT binary reproduction remain out of scope. The current 16-page build/placement check passes; repeat after later edits. |
| Shepherd: policy versus mechanism | For each existing policy, establish whether gpubpf implements it and, when the original artifact is runnable, compare against the original ad-hoc/unsafe/monolithic implementation. Audit headline claims in the abstract and introduction for policy/mechanism attribution and disclose any general-mechanism cost. | Matching SOTA remains a valid outcome; no result may imply that a policy improvement was caused by the mechanism. Novel-policy claims require concrete code and evidence, especially for agent-discovered policies. | Existing-policy coverage now has completed matched ports and explicit mechanism costs in the [reviewer-facing ledger](experiment/policy/reference/RELATED_POLICY_EXPRESSIBILITY.md). The draft attributes fusion/scheduling/cache gains to policies, not BPF. No existing-policy port is called an agent-discovered policy, and no formal native/BPF equivalence claim is made. Original transcript recovery and any genuinely new-policy claim remain open. |
| Shepherd: exposition and typography | Regroup scattered safety and SOTA-reimplementation answers under common labeled headings; fix Section 5.3 paragraph headings with two trailing periods and audit bibliography curly-brace escaping. | Preserve technical meaning and bibliography semantics; compile the paper and inspect the affected output. | Grouped safety/SOTA exposition and both reported typography fixes are in the latest paper source. The last documented build passes without undefined references/citations; later source additions still need the final build/placement and citation-metadata audit. |

The two author-response commitments that cannot be dropped are R6's RTX 5090
Table 1/NVBit result and R7's public prompts and benchmark harnesses. The fuller
revision plan additionally names the three R1 research artifacts and R3/R4
quantification. Evidence defects discovered during revision are treated as
requirements to repair, even when the original plan described the old figure
as merely being foregrounded.

## Historical August 31 machine state (superseded)

The following section is retained only to explain the old 610/7.1 preparation.
It is not the current host or experiment queue; the completed comparisons above
ran on Linux 6.15.11 and NVIDIA 575.57.08.

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
