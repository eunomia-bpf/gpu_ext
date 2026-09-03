# Background and related work: next experiments

Updated: 2026-09-03 UTC. Scope: the user's request to look for interesting work
**after finishing MoE-Infinity, XSched, and GPreempt**. Those three scoped
comparisons are [complete](revision-experiment-status.md#completed-comparisons--2026-09-03).
This is a bounded follow-on survey, not seven new reproduction results, a
whole-paper novelty assessment, or a commitment to run seven systems.
The original survey made no GPU experiment, driver change or service interruption.

Follow-on execution update: the user has now selected FineMoE, Hummingbird and
POD-Attention for implementation and experiments. Their PDFs below remain the
starting evidence; the three source/algorithm feasibility checks are running
in parallel. The subsequently completed
[GPreempt load study](../workloads/gpreempt/results-load-study-575-20260903.md)
now provides a foreground/background tradeoff under continuous BE supply:
original C and BPF both improve foreground p99 relative to native, while BE
goodput falls about 9%. This motivates the requested idle-interval experiment;
it is not already evidence of Hummingbird's benefit.

## What seems worth doing next

The priorities below are our inference from the papers and current results,
not measured benefits of a new BPF policy. Reproducing an existing decision rule
correctly remains a useful outcome; outperforming it is not required.

| Priority | Question and motivation | Smallest useful future comparison | Missing work / interpretation boundary |
| --- | --- | --- | --- |
| 1: adaptive expert prefetch | Can we retain useful prefetches while copying fewer unused experts? Current [MoE results](../workloads/moe-infinity/results-paper-v3-protected-575.md) have about 72% of completed prefetched copies unused before eviction. This counts copies, **not bytes or wasted runtime**. | Inspired by FineMoE: keep predictor, executor, cache budget and eviction rules fixed; compare all-positive, fixed-top-K and confidence-controlled prefetch sets. Use the same snapshots for native/BPF decision parity. Measure full-wall token/s, newly instrumented demand waits, useful/unused transferred bytes and adaptation across real A → B → A request groups, preserving policy history between groups. | Current activation counts do not supply FineMoE's full routing-probability maps or semantic features. Adding only a prefetch-set selector is a component port. First isolate the [baseline/executor asymmetry](../workloads/moe-infinity/results-paper-v3-protected-575.md), including temporary overload slots; the baseline gap is not yet causally explained. |
| 2: less costly foreground protection | Can a policy recover background throughput while protecting foreground latency? Our [separate driver-BPF arm](../workloads/xsched/performance-full-575-20260903.md) loses 39.5% BE throughput relative to XSched; this is not the same-policy HPF arm. The new GPreempt load study measures about 9% BE cost for both original C and BPF. | Inspired by Hummingbird: use real DNN clients, bursty foreground arrivals and saturated background work; compare HPF/fixed preemption with idle-interval-aware admission. Report foreground p99/SLO attainment **and** completed background work, with equal work and native/BPF policy inputs. UniBoost supplies a separate idea: require useful progress before reconsidering a costly switch. | Requires real queue-completion/idle hints and a safe launch-deferral or queue-control path, not discarded CUDA calls. Hummingbird's kernel splitter and UniBoost's KV-aware LLM scheduler are separate, larger integrations. GPreempt's completed continuous-supply result provides a comparison point, not proof of full GPU saturation. |
| 3: device-local decisions that actually choose work | Can bounded device-side BPF reproduce a useful SM-local task selector, rather than merely observe counters? POD-Attention offers a concrete original selector. | After an sm_120 port, compare serial attention, two-stream overlap, original POD selection and the same selector in device BPF, sharing numerical kernels and executor. Test several prefill/decode mixes, numerical correctness, exactly-once task execution and completion times; remove SM-local input as an ablation. | Need an execution interface returning an operation/task ID that controls real work. POD uses SM IDs and atomic tickets, not measured warp pressure or arbitrary hardware CTA placement. Kernel fusion benefits must not be attributed to the BPF mechanism. This is a larger integration than priority 1. |

These tests are proposals, not frozen protocols or implemented extensions.
Do not tune on the final measurement blocks or pool them with the completed
three-system campaign. Preserve failures and permit neutral/negative outcomes.

## Seven papers and their implementation boundaries

### FineMoE — EuroSys 2026

Hanfei Yu et al., *Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via
Fine-Grained Expert Offloading*. Iteration-level routing maps and semantic
similarity guide expert decisions. Section 4.3 selects a descending-probability
prefetch set using a similarity-dependent cumulative-probability threshold,
with at least K predicted experts, where K is the model's routing fan-out;
this does not guarantee inclusion of every actually routed expert.
This is a concrete alternative to
prefetching every positive-score candidate. [Author PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf).

The [official artifact](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) is
explicitly a **demo**, built on MoE-Infinity, with a stated GPU-memory
requirement of **at least 48 GB** and a Qwen1.5-MoE/LMSYS sample. It is not an
as-is reproduction on our 32 GB RTX 5090. Reuse the existing expert executor
only after matching the new selector's inputs and residency rules.

### HybriMoE — DAC 2025

Shuzhang Zhong et al., *HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management
for Efficient MoE Inference*. Combines intra-layer CPU/GPU placement,
impact-driven prefetch and a moving router-score cache policy that includes
near-selected experts, not only actual activations. This suggests a second
small comparison: reuse-score versus router-score cache decisions under
changing request mixes. [Paper, v1](https://arxiv.org/abs/2504.05897v1),
[official code](https://github.com/PKU-SEC-Lab/HybriMoE).

The existing score/eviction bridge is reusable, but currently lacks the
unselected experts' router probabilities. A cache-policy port would not
reproduce HybriMoE's CPU expert execution, placement queues or entire system.
No build or run of its artifact was attempted in this survey.

### Hummingbird — 2026 preprint

Tiancheng Hu et al., *Hummingbird: SLO-Oriented GPU Preemption at
Microsecond-scale*. Splits kernels through PTX transformation and replays the
correct block-index offsets; its scheduler fills detected idle intervals and
stops admitting low-priority work when high-priority work is ready. Larger idle
intervals permit consolidation. The original survey read
[v1](https://arxiv.org/html/2601.04071v1); implementation preparation now also
checks [v2, 2026-02-10](https://arxiv.org/abs/2601.04071v2), retained as a
[separate PDF](reference/2026-hu-hummingbird-v2.pdf). Section 4.3 requires
real bubble hints, waiting for high-priority GPU completion, split-kernel
admission and kernel-tick pacing. Merely shortening GPreempt's hint is not this
algorithm.

No runnable author artifact was confirmed in this bounded search. The
`microsoft/hummingbird` ML tensor compiler is unrelated. A driver policy
inspired by idle-interval admission is **not** a reproduction of its PTX
splitter, replay protocol and memory-management system. XSched's seconds-long
queueing p99 must not be presented as microsecond preemption latency.

### UniBoost — ICML 2026

Yueying Li et al., *Beyond Prediction: Tail-Aware Scheduling for LLM Inference*.
Uses attained-work-dependent priority and a memory-aware scheduler; MemGuard
quantizes progress so requests can do useful work between reconsiderations,
reducing KV swap churn. This motivates measuring switching cost rather than
assuming the most aggressive preemption is best. [Author page](https://yl3469.github.io/uniboost-icml26/),
[paper, Section 3.3](https://yl3469.github.io/uniboost-icml26/assets/icml_paper.pdf).

The inspected author-page Code link returned a placeholder on the same page;
no runnable author implementation was confirmed. Our TSG hooks do not expose
LLM request progress, KV residency or batching control. Transplanting the score
alone cannot reproduce the full scheduler.

### POD-Attention — ASPLOS 2025

Aditya K Kamath et al., *POD-Attention: Unlocking Full Prefill-Decode Overlap for
Faster LLM Inference*. Its fused kernel uses SM-local tickets to choose between
prefill and decode CTA work. This is a stronger device-policy example than a
callback counter. [Author PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf),
[official artifact](https://github.com/microsoft/vattention/tree/main/pod_attn).

The inspected [build configuration](https://github.com/microsoft/vattention/blob/main/pod_attn/setup.py#L105)
targets sm_80/sm_90; the [selector's SM-count comment](https://github.com/microsoft/vattention/blob/main/pod_attn/pod_attn/fused_fwd_kernel.h#L1313)
says it was only tested on A100. A 5090 port must validate SM-ID bounds and
shared-memory/kernel constraints. This is a **component-port candidate**, not
a ready full POD/Sarathi-Serve reproduction or proof of arbitrary CTA placement.

### MPK — 2026 v2

Xinhao Cheng et al., *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor
Programs*. Its persistent device runtime can coordinate tasks, dependencies
and shared-memory reuse, providing a possible later task-admission policy
example. [Paper, v2](https://arxiv.org/abs/2512.22219v2),
[official frozen artifact branch](https://github.com/mirage-project/mirage/tree/tgx-osdi26-ae).

**Citation/design reference on the present machine:** the inspected
[`linear_layer` architecture branches](https://github.com/mirage-project/mirage/blob/tgx-osdi26-ae/python/mirage/mpk/persistent_kernel.py#L1648)
accept 80–89, 90–99 and 100–119, then assert; CC 120 is not supported as-is.
An `-arch=native` compiler fallback does not establish operator compatibility.
A real operator port and task-state/execution interface would precede any
native-versus-device-BPF experiment.

### APEX — August 2026 preprint

Alish Kanani et al., *APEX: Adaptive Expert Prefetching for Memory-Efficient Edge
MoE Inference*. A pre-attention predictor and learned confidence distribution
choose extra experts beyond top-K. Only the correctness-preserving mode is a
candidate here: missed experts still execute on demand; do not substitute the
stall-free mode that changes expert selection. [Paper, v1](https://arxiv.org/abs/2608.11688v1).

Its reported performance comes from **synthesis-calibrated CHIPSIM
co-simulation**, not a measured RTX 5090 deployment. New prediction hooks,
training/calibration data and model parameters would be needed. No author code
was located in this bounded search; keep this as a lower-priority design
reference. [Evaluation methodology](https://arxiv.org/html/2608.11688v1#S5.SS2).

## Downloaded PDF inventory

All original seven downloads and the additional Hummingbird v2 snapshot were
checked with `pdfinfo` and first-page title inspection.
Sizes are the local downloaded files; this is an inventory, not evidence that
any artifact builds or runs. PDFs remain the original authors' work.

| Work / inspected snapshot | Local PDF | Bytes | Pages |
| --- | --- | ---: | ---: |
| FineMoE / author EuroSys 2026 PDF | [PDF](reference/2026-yu-finemoe.pdf) | 1,335,925 | 16 |
| HybriMoE / arXiv v1, 2025-04-08 | [PDF](reference/2025-zhong-hybrimoe.pdf) | 527,582 | 7 |
| Hummingbird / arXiv v1, 2026-01-07 | [PDF](reference/2026-hu-hummingbird.pdf) | 971,539 | 20 |
| Hummingbird / arXiv v2, 2026-02-10 | [PDF](reference/2026-hu-hummingbird-v2.pdf) | 1,126,558 | 20 |
| UniBoost / author ICML 2026 PDF | [PDF](reference/2026-li-tail-aware-scheduling.pdf) | 800,021 | 16 |
| POD-Attention / author ASPLOS 2025 PDF | [PDF](reference/2025-pod-attention.pdf) | 1,091,305 | 16 |
| MPK / arXiv v2, 2026-06-10 | [PDF](reference/2026-mpk-v2.pdf) | 829,938 | 18 |
| APEX / arXiv v1, 2026-08-12 | [PDF](reference/2026-kanani-apex.pdf) | 9,408,998 | 14 |

## Search scope and remaining uncertainty

Three parallel branches searched on 2026-09-03 UTC, then checked primary papers
and author repositories. Representative queries:

- Expert management: `"FineMoE" "github.com"`, `"HybriMoE" "github.com"`,
  `"APEX" "Adaptive Expert Prefetching" github`.
- Scheduling: `GPU scheduling preemption SLO memory bandwidth contention 2026 paper artifact`,
  `"Hummingbird" "GPU" "github" preemption`,
  `"Beyond Prediction" "Tail-Aware" github arxiv`.
- Device runtime: `POD Attention ASPLOS 2025 official code SM aware scheduling GPU A100`,
  `site:github.com mirage-project mirage mpk sm120 H100 CUDA persistent kernel`.

This adds seven candidates to the earlier
[policy expressibility survey](experiment/policy/reference/RELATED_POLICY_EXPRESSIBILITY.md);
it does not silently update that survey's historical runtime statuses. The
[three-comparison status](revision-experiment-status.md) is the current record.
No claim of exhaustive literature coverage, a novel algorithm, absent public
code everywhere, or unsupported hardware in general follows from this search.
In particular, an artifact's sm_120 port gap is not proof that the hardware
lacks the underlying feature. Source branches can change; pin an upstream
revision and record local patches before any future build or measurement.
