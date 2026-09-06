# Related-policy expressibility inventory

This inventory answers a narrow, auditable question: **which decisions from prior
GPU memory and scheduling papers can the current gpubpf ABI express?**  It does
not treat similarity of names as an implementation, and it does not treat a
successful local analogue as a reproduction of the original system.

The machine-readable source of truth is
[`related-policy-expressibility.json`](related-policy-expressibility.json).  It
currently contains **48 papers across seven policy families**.  Every row records
the primary publication/artifact URL, the paper's observations and actions, the
whole-policy classification, missing primitives, any corresponding in-tree
programs, and the strongest evidence level actually available.

The broader 59-source reading corpus and its 54 locally retained, first-page
checked PDFs are indexed by
[`docs/paper-material/policy-expressibility-papers/MANIFEST.md`](../../../paper-material/policy-expressibility-papers/MANIFEST.md).

## Classification boundary

The classification applies to the **whole policy or system** described by the
paper:

- `FULL`: current observations and actions preserve the policy's decision
  semantics.
- `ANALOGUE`: an in-tree policy targets the same goal, but its granularity or
  semantics differ materially.
- `PARTIAL`: only identifiable components map to current hooks.
- `NO`: a defining action is absent from the ABI.

There are intentionally no `FULL` paper rows in this revision.  The current ABI
fully represents several classic *subpolicies* (no-prefetch, a bounded
current-VA-block contiguous prefetch choice, and root-chunk head/tail ordering),
but the surveyed papers combine those with semantic inputs, proactive range
migration, exact placement, or runtime mechanisms.  Calling the whole systems
`FULL` would erase that distinction.

## Current mechanism envelope

The memory ABI exposes page-fault/current-VA-block prefetch context, root-chunk
activation/access, eviction preparation, and owner/root/generation metadata.  A
policy may select one contiguous region inside the current VA block or request
that one root chunk move to a used/unused list head/tail.  The separate
scheduler ABI provides task lifecycle/process identity, TSG timeslice control,
and supported whole-TSG preemption.

The important missing actions are arbitrary cross-block migration, exact victim
eviction, destination/tier/peer selection, pin/replicate/remote-map/coherence
control, DMA/network/queue control, and automatic tensor/expert/token/operator
semantics.  Those absences explain most `PARTIAL` and `NO` results.

## Evidence ladder

Evidence is attached to a **local mapping**, never silently inherited from the
paper's own evaluation:

1. `source`: primary source plus inspected local source; no build or runtime
   claim.
2. `build`: the named local analogue has a recorded successful build.
3. `engagement`: a real workload recorded the relevant local hooks/actions.
4. `performance`: a controlled run recorded an outcome metric for that local
   analogue.

Multiple rows now have bounded local performance evidence, including Expert
Buffering, MoE-Infinity, GPREEMPT, XSched, FineMoE, Hummingbird, POD-Attention,
the same-policy UVM comparison, and RTX 5090 observability. The eGPU row also
reaches `performance`, but through two deliberately separate local paths:
strict real-device callback engagement and a verification-disabled trampoline
microbenchmark. An evidence
level describes the strongest tested **local mapping**; it does not promote a
`PARTIAL` or `ANALOGUE` whole-system classification.  The four local PDFs
previously found to contain unrelated papers are excluded as evidence;
publication, author, and official artifact URLs are used instead.

## Reviewer-facing evidence ledger — 2026-09-03

This table separates baseline policy benefit from the cost of executing the
same decision through BPF.  “Native” means the matching non-BPF policy port,
not necessarily the authors' original binary.  All performance rows link a
complete repeated campaign and retain adverse outcomes.

| Policy or mechanism | Baseline -> native policy -> BPF | Strongest local evidence | Boundary and remaining gap |
|---|---|---|---|
| MoE-Infinity | 11.8964 -> 11.2233 -> 11.1900 token/s; BPF/native 0.996540 [0.989239, 1.005508] | [`performance`](../../../../workloads/moe-infinity/results-paper-v3-protected-575.md), five blocks / 15 cells | Activation matching, ranking, EAMC and eviction run in native and host-uBPF selectors. The baseline alone has a prefill overload shortcut, so baseline/policy is not a pure policy contrast. Different model/hardware; no original-system or equivalence claim. |
| XSched HPF | Native CUDA 76.8780 s / 10.2377 kernels/s -> original Level-1 XSched 26.9784 / 10.1497 -> BPF HPF 27.2502 / 10.1616 | [`performance`](../../../../workloads/xsched/performance-full-575-20260903.md), ten blocks plus controls | BPF makes HPF decisions; the original Level-1 frontend executes suspend/resume. The LC difference interval crosses zero. No Level-3, cross-XPU, pure-JIT-cost, or equivalence claim. |
| GPREEMPT | At continuous BE supply, native 1.7959 ms / 197.717 req/s -> original C 1.6148 / 179.967 -> BPF 1.6100 / 180.100 | [`performance`](../../../../workloads/gpreempt/results-load-study-575-20260903.md), 45 cells, plus the [`LC-knee sweep`](../../../../workloads/gpreempt/results-lc-knee-575-20260903.md) | Both policy arms protect LC at an approximately 9% BE cost; BPF/C intervals include zero. Host-mapped compatibility replaces GDRCopy; requested timeslice is not a proved hardware quantum. |
| Expert Buffering Section VI | Same-K FIFO 5.508 -> native 5.663 -> BPF 5.621 token/s | [`performance`](../../../../workloads/expert-buffering-policy/section-vi/results-performance-575-20260903.md), five blocks / 15 cells | Native/BPF decisions match; BPF/native is -0.74% [-1.40%, -0.20%]. Single-GPU policy port, not distributed expert buffering, gating control, or load balancing. |
| FineMoE dynamic set | Demand-only 5.1713, all-positive 3.6228 -> native 4.4994 -> BPF 4.5144 token/s | [`performance`](../../../../workloads/finemoe/results-performance.md), five blocks / 20 cells | Native/BPF implement the same host selector; their interval crosses one. Dynamic selection reduces unused speculation versus all-positive but remains 12.62% slower than demand-only. Not a full FineMoE reproduction. |
| Hummingbird idle admission | Native and fixed-GPreempt controls -> native C port -> host-uBPF port | [`performance`](../../../../workloads/hummingbird/results-575-20260903.md), 50 cells; [`bound-2 ablation`](../../../../workloads/hummingbird/pipeline/results-575-20260903.md), 40 cells | Conservative C/BPF ports lose about 19--20% BE throughput to fixed GPreempt. Bound 2 recovers about 15% versus bound 1 but does not establish unchanged LC protection. No released original scheduler arm or headline reproduction. |
| POD-Attention selector | FA serial/streams -> POD inline and matched CUDA adapter -> device-BPF | [`performance`](../../../../workloads/pod-attention/results-575-20260903.md), 250 cells; [`phase decomposition`](../../../../workloads/pod-attention/results-phase-full-575-01.md), 15 cells | BPF/CUDA costs 0.51--1.18% in nine shapes and is 0.44% faster in one; the fixed shape shows +1.78% [1.64%, 1.92%]. Fusion gains are POD policy gains. Current fresh-process BPF setup is about 271 s and strict verification was off. |
| Same-policy UVM no-prefetch | Built-in no-prefetch -> identical decision through gpubpf | [`performance`](../../../../workloads/uvm-policy-mechanism/results/analysis.md), 15 paired blocks | gpubpf adds 3.219% [2.247%, 4.202%] kernel time on one CPU-resident, non-first-touch fault path. This isolates mechanism cost, not an application-policy benefit. |
| Device callback and trampoline | Native kernel -> BPF return-only -> BPF per-thread counter | [`performance`](../../../../microbench/trampoline-scaling/results-575-20260903.md), 270 measurements; separate [`strict engagement`](../../../../workloads/bpftime-device-smoke/results-strict-575-20260903.md) | Return-only adds 0.0012--0.0022 ms at fixed geometry; counter cost grows with active work. The performance runtime disables verification and uses per-thread calls, so it does not prove once-per-warp or arbitrary-handler constant cost. |
| RTX 5090 observability | No probe -> matched NVBit and gpubpf kernel-return records / thread histogram / launch observation | [`performance`](../../../../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md), ten rotated blocks / 70 cells | Baseline is 37,586.3225 token/s. gpubpf/NVBit overhead is 90.7051%/99.6210% for `kernelretsnoop`, 2.9653%/10.3501% for `threadhist`, and 0.2208%/8.7959% for `launchlate`. All submitted P40 and earlier 5090 values remain retained. `kernelretsnoop` optimization is follow-on work, not a missing Table 1 row. |
| Raw non-composable map state | Native control -> instrumented producer -> host probe, plus overflow-negative control | [`engagement`](../../../../workloads/cross-layer-raw-map/results-full-575-02.md), five blocks / 15 cells | All 34,560 bounded tuples and all 2,560 deliberate drops reconcile. This is raw host readback, not a latency/bandwidth, on-chip-shard, automatic-placement, or unbounded-data result. |
| LMCache local disk | Recompute / original CPU / original disk, followed by matched native/BPF storage policy | [`performance`](../../../../workloads/lmcache-disk/results-575-perf-only-five-block-20260906.md), five blocks / 15 baseline cells | Recompute/CPU/disk output throughput is 30.6422/30.1168/28.5723 token/s and median TTFT is 67.1691/72.6468/96.3280 ms. Native/BPF recoverability and asynchronous GDS-decision arms remain active implementation work; the baseline campaign alone is not that comparison. |

The ledger is intentionally broader than the fixed 48-paper JSON matrix: POD,
Hummingbird, FineMoE, the UVM control, the raw-map check, trampoline scaling and
the RTX 5090 observability comparison are local policy/mechanism studies rather
than reclassifications of rows in that survey. The complete current Table 1
campaign covers all three tools; its unusually expensive `kernelretsnoop` arm
is being optimized without removing or replacing any earlier measurement.

## Survey matrix

The compact table below is an index.  The JSON contains the complete
observation/action and missing-primitive text.

### UVM and oversubscription

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [Towards High Performance Paged Memory for GPUs](https://research.nvidia.com/publication/2016-03_towards-high-performance-paged-memory-gpus) | 2016 | `PARTIAL` | `prefetch_adaptive_sequential`, `eviction_fifo` | No fault batching, arbitrary migration, or exact victim |
| [Adaptive Page Migration for Irregular Data-intensive Applications under GPU Memory Oversubscription](https://doi.org/10.1109/IPDPS47924.2020.00054) | 2020 | `PARTIAL` | `prefetch_adaptive_sequential`, `prefetch_reuse_dist` | No destination or cross-block range migration |
| [An Adaptive Framework for Oversubscription Management in CPU-GPU Unified Memory](https://doi.org/10.23919/DATE51398.2021.9473982) | 2021 | `PARTIAL` | adaptive prefetch + approximate LFU | Missing global telemetry and direct pre-eviction |
| [A Framework for Memory Oversubscription Management in Graphics Processing Units](https://rausavar.github.io/pubs/li_asplos19_final.pdf) | 2019 | `PARTIAL` | `prefetch_reuse_dist` | Cannot select zero-copy/global migration mode |
| [DeepUM](https://doi.org/10.1145/3575693.3575736) | 2023 | `PARTIAL` | stride/reuse predictors | No tensor/operator semantics or tensor-range action |
| [An Intelligent Framework for Oversubscription Management in CPU-GPU Unified Memory](https://arxiv.org/abs/2204.02974) | 2022 | `PARTIAL` | template/Belady-style page policy | No model feed, direct pre-eviction, or cross-block action |

### Prefetch and placement

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [Interplay between Hardware Prefetcher and Page Eviction Policy](https://doi.org/10.1145/3307650.3322224) | 2019 | `PARTIAL` | no-prefetch, adaptive prefetch, FIFO | Expressible subpolicies; no coordinated exact pre-eviction |
| [Page Placement Strategies for GPUs within Heterogeneous Memory Systems](https://research.nvidia.com/publication/2015-03_page-placement-strategies-gpus-within-heterogeneous-memory-systems) | 2015 | `NO` | — | No destination, remote mapping, or coherence action |
| [Mosaic](https://ghose.cs.illinois.edu/papers/17micro_mosaic.pdf) | 2017 | `NO` | — | Page-size and translation mechanisms are absent |
| [SUV](https://doi.org/10.1109/MICRO61859.2024.00030) | 2024 | `NO` | — | No compiler working-set feed, HBM reservation, or proactive ranges |
| [DREAM](https://doi.org/10.1145/3721145.3725748) | 2025 | `NO` | — | Device/NIC remote-memory data path is a new mechanism |
| [Forest](https://doi.org/10.1145/3695053.3731047) | 2025 | `PARTIAL` | adaptive-tree/stride prefetch | No object identity or cross-block tree/range action |
| [HELM](https://doi.org/10.1145/3712285.3759812) | 2025 | `PARTIAL` | reuse/adaptive prefetch | Selector is plausible; telemetry and placement actions are incomplete |

### MoE expert caching

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [Towards MoE Deployment / Expert Buffering](https://arxiv.org/abs/2303.06182) | 2023 | `ANALOGUE`; `performance` | native/BPF inactive-first/LIFO hot residency | Same-executor port only; no distributed gating, balancing, or original expert-buffer system |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | 2024 | `PARTIAL`; `performance` | native/BPF EAMC, rank, prefetch and eviction port | Host-uBPF application port; driver ABI still lacks native expert semantics and paper runtime |
| [Fiddler](https://arxiv.org/abs/2402.07033) | 2025 | `NO` | — | Defining action is CPU/GPU compute orchestration |
| [HOBBIT](https://arxiv.org/abs/2411.01433) | 2024 | `PARTIAL` | MoE prefetch + LFU | No precision selection, expert transfer, or token/layer events |
| [ProMoE](https://arxiv.org/abs/2410.22134) | 2024 | `PARTIAL` | `prefetch_moe_expert` | No proactive cross-block expert transfer |
| [PopFetcher](https://www.usenix.org/conference/atc25/presentation/zhang-junyi) | 2025 | `PARTIAL` | LFU + MoE prefetch | No peer/network action or expert identity |
| [MoE-Lightning](https://doi.org/10.1145/3669940.3707267) | 2025 | `NO` | — | Whole compute/transfer pipeline is outside the ABI |

### KV cache and weight tiering

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [FlexGen](https://proceedings.mlr.press/v202/sheng23a.html) | 2023 | `NO` | — | Tensor LP placement and GPU/CPU/disk I/O are runtime actions |
| [InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee) | 2024 | `NO` | — | No token/head KV identity or selective sub-page transfer |
| [vLLM / PagedAttention](https://doi.org/10.1145/3600006.3613165) | 2023 | `NO` | — | Logical KV allocation, sharing, and copy-on-write are absent |
| [PowerInfer](https://doi.org/10.1145/3694715.3695964) | 2024 | `NO` | — | Neuron prediction and CPU/GPU compute placement are absent |
| [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) | 2021 | `NO` | — | Distributed partitioning and NVMe pipelines are absent |
| [Capuchin](https://doi.org/10.1145/3373376.3378505) | 2020 | `PARTIAL` | LFU/reuse ranking | No tensor liveness, tensor-range action, or recomputation |

### GPU scheduling and QoS

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [GPREEMPT](https://www.usenix.org/conference/atc25/presentation/fan) | 2025 | `ANALOGUE`; `performance` | kernel-BPF role/timeslice plus host-uBPF hint decisions | Native/C/BPF load studies pass; host-mapped compatibility is not original GDRCopy/hardware reproduction |
| [GCAPS](https://arxiv.org/abs/2406.05221) | 2024 | `PARTIAL` | timeslice + whole-TSG preempt | No GPU-segment feed or real-time admission guarantee |
| [XSched](https://www.usenix.org/conference/osdi25/presentation/shen) | 2025 | `PARTIAL`; `performance` | host-uBPF HPF over original Level-1 executor | Bounded same-policy run passes; no Level-3, cross-XPU, or standalone gpubpf XQueue actuator |
| [REEF](https://www.usenix.org/conference/osdi22/presentation/han) | 2022 | `PARTIAL` | whole-TSG preempt | No kernel kill/restore, padding, or request mapping |
| [Salus](https://arxiv.org/abs/1902.04610) | 2020 | `PARTIAL` | process scheduler + PID quota | No iteration lanes or framework allocator |
| [Transparent GPU Sharing](https://www.usenix.org/conference/nsdi23/presentation/wu) | 2023 | `PARTIAL` | process timeslices | No per-submission gate or completion feedback loop |
| [Orion](https://doi.org/10.1145/3627703.3629578) | 2024 | `PARTIAL` | priority/timeslice intent | Needs per-operator profiles and lossless deferred launch queues |
| [Paella](https://doi.org/10.1145/3600006.3613163) | 2023 | `PARTIAL` | driver priority + device-event components | Needs compiler/runtime dispatch and completion protocol |
| [Tally](https://doi.org/10.1145/3669940.3707282) | 2025 | `PARTIAL` | whole-TSG preempt | No transparent thread-block scheduler or CUDA virtualization |
| [LithOS](https://arxiv.org/abs/2504.15465) | 2025 | `PARTIAL` | coarse process scheduling | No TPC steering/stealing, atomization runtime, or DVFS control |
| [Kernelet](https://arxiv.org/abs/1303.5164) | 2013 | `PARTIAL` | potential bounded block-filter component | Needs complete slice replay, profiling, and coexecution selection |
| [Bless](https://doi.org/10.1145/3689031.3696070) | 2025 | `PARTIAL` | timeslice intent | No per-kernel spatial quotas or bubble-filling admission |
| [TimeGraph](https://www.usenix.org/conference/usenixatc11/timegraph-gpu-scheduling-real-time-multi-tasking-environments) | 2011 | `PARTIAL` | priority and whole-TSG actions | No command-group admission/completion and real-time reservation accounting |
| [Gdev](https://www.usenix.org/system/files/conference/atc12/atc12-final319.pdf) | 2012 | `PARTIAL` | process timeslice + memory heuristics | No independent compute/DMA reservations or integrated GPU API |

### Userspace and device eBPF

| Paper | Year | Result | Available local component | Boundary |
|---|---:|---|---|---|
| [Extending Applications Safely and Efficiently / bpftime](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) | 2025 | `PARTIAL` | host hooks, maps, offline GPU-verifier tests | Driver hooks do not supply the whole EIM extension/resource contract |
| [eGPU](https://asplos.dev/pdf/bpftime_super.pdf) | 2025 | `PARTIAL`; `performance` | strict device-return counter plus trampoline microbenchmark | Strict engagement and verification-disabled performance are separate; observation/injection is not a scheduler |

### Multi-GPU and storage

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [G10](https://github.com/platformxlab/G10) | 2023 | `NO` | — | No storage destination, tensor lifetime, or async tensor I/O |
| [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin) | 2023 | `NO` | — | No collective/network scheduling or distributed routing |
| [Hierarchical Unified Virtual Memory](https://www.usenix.org/conference/atc22/presentation/choi-sangjin) | 2022 | `NO` | — | No peer destination, peer capacity, or remote mapping |
| [GPS](https://research.nvidia.com/publication/2021-10_gps-global-publish-subscribe-model-multi-gpu-memory-management) | 2021 | `NO` | — | No replication, global directory, or coherence action |
| [CARVE](https://research.nvidia.com/publication/2018-10_combining-hwsw-mechanisms-improve-numa-performance-multi-gpu-systems) | 2018 | `NO` | — | No peer topology, remote-cache allocation, or coherence control |
| [Griffin](https://doi.org/10.1109/HPCA47549.2020.00055) | 2020 | `NO` | — | No peer identity/destination or exact peer migration |

## Completed routes and still-missing mechanism surface

The earlier MoE-cache and scheduling-intent routes have now produced controlled
performance evidence, and no-prefetch has a same-policy mechanism-cost result.
Those successes do not erase the matrix boundaries.  The tested MoE policies
obtain expert semantics and transfers from their application executor; XSched
retains its original Level-1 XQueue frontend/actuator; and GPREEMPT retains a
two-context blocking-kernel executor with a compatibility signaling path.

The highest-value open expressibility questions are therefore mechanism gaps,
not more names in a baseline list: arbitrary cross-block/tier migration, exact
victim and destination choice, lossless command admission/completion,
asynchronous storage-transfer decisions, and multi-tenant policy isolation.
Papers classified `NO` identify these missing actions and must not be relabeled
from a similarly named local heuristic.

**Current GPREEMPT feasibility:** the old build-only status is superseded.
The 45-cell load study and 27-cell LC-knee sweep run native, original-C, and
actual BPF on RTX 5090.  The BPF arm executes kernel role/timeslice callbacks
and host-uBPF reset/hint/block/release decisions; the original-C arm executes
the same decisions without BPF.  Their behavior is close over the measured
loads, while both expose the LC-protection/BE-goodput tradeoff.  The status is
therefore **measured compatibility policy port**, not pending runtime.  It
remains `ANALOGUE`: host-mapped flags replace GDRCopy, the paper's original
models/hardware were not rerun, and the requested one-microsecond setting is
not direct proof of the effective hardware preemption quantum.

## bpftime routes and local checks

The old build-only/device-timeout status is superseded by two fresh strict
pairs.  In each positive cell, the R5 runtime admits the actual 13-instruction
program and records exactly 32,768 return callbacks while all 32,768 vector
outputs match.  In each negative cell, a lane-varying branch is rejected before
hook creation and fresh counter readback remains zero.  These runs establish a
narrow strict real-device path; they do not prove general verifier soundness.
The three earlier failed canaries remain retained and are not relabelled.

The separate trampoline campaign supplies performance evidence, but uses a
verification-disabled runtime.  Across 270 measurements, a return-only handler
adds 0.0012--0.0022 ms at fixed 4,096-block geometry, while a per-thread counter
body grows with active work.  The audited PTX uses ordinary per-thread
`call`/`call.uni`, not once-per-warp dispatch.  Strict engagement and measured
overhead therefore remain separate facts rather than one combined safety claim.

Host-uBPF ports in MoE-Infinity, XSched, GPREEMPT, Expert Buffering, FineMoE,
and Hummingbird also execute real JIT decisions.  They keep each workload's
native frontend and actuator, so they demonstrate bounded decision
expressibility, not that bpftime alone supplies XQueue, DMA, cache, or launch
execution.  More elaborate Orion/TimeGraph admission still requires operation
identity, completion events, and a lossless queue/dependency protocol; simply
overriding `cuLaunchKernel` does not implement that protocol.  Likewise,
Kernelet/Tally/LithOS still need exact slice replay, dependency preservation,
and spatial controls beyond the existing block predicate.

Primary artifacts provide future original-system baselines for
[Orion](https://github.com/eth-easl/orion),
[Paella](https://github.com/eniac/paella), and
[Tally](https://github.com/tally-project/tally), but none was built or run here.
TimeGraph/Gdev's legacy driver mechanisms are not drop-in RTX 5090 baselines.

## Offline validation

Run from the repository root:

```bash
python3 docs/experiment/policy/reference/validate_related_policy_expressibility.py
```

The check enforces the record floor, seven-category coverage, classification and
evidence enums, HTTPS URL shape, required fields, repository-relative path
containment/existence, and exclusion of known mismatched local source assets.
It is deliberately offline and does not make network availability a build gate.
