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

The page-level Expert Buffering analogue has local runtime evidence. The new
finite bpftime device-event workload has build evidence, explicitly separate
from the original eGPU paper's evaluation. The
four local PDFs previously found to contain unrelated papers are excluded as
evidence; publication, author, and official artifact URLs are used instead.

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
| [Towards MoE Deployment / Expert Buffering](https://arxiv.org/abs/2303.06182) | 2023 | `ANALOGUE` | `expert_buffering_policy`, `eviction_cycle_moe` | Page-level hot residency only; no gating or expert-atomic buffer |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | 2024 | `PARTIAL` | MoE prefetch + approximate LFU | No expert identity or expert-atomic transfer |
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
| [GPREEMPT](https://www.usenix.org/conference/atc25/presentation/fan) | 2025 | `ANALOGUE` | timeslice + whole-TSG preempt | [575 port pending](../../../driver_docs/sched/gpreempt-analysis/feasibility-575-20260902.md); missing role-to-TSG, hint and blocking-kernel protocol |
| [GCAPS](https://arxiv.org/abs/2406.05221) | 2024 | `PARTIAL` | timeslice + whole-TSG preempt | No GPU-segment feed or real-time admission guarantee |
| [XSched](https://www.usenix.org/conference/osdi25/presentation/shen) | 2025 | `PARTIAL` | timeslice + whole-TSG preempt | No XQueue/cross-XPU command suspend-resume runtime |
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
| [eGPU](https://asplos.dev/pdf/bpftime_super.pdf) | 2025 | `PARTIAL` | finite device-return counter workload | Observation/injection is not a scheduler; driver-policy feedback bridge remains explicit |

### Multi-GPU and storage

| Paper | Year | Result | Closest in-tree policy | Boundary |
|---|---:|---|---|---|
| [G10](https://github.com/platformxlab/G10) | 2023 | `NO` | — | No storage destination, tensor lifetime, or async tensor I/O |
| [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin) | 2023 | `NO` | — | No collective/network scheduling or distributed routing |
| [Hierarchical Unified Virtual Memory](https://www.usenix.org/conference/atc22/presentation/choi-sangjin) | 2022 | `NO` | — | No peer destination, peer capacity, or remote mapping |
| [GPS](https://research.nvidia.com/publication/2021-10_gps-global-publish-subscribe-model-multi-gpu-memory-management) | 2021 | `NO` | — | No replication, global directory, or coherence action |
| [CARVE](https://research.nvidia.com/publication/2018-10_combining-hwsw-mechanisms-improve-numa-performance-multi-gpu-systems) | 2018 | `NO` | — | No peer topology, remote-cache allocation, or coherence control |
| [Griffin](https://doi.org/10.1109/HPCA47549.2020.00055) | 2020 | `NO` | — | No peer identity/destination or exact peer migration |

## What can be experimented with now

The matrix suggests three honest experiment routes without claiming whole-system
ports:

1. **MoE cache-policy route:** compare the page-level Expert Buffering analogue,
   approximate LFU, and MoE-oriented prefetch under one frozen workload.  This
   tests a shared policy question (hot residency/reuse), not Huang et al.'s
   gating or expert-atomic implementation.
2. **UVM policy-component route:** compare no-prefetch, bounded adaptive/stride
   prefetch, and FIFO/LFU/MRU ordering.  These correspond to components studied
   by the HPCA'16, ISCA'19, adaptive-UVM, Forest, and HELM lines of work.  Report
   hook engagement separately from performance.
3. **Scheduling-intent route:** compare default scheduling with differentiated
   timeslices and supported whole-TSG preemption.  This exercises the policy
   intent shared with GPREEMPT/GCAPS/XSched, while explicitly excluding their
   missing driver/runtime protocols and real-time guarantees.

Papers classified `NO` are still useful: they identify concrete ABI extensions
needed for a future experiment.  They are not candidates for relabeling an
existing program.

**Current GPreempt feasibility:** the
[2026-09-02 source audit](../../../driver_docs/sched/gpreempt-analysis/feasibility-575-20260902.md)
replaces an absolute “infeasible” interpretation with **575/Blackwell port
pending, comparison not completed**. The official artifact still targets
550.120/sm_80, and its patch fails the read-only check on the 575 tree; those
negative results remain recorded. Basic wrapper and `sm_120` blocking-kernel
compilation succeeded, but no GPreempt driver or workload was run. Our present
BPF implementation remains an analogue, not an equivalent full-system port.
The separate historical paper feasibility table is retained unchanged.

## bpftime routes and local checks

The clean sibling `bpftime-r5` source at revision `36610ee` built its verifier
test target and passed **17 GPU-verifier test cases / 126 assertions** on the
CPU. The r5 `kernel_trace` and `threadhist` BPF examples, plus the development
tree's existing `atomizer` example, also compiled to BPF objects. None of those
checks executed a GPU kernel. The development source at revision `d6316fa`
already had tracked edits; this investigation did not modify that source.

Two short implementation routes emerge from the new papers:

1. **Keep a runtime's original frontend, move its bounded policy decision to
   userspace eBPF.** A same-XQueue HPF adapter is now available under
   `workloads/xsched/bpftime_hpf.*`; native XSched remains the reference, and
   the adapter uses bpftime's host-side uBPF JIT. This is not device-side BPF.
   More elaborate Orion/TimeGraph admission requires operation identity,
   completion events, and a lossless queue/dependency protocol; overriding a
   `cuLaunchKernel` return value does not implement that protocol.
2. **Add device observations, then evaluate one bounded block-policy
   component.** Device-return/thread maps can feed a host policy, but need
   explicit process/TSG correlation before they drive gpubpf. The existing
   atomizer's block-range predicate is a possible Kernelet/Tally/LithOS
   component only for independent-block kernels. Its partition divisor needs
   validation, and every slice must be replayed exactly once. Native slicing
   is the appropriate component baseline. This does not implement TPC
   steering, transparent CUDA virtualization, or whole-paper scheduling.

`workloads/bpftime-device-smoke/` provides a finite device-observation check:
eight launches of 4,096 threads, exact checking of all 32,768 output values,
and an independent requirement for 32,768 device-return callbacks (eight per
thread). It uses unique named shared memory, the shared GPU/struct-ops leases,
owned process-group cleanup, and pre/post GPU safety checks. It never removes
bpftime's default segment. Build and run from the repository root:

```bash
make -C workloads/bpftime-device-smoke -j2
python3 -B workloads/bpftime-device-smoke/run_smoke.py \
  --output workloads/bpftime-device-smoke/raw/575-canary-next
```

The existing CUDA runtime build is `../bpftime/build-cuda-pr503`; it is not
asserted to incorporate the r5 verifier. The three retained 575 canaries
(`raw/575-canary-root-01`, `-02`, `-03`) all **failed** the full engagement
criterion. Run 01 exposed the required `cuda__` GPU-program name prefix; run
02 exposed the matching loader-name update. Both names are now fixed. Run 03
matched/injected the target PTX and loaded its module; native and instrumented
workloads each checked all 32,768 values without mismatch, but the callback
criterion timed out. Attach success and preserved output correctness are not
proof of device-event engagement. All three runs removed their private shared
memory and returned to an idle GPU without new Xids. Counter snapshots and
failure-side correctness retention were subsequently added for diagnosis;
that diagnostic revision has only been rebuilt, not rerun on the GPU.

The three unchanged result files and parsed facts are retained in
[`workloads/bpftime-device-smoke/raw/canary-evidence.json`](../../../../workloads/bpftime-device-smoke/raw/canary-evidence.json).
A subsequent CPU-only call to the existing return-probe translation library
generated one unpredicated return call and no warp/lane filter. The per-thread
expectation remains 4,096 × 8; `call.uni` does not mean one invocation per warp.
The host lookup copies all 32,768 map bytes, and the probe retains ownership of
the CUDA allocation after the target exits. The leading unconfirmed hypothesis
is launch routing bypassing the patched CUfunction. Next-run counter snapshots
plus existing debug routing messages can distinguish that from a map/codegen
defect. Run 03's original patched PTX was not retained; the CPU reconstruction
is not retroactive execution evidence.

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
