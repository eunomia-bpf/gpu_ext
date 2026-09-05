# Supporting experiment plan: gpubpf UVM prefetch_stride_lfu for vLLM weights with explicit LMCache disk KV

## Research question

- Specific uncertainty tested: can the existing gpubpf UVM
  `prefetch_stride_lfu` policy reduce the incremental cost of UVM-managed
  vLLM weights while LMCache retains explicit local-disk KV transport?
- Arms:
  1. `stock` — stock `nvidia-uvm` module with the native LMCache disk
     configuration.
  2. `uvm` — custom gpubpf-capable `nvidia-uvm` module plus a weights-only
     UVM pool, no BPF policy.
  3. `uvm+lfu` — the identical UVM arm plus the existing
     `prefetch_stride_lfu` policy.
- Why it matters: `uvm` versus `stock` measures the unmanaged-UVM penalty,
  `uvm+lfu` versus `uvm` isolates the BPF policy contribution, and
  `uvm+lfu` versus `stock` answers whether the policy recovers that penalty.
  The result bounds the weight-management mechanism without touching the
  explicit disk-KV comparison.

## Paper-value admission (role)

- Planned role: **supporting**.
- Load-bearing uncertainty: whether `prefetch_stride_lfu` recovers the
  incremental cost of UVM weight management on this stack. A positive result
  supports the UVM-weight + explicit-disk-KV design; a negative result
  bounds the tested policy and workload and narrows the mechanism claim.
- Independent evidence added: first matched three-arm measurement of stock
  module, unmanaged UVM, and `prefetch_stride_lfu` on driver 575.57.08 with
  the paper's Qwen3-30B workload; no published result covers this
  configuration.
- Honest outcome stated up front: BPF may only recover part of the UVM loss
  and may remain below native LMCache; that is a valid, expected outcome,
  not a failure of the experiment.

## Environment and reused assets

- RTX 5090, driver `575.57.08`, kernel `6.15.11-061511-generic`, official
  vLLM `0.27.1+cu129`, LMCache `0.5.4`, `Qwen3-30B-A3B-FP8`.
- Same workload and server settings as the 575-v3 smoke protocol: 8 frozen
  prefix pairs, 1,536 cached tokens, 16 output tokens, sequential warm.
- Reused assets:
  - allocator source: `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`;
  - custom UVM module:
    `/opt/gpubpf/modules/575.57.08/6.15.11-061511-generic/nvidia-uvm.ko`;
  - BPF policy loader: `extension/prefetch_stride_lfu`;
  - UVM eviction monitor: the moe-infinity UVM monitor.

## Weights-only UVM boundary

- The UVM pool is a torch `CUDAPluggableAllocator` inside a
  `torch.cuda.MemPool`, active only in the `GPUWorker.load_model` weights
  context.
- KV allocation remains on the default allocator; the LMCache disk
  connector stays explicit (same connector, O_DIRECT disk tier, same cache
  layout) in all three arms.
- No arm changes the model, workload, token arrays, schedule, or timed warm
  phase.

## Comparison and baseline fairness

- `stock` is the main baseline (current native LMCache disk practice on this
  stack). A matched run is required because no published result covers
  575-series UVM weight management on this model and hardware.
- `uvm` is a component ablation (UVM without policy) that separates the
  UVM-management cost from the BPF policy effect.
- Fairness: identical server command shape, environment allowlist, model
  snapshot, prompt/schedule artifacts, memory budget, disk cache directory,
  and timed warm phase across arms; only the module, allocator scope, and
  BPF policy differ.
- Each arm must show its intended mechanism engaged (see completion); a
  non-engaged arm is invalid for its comparisons, not a win for another arm.

## Metrics and engagement evidence

- Primary: warm median TTFT (ms) and warm-phase output throughput (tok/s).
- Secondary: warm E2E latency (ms) and UVM migrated bytes (fault/migration/
  eviction counters from the module and the eviction monitor).
- Required per-arm evidence:
  - allocator live-byte and allocation log proving the UVM pool is used only
    for `load_model` weights and not for KV allocation;
  - nonzero UVM fault and migration counters during warm requests, plus
    eviction evidence where pressure exists;
  - LMCache disk KV engagement: 8/8 warm hits at 1,536 tokens and the
    48-file / 1,207,959,552-byte disk footprint with read evidence;
  - BPF deltas between request phases: `page_fault_calls`,
    `stride_detections`, `prefetches_issued`, `lfu_activations`,
    `lfu_accesses`, `lfu_sampled_updates`, `lfu_reorder_requests`,
    `eviction_prepares`.

## Execution

- One real preflight: the full 8-prefix workload on the UVM arms, verifying
  allocator logs, UVM counters, disk KV hits, and BPF deltas end to end
  before any timed block.
- Then 5 randomized paired blocks; each block runs all three arms in the
  block's frozen randomized order, with the seed recorded in the schedule
  artifact.
- Raw results under `raw/uvm-weight-<id>/block-NN/<arm>` with server logs,
  environment records, allocator logs, UVM/BPF counter dumps, and
  engagement records.

## Completion

- A block completes only when all three arms finish the full 8-prefix
  cold + warm protocol and every engagement check above passes.
- The experiment completes after 5 valid blocks, or earlier if the
  preflight or repeated cell failures show the path cannot run; partial
  blocks are never analyzed as complete.

## Interpretation

- Positive: `uvm+lfu` recovers most or all of the `stock`-to-`uvm` loss in
  TTFT and throughput; report the recovered fraction and residual cost.
- Negative/contradictory: `uvm+lfu` remains below `stock`; the existing
  `prefetch_stride_lfu` does not recover the UVM weight-management cost on
  this workload, which bounds the mechanism to the tested policy/workload
  scope.
- Mixed: partial recovery; report the policy-off (`uvm`) cost decomposition
  and the policy effect separately.
- No result here changes the explicit disk-KV comparison; that claim stands
  on its own protocol.
