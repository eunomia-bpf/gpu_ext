# 575 UVM Pressure Canary — 2026-09-06

Hardware: RTX 5090, 32607 MiB; NVIDIA 575.57.08; Linux 6.15.11-061511-generic.
Workload: Qwen/Qwen3-30B-A3B-FP8, 8 fixed prefixes, 1536 cached tokens, 16 output tokens.
No correctness/admission/engagement/filtering gates applied.

## Outcome

- Runs 01–03 were invalid for pressure measurement. Run 01 numbers are no-pressure
  reference only: 29.9408 native, 30.4887 BPF output tok/s.
- Run 04 is the one valid block: a paired canary with concurrent pressure.
  - Native: TTFT median 113.6388 ms, 1.7350 req/s, 27.7606 output tok/s.
  - Current BPF: TTFT median 170.1315 ms, 1.5216 req/s, 24.3455 output tok/s.
  - Relative BPF vs native: TTFT +49.71%, req/s -12.30%, output tok/s -12.30%.
- BPF loader: 2474 tracked chunks, aggregate debt pressure 26, saved 490, evicted 0.
  Only 11 KV disk-durable chunks appear in the largest captured 22 MiB allocation,
  against a 1 GiB explicit KV pool.
- Pressure result JSON is absent: the runner stopped the tenant after warm, before
  all requested 100 passes ended. READY/release plus loader counters support
  concurrent pressure, not full 100-pass completion.
- All four canary cache subdirectories were moved to Trash; raw logs remain.

## Attempts

1. 01 — Pressure launched after the model and both pressure processes already
   failed cudaSetDevice with OOM. Numbers: reference only, no pressure active.
2. 02 — Pressure prestarted and READY, but vLLM 0.98 rejected startup:
   30.37 GiB free < 30.73 GiB desired.
3. 03 — vLLM 0.95 started, then cold-phase OOM: model about 30.05 GiB plus
   pressure CUDA context about 498 MiB.
4. 04 — Valid one-block paired canary: identical 4 GiB CPU offload, gpu
   utilization 0.95, explicit 1 GiB KV pool, 4 GiB pressure tenant, requested
   100 passes with 50 ms pause. Both arms ready; pressure READY and released;
   8/8 warm. Per-arm numbers in Outcome above.

## Interpretation

- The current single-largest-allocation / migration-debt policy is negative
  evidence: under 4 GiB of concurrent pressure it cost ~12% throughput and
  ~50% TTFT median on this one block.
- This is not a final semantic policy result and not publication-level,
  due to a single block and the incomplete 100-pass pressure run.
- The 11-durable-chunks-in-22 MiB observation shows the visible allocation
  surface is far smaller than the explicit pool, so a single-largest signal
  mis-ranks what is actually recoverable.

## Next policy

- Direction: LMCache/framework publishes per-range lifecycle, disk
  recoverability, next-use deadline, and tenant. Verified hot-swappable gpubpf
  arbitrates driver-global UVM pages across processes, prioritizing cheap
  inactive recoverable pages while protecting active, soon-needed, and
  nonrecoverable pages. Do not claim this direction as first.
- Next canary: complete the full 100-pass pressure run to obtain the pressure
  result JSON, and collect at least one more block so the policy comparison is
  not single-block evidence.
