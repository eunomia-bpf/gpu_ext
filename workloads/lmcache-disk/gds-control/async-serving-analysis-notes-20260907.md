# Async-prefetch serving analysis — gds-async-prefetch-575-20260907-02 (2026-09-07)

CPU-only analysis of
`raw/gds-async-prefetch-575-20260907-02/raw.jsonl` (20 cells, all ready, no
cell errors, 160 warm completions; campaign exit 0). CLI:
`../analyze_gds_async_prefetch.py` (Markdown/JSON on stdout; root redirects).
Campaign params from `campaign.json`: 5 rotated blocks, 4 arms,
`prefetch_lead_ms=50.0`, warm stagger 250 ms, warm concurrency 4,
max-num-seqs 2, 768 MiB KV, 256 MiB GDS staging, 16 output tokens per warm
request. The earlier `gds-async-prefetch-575-20260907-01` (same 50 ms lead)
kept pathological ~3.3 s async TTFT; campaign 02 is the current record.

## Measured (raw.jsonl and the campaign server logs only)

- Throughput: arm medians across blocks (output tok/s): demand 58.6,
  eager 58.4, native 58.6, bpf 58.2; the overall range across all cells is
  57.4–59.2. These close medians come from a finite, arrival-dominated window.
  The fixed 250 ms stagger sets its duration to ~7×250 ms plus the last
  request's E2E (~2.19 s in all arms), so tok/s ≈ 128/2.19 ≈ 58 in every
  arm; the last-request tail and worker backpressure still contribute, so
  throughput is not independent of retrieval timing. The fixed arrival
  schedule limits interpreting these differences as saturated serving capacity.
- TTFT: per-cell median, then median across blocks (ms): demand 85.7;
  eager 99.8 (+14.1); native 116.1 (+30.4); bpf 116.1 (+30.4).
- Paired change vs `gds_demand_fifo`, same block only, complete pairs only:
  - eager: dTTFT +6.0…+16.6 ms, 0/5 blocks improving; dmean E2E
    −32.8…+16.1 ms, 4/5; dtok/s −0.7…+0.6, 2/5.
  - native: dTTFT +20.9…+42.0 ms, 0/5; dmean E2E −3.7…+21.0, 1/5;
    dtok/s −0.9…+0.1, 1/5.
  - bpf: dTTFT +24.9…+43.2 ms, 0/5; dmean E2E −7.6…+48.9, 1/5;
    dtok/s −1.2…−0.2, 0/5.
- Mean E2E per cell (ms): demand 455.6–490.7; eager 454.2–475.8;
  native 472.7–508.1; bpf 479.4–504.5. Pooled per-arm means: demand 474.1,
  eager 462.7, native 487.2, bpf 493.1. Eager has higher TTFT but lower mean
  E2E than demand: the client-observed post-first-token interval
  (E2E − TTFT) is marginally shorter for eager; that is an observed
  client-side interval, not an attribution to GPU decode. The TTFT penalty
  does not extend to the pooled mean whole-response latency, although one of
  the five eager/demand E2E pairs is adverse.
- native vs bpf: the two arms' marginal TTFT medians are nearly equal
  (116.12 vs 116.12 ms), but that marginal median hides per-block
  divergence. The same-block BPF/native ratios (root-reported pairs,
  reproduced with `--reference gds_async_native`) give a TTFT median of
  +3.5566% over a range of −13.3148% to +20.4513%, and a tps median of
  −0.8465%. The native decider mirrors the BPF rules and both arms use the
  same input definitions and application hints; live completion-derived
  estimates can differ. The paired ratios show no consistent BPF-side advantage
  and do not establish a tight mechanism-overhead bound.
- Server logs: every one of the 15 async cells logged exactly 48
  `LMCache WARNING: Ref count of MemoryObj … is negative: -1. Double free
  occurred somewhere. Setting ref count back to 0 as a hack`; all 5 demand
  cells logged 0 (720 warning lines total).

## Limitation

The 48 negative-refcount warnings per async cell are a known unfixed
refcount defect. The temporary repair task (originally assigned to Qwen
Next) terminated with HTTP 524; the same session is now GLM. Every
async-arm number above was measured with that defect active: the reset-to-0
hack perturbs staging-lifetime accounting. These numbers must not be called
clean final mechanism overhead. No logged counter connects the refcount
warnings to the TTFT deltas; their latency contribution is not measured,
and none is claimed.

## Why async TTFT is slower: measured anchors vs hypotheses

Measured anchors (source and log facts, not latency measurements):

- Artificial 50 ms hint: every arm sends
  `lmcache.prefetch_deadline_ns` = HTTP send time + 50 ms (runner parameter
  `prefetch_lead_ms=50.0`). It is an explicit application deadline, not a
  scheduler use-time prediction, and it is identical across arms. The demand
  arm ignores the field; in the async arms it is the only input that sets
  `HINT_PREFETCH_JIT`.
- Eager async costs: the eager arm (fifo decider, always submit) still runs
  the full async seam — async lookup path, admission lock, executor-thread
  cuFile handoff, future/claim plumbing — against the demand arm's
  synchronous first-token retrieval.
- No actual scheduler demand feed: `mark_demand()` is not connected to the
  vLLM scheduler and the framework continuation
  (`batched_get_non_blocking`) is explicitly not a demand signal; no arm
  receives real demand feedback.
- Estimates start at zero; before the first warm read completes,
  `estimated_transfer_ns=0` and the JIT branch cannot defer.

Hypotheses (consistent with the measurements, not proven by them):

1. The async integration may contribute to eager's +6…+17 ms TTFT difference;
   this does not establish an unavoidable overhead floor.
2. Native/bpf +21…+43 ms may combine async integration costs and JIT deferral: once
   completion-derived estimates exist, slack to the artificial 50 ms deadline
   can exceed the estimate, and the policy defers submission by
   min(slack−estimate, 10 ms). Because the deadline is unrelated to when the
   scheduler actually consumes the prefix, deferring the one retrieval that
   could have overlapped with a soon-scheduled request delays the first
   token with no offsetting demand-side benefit — a net TTFT penalty in this
   workload.
3. The fixed arrival spacing may mask capacity differences; TTFT and E2E
   therefore provide important additional signals in this design.

Not established by these numbers: that async prefetch helps under realistic
scheduler timing (0/5 blocks improved TTFT for every async arm), a stable BPF
performance advantage or tightly bounded overhead, or that the refcount defect
explains the TTFT delta.

## Reproduce

```
python3 analyze_gds_async_prefetch.py raw/gds-async-prefetch-575-20260907-02
python3 analyze_gds_async_prefetch.py raw/gds-async-prefetch-575-20260907-02 --format json
python3 analyze_gds_async_prefetch.py raw/gds-async-prefetch-575-20260907-02 --reference gds_async_native
```

A short-circuited `raw.jsonl` is reported with a PARTIAL status and the
missing cells listed; incomplete block pairs are rejected from paired
aggregation only, while all individual observations remain in the arm
medians and per-cell table.
