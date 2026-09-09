# Next kernelretsnoop optimization: GPU-local full records

## Queued continuation — 2026-09-08 22:25 PDT

The [SoA comparison](results-full-record-soa-20260908.md) is complete:
five paired blocks, 19.229% median throughput gain over AoS, but 27.621%
median loss relative to the uninstrumented control. These completed cells
must not be repeated. Both layouts retain all ten u64 fields and all
23,068,672 observed per-thread events. This is not the compact warp-record
variant, and neither replaces the original P40 or RTX 5090 Table 1 numbers.

The strongest next layout candidate is **32-slot grouped SoA (AoSoA)**.
Source inspection of `full-record-device-buffer/full_record_device_buffer_value.h`
shows that each current field plane has 256 * 16384 u64 elements: adjacent
fields for one event are 33,554,432 bytes apart. The current writer coalesces
adjacent-slot stores within each field, but accesses ten widely separated
field planes. Grouping fields within 32 consecutive logical slots could
reduce that address spread while retaining coalesced stores. These logical
groups are not an assumption that every possible launch geometry maps them
to exactly one physical warp.

For bank-local slot `s`, record index `k`, and field `f` in [0, 10), the
candidate u64 payload index is:

```text
(((k * (16384 / 32) + s / 32) * 10 + f) * 32 + s % 32)
```

All 32 slots' ten fields then occupy 2,560 contiguous bytes at a fixed `k`.
This changes only physical placement: retain 32 banks, 16,384 slots per bank,
256 records per slot, 335,675,408-byte bank values, all fields, the existing
per-thread timer calls, counter semantics, and whole-bank post-client drain.
No sampling, timestamp sharing, deduplication, reduced capacity, or manual
leader-only execution. Update writer and collector together in an opt-in
separate build; keep measured AoS/SoA binaries and the runtime unchanged.

This continues the same observability-cost question, now worded in the
current manuscript as RQ2 (Mechanism Cost): "What runtime overhead does
\\sys{} add for hooks, observability, and policy execution?" The directory
retains the historical RQ4 name. Its role is supporting: distinguish a
remaining layout cost from the competing explanation that callback/codegen
cost dominates. Better locality is not proof of faster execution; additional
index arithmetic, similar actual memory transactions, or callback overhead
could erase the expected benefit. No cache/TLB bottleneck is claimed from
source layout alone, and AoSoA is not claimed as a novel policy.

When a local-model slot becomes available, implement this scoped option and
reuse the existing `raw/full-record-soa-continuation-20260908.RBApri/paired.py`
workflow on pp512/tg0 TinyLlama, RTX 5090 / driver 575.57.08. The smallest
new matched comparison is current SoA versus grouped SoA, plus an
uninstrumented control, in five interleaved blocks. New controls belong only
to this changed-implementation comparison; none of the completed old cells
is overwritten or rerun. Report paired throughput ratios and baseline losses,
block variability, and post-client drain time separately. Existing collector
outputs are retained without a new counting, clock, or preflight campaign.
An actual implementation/launch error is repaired before retrying its failed
configuration; no timeout is imposed on the local-model work.

Positive results would support another record-preserving implementation
improvement; neutral or negative results would bound this layout hypothesis,
not show that BPF overhead is irreducible. The original P40 ratio is a
reference, not a value the measurements must be forced to match. Exact new
build/run commands and output directory await code handoff; no performance
claim or manuscript change is authorized by this plan.

**Queue, not a fourth active session:** the two XSched repairs and genuine
LMCache KV integration retain their current priority. Do not interrupt or
replace any live OpenCode generation to start this continuation.

## Current implementation update

The repaired 32-bank, record-major implementation now completes
[all five three-arm blocks](results-full-record-device-buffer-20260908.md).
Baseline/ring/GPU-local throughput medians are 38240.044/345.221/23267.229
token/s. Paired GPU-local/ring gain is 67.472x, while baseline-relative loss
remains 39.099%. Post-client bulk copy takes median 1490.922 ms, separately
reported. The first successful GPU-local cell was retained, not repeated.
The historical failure and proposal below remain; they do not describe the
current completion status. No old Table 1 value or manuscript was changed.

The first writer/collector builds at `743ef571`, but its first real prefill
aborts before throughput. The object BTF truncates the planned 1342701584-byte
bank to 268959760 bytes. The [actual run record](raw/full-record-device-buffer-first-20260908.BytFAr/README.md)
supersedes the eight-bank assumption below: local Qwen now changes banking
to 32 values of 16384 slots, preserving the total event capacity. The new
bank size is 335675408 bytes. Record-major indexing is also in implementation
to group adjacent-thread stores. No GPU-local full-record performance is
established yet; the original eight-bank proposal is preserved below.

The completed transport-2/3 comparison (`a9fb0a13`) improves median paired
throughput by 65.04%, but its 346.81 token/s remains far below the earlier
uninstrumented workload. Both layouts still write host-mapped memory. The
next hypothesis is that retaining the full event stream in GPU memory and
copying it in batches after prefill removes that hot-path transfer cost.

Reuse the existing GPU_ARRAY map (1503), CUDA IPC and collector-owned
allocation demonstrated by `onevalue-array-candidate`. Do not change the
existing runtime, old tool or old measurements for this first implementation.
This is the same kernel-return observation task, not a new paper workload.

## Required logical records and capacity

- Preserve all ten original u64 fields: block xyz, thread xyz, block
  dimensions xyz, and the per-thread timestamp (80 bytes).
- Record every thread; no leader-only filtering, timestamp substitution,
  count-only aggregation, sampling or deduplication.
- Preserve 524288 thread slots and 256 records per slot. The observed pp512
  run produces 23068672 records, but that is not a reason to shrink capacity.
- A single value for the entire allocation exceeds the device map-info
  signed-int value-size field. Use eight values/banks, each holding 65536
  thread slots, rather than changing that ABI. Per-bank records occupy
  1342177280 bytes, below signed-int range with counters and small headers.
- Reuse the existing per-thread linear coordinate mapping and report
  out-of-range/overflow counters. Do not silently generalize its supported
  geometry. Use u64 arithmetic before indexing across banks.

The collector performs eight whole-value lookups after the CUDA client
finishes, reusing one host destination allocation. It reports total bytes
and collection duration separately from llama.cpp prefill token/s. Device
storage totals about 10 GiB; required host shared memory must be specified
explicitly in the invocation. No new runtime bulk-copy API is needed.

This initial path is bounded capture with post-run collection, not an
unbounded concurrently drained streaming ring. Preserve that distinction.
The old full-record ring remains the comparison and fallback, not replaced.

## Execution after implementation

Use the existing pp512/tg0 TinyLlama workload and prefill metric. Compare
the unchanged full-record ring transport 3 and GPU-local full-record probe,
with an uninstrumented control only in the new matched comparison; retain
all old campaigns. Report allocation/collection costs separately, since
prefill-only speedup does not make those costs disappear. No clock-calibration
work, new paper, manuscript edit, or additional audit campaign is part of
this task. Existing verifier behavior is reported, not silently changed.

Implementation is delegated to local OpenCode after its current reusable
transport CLI task finishes; root reviews, builds, measures and publishes.
No new performance result is claimed by this plan.
