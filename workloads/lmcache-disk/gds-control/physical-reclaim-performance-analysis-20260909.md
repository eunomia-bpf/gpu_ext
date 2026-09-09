# Physical-reclaim disk/UVM serving analysis (pXYN4F, 2026-09-09)

Read-only interpretation of the completed pXYN4F campaign (`workloads/
lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F/`, main commit
`b8daf444`, 15/15 cells, exit zero, driver `95097e20`); facts cite
evidence, inferences are labeled hypotheses.

## Published results (exact)

- Median warm throughput (token/s): stock **73.574518**, native **68.608032**,
  BPF **65.422796**; per block B0 71.698241/69.291949/66.487342, B1
  70.572913/68.387745/68.733332, B2 73.961755/67.656516/65.422796, B3
  73.575320/69.575409/65.166163, B4 73.574518/68.608032/65.303293. Paired
  (within block, then summarized): BPF/native median **-4.047522%**
  (-6.337362%..+0.505334%, 4/5 adverse); BPF/stock **-11.241969%**;
  native/stock **-5.436485%** (all pairs adverse). "The policy does not
  improve throughput in this workload"; BPF/native is end-to-end, not an
  isolated BPF overhead measurement.
- Geometry (all arms): 384 MiB pool = 4096 tokens = 96 KiB/token; 8 cold
  prefixes 1536 tokens (48 prepared 24 MiB chunks); 8 warm prefixes
  1536/1024 tokens = 6/4 chunks; minimum warm demand 40 chunk restores;
  `max_num_seqs=2`; `recompute_ns_per_token=62502`.

## Observed behavior: rollback-to-0 vs prefix reload

- UVM counters present in 12/15 cells (missing: b0-BPF 0-byte file, b1-stock,
  b2-BPF; missing, not zero): policy cells all `restored=40` (the minimum,
  one restore per warm prefix chunk); stock cells 72/72/82/72 (b0/b2/b3/b4);
  `prepared=retained_total=48`; fallback/error counters 0.
- The stock excess is prefix *reload after re-admission*, not extra demand:
  b0-stock p1-warm loaded 6x, p3/p5/p7 2x each, p0/p2/p4/p6 once; per-request
  chunk totals = 72, matching the UVM counter exactly; b3-stock's excess is
  seven 1536-token reloads (40+42=82). Repeated preemption is arrival-
  dependent (2026-09-08: extra restorations sat on the policy arms), so the
  count difference is not a stable arm property.
- The stable arm property is the *re-admission route*. Stock rollbacks read
  "from N to 1024 tokens" (progress truncated to the prefix boundary; the
  prefix is kept and reloaded from disk, ~4 chunks per event). Policy
  rollbacks read "from N to 0 tokens" (b4-native: p4 2294->0, p6 1656->0,
  p7 1718->0, p0 2457->0, p2 1573->0, p2 2309->0): the armed
  ROUTE_FULL_RECOMPUTE forces a 0-hit lookup on re-admission, so the
  request recomputes its prefix and lost progress on the GPU.
- Zero-hit scope (root's `pXYN4F/lookup-log-counts.txt`): warm zero-hit lines
  1380 or 1414 in every policy cell, 0 in stock; the 8 cold-phase lines are
  separate, not warm. Warm positive-hit lines 8 policy vs 495/496 stock;
  retrieval completions 8 vs 15/16.
- Disk-restore cost, observed: b4-native `server.log` lines 331/335 record
  the first two warm requests' restores - p4-warm "Retrieved 1536 ... cost
  83.3271 ms, 1.6876 GB/s", p5-warm "Retrieved 1024 ... cost 58.2921 ms,
  1.6083 GB/s" - preceding their ~139/120 ms TTFTs, so those fast TTFTs
  include the disk restore. The other six warm TTFTs are 24.6-35 s,
  consistent with queueing under `max_num_seqs=2`; warm elapsed ~114-125 s.
- `preemption_log.count=0` is a parser artifact, not proof of no preemption:
  rollback warnings (7-8 stock, 4-6 policy per cell) plus the warm 0-hit
  bursts prove re-admission in both arms. Timing decomposition (waiting vs
  recompute vs queueing) is *unknown*: the ~1164.8 tokens/s prompt peak is a
  periodic logger window average, not isolated active prefill throughput,
  and no per-event recompute duration is derived.

## What each arm selects, and who restores

- Stock: vLLM default victim; re-admission runs the ordinary LMCache lookup,
  which restores the disk-backed prefix. Native: the adapter builds up to 8
  same-priority-class candidates (freeable exclusive blocks, captured
  `num_computed_tokens`, disk-backed bytes, coverage flag) and decides via
  the userspace ctypes entry in `kv_reclaim_native.so`. BPF: identical
  candidates/prices, decided in-kernel as struct_ops `gpu_kv_reclaim_choose`
  over UVM ioctl 83, recording only non-stock.
- Both arms execute the same `uvm_kv_reclaim_choose()` in `kv_reclaim_abi.h`
  (minimize estimated recovery cost per freeable KV byte within the worst
  priority class; ties/unusable telemetry fall back to the stock victim):
  identical inputs give identical decisions; differences are execution
  location and transport, not the rule.
- Actual restoration, in all arms: the nvidia_uvm driver's CPU-staged
  fault/hydrate path for sealed on-disk GPU-promoted ranges (ioctls 84-87,
  2 MiB VA blocks), driven by `ChunkBacking.restore()` in
  `lmcache_diskuvm_backing.py` (`gpu_fault_read` -> D2D copy -> re-OFFLOAD
  releasing GPU/CPU PTEs; no file write). The policy arms choose victim and
  route only; they perform no restoration.
- Prices: recompute is fixed 62502 ns/token ("median TTFT per prompt token
  in single calibration phase, not pure GPU compute"); the disk price is
  `_backing.read_stats().mean_ns_per_kib` (None until a real read; capture
  wraps the GDS load path `_load_gds`, root-verified; no claim the DiskUVM
  restore path bypasses capture). `kv_reclaim_abi.h` biases: unknown coverage
  prices the disk prefix as zero; unknown disk rate prices the disk route
  unbounded; both push toward FULL_RECOMPUTE.

## What the gap does and does not isolate

- Per event: a stock re-admission costs one ~58-83 ms disk reload plus
  recompute of the lost progress only; a policy re-admission costs a forced
  0-hit plus unknown-duration waiting plus recompute of prefix and lost
  progress. The recompute price does price that lost progress *in its token
  count*; what it omits is waiting, contention, and repeated future cycles.
  (Hypothesis: the stable arm difference is this re-admission route.)
- BPF/native -4.047522% (4/5 adverse) cannot be decomposed from these
  traces: no kernel-side BPF decision record is retrievable (non-stock
  only), and the adapter diagnostics that would echo routes/prices never
  arrived (every policy cell `arrived=false, expected=true,
  waited_s=10.008`), so struct_ops overhead is not separable.

## One feasible next change (no implementation)

Re-evaluating "when schedulable, using the actual num_computed_tokens" is
circular: the route affects admission and after preemption the live
`num_computed_tokens` is 0 - a schedulable-time decision sees a zeroed state
of its own making. The feasible change: preserve the logical computed extent
captured at preemption time; decide the route exactly once, before the
re-admission lookup starts, using that preserved extent plus the latest
coverage and read cost; keep that decision stable while the async lookup
proceeds (the existing DISK_PREFIX path already runs the real lookup/restore
as upstream with a stable route; FULL_RECOMPUTE applies the 0-hit with no
lookup at all).

Existing mechanism: the per-request `_pending_route` entry
(adapter:689-707) spans preemption to consumption with stable route
application, and the preemption-time candidate snapshot captures
`num_computed_tokens` (adapter:555-577). Missing: (a) the pending entry
stores only route/route_name/estimated_ns/decision_seq - the preserved
extent is not retained; (b) the hooked lookup for a pending FULL_RECOMPUTE
returns 0 without any decision call (adapter:398-404), so the route is never
re-derived from fresh coverage/read cost. Both are source-local,
unimplemented, untested; this changes when and against what the single
decision is made, not which prices are used, and does not duplicate the
completed total-cost and ratio campaigns.

## Limitations

- CPU compile overlaps: block0 native 06:51:18, block4 BPF 07:24:43 (CPU-
  only, no GPU launch); effects unmeasured, not assumed zero; cells kept.
- Missing data stays missing: 3 UVM counter cells; adapter diagnostics in
  every policy cell.
- Prior campaigns, numbers preserved: total-cost 2026-09-08 (20 policy
  cells, `4381f660`): medians stock 71.916482, native cost/byte 64.743942,
  native total cost 68.964702, BPF total cost 68.834700 - both below stock,
  both completed. Grace 2026-09-08: BPF/native near zero at the mean; extra
  restorations arrival-dependent, shared between native and BPF. Driver
  `95097e20` removes a serving-OOM *hypothesis* on this workload; completion
  does not prove OOM is impossible elsewhere.
