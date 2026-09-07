# Reusable ioctl buffer (RQ2 support): decision-level evidence and plan review (20260907)

Analysis-only review of the supporting RQ2 experiment: the same BPF policy
program, comparing the current per-call
136-byte ioctl buffer allocation against an opt-in reusable buffer/persistent
ctypes view, with a native same-algorithm control. Real-driver paired decision
timing was ordered before any end-to-end storage comparison; it has now run
and this note reviews that evidence. Nothing here creates code, reruns any
cell, or launches GPU work. Synthetic ctypes packing probes attempted while
drafting failed and are not performance evidence: this note cites only the named files and
completed reports. No file/content hashes, checksums or digests are generated,
compared or recorded for this note.

## Standing constraints

- The completed 25-cell 200 ms policy-budget matrix stands (BPF 200/10 read
  p99 paired median -51.011%, all five pairs improving on both metrics); it is
  never rerun and is not used as a contrast for new numbers. Its own stated
  next work is integration, not repetition.
- Each admission still invokes its actual native/BPF decider exactly once;
  decisions are unchanged. The switch is default-off
  (`LMCACHE_GDS_IOCTL_REUSE=0` default) and was implemented separately
  (replay metadata records implementation `b0eeb67c`); its files were not
  edited here.
- No arbitrary performance threshold is used anywhere; no extra gate and no
  separate correctness campaign; no additional GPU studies by this note.

## Landed evidence: real-driver paired decision replay

Artifacts (read in full by this note):
`/home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/gds-ioctl-reuse-575-20260907/decisions.json`,
`.../analysis.json`, `.../command.txt`. Recorded scope string: "real command82
decision calls, no storage I/O".

Procedure (command.txt): 6 blocks x 3 arms (native, legacy_bpf, reuse_bpf)
rotated by `(block+position)%3`; per cell 100 warmups then 1,000 timed
`decide()` calls over a fixed four-request cycle (24 MiB demand read;
live-demand write queue_depth=8 slack 10 ms; live-demand write queue_depth=0;
speculative read pressure 900 slack 5 ms); `LMCACHE_GDS_IOCTL_KEEP_GIL=0`;
the same nvidia_uvm command-82 driver path throughout. Records keep
elapsed_ns plus action, defer_ns, priority, batch_target per request.

Cell medians, ns: native 467--490.5 (median-of-medians 472.5, p99 median
1495); legacy_bpf 1526.5--1596 (median 1575.5, p99 median 2670); reuse_bpf
1174--1223 (median 1190, p99 median 2234). The between-arm median difference
is 1575.5 - 1190 = 385.5 ns: reuse saves about 0.39 us per decision call.

| Paired reuse_bpf/legacy_bpf | Blocks 0--5 (%) | Median (%) |
| --- | --- | ---: |
| Median change | -24.937, -25.756, -23.766, -22.595, -24.634, -21.651 | -24.2001 |
| p99 change | -18.198, -17.613, -16.144, -17.290, -16.938, -10.637 | -17.114 |
| Loop decision rate | +30.924, +31.331, +6.836, +25.755, +28.610, +0.970 | +27.182 |

All six paired medians improve. The ranges are the six observations, not
confidence intervals; the block-5 loop-rate pair (+0.970%) shows the loop-rate
metric itself is noisy (block-0 native sits at 1.33 M calls/s against
1.51--1.56 M in the other native cells).

Decision-stream identity (recorded data, re-verified by read-only recompute of
decisions.json): across all 6 x 1,000 request rows the legacy_bpf, reuse_bpf
and native records agree exactly on action, defer_ns, priority and
batch_target — 6,000/6,000 triples, zero disagreements, zero recorded errors.
This agreement covers the four exercised request classes, not the entire
correctness surface of the switch or its concurrent use. It additionally
confirms that the driver/BPF program reproduces the native same-algorithm
output on every exercised request class.

## Prior decision-stage fact base (existing reports)

Instrumented concurrent campaign (results-575-gds-admission-timing-20260907.md;
medians of five per-cell medians, event-driven 10 ms executor, us):

| Stage | Native | BPF |
| --- | ---: | ---: |
| Request provider | 7.908 | 8.166 |
| Decision-lock wait | 0.234 | 0.288 |
| Decider call | 1.827 | 17.962 |
| Total admission | 11.973 | 30.140 |

- BPF per-cell decider medians 16.929--19.023 us; BPF maxima 3,008--3,140 us
  (decider call), read lock-wait max 5,884.749 us, admission max 5,914.727 us;
  native maxima 1.004/78.735 us. The cause of the concurrent-context tails is
  unrecorded.
- C++ executor reference (raw/gds-policy-campaign-20260906-summary.json, via
  the live-feedback bottleneck analysis): mean per-request decision time BPF
  1.004594 us, native 0.063484 us, FIFO 0.039563 us. Different executor,
  controlled inputs, no Python: it is a separate low-level reference near
  1 us mean, not an upper bound on the current driver/BPF execution share.
- End-to-end scale (completed write-budget campaign, 200 ms arms): BPF-200
  read p50 50.205 ms, p99 118.097 ms, 244--262 decisions/cell; paired
  read-p99 changes vs native at the same budget span -39.718% to +33.849%,
  with storage-throughput sequence drift documented in earlier campaigns
  (e.g. 4,952--5,565 MiB/s, then 848 MiB/s, later 1,247--2,059). Drift is an
  observed sequence; no cause was isolated.

## Where the per-call allocation lives (source facts, current tree)

- `PolicyRequest.pack()` builds a fresh 136-byte bytearray every call
  (lmcache_gds_policy_adapter.py:119--129), invoked inside
  `BpfDecider.decide` (:255--271) under `BpfDecider._lock`, itself inside the
  adapter's serialized `_decision_lock` (lmcache_gds_backend_adapter.py:511--518).
- `_uvm_ioctl` re-evaluates `(ctypes.c_char * 136).from_buffer(buffer)` per
  call (:228--233): ctypes array-type lookup plus a new view
  object per admission. Output parsing via `_PARAM_OUT.unpack_from` is retained
  unchanged under reuse.
- `NativeDecider` creates a `Decision` object but no ioctl buffer/view, so the
  switch does not change its source path: native is an unchanged reference, not an
  allocation comparison. Its per-decision cost also changes between contexts
  (472.5 ns isolated vs 1.827 us concurrent decider median), so context — not
  buffer reuse — differs between those measurements in both arms.
- Reuse replaces per-call pack/view construction with `pack_into` over a
  persistent buffer plus one-time view creation, with output zeroing between
  calls and unchanged lock discipline; those invariants are what the recorded
  replay validates (implemented in `b0eeb67c`; not re-derived here).

## What the measured saving can and cannot explain

- Measured: allocation reuse removes about 0.39 us of a 1.5755 us isolated
  command-82 decision call (paired median -24.2001%, six of six blocks;
  p99 -17.114%; loop-rate +27.182% median, +0.970% to +31.331%).
- Cross-context comparison, not a causal decomposition: the isolated legacy
  call is 1.5755 us while the concurrent instrumented decider median is
  17.962 us. Both arms' recorded costs differ in the concurrent context,
  including native, which has no ioctl buffer. These are different campaigns
  and measurement contexts; they do not isolate the cause of the difference
  or establish how much concurrent cost allocation contributes. Subtracting
  medians across campaigns is explicitly not evidence and this note names no
  cause for it.
- Scale: 0.385 us is about 0.0003% of the BPF-200 read p99 (118.097 ms).
  Assuming the isolated saving applied uniformly to 244--262 decisions gives
  about 0.10 ms per cell; this is illustrative arithmetic, not an upper bound
  or measured concurrent saving. Queueing effects need not be additive.
- Verdict: allocation reuse is a useful, reproducible optimization at the
  decision-call level. It does not explain the historical concurrent decisions
  or storage tails. These isolated admission savings do not establish
  end-to-end benefit; no end-to-end superiority claim is made. Whether a
  concurrent effect is detectable requires its own matched measurement.

## Residual bottleneck candidates (measured facts vs hypotheses)

Facts (existing records):
1. The matched budget change establishes an end-to-end parameter effect:
   200 ms cells record 244--262 decisions and 0--1 budget-exhausted writes of
   96 alongside ~51--54% read-p99 gains. The counts support the interpretation
   but do not isolate decision volume as the sole cause.
2. Rare multi-ms concurrent outliers exist (decider max 3.0--3.1 ms; read
   lock-wait max ~5.9 ms) against sub-microsecond median lock waits; cause
   unrecorded.
3. Storage-state sequence drift between cells is large (multi-fold write
   throughput collapses within single campaigns) and unexplained; its observed
   variation is much larger than the isolated-call saving.
4. Keep-GIL across the ioctl was tested and is not a reliable improvement;
   GIL handoff is not established as a tail driver.
5. New here: the isolated driver+BPF+Python call median is ~1.58 us (legacy)
   with ~0.39 us saved by the buffer/view/callable-reuse change, and
   decision streams are identical across arms — one call per admission,
   unchanged outputs.

Hypotheses (not proven by existing records):
1. What fills the concurrent-context decider median (~18 us) — prior analyses
   propose decision-lock hold-time amplification under the 64 reader threads,
   thread-scheduling/GIL interaction, and driver-side per-call state. The
   concurrent stage table (provider ~8.2 us, lock wait ~0.29 us, decider
   17.96 us) leaves the within-stage composition unmeasured; no concurrent
   sub-stage decomposition exists, and this note does not infer one.
2. Single asyncio-loop-thread congestion (deferred saves, re-decisions and
   completions share one thread) inflating write offer->submitted and
   shadowing reads (live-feedback analysis; no loop-stage data).
3. LMCache-internal pool/lock occupancy under 64 reader threads (unexamined).
4. SSD-internal state behind the sequence-linked throughput drift (no
   during-run trace exists).

Allocation reuse is now measured and small in this isolated run. The other
candidates remain unmeasured; this does not prove a causal ranking or rule out
queueing amplification.

## Plan review and recommendation

- The executed order is endorsed: real-driver paired decision timing (no
  storage I/O) answered the allocation question first, on the replayable
  decision surface, before any storage comparison could over-claim. RQ2's
  storage-side evidence remains the completed 200 ms matrix; this switch
  contributes mechanism accounting only.
- Smallest meaningful comparison: the landed 6-block x 3-arm replay is it,
  and it is sufficient at decision scale. Exact metrics recorded: per-request
  elapsed_ns; per-block medians and nearest-rank p99; paired median, p99 and
  loop-rate change percentages; decision fields per request; error counts.
  A read-only recompute from decisions.json reproduced analysis.json exactly
  (472.5/1575.5/1190 ns; paired median -24.2001%).
- No threshold and no gate: keeping the default-off switch is a descriptive
  judgment over the Stage table above, not a numeric pass/fail. No separate
  correctness campaign: default-off leaves ordinary behavior untouched, and
  the 6,000/6,000 decision-stream agreement is retained as scoped evidence,
  not a proof covering every input or concurrent execution.
- No further GPU runs are recommended, and the storage matrix is not repeated
  to chase noise: paired read-p99 variability of -39.718% to +33.849% at the
  same budget makes a small storage-scale reuse effect difficult to resolve.
  The open questions that matter more (concurrent-context decision
  decomposition, loop-thread and storage-state attribution) belong to their
  own reports' plans, not to this switch.

## Limitations

- The replay is single-threaded and isolated: it measures the per-call
  decision cost including packing/view work, not the concurrent storage-cell
  regime; its percentages must not be transplanted onto concurrent admission
  or end-to-end metrics.
- The four-request cycle exercises the demand, live-demand defer, live-demand
  submit and speculative branches; agreement covers these exercised classes
  only (24 MiB objects), not every flag combination or ABI edge.
- The instrumented stage table comes from event-driven 10 ms cells; 200 ms
  cells were not stage-instrumented. Budget changes can alter concurrency and
  timing; isolated replay numbers must not be transplanted to 200 ms cells.
- Six paired blocks on one machine and driver build; ranges are observations,
  not confidence intervals. All records, including the noisy block-5 loop
  rate, are retained in decisions.json.

## Artifacts

- Replay raw records:
  /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/gds-ioctl-reuse-575-20260907/decisions.json
  (18 cells x 1,000 records, implementation `b0eeb67c`, 100 warmups).
- Replay summary and paired analysis:
  /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/gds-ioctl-reuse-575-20260907/analysis.json
- Replay command:
  /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/gds-ioctl-reuse-575-20260907/command.txt
- Source-of-record reports:
  results-575-gds-admission-timing-20260907.md,
  results-575-gds-write-budget-20260907.md,
  results-575-gds-live-event-driven-20260907.md,
  results-575-gds-gil-handoff-20260907.md,
  gds-control/live-feedback-bottleneck-analysis-20260907.md
  (under /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/).
- Policy and adapter sources:
  gds-control/lmcache_gds_policy_adapter.py,
  gds-control/lmcache_gds_backend_adapter.py,
  gds-control/gds_policy.bpf.c
  (under /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/).
