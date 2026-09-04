# Independent review of the stock TSC/PTIMER admission result

Date: 2026-09-04

Scope: commits `5be05048`, `acbd5e44`, `663841a8`, and `51a1e397`

Evidence: `../raw/tsc-ptimer-575-02/records.jsonl` and the NVIDIA
575.57.08 source

Method: direct raw-record recomputation and source/ABI inspection; the project
analyzer was not invoked, and no GPU operation was run

## Verdict

The result is a **valid negative admission control** for the tested public
TSC/PTIMER path.  It supports the narrow statement that this source-derived
method did not pass the frozen 1,500 ns certification gate, so it cannot admit
the dependent `launchlate` experiment.  It does **not** show that the physical
clock-pairing error is at least 7,073 ns: 7,073 ns is a conservative admission
bound, not an empirical lower bound on hardware error.

The exact `/3` bound was tightened after the retained pilot and then evaluated
in a fresh run.  The public path and 1,500 ns threshold were already fixed, but
the final bound formula was therefore not untouched from the initial pilot.
This history is disclosed in `result.md`; the repair makes admission easier,
not harder, and the fresh corrected run still fails by a wide margin.

## Direct raw-record recomputation

The JSONL contains 211 records: 210 valid samples and one summary.  There are
no invalid-sample or invalid-batch records.  The samples cover every
`(batch,index)` pair in the intended 15 batches by 14 interior indices
(`index=1..14`) exactly once.  The stderr file is empty.

Using the recorded TSC frequency of 3,686,400,000 Hz, I independently
recomputed each width as

```text
ceil((tsc_high - tsc_low) * 1e9 / tsc_hz) + 2 * 32 ns.
```

All 210 recomputed widths equal their stored values.  Sorting those values and
using the runner's integer median convention gives:

- minimum: 5,641 ns;
- median: 7,073 ns;
- maximum: 10,069 ns.

The first and last accepted samples give 3,697,024,129 elapsed TSC cycles and
1,002,902,816 elapsed PTIMER nanoseconds.  Exact integer evaluation of

```text
abs(dt_cycles * 1e9 - dg_ns * tsc_hz) * 1e9
---------------------------------------------------
                 dt_cycles * 1e9
```

gives 20,776 ppb.  TSC midpoints and PTIMER values have zero non-increasing
adjacent pairs.  Thus the independently reconstructed gates are:

- accepted samples at least 200: pass (210);
- rejected samples, regressions, and detected migrations equal zero: pass;
- cleanup and final RM status reported successful: pass;
- median conservative width at most 1,500 ns: **fail** (7,073 ns);
- endpoint rate error at most 10,000 ppb: **fail** (20,776 ppb);
- overall gate: **fail**.

The raw file is sufficient to recompute sample accounting, widths,
monotonicity, endpoint rate, and the failed gate.  Cleanup and migration are
necessarily checked against the probe's emitted status rather than an external
observer.

## Conservative-bound proof

The 575.57.08 implementation of
`tmrGetGpuAndCpuTimestampPair_GM107` performs three ordered GPU low-register
reads between four TSC reads (`c-G-c-G-c-G-c`).  On x86, each read is separated
by `portAtomicTimerBarrier()`, implemented as `lfence`.  The implementation
selects the shortest of the three adjacent TSC gaps and returns its integer
midpoint with the GPU read enclosed by that gap.

For one retained sample, let `p` and `n` be the returned midpoints from the
immediately preceding and following helper calls, and let `c0..c3` be the four
TSC values in the current helper call.  The calls execute serially, so

```text
p <= c0 <= c1 <= c2 <= c3 <= n.
```

If `d` is the selected shortest gap, then

```text
d <= (c3 - c0) / 3 <= (n - p) / 3.
```

The probe uses `ceil((n-p)/3)`, then splits that width below and above the
returned floor midpoint with floor/ceiling halves.  This contains both true TSC
endpoints of the selected gap.  Clamping to `p` and `n` cannot remove a true
endpoint because the entire current zipper lies between them.  Adding one
32 ns PTIMER-resolution allowance at each side only widens the interval.
Consequently, the formula is source-justified and conservative, though it can
be substantially looser than the unknown selected gap.

The arithmetic does not expose a result-changing overflow or rounding defect
for this run.  Products use 128-bit intermediates, cycle-to-nanosecond
conversion rounds upward, and the captured TSC magnitudes are far below the
64-bit midpoint-overflow boundary.

## ABI inspection

The hand-written control ABI agrees with the 575.57.08 public definitions used
by command `0x20800406`:

- `cpuClkId` and `sampleCount` are two 8-bit fields;
- the sample array begins at offset 8, has 16 entries, and each entry is two
  aligned 64-bit values;
- the resulting correlation parameter object is 264 bytes;
- the allocation, control, free, device, subdevice, and xfer structures have
  the field order, alignment, and sizes enforced by the probe's static
  assertions;
- TSC source ID `0x02` selects the CPU TSC with the processor field left at its
  backward-compatible CPU value.

The locally added endpoint-v1 command is separate from this stock command and
does not modify the three-read TSC helper.  Successful RM status plus 210
monotonic nonzero returned pairs provide an additional semantic ABI check.

## Non-fatal deviations and claim boundary

The plan says that an endpoint affine rate must have residual intervals that
cover every retained point, but the C runner does not enforce that condition.
I evaluated it independently: the exact affine line through the first and last
sample midpoints lies within all 210 recorded `[tsc_low, tsc_high]` intervals;
the largest absolute center residual is about 69.455 ns at the nominal TSC
rate.  This is a real runner/plan deviation, but the omitted condition passes
and the earlier precision gate independently fails, so it is non-fatal to this
negative admission decision.

The raw artifact records `tsc_hz` but does not capture the planned Linux
clocksource and CPU-feature checks.  A read-only review on the same host found
the active clocksource `tsc` and the four requested flags (`constant_tsc`,
`nonstop_tsc`, `tsc_known_freq`, and `rdtscp`).  Their absence from the retained
raw metadata is a reproducibility-packaging limitation, not a route by which
the 7,073 ns certified bound could satisfy the 1,500 ns gate.

Accordingly, the defensible wording is:

> The corrected stock public TSC/PTIMER path failed to certify a conservative
> median uncertainty of at most 1.5 us, so `launchlate` remained unadmitted.

Do not rewrite this as either an observed 7.073 us hardware error or proof that
no future stock-stack method could meet the threshold.
