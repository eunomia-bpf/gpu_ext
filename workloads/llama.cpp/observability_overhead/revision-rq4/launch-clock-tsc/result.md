# Stock-driver TSC/PTIMER launch-clock result

Date: 2026-09-04  
Hardware: NVIDIA GeForce RTX 5090, driver 575.57.08  
Host: Linux 6.15.11, Intel Core Ultra 9 285K, CPU 23  
Result: **valid negative control; launchlate remains invalid and omitted**

## Outcome

The public, no-module-change TSC/PTIMER recovery path does not meet the frozen
precision gate.  The corrected run accepted all 210 interior samples with zero
RM rejection, clock regression, or CPU migration, and it released its private
RM objects.  Its conservative median TSC/PTIMER bracket is **7,073 ns** and its
maximum is **10,069 ns**, both well above the unchanged **1,500 ns** admission
limit.  The endpoint rate point estimate differs by **20,776 ppb**, also above
the unchanged 10,000 ppb gate, although the failed precision bound alone is
sufficient to stop the path.

The tested hypothesis is contradicted.  The planned RM--CUDA--RM
`%globaltimer` identity control and all 220-launch correctness/performance cells
were intentionally not run because precision is an earlier mandatory gate.
There is no launch-latency or throughput result here, and no kernel duration is
used as a substitute.

## Evidence replay

The corrected command was:

```bash
flock /tmp/gpubpf-revision-gpu0.lock -c \
  'flock /tmp/gpubpf-revision-struct-ops.lock -c \
  "launch-clock-tsc/tsc_ptimer_sanity --batches 15 --cpu 23 \
  --pause-ms 1000 > raw/tsc-ptimer-575-02/records.jsonl \
  2> raw/tsc-ptimer-575-02/stderr.log"'
```

The raw log has 210 sample records and one summary.  The independent analyzer
recomputes every interval width, count, monotonicity condition, median, maximum,
cleanup condition, and the final failed gate from those records:

```bash
python3 launch-clock-tsc/analyze_tsc_ptimer.py \
  raw/tsc-ptimer-575-02/records.jsonl
```

A separate [raw-only review](independent-review.md) confirms the result without
calling that analyzer. It also records one non-fatal plan deviation: the runner
did not gate affine residual coverage, while the independent calculation finds
all 210 points covered. Precision and endpoint-rate gates fail independently.
The 7.073 us value is a conservative certification bound, not a measured lower
bound on physical clock error.

The first retained probe used the whole neighboring-midpoint span and reported
a 21,089 ns median.  That was safe but unnecessarily loose.  Before interpreting
the result, the implementation applied the NVIDIA source's stronger fact that
the selected gap is the shortest of three gaps in a `c-G-c-G-c-G-c` zipper;
the corrected `/3` bound produced the 7,073 ns value above in a fresh directory.
Neither retained run passes.

## Feasibility boundary

- **Public RM, PLATFORM_API:** already measured at 4,730 ns median with direct
  transport, above the same precision gate.
- **Public RM, TSC (this result):** uses exactly the clock available to a safe
  private host helper and avoids a RAW-to-TSC conversion, but its exposed
  midpoint still cannot bound the selected GPU read tightly enough.
- **Versioned RM endpoints-v1:** measured at 759 ns when its matching module was
  loaded, but the currently loaded stock module rejects command `0x20800408`.
  Using it would require the prohibited module lifecycle operation.
- **CUPTI:** CUDA 12.9 documents that GPU activity timestamps are linearly
  converted into a normalized CPU timestamp domain.  CUPTI can therefore time
  API callback to CUPTI kernel-start activity, but that replaces gpubpf's
  `%globaltimer` device-entry observation with CUPTI's endpoint; it does not
  calibrate the native gpubpf pair under a documented raw-clock contract.
- **CUDA events:** both event endpoints live in stream/device execution time and
  cannot timestamp the CPU launch boundary.

Thus there is no currently safe, stock-stack route to a valid native
launch-boundary-to-gpubpf-device-entry Table 1 row.  The correct paper action is
to keep `launchlate` omitted and retain the already valid two-tool subset.

## Result review

- run status: **valid control**
- tested hypothesis: **contradicted**
- research value: **supporting boundary evidence**
- paper impact: **mechanism/stack boundary**
- next paper decision: keep `launchlate` out of Table 1; do not replace its
  endpoint with kernel duration or CUPTI activity and do not change modules in
  this safe follow-up
