# Plan review

OpenCode session `ses_f95e8ff8effer5oWm1VTCvVIuq` reviewed the plan read-only with snapshots, tools, editing, shell access, and network access disabled.

## Round 1

Verdict: required fixes.

- The original text did not explain that the 32-callback marker was a separate untimed 32-thread kernel.
- The original fixed-total/fixed-block matrix could not isolate the claimed block-only effect because block size changed with block count.
- The reviewer also requested a clearer per-thread map/no-race argument.

Repairs: the marker and its limited role are now explicit. The matrix now fixes 256 threads per block and active work while varying launched block footprint, then fixes launch geometry while varying an active whole-warp prefix. The map oracle now specifies one serially updated slot per logical thread and complete run-length comparison.

## Round 2

Verdict: required fixes.

- The reviewer read "20 attached runs" as scale-cell runs rather than application processes and found the count inconsistent with 270 measurements.
- One stale sentence still said 11 rather than nine cells per process.
- The reviewer asked whether the largest grid implied simultaneous residency.

Repairs: the completion rule now distinguishes application processes from cell measurements; the stale count is fixed; the plan states that scheduling waves are expected and that device launch limits are checked.

## Round 3

Verdict: ready.

The reviewer confirmed that both original blockers were closed, the process/measurement arithmetic was consistent, and scheduling-wave/device-limit language addressed the capacity concern. It found only three stale axis labels, which were mechanically changed to `block-footprint` and `active-warp` after the ready verdict. No GPU execution or result claim is part of this review.
