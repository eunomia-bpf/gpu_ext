# Independent design review request

Review the sibling `plan.md` as a read-only systems reviewer.  The target is a
stock NVIDIA 575.57.08 driver on one x86 socket.  The public RM source implements
`cpuClkId=TSC` by repeating a `c G c G c G c` zipper, returning the midpoint of
the closest TSC pair and the enclosed PTIMER value; the public command loops for
up to 16 requested samples.  The host is pinned to one CPU and advertises
`constant_tsc`, `nonstop_tsc`, `tsc_known_freq`, and `rdtscp`.

Decide whether neighboring returned TSC midpoints conservatively bound the
unknown selected endpoints for each interior sample, whether this can support
an affine TSC-to-PTIMER launch-boundary-to-device-entry measurement, and what
minimum correctness gates are missing.  Reject any use of kernel duration,
midpoint-only precision, unbounded clock assumptions, changed histogram bins,
or modified NVIDIA modules.  Do not use tools or modify files.  End with
`VERDICT: PASS` only if no scientific blocker remains; otherwise end with
`VERDICT: BLOCKED` and the exact defect.
