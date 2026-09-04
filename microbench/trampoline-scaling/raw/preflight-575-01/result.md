# Device-trampoline preflight attempt 01

Status: **stopped before any arm was accepted**.

The native application completed its one-cell correctness run with 1,048,576
checked values and zero mismatches. Its measured launch was not admitted into
the result, because the post-run safety gate still observed UVM reference
count 4 at the fixed 60-second deadline. The GPU was otherwise idle and no
compute process, kernel anomaly, struct_ops object, power-cap sample, or
correctness failure was observed.

The runner's outer finalizer subsequently recorded UVM reference count zero,
and an independent audit immediately after exit also observed zero. This is a
fail-closed settle-timeout result, not trampoline performance evidence. A
follow-up may lengthen the fixed UVM-settle deadline before rerunning all three
arms; this attempt must not be pooled with a successful preflight.
