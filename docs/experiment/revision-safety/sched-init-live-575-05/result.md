# Scheduler-init lifecycle attempt 05

Status: **16/16 matrix cells passed; original driver restored**.

The candidate passed admission, reversible module replacement, service
restoration, and a native numerical preflight covering 32,768 values across
eight launches with zero mismatches. Two randomized blocks then completed all
eight rows: native unattached, no request, legal, duplicate, invalid
interleaving, conflicting, independent interleaving, and independent
timeslicing. Every row passed its event-join, ownership, cleanup, monitoring,
and GPU-safety gates. No cell observed a foreign compute process, owned-group
survivor, newly surviving BPF object, abnormal kernel event, or throttling.

The lifecycle removed the candidate, inserted the known-good 575.57.08 module
subset, restored GDM and persistence services, and passed final validation
with no recovery or finalization errors. The independent post-run audit
observed Linux 6.15.11, 15 MiB GPU memory use, 0% GPU utilization, a 400 W
power limit, UVM reference count zero, no attached struct_ops, and only the
restore core's GSP completion hook.

This closes the scheduler-init safety matrix. The result validates the
candidate transition interface and its fail-closed cleanup behavior; it is a
safety/correctness result, not a performance comparison.
