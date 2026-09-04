# Scheduler-init lifecycle attempt 04

Status: **2/16 matrix cells completed; original driver restored**.

The candidate passed admission, replacement, service restoration, and native
preflight. Both `native_unattached` and `bpf_no_request` then passed their full
functional and safety gates. The `bpf_legal` cell stopped only at final BPF
cleanup: seven programs present in its before inventory had disappeared by the
after inventory; no program, map, or link was added. These were ambient
objects, not owned leaks, so exact equality was stricter than the cleanup
property being tested.

The lifecycle removed the candidate, inserted the known-good module subset,
restored both services, and passed final validation with no recovery or
finalization errors. The immediate post-run audit independently observed
NVIDIA 575.57.08, 15 MiB memory use, 0% GPU utilization, a 400 W power limit,
UVM reference count zero, and no attached struct_ops.

The cleanup rule must continue to reject every newly surviving BPF object, but
may safely record disappearance of a pre-existing ambient object. The next
runner revision implements precisely that one-way subset rule; struct_ops
cleanup remains exact-empty.
