# Scheduler-init lifecycle attempt 03

Status: **1/16 matrix cells completed; original driver restored**.

The candidate again passed admission, module replacement, service restoration,
and the native numerical preflight. The first `native_unattached` cell passed
all functional, monitoring, cleanup, stable-BPF-inventory, and GPU-safety
checks. The following `bpf_no_request` cell stopped before target release: the
owned struct_ops map had the loader-reported ID, but this kernel's `bpftool link
show` did not enumerate a struct_ops link, so an exact nonempty-link assertion
failed closed.

The lifecycle then removed the candidate, inserted the known-good module
subset, restored both services, and passed final validation with no recovery
or finalization errors. The immediate post-run audit independently observed
NVIDIA 575.57.08, 15 MiB memory use, 0% GPU utilization, a 400 W power limit,
UVM reference count zero, no attached struct_ops, and only the restore core's
GSP completion hook.

The next runner revision should keep exact map-ID/PID ownership and require an
exact link ID when the kernel enumerates one, while recording rather than
rejecting the kernel's established “link not enumerated” behavior. This is the
same fail-closed ownership rule already used by the completed MoE runner.
