# Scheduler-init lifecycle attempt 02

Status: **candidate exercised and original driver restored; matrix incomplete**.

The candidate passed artifact/BTF admission, replaced the known-good 575 stack,
restored both services, and passed the native CUDA numerical preflight. The
first native-unattached matrix cell also reached its functional validation, but
the cleanup gate rejected the cell because the all-BPF inventory included
short-lived maps created by each `bpftool map show` process. Their IDs changed
between the before and after queries, so the matrix correctly stopped and did
not count the cell.

The lifecycle record proves that recovery removed the candidate, inserted the
known-good module subset, restored GDM and NVIDIA persistence, and passed its
final validation with no recovery or finalization errors. The immediate
post-run audit independently observed NVIDIA 575.57.08, 15 MiB memory use, 0%
GPU utilization, a 400 W power limit, UVM reference count zero, no attached
struct_ops, and the restore core's GSP completion hook without the candidate's
constructor diagnostic hook.

This attempt contributes zero completed matrix cells. The next runner revision
must distinguish stable BPF objects from the introspection process's own
ephemeral iterator maps while continuing to reject persistent leaked programs,
maps, links, and struct_ops objects.
