# Same-address disk-backed UVM: source checkpoint

Driver source: `revision/gpu-storage-decision-575`, commit `c8e2831d`
(initial implementation checkpoint `db156f27`).
Status: source committed and pushed; **not loaded or performance-measured**.
Existing LMCache/cuFile results are unchanged. This prototype uses CPU staging,
not NVMe-to-GPU P2P, and does not yet establish automatic LMCache offload.

The local implementation adds a file-backed, sealed read-only managed range:
REGISTER attaches the file, OFFLOAD queues staging and file writes, and QUERY
reports page counts. CPU/GPU population paths restore disk-backed bytes into
CPU chunks at the same managed address. Shared backing state survives range
splits; workers retain range-backing, block and chunk references. Writable KV
ranges and direct-storage DMA are not supported by this checkpoint.

Ten source files changed: `uvm_disk_backing.c/.h`, `uvm.c`, `uvm_ioctl.h`,
`uvm_va_block.c/.h`, `uvm_va_range.c/.h`, `uvm_forward_decl.h`, and
`nvidia-uvm-sources.Kbuild`. No binary or disk-cache payload was committed.

## Build observation, not a live result

The implementation session ran several `make -j2` commands without waiting for
the requested CPU build window. Their observed span was 2026-09-08
05:29:40--05:34:17 PDT. The experiment owner was notified through the shared
coordination record so overlapping timing cells can be identified and retained
with the overlap disclosed.

Initial compilation errors included wrong VA-space locking and a wrong
`uvm_va_block_find` signature; the source was subsequently corrected.
The final reported `make-exit=0` came after `make | tail`, so it is not a
reliable make exit status. A newly linked `kernel-open/nvidia-uvm.ko`
(63,029,672 bytes, 05:32:53 PDT) contains `uvm_disk_backing_register`,
`uvm_disk_backing_offload_block`, `uvm_disk_backing_read_pages`, and
`uvm_disk_backing_query`. Root's `git diff --check` passed.

Next: capture the build's actual exit status in a coordinated window, then use
a small userspace same-address disk-offload/restore performance client. Do not
load this module during another task's GPU measurements. End-to-end behavior,
LMCache integration and performance remain unfinished; no success or speedup
is inferred from the source handoff.

## Completion-order follow-up — 2026-09-08 06:48 PDT

The local implementation has now separated recording the file-write outcome
from publishing offload completion. Pending pages remain pending through
unmapping and CPU-chunk removal. A successful write whose reclamation fails
clears the on-disk state and sets the existing error state; the unmap tracker
result is checked before removing a chunk. This addresses a concrete timing
problem in the checkpoint: observing pending=0 before reclamation could time a
resident access as if it were a disk restore.

This follow-up is still an unbuilt working-tree change in
`kernel-open/nvidia-uvm/uvm_disk_backing.c`; `git diff --check` passes. The
same-address client is present at `disk-uvm/disk_uvm_perf.cu`, with build/run
instructions still being completed. Neither change has produced new disk
performance numbers. The GPU scan owned by another task remains live, so no
module reload or overlapping GPU measurement was started here.

## Published source handoff — 2026-09-08 07:03 PDT

Driver correction `c8e2831d` is now committed and pushed. It changes only
`uvm_disk_backing.c` (91 insertions, 24 deletions); the staged diff check
passes, and the corrected source has not been rebuilt or loaded.
The client, Makefile and run instructions are committed and pushed in main
`d1bc712d`. Root repaired the client's directory traversal to use
`opendir`/`readdir`/`closedir` with `DIR *`; the build command selects CUDA
12.9 and `sm_120`. Client compilation and same-address disk measurements are
still pending. No cache payload or compiled binary was included.
