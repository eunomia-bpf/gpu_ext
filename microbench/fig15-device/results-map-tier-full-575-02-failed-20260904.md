# Device-map full attempt 02: retained pre-READY allocation failure

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-full-575-02`

This attempt is **invalid and contributes no performance result**. The first
9 paired blocks (72 arm processes) completed. The first arm of block 10,
`device_update`, then created its private 256 MiB shared-memory segment but
exited before `FIG15_READY` while calling `bpf_object__open_file`. The loader
reported `-12` (`Cannot allocate memory`), but subsequent source audit found
that this diagnostic incorrectly substituted `-ENOMEM` whenever libbpf
returned null instead of preserving `errno`. The actual libbpf error was
therefore lost. The machine still reported roughly 115 GiB of available RAM,
63 GiB available in `/dev/shm`, an idle GPU, and no OOM or Xid record, but these
observations cannot recover the missing call-level error. This attempt proves
only a pre-READY object-open failure after shared-memory and CUDA setup; it
does not prove allocation exhaustion.

The attempt also exposed an incomplete form of the earlier cleanup repair.
`wait_for_ready` observed the private segment before the loader exited, but its
local identity was lost when it raised, so the caller's `finally` block still
refused reclamation. The repair now stores the observation in caller-owned
state before any readiness error can escape. A CPU-only regression test covers
this exact create-then-fail sequence. After the loader was confirmed dead and
the sole named segment was confirmed to be a regular file owned by the runner
with no process holding it, only that temporary IPC object was removed. It is
automatically recreated by a fresh run.

The loader diagnostic now saves `errno` immediately after the libbpf call and
uses `EIO` only when a null return carries no errno. A nonexistent-object CPU
check confirms that it now reports `ENOENT` rather than the earlier invented
`ENOMEM`; a source regression test prevents that fallback from returning.

No prefix, partial block, or timing value is used. A future full run must use a
new directory and start again from block 1 after the allocation failure is
understood or bounded by an independently validated batching protocol.
