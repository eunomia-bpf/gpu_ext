# Device-map full attempt 01: retained pre-READY loader failure

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-full-575-01`

This attempt is **invalid and contributes no performance result**. It completed
the first 12 paired blocks and the first three arms of block 13. On block 13,
order 4 (`host_update`), the loader created its private shared-memory segment
but exited before `FIG15_READY` while opening the BPF object. The runner then
exposed a cleanup defect: it recorded the segment identity only after READY,
so it correctly refused to unlink an identity it had not captured.

No partial block, prefix, or timing value is used. After confirming that the
unique 256 MiB segment was a regular file owned by the runner's user, that its
loader process no longer existed, and that no process held it, the segment was
removed; no other shared-memory object was touched.

The repair now captures the private segment's device/inode/owner tuple as soon
as it appears, before waiting for READY, and still refuses cleanup if that
identity later changes. The loader also reports the libbpf error code and text
on an early object-open failure. Offline positive and symlink-rejection tests
cover the new cleanup boundary. The entire 16-block campaign must restart in a
new directory.

