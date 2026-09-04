# Device-map full attempt 04: retained libbpf EINVAL

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-full-575-04`

This attempt is **invalid and contributes no performance result**. It completed
the first three paired blocks and the first three arms of block 4 (27 arm
processes). The next `device_update` loader initialized private shared memory
and CUDA but exited before `FIG15_READY` while opening the unchanged BPF
object. With the corrected errno handling, libbpf reported `EINVAL` (`-22`).
The caller-owned cleanup state reclaimed the private segment automatically.

The failure precedes BPF load, attach, application launch, and timing, so it is
not an observed failure of the device-update policy. The existing log did not
contain libbpf's internal open trace; the loader now installs a diagnostic
libbpf callback before opening the object. This setup-only logging precedes the
timed CUDA interval and is applied uniformly. A future failure can therefore
identify the failing ELF/BTF/open stage instead of preserving only `EINVAL`.

No prefix, partial block, or timing value is used. Any next full campaign starts
again at block 1 in a fresh directory with the unchanged 128-arm schedule and
no failed-cell retry.
