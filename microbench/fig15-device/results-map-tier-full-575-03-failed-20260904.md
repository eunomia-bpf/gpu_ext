# Device-map full attempt 03: retained object-open failure

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-full-575-03`

This attempt is **invalid and contributes no performance result**. It completed
10 paired blocks and the first seven arms of block 11 (87 arm processes). The
last `device_update` loader created and initialized its private shared-memory
segment and initialized CUDA, but exited before `FIG15_READY` while opening the
BPF object. The repaired caller-owned cleanup state reclaimed the segment
without manual intervention; no application process or timed cell started.

The loader printed `-12` (`Cannot allocate memory`), but that value came from a
diagnostic bug that substituted `-ENOMEM` for every null libbpf return. The
underlying errno was lost, so this attempt establishes an object-open-stage
failure only. It does not establish host, shared-memory, GPU-memory, or
arm-specific exhaustion. The same BPF object passed a fresh eight-arm
preflight immediately before this campaign.

No prefix, partial block, or timing value is used. The diagnostic is repaired
before any further run; the next campaign must start at block 1 in a new
directory with the same frozen schedule and no failed-cell retry.
