# Device-map full attempt 05: retained lazy-startup interleaving

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-full-575-05`

This attempt is **invalid and contributes no performance result**. It completed
the first paired block and the first `rpc_lookup` arm of block 2 (9 arm
processes). The next `host_lookup` loader exited before `FIG15_READY` with
libbpf `EINVAL`; cleanup reclaimed its private segment automatically.

The newly enabled libbpf trace localized the interruption. Libbpf had parsed
the ELF sections, programs, externs, and the first two map definitions. Between
extern collection and map parsing, the interposer lazily initialized bpftime
shared memory and CUDA. Parsing then stopped immediately after the
`host_values` definition, before BPF load, map creation, attach, application
launch, or timing. Successful loaders parse all four identical definitions
after the same initialization has already completed.

The next loader revision forces that synchronous one-time initialization with
an ordinary `/dev/null` open/close before calling libbpf, and emits a required
`FIG15_SERVER_PRIMED` record. This setup-only repair is uniform across attached
arms and precedes the timed CUDA interval. No prior prefix is reused; the next
campaign starts from block 1 with the same schedule and no failed-cell retry.
