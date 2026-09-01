# Transition validator Phase B scheduler results

Date: 2026-08-31

Disposition: `PARTIAL`. The scheduler production path, both driver lines, and
the public BPF ABI are integrated and build. Actual verifier-load fixtures and
runtime scheduler validation remain open, so this is not Phase B `PASS`.

## Implemented path

The scheduler callback now runs in `kchangrpapiConstruct_IMPL()` immediately
after the native `MEDIUM` interleave default. The earlier callback in
`kchangrpInit_IMPL()` was removed because the enclosing constructor overwrote
its interleave result.

The callback receives a 32-byte immutable input structure. Presence,
repeat/conflict state, and requested values live in a surrounding
driver-private decision structure. Setter kfuncs record into that private
state; direct BPF writes are rejected. After the callback, the caller rebuilds
an observed TSG/runlist/phase snapshot, validates each request independently,
and commits accepted values through the two native setters. Rejection of one
field does not suppress an accepted value for the other field. `LOW=0` is a
legal explicit request and no numeric value represents absence.

The public BPF header now matches the input-only ABI, uses native interleave
values `0/1/2`, and declares both setters' integer result type. Compile-time ABI
checks cover the public layout.

## Offline execution

The production-header host target passes 12 cases and 145 assertions on both
575 and 610. The two affected BPF applications, `gpu_sched_set_timeslices` and
`gpu_sched_trace`, compile after the ABI change.

All five 575 modules build against Linux 6.14 headers. That build lacks module
BTF, so it is compile evidence only and cannot establish struct_ops or kfunc
admission.

The same implementation was ported to `port/nvidia-610.43.02`. One native API
difference was resolved: 610 removed the old minimum-timeslice query and its
setter accepts the full `NvU64` range, so the port validates against a zero
lower bound. All five 610 modules build and link against Linux 7.1.12 headers.
Using the running kernel's base BTF and its native module-BTF generation path,
the fresh `nvidia.ko` contains a `.BTF` section. A BTF dump contains:

- `struct nv_gpu_task_init_ctx`, size 32 with five input members;
- `struct nv_gpu_sched_ops` and its generated struct_ops wrapper; and
- `bpf_nv_gpu_set_timeslice` and `bpf_nv_gpu_set_interleave`.

No module was loaded and the live-preflight count remains zero.

## Independent review and remaining blocker

Fresh review accepted the relocation, separate expected/observed snapshots,
input/private-state separation, presence and conflict semantics, independent
native commits, error routing, and extension ABI. It returned `BLOCK` for a
Phase B pass because the required actual verifier fixtures do not yet exist.

The next focused result must call `bpf_object__load()` without attaching and
attempt exactly the five scheduler fixtures frozen in the plan: two expected
write rejections and three expected admissions for immutable read, timeslice
setter, and explicit `LOW=0` setter. Module-BTF presence removes the earlier
575 artifact blocker but does not substitute for those verifier outcomes.

