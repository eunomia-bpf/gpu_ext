# Transition validator Phase B scheduler results

Date: 2026-08-31

Disposition: `PARTIAL`. The scheduler production path, both driver lines, and
the public BPF ABI are integrated and build. The five verifier-load fixtures
are implemented and independently reviewed, but their kernel load outcomes and
runtime scheduler validation remain open, so this is not Phase B `PASS`.

Independent fixture review: [phase-b-scheduler-review.md](phase-b-scheduler-review.md).

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

Five focused scheduler verifier fixtures now build through the extension's
normal BPF toolchain. All five expose the exact 32-byte public callback context.
The two negative objects emit direct stores to public input offset 16 and the
driver-private decision wrapper offset 32. The three positive objects perform
immutable input reads, request a 100-us timeslice through its kfunc, and request
explicit `LOW=0` interleave through its kfunc.

The userspace fixture runner calls `bpf_object__load()` but never attaches. It
must admit all three positive controls before attempting either negative; only
then does `-EACCES` count as the expected direct-write denial. It preserves a
separate raw verifier log for every attempted object and requires exactly five
attempts, three admissions, two rejections, and five matching outcomes. Fresh
independent review passed the fixture ABI, emitted instructions, load ordering,
rejection classification, log preservation, no-attach boundary, and build
dependencies.

The runner's local precondition check exits before any load because this shell
is not root. More importantly, the running official NVIDIA module does not
export `/sys/kernel/btf/nvidia`; replacing that in-use display module was not
attempted. Therefore these are implemented and reviewed load fixtures, not
actual verifier admission/rejection results, and the live-preflight count
remains zero.

## Independent review and remaining blocker

Fresh review accepted the production integration and the load-fixture harness.
Phase B remains blocked on executing those fixtures against a running custom
module, then exercising the scheduler callback and native commit path. The
fresh module-BTF artifact proves that the 610 build carries the required types
and kfuncs; it does not substitute for the five verifier outcomes.
