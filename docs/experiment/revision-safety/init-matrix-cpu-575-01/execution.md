# Scheduler-init fixture CPU/build record

2026-09-03. Root authorized these CPU17-only commands after the POD full run
finished. No default Make target, BPF load/attach, module operation, CUDA/GPU
workload, or unrelated test was run. Working directory:
`/home/yunwei37/workspace/gpu/gpu_ext`.

| Command | Exit | Original log | Result |
| --- | --- | --- | --- |
| `taskset -c 17 make -C extension test_revision_init_fixtures` | 0 | [cpu.log](cpu.log) | **7 fixtures, 28 synthetic cases, 705 assertions** using the actual shared 575 recorder/validator |
| `taskset -c 17 make -C extension revision_init_fixtures` (initial) | 2 | [bpf-build.log](bpf-build.log) | `BPF_PROG` already supplies a `ctx` parameter; the new typed argument collided with it |
| Same explicit compile-only target after correction | 0 | [bpf-build-02.log](bpf-build-02.log) | All seven BPF objects compiled; renamed only that callback argument to `policy_ctx` |

The failed build log is retained, not overwritten. CPU-tested shared request
sequences did not change for the BPF macro-name correction. Source header root
is `../gpu_ext-kernel-575/kernel-open/common/inc`, source revision `849ea75d`;
no 610 headers were substituted. CPU output explicitly says
`scope=production_shared_recorder_validator native_execution=0`.

Read-only ELF relocation inspection found the intended timeslice/interleave
kfunc call-site counts respectively: no-request 0/0, legal 1/1,
invalid-interleave 0/1, duplicate 2/2, conflict 3/3, independent-interleave 1/1,
independent-timeslice 2/1. These are compiled request call sites, **not observed
native setter calls**. No-request contains no scheduler setter relocation.

Built inventory under `extension/.output/` (ordinary file sizes):

| File | Bytes |
| --- | ---: |
| `revision_init_cpu_test` | 27648 |
| `revision_init_no_request.bpf.o` | 6328 |
| `revision_init_legal.bpf.o` | 7008 |
| `revision_init_invalid_interleave.bpf.o` | 6696 |
| `revision_init_duplicate.bpf.o` | 7240 |
| `revision_init_conflict.bpf.o` | 7480 |
| `revision_init_independent_interleave.bpf.o` | 7024 |
| `revision_init_independent_timeslice.bpf.o` | 7160 |

The complete source matrix and precise missing diagnostic-hook proposal are in
[extension/revision_init.md](../../../../extension/revision_init.md).
**Native-init validation/native-setter/constructor observations remain open.**
Current RM functions are not directly probeable on this 575/6.15 path; no
known-failing probe attempt or shadow-based success claim was substituted.
