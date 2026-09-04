# CPU loader failure-handling plan

Date: 2026-09-04

## Question

Does the verifier-enabled bpftime syscall server reject an invalid program at
the real `BPF_PROG_LOAD` boundary in `STRICT` mode without allocating a program
slot, and what are the explicit boundaries of `WARNING`, `NO_VERIFY`, and the
unset default?

This is a supporting failure-handling test for Reviewers B/F. It is distinct
from the existing direct `verify_gpu_program` rejection matrix and GPU attach
tests. It does not exercise the Linux kernel verifier, the GPU SIMT attach
path, a driver transition, or GPU execution.

## Frozen matrix

Each arm runs in a fresh process with a private shared-memory name. The invalid
program writes eight bytes below the 512-byte eBPF stack; its matched control
has the same instructions but writes inside the stack.

| Arm | Sequence | Required outcome |
| --- | --- | --- |
| strict-invalid | invalid, then valid | invalid returns `-1/EINVAL`; valid loads |
| strict-control | valid only | valid program ID equals strict-invalid's valid ID, showing that rejection did not consume a slot |
| warning | invalid, then valid | both load; log contains the verifier warning |
| no-verify | invalid, then valid | both load; log records skipped verification behavior through the configured mode |
| default | invalid, then valid | both load, documenting that the unset default is warning-only |

Run three fresh repetitions. The run is valid only if all 15 cells satisfy the
frozen outcomes, the STRICT diagnostic identifies verifier failure, no arm
uses the kernel, and every private shared-memory object is removed after its
process exits. A rejected invalid program without an accepted matched control
is not a pass.

## Command and implementation boundary

Run from the `gpu_ext` root:

```text
docs/experiment/revision-safety/loader-failure-cpu-575-01/run.sh \
  ../bpftime-r5/build-r5-v2
```

The selected build has `ENABLE_EBPF_VERIFIER=YES` and
`BPFTIME_ENABLE_CUDA_ATTACH=OFF`. The harness calls the libc `syscall` symbol,
which the official syscall-server preload library interposes. It neither uses
sudo nor opens a GPU or driver device.
