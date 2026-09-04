# CPU loader failure-handling results

Date: 2026-09-04

Run status: **valid for the stated CPU userspace-loader scope**. All 15 cells
across three fresh repetitions met the frozen gates.

## Results

| Mode and sequence | Invalid program | Matched valid program | Interpretation |
| --- | --- | --- | --- |
| `STRICT`: invalid, then valid | `-1`, `errno=EINVAL` | ID 3 | rejected before program allocation |
| `STRICT`: valid only | not submitted | ID 3 | fresh-process allocation control |
| `WARNING`: invalid, then valid | ID 3 | ID 4 | warning is emitted, then both are admitted |
| `NO_VERIFY`: invalid, then valid | ID 3 | ID 4 | both are admitted without a verifier warning |
| unset default: invalid, then valid | ID 3 | ID 4 | default behavior is warning-only |

Every row repeated identically three times. In each strict pair, the legal
program after rejection received the same ID as the fresh-process legal-only
control. The strict logs contain the verifier failure and no load record for
`unsafe_stack`; the warning/default logs contain verifier warnings and load
records. This supports the narrow conclusion that explicit strict-mode
rejection did not allocate a program slot. It does not claim the absence of
all possible transient loader work.

## Execution boundary

- bpftime source revision: `ea9907d1df4b`, branch
  `revision/r5-safety-evidence`.
- Runtime build: `ENABLE_EBPF_VERIFIER=YES` and
  `BPFTIME_ENABLE_CUDA_ATTACH=OFF`.
- Host: Linux 6.15.11, x86-64; effective capability mask was zero.
- Entry path: libc `syscall` interposition, `BPF_PROG_LOAD`, and the official
  bpftime syscall-server verifier/admission implementation.
- Isolation: each cell used a private shared-memory name. The runner removed
  that exact object on both success and failure; the post-run search found no
  matching object.

The retained command was:

```text
docs/experiment/revision-safety/loader-failure-cpu-575-01/run.sh \
  ../bpftime-r5/build-r5-v2
```

Raw stdout, stderr, environment, and the machine-checked summary are under
[`raw`](raw/). The summary reports `PASS cells=15 repetitions=3`.
`raw-initial-01/` retains the first same-sized implementation run made before
the runner gained failure-path cleanup and source-revision recording; it is
not used for the reported outcome.

## Claim boundary

This is supporting evidence for one failure-handling boundary: a
verifier-enabled userspace loader can reject an unsafe eBPF program before
program allocation when explicitly configured `STRICT`. It also establishes
an important qualification: `WARNING` is the default and is not fail-closed.

This does **not** execute or validate the Linux kernel verifier, GPU SIMT
attachment, PTX generation, a GPU callback, a kernel-driver transition, or a
native actuator. It therefore cannot expand the paper's broader safety claim
or replace the existing strict-device and live-transition controls.

Independent OpenCode review returned `PASS` with no blocking findings; see
[`opencode-review.md`](opencode-review.md).
