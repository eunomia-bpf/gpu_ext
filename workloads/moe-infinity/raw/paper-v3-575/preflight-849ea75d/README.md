# Three-mode enhanced correctness preflight

All three modes passed sequentially on 2026-09-03, 01:25:35–01:29:30 UTC.
This is correctness and real engagement evidence, **not paired performance**.
The source/driver declaration is the coordinator's `849ea75d` stage at
`/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11`; every admission
retains ordinary revisions and file metadata. GPU power was fixed at 400 W.

Each fresh server ran the same two nonstream 512-input/64-output requests,
then a complete SSE request repeating the second prompt. All three responses
matched the retained same-frontend goldens exactly. Every SSE contains 64 token
frames plus DONE, with independent engine and serving-metric increments of
64 tokens. Expert row sizes 1/256/257/353 and four accumulation arrival orders
had zero maximum absolute and relative numerical error in every mode.

| Mode | Actual selector execution | EAMC match calls | Rank calls | Scored evictions | Completed prefetches |
| --- | --- | ---: | ---: | ---: | ---: |
| `native-off` | Original native demand/cache path | 0 | 0 | 0 | 0 |
| `paper-native` | Native paper-port selectors | Native | Native | 12,731 | 3,900 |
| `paper-bpf` | Three real host uBPF JIT selectors | 4,608 | 6,912 | 13,042 | 4,251 |

The paper-native controller completed three requests, retained six phase EAMs,
and matched 4,608 predictions. BPF counters are all zero for both native modes.
The BPF run completed the same three requests and six EAMs; same-snapshot match,
rank and scored-eviction verification had zero mismatches. All three JIT program
summaries report zero errors. Asynchronous transfer scheduling can change
cache snapshots across runs; cross-run victim/transfer counts are not claimed
identical, and this does not replace the same-snapshot equivalence checks.

Completed-prefetch accounting after the final drain conserves outcomes:

| Mode | Transfer bytes | First-use hits | Unused-prefetch evictions | Unused residents |
| --- | ---: | ---: | ---: | ---: |
| `paper-native` | 51,661,209,600 | 2,312 | 1,582 | 6 |
| `paper-bpf` | 56,310,718,464 | 2,396 | 1,844 | 11 |

All server exit codes were zero. Each post-cleanup snapshot shows GPU 2 MiB,
0% utilization, no compute clients, UVM refcount zero, empty struct_ops, and
no new Xid or RM unhandled-interrupt warning. The final runner session exited
zero and released both GPU/struct_ops leases.

Metadata correction: the historical canary launcher wrote the top-level
`execution_domain` field as `host-ubpf-jit` for every mode. This field is wrong
for `native-off` and `paper-native`; their retained `mode`, launch environment,
dispatcher mode and zero BPF-call counters establish native execution. Raw
records are preserved unchanged. The launcher was corrected after all three
runs; only `paper-bpf` is host uBPF JIT. This label repair does not change
execution, workload, numerical outputs or policy decisions.

These runs precede a separately planned vectorized BPF argument-packing
optimization. They do not certify the optimized bridge until its own real
correctness run passes. They also do not reproduce the authors' original
hardware/model results or execute the activation policy in kernel UVM hooks.
