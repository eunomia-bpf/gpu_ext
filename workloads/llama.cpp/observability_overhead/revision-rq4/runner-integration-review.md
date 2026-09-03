# Table 1 lossless-runner integration review

Status: **READY for a fresh GPU correctness preflight.** This is CPU/build
readiness only; it is not a Table 1 measurement.

The runner is paired with the bpftime collector in commit `052513a` on branch
`revision/table1-575`. It now requires the deterministic 47-byte application
output and the collector's exact correctness record: 720,896 events, 220
selected launches, 22,528 unique coordinates, multiplicities
1,024-at-220/1,024-at-44/20,480-at-22, no other multiplicity or segment
mismatch, exact-oracle enabled and passed, equal committed/runtime/host/nonzero
counts, and zero drop, dirty, pending, or second-drain counters.

Timed cells explicitly disable the correctness-only oracle. They retain the
internal lossless and complete-coordinate gates without assuming uniform
multiplicity or a fixed event total, then require the gpubpf and NVBit exit
collectors to agree on event and selected-launch counts within each block.

Root independently ran the combined offline suites: 39 tests passed. The help
entry point and scoped diff check also passed. A separate read-only reviewer
and OpenCode 1.18.27 session `ses_f9650f15affeCvlQMfGq3gsDkE` both returned
`READY`. OpenCode ran with `snapshot:false`, sharing disabled, and write,
shell, network, task, and external-directory access denied; a first all-tools-
denied response attempted a read and produced no verdict, so the same session
was continued with read-only access before its verdict was accepted.

The remaining gate is a real RTX 5090/575.57.08 preflight. CPU tests and PTX
inspection do not prove lossless behavior under actual multi-stream GPU
concurrency, and no performance claim is made here.
