# Predictive-prefetch preflight 01: retained harness-gate failure

Date: 2026-09-03

Status: **failed preflight; not a performance result**.

The coordinator acquired both shared experiment leases and launched the first
scheduled arm, `bpf-prefetch-on`, on the RTX 5090 with NVIDIA 575.57.08, Linux
6.15.11, and the declared protected driver stage
`/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11`.

The retained request completed normally: its independent request record marks
the golden-output check passed, contains 65 SSE frames (64 token frames plus
DONE), and reports `finish_reason=length`. The server exited 0, cleanup recorded
no error, and the measured interval advanced 64 engine tokens and 64 engine
steps with 12,606 expert-cache accesses (10,841 hits and 1,765 misses).

The cell then failed closed because the inherited MoE engagement validator
required 512 generated tokens. This preflight intentionally executes one
64-token request; the planned full factorial cell executes six requests, or
384 tokens. Consequently the outer result contains no accepted cell and the
remaining three arms were not launched.

This directory is retained as negative harness evidence. It must not be
completed, relabeled, pooled with later attempts, or cited as policy-performance
evidence. A corrected harness must use a new output directory and independently
pass all four arms before any five-block timing campaign begins.
