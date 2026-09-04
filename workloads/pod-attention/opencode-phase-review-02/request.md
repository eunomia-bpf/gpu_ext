# POD phase follow-up: strict read-only readiness review

Act as an independent systems/code reviewer.  The attached files are the
entire review surface.  Do not use tools, edit files, run commands, browse,
delegate, or infer that any GPU work has run.  This request is for CPU/source
readiness only.

Return exactly one leading verdict, `READY` or `REQUIRED FIXES`, followed by a
concise rationale.  Treat a defect as blocking only if it can invalidate or
misrepresent a future real run.  Cite exact files/functions for every blocker.

Audit the following:

1. The formal plan is exactly 15 fresh-process cells: three arms
   (`pod_inline`, `pod_cuda`, `pod_bpf`) in seeded randomized order within each
   of five paired blocks.  The excluded preflight is exactly three cells.
2. One previously valid Llama-3-8B / batch-32 shape is frozen, with 10 warmups
   and 100 formal samples (three in preflight).
3. Required monotonic markers cover coordinator start/loader readiness/client/
   cleanup and child process/imports/first diagnostic launch+sync/warmup/steady
   completion, and missing, malformed, or cross-process-invalid markers fail.
4. The inherited numerical comparison and FP32 characterization, CTA/atomic
   exactly-once audit, BPF engine 2, launch-bridge counts, first successful
   per-CUfunction marker, shared-memory opt-in, exact six-target loader,
   driver/telemetry/runtime/post-safety checks, private cleanup, and exclusive
   leases remain mandatory.
5. Six registered targets are not misrepresented as six launches: the one
   fixed shape must record its one actual adapter launch and the five inactive
   alternatives remain explicitly unobserved.
6. `--dry-run` is genuinely offline: it prints the exact plan but performs no
   artifact inspection, output write, lease acquisition, process launch, or
   GPU operation.  The tests meaningfully enforce that boundary and important
   fail-closed mutations.
7. Documentation never upgrades the result into generic attachment cost,
   strict-verifier admission, a constant trampoline cost, or full POD system
   performance.  A future real preflight requirement is not itself a
   CPU-readiness blocker.

Also flag output-overwrite risk, unsafe private-segment cleanup, accidental
changes to prior raw/results, or a phase label that claims more isolation than
its markers establish.
