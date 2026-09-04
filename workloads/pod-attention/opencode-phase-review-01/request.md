# POD phase-study CPU implementation review request

You are an independent read-only systems/code reviewer. Do not call tools,
edit files, run commands, use the network, launch a GPU job, or delegate work.
The complete relevant sources and protocol are attached directly. No GPU cell
has run for this follow-up; review implementation readiness only.

Return `READY` or `REQUIRED FIXES`, with blockers first and exact file/function
references. Audit these questions:

1. Does the independent runner create exactly three real preflight cells and
   five interleaved fresh-process blocks across `pod_inline`, `pod_cuda`, and
   `pod_bpf`, with one fixed shape and unchanged 10-warmup/3-or-100-sample
   behavior?
2. Do Linux monotonic timestamps correctly bound parent setup/loader/client/
   cleanup and child process-main/stdlib imports/runtime imports/pre-first
   diagnostic/post-first-sync/warmup/steady completion? Are all required
   markers fail-closed and cross-process ordered?
3. Does the launch bridge capture the actual successful first POD CUfunction
   launch without adding a clock read to recurring launches, while retaining
   its shared-memory and launch-count ABI/gates?
4. Are the existing hard FP16 output comparison, full FP32 characterization,
   BPF engine 2, exact CTA/atomic audit, bridge counts, shared-memory opt-in,
   575 driver, telemetry, private segment, orderly detach, runtime inventory,
   and post-safety gates still mandatory?
5. Is the six-target limitation represented honestly: loader readiness covers
   six registered exact targets, while this fixed shape records exactly one
   actual first launch and leaves the five inactive targets explicitly null?
   The code must not claim that all six alternatives launched.
6. Do the CPU tests substantively reject missing/out-of-order markers, wrong
   engine, wrong/absent first launch, non-target kernel, incomplete loader,
   dirty cleanup, changed runtime/preflight, wrong matrix, and bridge failures?

Also flag any measurement confound, accidental change to the frozen 250-cell
campaign, output-overwrite risk, unsafe cleanup behavior, or C++ concurrency/
ABI defect. Distinguish a run-time GPU validation requirement from a CPU-code
readiness blocker.
