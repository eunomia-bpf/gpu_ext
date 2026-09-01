# Transition validator Phase B prefetch results

Date: 2026-08-31

Disposition: `PASS` for the offline UVM prefetch integration subset and
`PARTIAL` for Phase B overall. PMM membership/commit integration, its two BPF
load fixtures, kernel-native PMM execution, and live prefetch fallback remain
open.

Independent review: [phase-b-prefetch-review.md](phase-b-prefetch-review.md).

## Production integration

Both initial and iterator callbacks now receive a fresh 24-byte
`uvm_bpf_prefetch_decision` instead of a writable narrow output region. The
setter records original `u64` endpoints in callback-local state, preserving
presence and repeat/conflict history. Direct writes to callback and pointed
BTF context state are rejected by the struct_ops verifier boundary.

`compute_prefetch_region()` validates the signed raw callback action before
routing it. A legal `BYPASS` commits only a validated absolute half-open region;
an invalid action or invalid candidate runs native DEFAULT behavior. An
`ENTER_LOOP` callback receives an absolute checked current region and a fresh
decision on each iteration. Only legal iterator `BYPASS` output commits, and
the last legal selection wins.

The native and iterator paths both translate bitmap-tree-relative endpoints
through the shared widened checked translator. Addition overflow, subtraction
underflow, bounds, and `uvm_page_index_t` width are validated before narrowing.
`(0, 0)` remains the sole legal empty encoding.

The registered raw integer-VA-space `bpf_gpu_migrate_range()` interface was
removed. It had no native object ownership, generation, invalidation, or
teardown guarantee. Fourteen experimental sources that still reference that
interface are retained as unavailable source artifacts and excluded from the
default validated BPF application set.

## Offline evidence

- The 575 implementation is committed on `test-sched`; all five modules build
  against Linux 6.14 headers.
- The same change is ported to `port/nvidia-610.43.02`; all five modules build
  against Linux 7.1.12 headers.
- Kernel-native module BTF generation for the fresh 610 `nvidia-uvm.ko` exposes
  `struct uvm_bpf_prefetch_decision` at size 24 and the three-argument
  `bpf_gpu_set_prefetch_region` with two `u64` endpoints. It does not expose
  `bpf_gpu_migrate_range`.
- The extension's complete default build passes for 35 BPF applications. All
  30 prefetch callback implementations use the decision-context ABI; their
  setter calls pass that context. A representative built object exposes the
  same 24-byte layout with attempted/conflict at byte offsets 0/1 and endpoints
  at offsets 8/16.
- The 14 sources containing the removed migration call exactly match the
  Makefile's deferred set and do not enter the default application build.

No module was loaded, no BPF program was attached, and no live preflight was
consumed. Build and BTF inspection do not establish runtime admission or
fallback behavior.
