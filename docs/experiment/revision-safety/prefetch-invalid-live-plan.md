# Q2 live invalid-prefetch fallback plan

Status update, 2026-09-03: the replacement
[`prefetch-invalid-575-02`](prefetch-invalid-575-02/result-review.md) campaign
completed all three controls and exact old-runtime restoration. This document
preserves the pre-run plan and acceptance boundary. The earlier
[`prefetch-invalid-575-01`](prefetch-invalid-575-01/results.md) attempt remains
a closed admission failure with zero completed controls.

## Paper question and hypothesis

This experiment tests the transition-validation layer requested by Reviewers B
and F. The hypothesis is that a valid BPF policy returning unsupported initial
prefetch action `99`, while requesting the legal empty region `(0,0)`, is
routed by the production validator to the unmodified native traversal and
leaves a real managed-memory workload numerically correct.

This is decisive evidence only for invalid initial-prefetch-action containment.
It does not measure performance, completed DMA, physical PCIe traffic, or every
resource transition supported by gpubpf.

## Minimal diagnostic scope

The 575 UVM diagnostic may change only:

1. `uvm_bpf_struct_ops.h`: copied scalar context and phase definitions;
2. `uvm_bpf_struct_ops.c`: one `noinline` void/const-pointer observation hook;
3. `uvm_perf_prefetch.c`: SELECTED and FINISHED calls plus a native-loop count.

It must not change policy dispatch, the validator, effect selection, branch
conditions, actuation, or the returned region. The context must contain no raw
kernel pointers or address-derived tokens. The observer pairs at most one
SELECTED frame with one FINISHED frame for each full `pid_tgid`; nesting,
duplicate starts, or unmatched finishes reject the cell. The diagnostic build
is functional instrumentation and must never be used for a performance cell.

## Fixed real controls

Each mode runs a fresh 8 GiB / 64 KiB `uvm_fault_stream` target and checks all
131,072 output values. Order is fixed because this is a functional control,
not a timing comparison.

| Mode | Policy | Required selected state | Required completed state |
| --- | --- | --- | --- |
| Native | no struct_ops policy | action 0, no request, `NOOP_DEFAULT`, `NATIVE` | native complete, native iterations greater than zero |
| BYPASS | action 1 and legal `(0,0)` | `APPLY`, `BYPASS` | no native iteration; output `(0,0)` |
| Invalid99 | action 99 and legal `(0,0)` | `APPLY`, `NATIVE` | native complete, iterations greater than zero; output within copied bounds |

`APPLY` in the last two rows describes region validation. It must not be
reported as acceptance of action 99. Native iteration count is loop-body
execution, not total calls made by traversal macros.

## Admission and acceptance gates

Before releasing any target:

- build the exact new UVM module and retain revision, file inventory, sizes,
  compiler output, and tests without content hashes;
- confirm the hook has the expected module BTF prototype and function-entry
  instrumentation, then prove actual fentry admission;
- require Linux 6.15.11, NVIDIA 575.57.08, both existing lease inodes, 400 W,
  no compute client, zero UVM references, and no pre-existing struct_ops link;
- load only the diagnostic observer and, for BYPASS/invalid99, the fixed policy.

Every completed cell requires nonzero and exactly paired SELECTED/FINISHED
events, stable copied selection fields, empty outstanding-frame state, no map,
identity, nesting, recursion, link, or program-run-count error, the mode-specific
state above, and exact numerical target readback. A foreign GPU client, Xid,
missing phase, timeout, attachment failure, surviving owned process, incomplete
cleanup, or service/module restoration failure rejects the cell and stops the
sequence. Records go to a fresh `prefetch-invalid-575-02` directory; attempt01
is never overwritten.

## Outcome interpretation

- Positive: all three controls satisfy every gate. Claim only that an invalid
  initial action falls back to native traversal with correct workload output.
- Mixed: controls execute but a diagnostic or cleanup gate fails. Preserve the
  run and make no safety claim.
- Negative: invalid99 bypasses native traversal, returns an illegal region, or
  corrupts output. Report the counterexample and do not retry it away.
- Inconclusive: build, BTF, fentry, environment, or cleanup admission fails
  before all controls complete. Preserve the failure and keep Q2 open.

After this bounded result, the next mandatory experiment is RTX 5090 Table 1.
Its current event-collection and clock-domain gates must be repaired before any
formal timing run.
