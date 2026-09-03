# Proposed 575 prefetch diagnostic seam

Status: **source proposal only, not applied, built, loaded or admitted**.
The closed [attempt01](../../docs/experiment/revision-safety/prefetch-invalid-575-01/results.md)
still has zero completed controls and no live fallback evidence. This patch
does not repair or change that frozen fixture.

## Why existing named functions are insufficient

The actual 575 source is sibling `gpu_ext-kernel-575`. Source and the retained
`kernel-open/nvidia-uvm/uvm_perf_prefetch.o` show:

- `uvm_perf_prefetch_bitmap_tree_iter_get_count` returns `NvU16` and
  `iter_init` returns void, but both are inlined in `compute_prefetch_mask`:
  its disassembly has no calls to either. Their retained BTF names cannot
  establish execution of those inlined operations.
- `get_range` calls remain in the native branch, but actual attempt01 rejects
  its STRUCT return before attachment. Do not change its prototype, use
  offsets, or bypass that gate.
- Calls to scalar-returning `uvm_page_mask_region_weight` remain, but its
  FUNC is absent from attempt01's saved `loaded-uvm-btf.txt`. The saved BTF
  does contain `get_count`, `iter_init`, and `compute_prefetch_mask`.
  A generic bitmap-weight observation would not expose the selected effect.
- The existing wrapper returns before validation. `compute_prefetch_mask`
  fills a merged output mask, not the per-decision validation/effect/traversal
  boundary. Neither alone proves the requested chain.

This is source/object inspection, not an attachment attempt. No current
non-STRUCT, BTF-based attachment point found here supplies both requested
boundaries.

## Exact patch scope and interface

[driver-diagnostic.patch](driver-diagnostic.patch) changes only these proposed
driver paths:

1. `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h`: a scalar-only diagnostic
   context containing no kernel addresses, two phase values and one
   void-returning declaration.
2. `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c`: one `noinline`, const-pointer
   hook with the existing barrier-only pattern used by
   `nv_gpu_sched_gsp_control_complete`.
3. `kernel-open/nvidia-uvm/uvm_perf_prefetch.c`: copy actual local state after
   initial effect selection; count native-loop body executions; copy actual
   returned region after the existing branch completes.

Both C files already belong to `nvidia-uvm-sources.Kbuild`; no new Kbuild
source, kfunc, struct_ops member, parameter, lock, allocation or global counter
is added. The validator and actuator code, branch conditions and return value
are unchanged. There are two diagnostic calls per completed region decision,
including BYPASS, plus one local increment per native iteration. This adds
instrumentation overhead: use it for the functional safety controls, not as a
transparent replacement for existing performance measurements.

The single prospective fentry target is
`void uvm_bpf_prefetch_diagnostic(const struct uvm_bpf_prefetch_diagnostic_ctx *ctx)`.
It takes no by-value structure. The observer copies the driver-filled context;
it never supplies it or modifies driver fields. `const` documents the
interface, not a substitute for the tracing verifier's write restrictions.

| Phase | Actual evidence | Fields not yet meaningful |
| --- | --- | --- |
| SELECTED | Original-width raw action/request, original region-validation result, selected initial effect, legal bounds, page and invocation identity | Returned region; native completion |
| FINISHED | Same immutable selection evidence plus the actual narrowed return region and native-loop completion/iteration count | No final hint or DMA result exists here |

`initial_region_result` is copied before the native branch can reuse
`region_result` for address translation. It is **region** validation, not a
separately exposed action-validator return. For invalid99 with legal (0,0),
expect region APPLY and actual effect NATIVE; do not label APPLY as acceptance
of action99. No validator is rerun solely to synthesize a diagnostic result.

The diagnostic publishes no pointer or address-derived identity. Pair only one
outstanding frame per full observer `pid_tgid`; reject nested or duplicate
starts and unmatched completion, and assign any persistent sequence in the
observer. This does not independently identify the target TGID/VA space: the
same exclusive-window limitation as the original fixture remains.

FINISHED occurs after the whole selected branch; it is also emitted for
BYPASS to make zero traversal an explicit completed control. Only the native
branch increments `native_iterations` (after its real get_range call) and
sets `native_completed=1` after its loop. The count is loop-body executions,
**not** get_range calls: the traversal macro makes additional range calls while
computing counters. The output is the actual `compute_prefetch_region`
return, not the compute-mask output, final filtered prefetch hint or DMA mask.

## Review and admission still required

No patch-application check, compilation or test was run in the concurrent EB
formal window. Root can first use `patch --dry-run --batch -p1 -i` with this
patch's absolute path from an explicitly selected 575 source copy, then review
the three-file diff. The active driver source and frozen observer/runner were
not changed. A separate coordinated build/load window must inspect the new
symbol's BTF and entry instrumentation and prove actual fentry admission;
a void/pointer signature does not itself prove admission.

Only then prepare a separately versioned observer/loader using this hook.
Require one SELECTED/FINISHED pair per actual wrapper return, nonzero matched
counts, stable copied selection fields, legal actual output bounds, no missed
frame/map/recursion events, complete target readback and owned cleanup:

| Control | Raw action / initial region result / effect | Native evidence | Actual region |
| --- | --- | --- | --- |
| Native | 0 / NOOP_DEFAULT / NATIVE | completed=1; iterations>0 | empty or within recorded bounds |
| Legal BYPASS | 1 / APPLY / BYPASS | completed=0; iterations=0 | (0,0) |
| Invalid99 | 99 / APPLY / NATIVE | completed=1; iterations>0 | empty or within recorded bounds |

Retain actual callback counts independently. If a compute-mask observation is
also retained, admit it separately and call it **pre-filter compute-mask
output**. Do not reconstruct a requested region and label it an observed mask.
Any missing phase, admission failure, instrumentation loss, foreign client,
numerical failure or incomplete cleanup rejects the run. Use fresh output
directories; preserve attempt01 and all existing 610/575 evidence unchanged.
