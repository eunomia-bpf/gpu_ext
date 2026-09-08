# Unified six-configuration figure: missing-cell supplement

Requested by the user through the paper task on 2026-09-08. This document
records experiment ownership only; this session does not edit or commit paper.

The unified figure should show one each of No policy, Scheduler,
Prefetch(0,20), Prefetch(20,80), Evict(20,80), and Combined for each original
workload, with the Single1x reference. Do not duplicate Scheduler, add a
dagger, or mix old/new timing groups to manufacture this presentation.

## Existing usable cells

The completed results_combined_575_20260908.Mwmymz campaign contains all
five blocks for each of HotSpot, GEMM, and sparse K-Means:

| Existing arm | Unified label |
| --- | --- |
| baseline | No policy |
| sched_only | Scheduler |
| memory_only, prefetch_eviction_pid20/80 | Evict(20,80) |
| combined | Combined |

Those60 cells and their120 tenant exits are complete. No rerun is needed.
Use HotSpot/GEMM size factor0.6 and sparseK-Means0.9, one measured iteration,
with the original benchmark's built-in warmups unchanged.

## Missing cells, not yet run

Prefetch(0,20), Prefetch(20,80), and Single1x each need five observations per
workload:45 new cells total. These must retain the same stopped-before-CUDA
launch convention and completion origin as the completed campaign. For the
two-tenant prefetch arms, attach the original prefetch_pid_tree tool before
common release. For Single1x, release the sole stopped tenant and measure
its whole-process completion from that release, retaining its raw output.

Old run_single_experiment starts its timer before Popen and deletes its
temporary raw log. Old concurrent policy runs also have different launch/
attachment timing. Their published completion numbers therefore cannot be
directly substituted for the missing same-origin cells. Preserve them as
historical results rather than deleting or relabeling them.

## Implementation and execution queue

The current OpenCode runner owner will finish its bounded Table1 patch,
then extend the existing run_policy_comparison.py combined path with
opt-in selection of the missing arms and the single-tenant variant. Preserve
the current four-arm default and existing results. Do not create a new
benchmark, invoke global cleanup, or repeat completed cells.

Root will run the missing cells serially under GPU and struct-ops locks,
using the same saved candidate core and current ff68a1d4 UVM as the earlier
campaign, and restore the prior modules/services/storage-policy loaders.
The saved module location and prior lifecycle are documented in the
completed campaign README. No new driver build is required for this
supplement, and the unfinished disk-backed UVM code must not be substituted.

All three permitted OpenCode slots are presently occupied. This runner
extension precedes a natural slot handoff to the existing XSched host/device
session; it does not authorize a fourth session or termination for silence.
No supplement implementation, new45-cell result, or completed unified plot
is claimed by this plan. Paper integration stays with the paper owner.
