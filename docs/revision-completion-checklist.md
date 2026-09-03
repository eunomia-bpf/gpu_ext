# Revision completion checklist

Updated 2026-09-03 UTC. The user requests completion of the remaining items,
not only a status audit. The [complete review archive](paper/asplos-27-rebuttal/README.md)
define the scope. The dated plan in the paper repository is not evidence that
these commitments have been met. Experimental completion, paper integration,
and public artifact availability are separate checks.

Current integration update: the active paper source now includes safety
pseudocode/algorithm/examples/TCB, the capability table, completed policy
comparisons and four-panel scheduling figure, discussion additions and
typography repairs. The abstract/introduction now distinguish policy benefits
from the measured mechanism cost. A fresh three-pass LaTeX/BibTeX build succeeds
without undefined references/citations, and the two scheduling/memory figures
were rendered and inspected. The checked source draft is pushed as paper commit
`ee1623e`; its [build review](paper/asplos-27-rebuttal/revision-build-review.md)
records the integrated checkpoint's remaining issues. Condensation produced
**16 total pages, with conclusion on page 14**, still exceeding the working
original-body-plus-two-page target. Hummingbird, POD and the completed narrow
strict-device controls are included in that successfully built and visually
checked source. Review archive commit `46d2f70` adds the missing review updates
and meta-review. This is not closure of missing driver safety tests, Table 1,
disk measurements, agent-log release, or final figure-expansion work. See
[safety evidence and gaps](revision-safety-design.md) and
[remaining artifact execution](revision-remaining-artifacts.md).

The user has explicitly **paused LMCache**, retaining its failed correctness
records and the original storage-tier commitment as deferred, not complete.
Two requested read-only OpenCode subagents have returned; their full reports
and main-thread corrections are [recorded here](experiment/revision-opencode-review-20260903.md).

## Current evidence and remaining work

| Commitment | Evidence available | Still required |
| --- | --- | --- |
| MoE-Infinity and XSched comparisons | Complete paired studies with baseline, native algorithm and actual host-BPF decisions; results/costs are integrated, built and pushed. See the [experiment handoff](revision-experiment-status.md). | Preserve boundaries: MoE uses a paper-algorithm reimplementation; XSched uses the original Level-1 executor, not Level-3. MoE baseline also has a different prefill cache shortcut, so its contrast is not pure policy effect. |
| GPreempt policy comparison | Original 15-cell study and 45-cell contention study complete; both C and BPF improve foreground p99 with background costs. The [load study and four-panel figure](../workloads/gpreempt/results-load-study-575-20260903.md) are integrated, built, inspected and pushed. | Host-mapped compatibility port, not original GDRCopy/hardware reproduction. The new response metric replaces the unsupported historical scheduling-latency proxy; raw historical records remain. |
| LMCache local-disk backend | Repaired 575 traced preflight proves storage engagement. All three correctness arms then ran: all eight cold outputs match, and CPU/disk warm outputs match each other, but **all eight differ from recompute**. Both raw directories' 275 non-cache records are committed (`013d4b4`); native cache payloads remain local. | **Paused by user direction.** No formal cells ran and no valid performance result is claimed. Shared KV layout/stride mismatch is a source-backed candidate, not a confirmed root cause. Preserve the promised storage discussion and explain the deferred measurement. |
| Expert Buffering hot residency | Historical 20-cell page-profile analogue remains separate. [Section VI selector/state](../workloads/expert-buffering-policy/section-vi/plan.md): nine tests / 2,131 native and JIT decisions, plus 1,432 state checks / 650 pairs. [Private four-point adapter](../workloads/expert-buffering-policy/section-vi/adapter-step-b.md) passes 61 fake-device and eight source/cleanup checks; its [actual 19-unit offloader build and exact-path import](../workloads/expert-buffering-policy/section-vi/adapter-build-01.md) now pass with CUDA uninitialized. | Three-arm real original-HF/logit correctness, physical residency/eviction/copy engagement, live-input native/JIT parity and five paired performance blocks remain. CPU fixtures and a successful build are not GPU reproduction evidence. |
| Transition-validation pseudocode, SIMT algorithm, rejected policies, failure taxonomy, TCB | Code-grounded exposition is in the draft. [Two actual strict counter pairs pass](../workloads/bpftime-device-smoke/results-strict-575-20260903.md), independently audited: 32,768 callbacks per positive, explicit rejection and zero counters per negative. Scheduler-init shared-validator fixtures pass 28 cases / 705 assertions. | Native scheduler-init and invalid-prefetch transition observations remain open; CPU/compile-only fixtures are not those tests. The verification-disabled performance runtime is not retroactively verified by the separate strict controls. |
| Three-way expressibility table | Draft separates user space, modified driver and current gpubpf capabilities, with actuator/trust boundaries; built, visually checked and pushed. | Preserve those execution-domain and enforcement qualifications in the final revision. |
| Expand Fig. 13 and distinguish policy from mechanism | Matched-policy subsection and revised attribution are integrated. Historical Fig. 13 uses six audited CSVs and labels its single-round/engagement limitations; the corrected full-width rendering has been inspected. | The fresh engaged, independently timed memory/scheduling matrix remains unmeasured. Do not turn old observations into causal scheduler evidence. |
| RTX 5090 Table 1 | Independent 575 runtime builds; 34 CPU tests and three actual-header logger checks pass. [Three retained preflights](../workloads/llama.cpp/observability_overhead/revision-rq4/preflight-575-results.md) remain failed. The [new three-cell GPU diagnostic](../workloads/llama.cpp/observability_overhead/revision-rq4/targeted-diagnostic-results.md) confirms equal 720,896-event / 22,528-nonzero-slot histograms after repairing our NVBit EXIT predicate, plus full 1,048,576-slot BPF readback. All owned cleanup/safety checks pass; BPF still fails exact stdout because of one PTX-pass log line. | Zero timing cells. Repair the remaining stdout source, validate complete event-channel collection and per-launch semantics, and repair both launch-latency paths. The old apparent BPF histogram deficit was not a valid overhead comparison. |
| Agent prompts and benchmark harnesses | Public harnesses, the missing-session inventory and separately labelled newly authored reproduction templates are committed and pushed (`1e4564c`). | Recover and redact actual original transcripts before claiming the original-prompt release complete. The author has been asked for the backup. |
| Discussion and organization | Draft groups stale-state thrashing, CXL tiers, tenant scope, trampoline scaling, portability and software co-location versus static partitioning; built and pushed. | Proposed mitigations remain distinct from implemented/tested guarantees; current page budget remains unmet. |
| Typographic fixes | Active-source double punctuation and printed bibliography braces repaired and checked in a fresh build. | Existing bibliography metadata warnings are not a completed citation audit. |

## Additional requested experiments

| Experiment | Status and next action |
| --- | --- |
| FineMoE dynamic prefetch | **Measurements and planned plot complete:** 20 cells, five blocks. [Report](../workloads/finemoe/results-performance.md) retains reduced unused transfers versus all-positive, a throughput loss versus demand-only, and unresolved BPF/C difference. The [two-panel plot](../workloads/finemoe/figures/dynamic-set-comparison.pdf) and exact 20-row CSV reconstruct every point from saved worker records and check against the independent analysis; three focused plot tests and eight existing scheduling-plot tests pass. The 7-inch vector PDF was rendered and inspected. No new GPU measurement occurred; this new plot is a workload artifact, not yet an additional paper float. |
| Hummingbird idle scheduling | **Original conservative-port study complete and published (`2658e0e`):** full 50-cell raw comparison and [independent audit](../workloads/hummingbird/raw-audit.md). C/BPF both lose roughly 19–20% background goodput to fixed GPreempt; all results and incomplete-coverage controls remain. The stronger per-piece completion fence is an unresolved confound, not a confirmed cause. The new [fixed-bound ablation preflight](../workloads/hummingbird/pipeline/preflight-results.md) passes 8/8 GPU cells and independent audit, with actual depth-2 engagement and complete event retirement. Its 40 formal cells remain pending; no formal recovery benefit is claimed. |
| POD-Attention device task choice | **Complete:** five blocks, ten shapes, 250 operator cells / 25 arm processes; [report](../workloads/pod-attention/results-575-20260903.md) and [independent audit](../workloads/pod-attention/raw-audit.md) reconcile every cell. Nine shapes show 0.51–1.18% BPF/CUDA latency cost; substantial whole-process cost and FP32 characterization exceedances remain explicit. Strict verifier enforcement/full-system reproduction are not claimed. Earlier failures remain retained. |

These additional experiments do not silently replace safety, deferred local-disk, Table 1,
artifact-release or paper-integration commitments. A negative but valid result
may complete a comparison; missing execution cannot.

## Execution and publication rules

The authorized paper-integration outline is: add source-backed safety
exposition beside design/implementation; add a matched-policy subsection,
execution-domain capability table and completed scheduling figure; correct
failure classification and policy/mechanism attribution; group deployment
limits in discussion; and fix the reported typography issues. Missing live
tests remain explicit TODOs, not completed claims. Preserve existing labels
and the user's paper revision. This outline is not an approval gate.

Opening correspondence: retain the existing context, workload-dependence,
agent motivation, interface requirements, related approaches, system and
challenge paragraphs in their current order. Update only the interface-contract
and result/contribution paragraphs, then derive the abstract's corresponding
claims from the revised body: headline gains compare policies; the 610 UVM
experiment measures a 3.219% same-policy cost; matched ports do not establish
equivalence; and verifier tests and enabled runtime enforcement are distinct.
No new result is introduced only in the abstract. Full build/placement review
waits for the exclusive GPU timing batch to finish.

GPU experiments are serialized under the existing shared leases; source review,
lightweight CPU tests and non-overlapping documentation work run in parallel.
Keep all failed and interrupted attempts, use new output directories, and do
not change frozen thresholds or omit adverse cells to obtain a favorable result.
Use actual correctness/engagement records, source revisions and explicit file
inventories, never file/content digests. Preserve unrelated worktree changes.
Commit and push scoped implementation, records and documentation after review;
do not describe local ignored artifacts as publicly released.
