# Revision completion checklist

Updated 2026-09-03 UTC. The user requests completion of the remaining items,
not only a status audit. The [verbatim author response and shepherd comments](revision-shepherd-comment.md)
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
`b5e2f25`; its [build review](paper/asplos-27-rebuttal/revision-build-review.md)
records that checkpoint's remaining issues. Subsequent condensation produced
**16 total pages, with conclusion on page 14**, still exceeding the working
original-body-plus-two-page target. Hummingbird, POD and the completed narrow
strict-device controls are now added to source; a fresh final build and review
remain needed. This is not closure of missing driver safety tests, Table 1,
disk measurements, agent-log release, or final figure-expansion work. See
[safety evidence and gaps](revision-safety-design.md) and
[remaining artifact execution](revision-remaining-artifacts.md).

## Current evidence and remaining work

| Commitment | Evidence available | Still required |
| --- | --- | --- |
| MoE-Infinity and XSched comparisons | Complete paired studies with baseline, native algorithm and actual host-BPF decisions; scoped results and costs are integrated in the paper draft. See the [experiment handoff](revision-experiment-status.md). | Fresh paper build/review and publication. Do not call component ports full original-system reproduction. |
| GPreempt policy comparison | Original 15-cell study and 45-cell contention study complete; both C and BPF improve foreground p99 with background costs. The draft includes the [load study and four-panel figure](../workloads/gpreempt/results-load-study-575-20260903.md). | Fresh build/placement review and publication. The new response metric replaces the unsupported historical scheduling-latency proxy; raw historical records remain. |
| LMCache local-disk backend | One real 610 cold/warm disk check. On 575, the first complete-workload startup failed; a same-environment tiny Triton test fails with bundled 13.1 but passes all 4,096 outputs with 12.9. All three arms now pin the latter; [diagnosis](../workloads/lmcache-disk/compatibility-575.md) and 27 CPU tests are retained. | Repaired traced preflight 02 is running, followed by three correctness cells and ten paired blocks. No 575 storage/performance result is claimed yet. |
| Expert Buffering hot residency | Four-arm, five-block page-granular/profile-guided analogue; a [read-only raw audit](../workloads/expert-buffering-policy/raw-audit.md) reconciles the saved timing arithmetic and documents the EOS protocol change, weak correctness oracle and missing restoration evidence. | The promised policy implementation remains open: the original Section VI uses current-batch activity, expert-atomic residency and inactive-first/LIFO eviction, not a static hot-page profile. Implement and validate that policy before claiming reproduction; publish the retained old raw records with their limitations separately. |
| Transition-validation pseudocode, SIMT algorithm, rejected policies, failure taxonomy, TCB | Code-grounded exposition is in the draft. [Two actual strict counter pairs pass](../workloads/bpftime-device-smoke/results-strict-575-20260903.md), independently audited: 32,768 callbacks per positive, explicit rejection and zero counters per negative. Scheduler-init shared-validator fixtures pass 28 cases / 705 assertions. | Native scheduler-init and invalid-prefetch transition observations remain open; CPU/compile-only fixtures are not those tests. The verification-disabled performance runtime is not retroactively verified by the separate strict controls. |
| Three-way expressibility table | Active draft now separates user space, modified driver and current gpubpf capabilities, with actuator/trust boundaries. | Fresh build/readability review and publication. |
| Expand Fig. 13 and distinguish policy from mechanism | Matched-policy subsection and revised headline attribution are drafted. Historical Fig. 13 source now pins six audited CSVs and labels its single-round/engagement limitations. | Render and inspect the corrected full-width figure; the fresh engaged, independently timed memory/scheduling matrix remains unmeasured. Do not turn those old observations into causal scheduler evidence. |
| RTX 5090 Table 1 | Matching seven-arm harness is prepared. The strict smoke exposed a real per-thread map readback truncation, fixed in independent R5 commit `ea9907d`. | Apply the same metadata repair in a controlled Table 1 runtime and strengthen its readback oracle before executing. Do not substitute unrelated application throughput or an incomplete histogram. |
| Agent prompts and benchmark harnesses | Public harnesses, the missing-session inventory and separately labelled newly authored reproduction templates are committed and pushed (`1e4564c`). | Recover and redact actual original transcripts before claiming the original-prompt release complete. The author has been asked for the backup. |
| Discussion and organization | Draft now groups stale-state thrashing, CXL tiers, tenant scope, trampoline scaling, portability and software co-location versus static partitioning. | Fresh paper build/review and publication; proposed mitigations remain distinct from implemented/tested guarantees. |
| Typographic fixes | Active-source double punctuation and printed bibliography braces repaired. | Verify a fresh build. |

## Additional requested experiments

| Experiment | Status and next action |
| --- | --- |
| FineMoE dynamic prefetch | **Complete:** 20 cells, five blocks. [Report](../workloads/finemoe/results-performance.md) retains reduced unused transfers versus all-positive, a throughput loss versus demand-only, and unresolved BPF/C difference. Integrate the bounded result; no favorable-result rerun is required. |
| Hummingbird idle scheduling | **Complete and published (`2658e0e`):** full 50-cell raw comparison and [independent audit](../workloads/hummingbird/raw-audit.md). C/BPF both lose roughly 19–20% background goodput to fixed GPreempt; all results and incomplete-coverage control requests are retained. |
| POD-Attention device task choice | **Complete:** five blocks, ten shapes, 250 operator cells / 25 arm processes; [report](../workloads/pod-attention/results-575-20260903.md) and [independent audit](../workloads/pod-attention/raw-audit.md) reconcile every cell. Nine shapes show 0.51–1.18% BPF/CUDA latency cost; substantial whole-process cost and FP32 characterization exceedances remain explicit. Strict verifier enforcement/full-system reproduction are not claimed. Earlier failures remain retained. |

These additional experiments do not replace safety, local-disk, Table 1,
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
