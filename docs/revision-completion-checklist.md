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
from the measured mechanism cost. These are **draft source changes pending
fresh compilation and review**, not closure of missing safety tests, Table 1,
disk measurements, agent-log release, or final figure-expansion work. See
[safety evidence and gaps](revision-safety-design.md) and
[remaining artifact execution](revision-remaining-artifacts.md).

## Current evidence and remaining work

| Commitment | Evidence available | Still required |
| --- | --- | --- |
| MoE-Infinity and XSched comparisons | Complete paired studies with baseline, native algorithm and actual host-BPF decisions; scoped results and costs are integrated in the paper draft. See the [experiment handoff](revision-experiment-status.md). | Fresh paper build/review and publication. Do not call component ports full original-system reproduction. |
| GPreempt policy comparison | Original 15-cell study and 45-cell contention study complete; both C and BPF improve foreground p99 with background costs. The draft includes the [load study and four-panel figure](../workloads/gpreempt/results-load-study-575-20260903.md). | Fresh build/placement review and publication. The new response metric replaces the unsupported historical scheduling-latency proxy; raw historical records remain. |
| LMCache local-disk backend | One real cold/warm disk check; token correctness and actual files observed. | Repeated storage-tier comparison remains absent. A smoke check does not complete the named local-disk commitment. Resolve and execute the retained full protocol after the active GPU studies. |
| Expert Buffering hot residency | Four-arm, five-block page-granular/profile-guided analogue exists. | Describe its non-expert-atomic, non-current-batch scope; audit/publicize the raw records before using the result. Do not silently relabel it the original algorithm. |
| Transition-validation pseudocode, SIMT algorithm, rejected policies, failure taxonomy, TCB | Exact source paths were audited and code-grounded material added to the draft. Strict device positive/negative integration is being implemented separately. | Live strict-device and remaining driver-transition tests; fresh paper build/review. The verification-disabled performance runtime is not retroactively verified by prose or CPU tests. |
| Three-way expressibility table | Active draft now separates user space, modified driver and current gpubpf capabilities, with actuator/trust boundaries. | Fresh build/readability review and publication. |
| Expand Fig. 13 and distinguish policy from mechanism | Matched-policy subsection and revised headline attribution are drafted. Historical Fig. 13 source now pins six audited CSVs and labels its single-round/engagement limitations. | Render and inspect the corrected full-width figure; the fresh engaged, independently timed memory/scheduling matrix remains unmeasured. Do not turn those old observations into causal scheduler evidence. |
| RTX 5090 Table 1 | Hardware and multiple application measurements available. | Complete the matching Table 1 microbenchmark protocol; do not substitute unrelated application throughput for missing table cells. |
| Agent prompts and benchmark harnesses | Existing public harnesses are indexed; a complete missing-session inventory and newly authored, separately labelled reproduction templates are now written. | Publish the inventory/templates; recover and redact actual original transcripts before claiming the original-prompt release complete. The author has been asked for the backup. |
| Discussion and organization | Draft now groups stale-state thrashing, CXL tiers, tenant scope, trampoline scaling, portability and software co-location versus static partitioning. | Fresh paper build/review and publication; proposed mitigations remain distinct from implemented/tested guarantees. |
| Typographic fixes | Active-source double punctuation and printed bibliography braces repaired. | Verify a fresh build. |

## Additional requested experiments

| Experiment | Status and next action |
| --- | --- |
| FineMoE dynamic prefetch | **Complete:** 20 cells, five blocks. [Report](../workloads/finemoe/results-performance.md) retains reduced unused transfers versus all-positive, a throughput loss versus demand-only, and unresolved BPF/C difference. Integrate the bounded result; no favorable-result rerun is required. |
| Hummingbird idle scheduling | Real calibration, 20-cell pattern qualification and ten-cell preflight passed. The same custom modules were restored after the reboot and both required-RPC canaries passed. The unchanged full 50-cell comparison is running; no final benefit claim yet. |
| POD-Attention device task choice | Original operators and adapters built; numerical protocol clarified before POD measurements. Preflight 03 was interrupted by the 09:04 UTC reboot. Wrapper fix passes 37 CPU tests; actual device-BPF numerical/engagement and performance comparisons remain open. |

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
