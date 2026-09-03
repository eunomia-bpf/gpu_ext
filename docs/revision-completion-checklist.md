# Revision completion checklist

Updated 2026-09-03 UTC. The user requests completion of the remaining items,
not only a status audit. The [verbatim author response and shepherd comments](revision-shepherd-comment.md)
define the scope. The dated plan in the paper repository is not evidence that
these commitments have been met. Experimental completion, paper integration,
and public artifact availability are separate checks.

## Current evidence and remaining work

| Commitment | Evidence available | Still required |
| --- | --- | --- |
| MoE-Infinity and XSched comparisons | Complete paired studies with baseline, native algorithm and actual host-BPF decisions; compatibility limits documented in the [experiment handoff](revision-experiment-status.md). | Integrate scoped results, mechanism costs and limitations into the active paper. Do not call component ports full original-system reproduction. |
| GPreempt policy comparison | Original 15-cell study and 45-cell contention study complete. Both C and BPF reduce foreground p99 versus baseline under the three fixed loads, with background costs retained. | Integrate the [load study and four-panel figure](../workloads/gpreempt/results-load-study-575-20260903.md); distinguish response time from the old service-time metric. |
| LMCache local-disk backend | One real cold/warm disk check; token correctness and actual files observed. | Repeated storage-tier comparison remains absent. A smoke check does not complete the named local-disk commitment. Resolve and execute the retained full protocol after the active GPU studies. |
| Expert Buffering hot residency | Four-arm, five-block page-granular/profile-guided analogue exists. | Describe its non-expert-atomic, non-current-batch scope; audit/publicize the raw records before using the result. Do not silently relabel it the original algorithm. |
| Transition-validation pseudocode, SIMT algorithm, rejected policies, failure taxonomy, TCB | Implementation and historical safety tests exist; current paper contains only part of the promised exposition. | Verify the exact deployed/source paths, reconcile the verifier description, add code-grounded material to design/implementation and test any newly exposed implementation gap. |
| Three-way expressibility table | Policy/source mappings exist across workload reports. | Add user-space / modified-driver / gpubpf capability table, with precise actuator and trust boundaries; a commented-out old table does not qualify. |
| Expand Fig. 13 and distinguish policy from mechanism | Matched-policy measurements now available; bounded component comparisons do not all exercise driver/device BPF. | Integrate per-policy execution substrate, benefits, overheads and negative results; revise headline attribution in abstract/introduction. |
| RTX 5090 Table 1 | Hardware and multiple application measurements available. | Complete the matching Table 1 microbenchmark protocol; do not substitute unrelated application throughput for missing table cells. |
| Agent prompts and benchmark harnesses | Public harness code exists; original exploration transcripts are not all present. | Inventory and publish the prompts/logs that actually exist, disclose missing originals, and provide reproducible current prompts without presenting reconstructions as original records. |
| Discussion and organization | Partial material on stale state, tenant scope and portability exists. | Add explicit stale-state thrashing, CXL tiers, per-tenant policy, trampoline scaling, portability and software co-location versus static-partitioning discussion, grouped by topic. |
| Typographic fixes | Double punctuation and printed bibliography braces confirmed in the current paper sources. | Fix active sources and verify a fresh build. |

## Additional requested experiments

| Experiment | Status and next action |
| --- | --- |
| FineMoE dynamic prefetch | **Complete:** 20 cells, five blocks. [Report](../workloads/finemoe/results-performance.md) retains reduced unused transfers versus all-positive, a throughput loss versus demand-only, and unresolved BPF/C difference. Integrate the bounded result; no favorable-result rerun is required. |
| Hummingbird idle scheduling | Real calibration, 20-cell pattern qualification and ten-cell preflight passed. Full 50-cell comparison remains open. The 09:04 UTC reboot returned the host to stock modules; restore and validate the existing custom interface before measurement. |
| POD-Attention device task choice | Original operators and adapters built; numerical protocol clarified before POD measurements. Preflight 03 was interrupted by the 09:04 UTC reboot. Wrapper fix passes 37 CPU tests; actual device-BPF numerical/engagement and performance comparisons remain open. |

These additional experiments do not replace safety, local-disk, Table 1,
artifact-release or paper-integration commitments. A negative but valid result
may complete a comparison; missing execution cannot.

## Execution and publication rules

GPU experiments are serialized under the existing shared leases; source review,
lightweight CPU tests and non-overlapping documentation work run in parallel.
Keep all failed and interrupted attempts, use new output directories, and do
not change frozen thresholds or omit adverse cells to obtain a favorable result.
Use actual correctness/engagement records, source revisions and explicit file
inventories, never file/content digests. Preserve unrelated worktree changes.
Commit and push scoped implementation, records and documentation after review;
do not describe local ignored artifacts as publicly released.
