# LMCache local-NVMe revision-2 review

Status: **offline repair passed final independent review; no GPU execution
authorized**.

The reviewer must inspect `plan-v2.md`, `run_lmcache_disk.py`,
`test_runner.py`, `prompts.json`, `schedule.json`, and
`artifacts-current.json`. The first review ran read-only: ten CPU tests passed,
the checked-in prompts exactly regenerated from the pinned dataset/tokenizer,
and no GPU was launched.

## Round 1 findings

The reviewer blocked execution for the following reasons:

1. Pending review prose accidentally contained the runner's approval sentinel,
   while the parser accepted a substring. Preflight and smoke did not check it.
2. `--prompts` could select inputs other than the file validated and recorded
   by admission.
3. The runner did not itself enforce exact prompt regeneration from the pinned
   ShareGPT rows and tokenizer.
4. The first ten schedule positions were badly unbalanced and the runner did
   not regenerate the schedule from its seed.
5. The throughput-tradeoff rule treated an interval crossing the -5% boundary
   as proof of regression.
6. Analysis omitted declared output-token rate and disk-versus-CPU rate
   comparisons.
7. Resume/analysis trusted ordinary file metadata without re-parsing the raw
   server log, trace, request usage, engagement, and schedule semantics.
8. Store/retrieve parsing checked only numerators, not the corresponding total
   and required-token denominators.
9. The plan needed an explicit hypothesis, competing interpretations,
   paper-value decision, metric sources, and fixed executable commands.
10. TTFT used the first non-empty text fragment rather than the first generated
    token event; eviction/staging-allocation failures were underspecified.
11. The custom manifest, pass-marker, completion-marker, approval-parser, and
    promotion/resume layer violates the experiment skill's ban on
    project-authored experiment-control interfaces. The workflow must use the
    source-native server path, ordinary raw outputs, and a recomputable analysis
    adapter instead.
12. Revision 1 already consumed three real preflight launches. An unchanged
    scientific question and a new namespace do not reset that cap; another
    launch needs explicit higher-level authorization or a genuinely different
    experiment.

The 0.98 budget remains only a plausible, safe feasibility estimate, not a
capacity proof.

FINAL DECISION: BLOCK

## Round 2 findings and response

The second read-only review confirmed removal of the custom promotion/control
layer, exact prompt regeneration, the seeded Latin-cycle schedule, complete
store/retrieve denominators, generated-token TTFT, semantic raw-output
revalidation, both-baseline rate analysis, the corrected confidence-interval
rule, preservation of the exhausted attempt cap, and absence of active content
fingerprint logic.

It requested final offline repairs for launch-order evidence and anti-selection,
additional allocation/eviction wording, the official metric source and
supporting paper-value rationale, stable environment semantics, per-prefix
store-state semantics, stale README/module naming, distinct comparison command
paths, and warm-phase sequencing/exclusions. These items were repaired without
launching a model server.

ROUND-2 DECISION: OFFLINE REPAIR BLOCK; GPU LAUNCH BLOCK

## Round 3 — final allowed follow-up

The final reviewer inspected commit `4e0a28a`, ran all 15 CPU-only structural
tests, and found every round-2 blocker repaired. The review specifically
confirmed exact prompt regeneration; schedule and actual timestamp/order
checks; contiguous attempts, ordinary failure records, stopping and completed-
position balance; allocation/eviction patterns; per-prefix persistence state;
denominators; exact output comparison; warm-phase semantics; both-baseline
rates; stable source/runtime/storage/model semantic checks; corrected
README/module/source scans; and distinct comparison command paths. The official
vLLM benchmark documentation supports the TTFT measurement point and throughput
terminology used in the plan.

No GPU, CUDA workload, or model server was launched during review.

FINAL OFFLINE REPAIR DECISION: PASS

FINAL GPU LAUNCH DECISION: BLOCK — revision 1 exhausted the three-attempt cap,
and no higher-level authorization exists.
