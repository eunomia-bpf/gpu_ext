# LMCache local-NVMe revision-2 review

Status: **blocked in independent review; no GPU execution authorized**.

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
