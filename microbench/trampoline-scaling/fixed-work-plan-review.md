# Fixed-work plan review

OpenCode session `ses_f937458a3ffeoxgiDrDt3sJOdn` reviewed the proposed
fixed-work experiment with the local
`spark-gateway/qwen3.8-27b-nvfp4-200k` model. Snapshots and sharing were
disabled, all permissions were denied, and write, edit, shell, web-fetch, and
task tools were explicitly disabled.

Verdict: **READY**.

The reviewer found no blocking defect. It confirmed that the plan holds total
threads and dynamic warps fixed, treats reciprocal block/thread dimensions as
block organization rather than an isolated block-count variable, uses a
predeclared normalized difference-in-differences and equivalence bound, pairs
and randomizes the ten blocks, and retains complete correctness, engagement,
cleanup, and safety gates. It also confirmed that the claim boundary does not
assume warp-leader dispatch or universal block-count independence.

## Independent evidence audit follow-up

A later independent implementation audit rejected the first analyzer because
it trusted derived dictionaries in `result.json` and tested only the two
endpoints. Those blockers were repaired before any fixed-work GPU run:

- the runner now writes per-arm lifecycle and paired safety records in addition
  to the existing application, loader/map, agent, and telemetry logs;
- the analyzer reopens those raw files, recomputes every correctness,
  engagement, cleanup, telemetry, and safety gate, and uses raw application
  events as the only timing source;
- the endpoint DiD remains primary, while four predeclared Bonferroni-adjusted
  98.75% contrasts form an all-five organization guard with at least 95%
  family-wise coverage; and
- the seed-fixed arm order is position-balanced to the best possible 4/3/3
  allocation across ten blocks.

CPU-only failure injections cover missing and modified raw evidence, including
a +5.1 ms effect at the middle organization that leaves the endpoint primary
unchanged but must contradict the all-five guard. This section records the
repair, not a GPU result.

OpenCode session `ses_f93484576ffeKGC1F9CVot0fEg` then reviewed the repaired
runner, analyzer, tests, plan, and README with the same local Qwen model and
deny-all configuration. Its final verdict was **PASS**: it found no remaining
schema, resume, raw-evidence, path-reuse, statistical, ordering, or claim-scope
blocker. This is a read-only source review, not experimental evidence.
