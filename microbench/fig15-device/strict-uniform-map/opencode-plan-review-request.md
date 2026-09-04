# Read-only plan review request

Review `plan.md` as a GPU systems experiment reviewer. Do not edit files, run
commands, use tools, or request broader experiments.

The key distinction is intentional: a prior WARNING-mode microbenchmark used
lane-varying keys and is rejected by the current STRICT verifier. This new
experiment uses one uniform constant key and value in every lane, keeps the
same scalar-per-thread runtime, and compares CUDA device-global map type 1503
against directly host-mapped type 1513 under real STRICT admission.

Identify only defects that would invalidate strict admission evidence,
correctness, the matched placement comparison, schedule balance, or paired
inference. Explicitly check that the claims do not imply per-lane support,
warp-leader aggregation, verifier soundness, invocation cardinality, pure
hardware latency, or application performance. End with exactly
`VERDICT: PASS` or `VERDICT: FAIL`.
