# Independent reviews for proposal 2

## Round 1 — APPROVE WITH REQUIRED REPAIRS

The reviewer approved proposal 2's aggregate output-token throughput estimand
and observational atomic counter getter, with two required repairs:

1. Remove `num_offloaded_experts` from snapshots because its current property
   calls `IsTensorOffloaded()`, which mutates native state; bind the wrapper
   directly to the dispatcher getter and forbid all mutating/reset paths.
2. Freeze the exact valid-block stopping point, paired-ratio estimator,
   bootstrap sample estimator, seed, confidence level, and quantile method.

Both repairs were incorporated into proposal 2 revision 2.

## Round 2 — APPROVE WITH REQUIRED REPAIRS

The reviewer confirmed that the mutating residency queries were removed and
the stop/estimator/interval rules were substantially frozen, with three final
execution repairs:

1. Change the actual server command to the stats-wrapper module and freeze its
   exact equivalent of the official `__main__` initialization and uvicorn
   sequence.
2. Give the two-pass smoke its correct 1,024-token post-warm-up engagement
   delta instead of the timed block's 512-token gate.
3. Freeze the precise NumPy RNG API and persist the 10,000-by-5 bootstrap index
   matrix.

All three repairs were incorporated into proposal 2 revision 3.

## Round 3 — APPROVE

The reviewer confirmed that the real wrapper command and official-equivalent
startup sequence, smoke/timed token accounting, and exact persisted bootstrap
RNG/index matrix are fully specified. No scientific, fairness, executability,
or observational-instrumentation blocker remains. Approval permits offline
implementation and a later admitted GPU preflight; it is not a performance
result.
