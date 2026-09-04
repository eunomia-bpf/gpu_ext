# Independent result review

## Scope and method

This review inspected the experiment plan, BPF source and disassembly, loader,
runner, runtime map-helper paths, and every raw file in
`raw/strict-uniform-map-full-575-01`.  The raw values were parsed and
recomputed independently; `analyze_strict_uniform_map.py` was not imported or
invoked.  The existing `analysis.md` and `analysis.tsv` were read only after
the independent recomputation.

An additional OpenCode 1.18.27 review was attempted with
`spark-gateway/qwen3.8-flash-next-nvfp4-220k`, GPU visibility disabled,
snapshots and sharing disabled, and every tool permission denied.  The bounded
60-second invocation emitted only `step_start`, with no text or
`step_finish`.  It therefore supplied no advisory verdict and is not evidence
for this review.

## Completion and mechanism engagement

- The raw schedule contains exactly 72 fresh-process cells: 12 blocks and six
  arms per block.  Each arm occurs 12 times and exactly twice in every order
  position.  Every `run_id` equals its block, and the recorded order exactly
  matches the frozen seed-1797 cyclic/reverse construction.
- All 72 application logs identify the RTX 5090, record eight warmups and 64
  timed launches, and report all 32 CUDA outputs correct.  The directory and
  file inventories are complete, with no missing or extra cell.
- All 60 attached cells have return code zero and `STRICT` in their execution
  record.  Each has exactly one target-PID verifier timing, one target-PID
  `mode=STRICT` acceptance, and three target-PID map descriptors with types
  1503, 1513, and 1503.  The instruction counts are consistently 2 for no-op,
  14 for either update, and 22 for either lookup.  No failure, skip, verifier-
  unavailable, runtime error, or critical record occurs.
- Each attached cell has one `matched=1` target transformation, one patched
  module load, and one successful attach.  The selected program name is the
  planned arm in every cell.  The repeated program-selection log line comes
  from the runtime's two discovery stages; it does not produce a second
  transformation or attach.
- Loader logs bind the BPF relocations to `device_values` for the type-1503
  arms, `host_values` for the type-1513 arms, and the same device-resident
  `observed_values` sink for both lookup arms.  Device/host pairs have equal
  BPF instruction counts and equal transformed PTX sizes: 2808 bytes for both
  updates and 3480 bytes for both lookups.
- All 48 map-operation cells pass the independent final-effect oracle.  Update
  arms change their source map from its zero initialization to the planned
  nonzero value.  Lookup arms preserve the independently initialized source
  value and change the zero-initialized common sink to that value.  This proves
  that at least one attached callback completed its planned operation; as the
  plan states, the idempotent oracle does not prove invocation cardinality.
- The 60 target PIDs and 60 private shared-memory names are unique.  All target
  processes have exited and all recorded private segments are absent after the
  run.

## Independent recomputation

The independent computation used CUDA event milliseconds divided by 64,
within-block log ratios, paired medians, 10,000 seeded pair resamples, and the
predeclared 97.5% intervals for the two co-primary comparisons.

| Arm | median microseconds per launch |
|---|---:|
| native | 2.014250 |
| strict no-op | 3.928500 |
| device update | 3.860250 |
| host update | 3.882750 |
| device lookup | 3.841500 |
| host lookup | 4.148250 |

| Co-primary comparison | pairs | host/device ratio (97.5% interval) | host-device microseconds (97.5% interval) |
|---|---:|---:|---:|
| update | 12 | 1.000849 [0.989076, 1.014265] | 0.003250 [-0.042500, 0.055000] |
| lookup | 12 | 1.077766 [1.064425, 1.083279] | 0.298500 [0.251000, 0.319000] |

The update ratio is above one in 6 of 12 blocks and is unresolved.  The lookup
ratio is above one in all 12 blocks.  These values reproduce the retained
analysis output without using its implementation.

## Interpretation boundary

The run validly demonstrates that the current STRICT verifier, JIT, CUDA
attach path, and both GPU map placements compose end to end for these uniform
programs.  It finds a lookup-specific cost for the complete type-1513
host-mapped implementation relative to type 1503, but no detected update
difference.

The lookup result is not a pure host-versus-device DRAM or PCIe measurement.
The actual helper path includes cache behavior and the host-map coherence
fence, while both lookup arms also perform the same device-map sink update.
Those are legitimate parts of the compared map-type implementations but must
remain inside the claim.  The result also does not establish per-lane strict
admission, warp-leader aggregation, callback cardinality, verifier soundness,
application-level speedup, or a zero-cost mechanism.  It must remain distinct
from the earlier WARNING-mode, lane-varying placement experiment.

## Judgments

- Run status: **valid**.
- Tested joint hypothesis: **inconclusive**; lookup supports the expected
  direction, while update is unresolved.
- Research value: **supporting**.
- Paper impact: **additional RQ4 evidence and a mechanism/workload boundary**.
- Next paper decision: report the real strict-admitted uniform execution and
  the lookup-specific end-to-end map-type result; state that update matched
  within uncertainty and preserve all scope limitations above.

No blocking defect was found in strict-admission evidence, callback
engagement, map-effect correctness, matched timing, schedule balance, or the
paired inference.
