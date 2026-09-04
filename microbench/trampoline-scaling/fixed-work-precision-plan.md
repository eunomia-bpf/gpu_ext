# Experiment Plan: RQ4 fixed-work precision follow-up

## Research Question

- RQ exactly as written in the paper: "What is the overhead of gpubpf's core
  mechanisms, including its observability capabilities and device-side
  extensions?"
- Specific uncertainty: Does the return-only device hook change execution time
  by more than +/-1% across five CUDA block organizations when every kernel
  still launches exactly 131,072 threads (4,096 whole warps), performs 16 hook
  repetitions per thread, and produces the same output?
- Role: supporting. This is a prospective precision follow-up to the valid but
  inconclusive `fixed-work-full-575-01` campaign, not a replacement chosen
  after inspecting a favorable subset.

## Why another run is admitted

The prior run used one CUDA-event interval over eight launches. Native batches
were only 29.920--40.960 us, while the +/-1% equivalence margin corresponds to
roughly 0.33--0.39 us. Across ten randomized blocks, the endpoint paired
difference-in-differences (DiD) had a standard deviation of 6.90%, and the four
all-five contrasts had standard deviations of 6.39%, 9.62%, 5.04%, and 6.90%.
Thus the 95% endpoint interval was [-5.47%, +3.13%] and the multiplicity-adjusted
guard intervals were wider still. This diagnoses insufficient precision; it
does not show that the organization effect is zero.

The highest-value repair is to lengthen each device-timeline interval rather
than weaken the bound or change the kernel. Each timed observation will
aggregate 512 identical launches, 64x the prior count. Fixed per-interval event
noise then occupies a smaller fraction of elapsed time, while independent
per-launch jitter should average down. Forty-eight fresh randomized blocks
retain between-process variability and give exact arm-position balance.

## Exploratory power planning (not evidence)

The following calculation selects a fixed budget; it is not a result and will
not be recomputed to stop the run early. The worst observed contrast standard
deviation was 9.62%. If independent launch-level noise decreases no faster than
the square root of the 64x aggregation factor, its projected standard deviation
is 1.203%. For a median under an approximately symmetric distribution, the
normal-reference standard-error multiplier is 1.253. A 98.75% two-sided guard
interval uses approximately 2.498 standard errors. Requiring 95% marginal
power at a true zero effect adds 1.960 standard errors, yielding

`ceil((1.253 * 1.203 * (2.498 + 1.960) / 1.0)^2) = 46` blocks.

We round up to 48 so all six arm permutations occur eight times. By the union
bound, 95% marginal power for each of four guards implies at least 80% power
for all four when their true effects are zero. This is deliberately labeled
exploratory: the old run cannot prove the square-root scaling assumption, the
median approximation may be imperfect, and a nonzero effect near a boundary
will have lower power. The confirmatory run remains valid and may remain
inconclusive if aggregation does not reduce variance enough.

## Frozen comparison and workload

- Main comparison: identical native CUDA binary versus the same binary with
  the current gpubpf/bpftime return-only device handler attached.
- Engagement control: the same exact-counter handler as the prior experiment;
  it is not a competing baseline and is excluded from the equivalence test.
- Organizations: 128x1,024, 256x512, 1,024x128, 2,048x64, and 4,096x32.
- Per-kernel invariants: 131,072 launched and active threads, 4,096 dynamic
  warps, 16 arithmetic/hook repetitions per thread, identical seed rule,
  output allocation, hook site, and integer oracle.
- Aggregation: 16 untimed warmups followed by one CUDA-event interval around
  exactly 512 timed launches for each organization. Aggregation changes only
  the number of identical kernel launches in the measured interval; it does
  not change work per kernel or the tested mechanism.
- Repetitions: one three-arm, middle-organization preflight at the full
  warmup/launch/repetition settings, then exactly 48 full paired blocks. Each
  full block has three fresh arm processes and five measurements per process,
  for 144 processes and 720 timed cells.
- Randomization: seed 1797 fixes all assignments before execution. Every six
  consecutive blocks contain all six arm permutations in randomized order, so
  every arm occupies every position exactly 16 times. Cell order is independently
  randomized once per block and shared by its three arms.

## Metrics and multiplicity

- Primary: within each block, compute the endpoint DiD
  `[(noop_4096x32 - native_4096x32) -
  (noop_128x1024 - native_128x1024)]`, normalized by the mean of the two native
  endpoint batch times. Report the median and a seed-1797 paired-bootstrap 95%
  percentile interval over all 48 blocks.
- Predeclared all-five guard: compare each of the other four organizations with
  128x1,024 using the same normalized paired DiD. Report four independently
  seeded 98.75% percentile intervals. Bonferroni therefore gives at least 95%
  family-wise coverage for these four guards.
- The +/-1% margin, median estimand, endpoint primary, four guard contrasts,
  bootstrap sample count (10,000), confidence levels, and seeds are unchanged
  from the original fixed-work analysis.
- The primary and all four guards must lie wholly inside [-1%, +1%] to support
  the bounded claim. An interval wholly outside the margin contradicts it. Any
  boundary-crossing interval makes the conclusion inconclusive.
- No subgroup, alternate estimator, block deletion, precision-triggered retry,
  sample-size reassessment, or optional stopping is permitted. Full analysis
  occurs only after all 48 blocks finish. A failed infrastructure cell is
  retained; a systematic implementation defect may be repaired only by
  rerunning the entire affected campaign under the same frozen design.

## Correctness and independent evidence

All existing fail-closed gates remain required: exact output for all 131,072
active values plus untouched canaries; exact counter value
`(16 + 512) * 16 = 8,448` for every active thread and zero outside the active
range; target-stub and marker-fallback transformation; module load and two-link
attach/detach; unique private shared memory and cleanup; distinct telemetry;
paired per-arm and campaign safety snapshots; no Xid, abnormal logs, process
survivors, UVM references, or struct-ops residue.

The independent analyzer treats `result.json` only as the frozen schedule and
raw-directory locators. It reopens application, loader/map, agent-bootstrap,
telemetry, safety, and lifecycle files and recomputes all measurements and
gates. Derived runner summaries are not evidence.

## Execution and stopping rule

After acquiring the existing read-only GPU and struct-ops leases:

```sh
python3 run_fixed_work_precision.py \
  --phase preflight \
  --output raw/fixed-work-precision-preflight-575-01
python3 run_fixed_work_precision.py \
  --phase full \
  --output raw/fixed-work-precision-full-575-01
python3 analyze_fixed_work_precision.py \
  --result raw/fixed-work-precision-full-575-01/result.json
```

The only successful stopping point is one complete 48-block run plus independent
raw replay. The runner stops fail-closed on an invalid cell for diagnosis; it
does not inspect performance to decide whether to continue. There is no
sequential testing or sample-size extension beyond 48 blocks.

## Paper decision and scope

- Supported: state only that no material block-organization effect was detected
  within +/-1% for this synthetic kernel, current return-only handler, RTX 5090,
  and the five fixed-work organizations.
- Contradicted: report the affected organization and direction and remove any
  block-organization-independence wording.
- Inconclusive: retain all estimates and intervals and make no equivalence
  claim; additional post-hoc runs do not get pooled into this decision.
- In every outcome, this remains verification-disabled runtime measurement. It
  does not establish once-per-warp dispatch, arbitrary-handler constant cost,
  application-level overhead, or universal independence from block count.
