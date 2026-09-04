# Experiment Plan: RQ4 Fixed-work block organization

## Research Question
- RQ exactly as written in the paper: "What is the overhead of gpubpf's core mechanisms, including its observability capabilities and device-side extensions?"
- Specific uncertainty tested here: Whether changing CUDA block organization adds a material device-hook penalty when total threads, dynamic warps, per-thread arithmetic, hook sites, hook repetitions, launches, and outputs are fixed.
- Why the answer matters: Reviewer D asks whether trampoline overhead scales with extreme block counts. The existing experiment fixes active hook work but launches additional early-return threads, so it cannot by itself isolate a fixed-total-work organization effect.

## Paper-Value Admission
- Planned role: supporting.
- Largest credible paper story this experiment could unlock: A matched RTX 5090 bound on the incremental cost of the current minimal device hook across a 32x block-organization range at identical total work.
- Strongest reviewer reject argument or load-bearing uncertainty addressed: The current low-overhead point and fixed-active-prefix experiment may hide an interaction between instrumentation and CUDA block scheduling or occupancy.
- Independent evidence added beyond existing runs and published results: The existing block-footprint axis holds active work fixed but increases launched inactive threads. This experiment holds both launched and active threads fixed and changes only the block/thread decomposition.
- Why the result is not tautological, already settled, or dominated: The attached-minus-native contrast can still change with block scheduling, occupancy, register pressure, or instrumentation waves even though the number of dynamic hook instructions is fixed.
- Paper decision if positive: State only that no material block-organization penalty was observed for this kernel and tested RTX 5090 range; retain the separate active-warp result for invocation scaling.
- Paper decision if contradictory, mixed, or inconclusive: Report the organization-sensitive boundary and remove any block-count-independence wording.
- Best alternative experiment and why this one has higher decision value: Another application-level observability run would add breadth but would not repair the causal ambiguity in the reviewer-facing trampoline claim.

## Expected And Alternative Outcomes
- Current expected answer: The minimal return-only handler's attached-minus-native time will remain within the predeclared materiality bound across the fixed-work organizations.
- Strongest competing explanation: Instrumentation changes occupancy or scheduling-wave behavior, producing an organization-dependent increment despite identical dynamic work.
- Result that would contradict the expectation: The endpoint contrast between 128 blocks x 1,024 threads and 4,096 blocks x 32 threads has a 95% interval wholly outside the materiality bound; any of the four all-five guard intervals is wholly outside the same bound; or any raw correctness, engagement, lifecycle, telemetry, or safety gate fails.

## Published Precedent And Real Assets
- Closest published protocol: CUDA event elapsed time provides device-timeline batch latency; dynamic instrumentation work conventionally reports matched kernel slowdown or latency increment.
- Official system/model/data/benchmark/tool and version: RTX 5090, NVIDIA driver 575.57.08, CUDA 12.9, and the selected bpftime Table 1 CUDA/PTX runtime.
- What is reused: The existing deterministic CUDA kernel, explicit hook stub, native/no-op/counter arms, libbpf loader, target/marker engagement gates, exact output and map oracles, safety lifecycle, telemetry, and paired runner.
- Necessary deviations or custom glue: Add a fixed-work matrix, block-local cell randomization, and an independent difference-in-differences analyzer. No driver or runtime interface changes are required.

## Comparison
- Proposed system or method: The current gpubpf/bpftime device trampoline with a return-only device BPF handler.
- Main baselines and the competing position each represents: The identical CUDA binary without the bpftime agent is the native control and represents no dynamic policy instrumentation.
- Why each main baseline needs a matched run instead of citation alone: The question is an RTX 5090 interaction between this runtime and block organization; published or prior unmatched timings cannot isolate it.
- Controls or ablations, labeled separately: A counter-handler arm validates exact target callback coverage and exposes state-update sensitivity, but is secondary rather than a competing baseline. A separate 32-thread marker validates fallback attachment.
- Conclusion if each main baseline matches or wins: A native win quantifies instrumentation cost. Statistical similarity supports only a bounded no-material-penalty statement, never a mechanism speedup.
- Information, tuning, and compute fairness: Every cell launches 131,072 active threads (4,096 warps), performs the same arithmetic and 16 explicit hook repetitions per thread, uses the same launches and output oracle, and differs only in reciprocal block/thread dimensions. Arms use a seed-1797 randomized balanced schedule: the first nine assignments give every arm each position three times, and the tenth makes the unavoidable imbalance at most one. One cell order is randomized per block and shared by all three arms.
- Split or leakage rule when relevant: Validity is determined exclusively by frozen correctness, engagement, cleanup, and safety gates. Performance never selects retries or exclusions.

## Workloads And Metrics
- Real workloads or tasks: One deterministic CUDA kernel at five organizations: 128x1,024, 256x512, 1,024x128, 2,048x64, and 4,096x32. Each launches exactly 131,072 threads and 4,096 whole warps; every thread reaches the same hook site.
- Primary metrics: Within each randomized block, compute `(noop - native)` at both endpoints and their difference-in-differences. Normalize the endpoint contrast by the mean endpoint-native batch time. The primary uncertainty is a seed-1797 paired-bootstrap 95% interval over ten blocks.
- Predeclared all-five organization guard: For each of cells 1--4, compute the paired difference between that cell's `(noop - native)` and cell 0's `(noop - native)`, normalized by the mean native time of the two organizations. Bootstrap each ten-block median independently with a fixed distinct seed. Four two-sided 98.75% percentile intervals apply a Bonferroni correction, giving at least 95% family-wise coverage. The guard passes only if every interval lies within +/-1%. Any interval wholly outside is contradictory; a boundary-crossing interval is inconclusive. The tested hypothesis is supported only when both the endpoint primary and the all-five guard pass.
- Correctness check or ground truth: All output slots match the independent integer oracle; all cell geometry and fixed-work fields match the frozen matrix; the counter map exactly reports every warmup and timed callback for every logical thread; marker, PTX transformation, module, attach, detach, private-segment, telemetry, UVM, Xid, and survivor gates pass.
- Repetitions, seeds, and uncertainty: One real three-arm preflight at the middle organization, then ten randomized paired blocks. Balanced arm assignment, cell-order randomization, and all bootstrap intervals use distinct deterministic streams derived from seed 1797.
- Cost estimate when material: Thirty full-run processes containing 150 timed cells, expected to finish within 30 minutes and bounded by the existing one-hour runner deadline.

## Planned Runs
| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| preflight | dependency | 1,024x128 fixed-work cell | native / no-op / counter | 1 paired block | Proves the actual path only |
| full | baseline | five fixed-work organizations | native | 10 | Matched denominator |
| full | proposed | five fixed-work organizations | return-only handler | 10 | Primary block-organization effect |
| full | control | five fixed-work organizations | exact counter handler | 10 | Engagement and state-update boundary |

## Execution
- Authoritative command or workflow: Build with the existing `Makefile`, then run `python3 run_fixed_work.py --phase preflight --output raw/fixed-work-preflight-<id>` followed by `python3 run_fixed_work.py --phase full --output raw/fixed-work-full-<id>`. Analyze only a complete full result with `python3 analyze_fixed_work.py --result raw/fixed-work-full-<id>/result.json`.
- Real preflight case: All three arms at 1,024 blocks x 128 threads, one warmup, two timed launches, and two hook repetitions.
- Full completion rule: All 30 arm processes and 150 cell measurements pass; all fixed-work invariants, exact output/counter evidence, target/marker engagement, lifecycle, telemetry, and final safety gates pass. The offline analyzer must independently reopen and validate every admitted arm's raw application log, loader/map log, agent-bootstrap log, telemetry CSV, paired safety snapshots, and lifecycle record. `result.json` supplies only the frozen schedule and raw-directory locators; its derived `valid`, measurement, engagement, telemetry-summary, and safety dictionaries are not analysis evidence.
- Raw-result path: The selected new `raw/fixed-work-*` directory contains the ordinary runner result; a distinct directory per arm with `application.log`, `lifecycle.json`, and paired safety snapshots; attached-arm `loader.log` and `agent.log`; a distinct telemetry CSV per arm; summary files; and independent analysis outputs.
- Checkpoint or recovery approach: The existing fail-closed per-arm checkpoint/resume path is reused. Resume accepts only the frozen profile, schedule, per-block cell order, source metadata, and already-valid arms.

## Interpretation
- Positive result: The endpoint 95% interval and all four Bonferroni-adjusted 98.75% intervals lie within +/-1%; report no material organization effect across all five tested organizations in this bounded kernel/GPU range.
- Negative or contradictory result: The endpoint interval or any all-five interval is wholly outside +/-1%; report the affected organization and measured direction and remove block-independence wording.
- Mixed or inconclusive result: No interval is wholly outside but at least one overlaps an equivalence boundary; report all estimates and intervals without an independence claim.
- Target paper figure or table: A compact supporting panel showing paired no-op increment across block organizations, annotated with the endpoint contrast and interval.

## Reproducibility Notes
- Software and data versions: Record ordinary Git revisions, runtime configuration, GPU/driver/CUDA properties, commands, seeds, and file inventory/metadata; never use content digests.
- Config and seed notes: Full runs use ten blocks, seed 1797, a balanced three-arm order, two warmups, eight timed launches, and 16 hook repetitions. Every dimension is a whole-warp multiple. The endpoint interval is 95%; the four all-five intervals are 98.75% each under the predeclared Bonferroni family-wise guard.
- Known deviations: Block count and threads per block must vary reciprocally to hold total work and warps fixed, so the independent variable is block organization, not block count in isolation. The current runtime uses an ordinary PTX `call`/`call.uni`; this experiment neither assumes nor proves scalar warp-leader execution. It is a synthetic verifier-off mechanism result and excludes attach/JIT time.
