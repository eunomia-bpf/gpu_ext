# Experiment Plan: RQ4 Device-hook scaling

## Research Question
- RQ exactly as written in the paper: "What is the overhead of gpubpf's core mechanisms, including its observability capabilities and device-side extensions?"
- Specific uncertainty tested here: Whether the incremental execution time of the current bpftime-backed device hook grows with dynamic active warps, and whether increasing the launched block footprint adds a separate cost when active work, active hook invocations, threads per block, launches, and hook sites are fixed.
- Why the answer matters: Reviewer D explicitly asks whether application-level trampolines scale for kernels with extreme block counts and heavy thread utilization. The current RTX 5090 microbenchmark is a single 32-element point and cannot answer that question.

## Paper-Value Admission
- Planned role: supporting.
- Largest credible paper story this experiment could unlock: A matched RTX 5090 scaling curve that bounds device-hook cost over 16x launched-block variation and 16x active-warp variation, while separating a minimal handler from a representative map-update handler.
- Strongest reviewer reject argument or load-bearing uncertainty addressed: The reported small single-point overhead may hide nonlinear costs at large grids or many resident warps.
- Independent evidence added beyond existing runs and published results: Existing Table 1 work measures application throughput for three tools; this experiment directly varies the number and organization of dynamic hook invocations in the same CUDA kernel.
- Why the result is not tautological, already settled, or dominated: PTX injection, handler code, register pressure, occupancy, and map traffic can make scaling architecture-dependent. The selected runtime source also inserts an ordinary device call and does not itself prove the paper's once-per-warp description.
- Paper decision if positive: Add a bounded scaling result and use wording supported by the observed invocation granularity.
- Paper decision if contradictory, mixed, or inconclusive: Report the nonlinear or block-sensitive boundary, narrow the mechanism claim, and investigate/implement warp-leader dispatch before making a once-per-warp claim.
- Best alternative experiment and why this one has higher decision value: Another end-to-end observability workload would add breadth but would not isolate the exact block/warp scaling concern.

## Expected And Alternative Outcomes
- Current expected answer: Incremental time will track dynamic hook invocations/active warps, while the fixed-active-work series will show little sensitivity to a larger launched-block footprint. The source audit does **not** justify predicting one scalar handler execution per warp.
- Strongest competing explanation: Register pressure, scheduling waves, PTX patching, or map contention introduces nonlinear growth or an independent block-count penalty.
- Result that would contradict the expectation: A systematic increase across the block-footprint series, superlinear growth across the active-warp series, or failure of exact callback/correctness accounting at scale.

## Published Precedent And Real Assets
- Closest published protocol: CUDA event elapsed time is the standard device-timeline method for timing work submitted to a stream; NVBit evaluates dynamic GPU instrumentation overhead as application/kernel slowdown.
- Official system/model/data/benchmark/tool and version: RTX 5090, NVIDIA driver 575.57.08, CUDA 12.9, and the selected `bpftime-table1-575` CUDA/PTX runtime revision.
- What is reused: bpftime's native kprobe-entry PTX pass, loader/syscall-server path, agent preload path, per-GPU-thread array map, and CUDA event timing API.
- Necessary deviations or custom glue: A finite deterministic kernel, two small eBPF handlers, an ordinary libbpf loader, and a runner that interleaves arms and validates exact output/counter records. No new runtime or driver interface is introduced.

## Comparison
- Proposed system or method: The current gpubpf/bpftime device trampoline with (1) an attached return-only handler and (2) a representative per-thread array-map increment.
- Main baselines and the competing position each represents: The exact same binary without the bpftime agent is the native control. Its compiled dummy hook stub keeps the source and call site identical while omitting runtime PTX replacement.
- Why each main baseline needs a matched run instead of citation alone: The question is the incremental cost on this RTX 5090 and selected runtime, which published numbers cannot supply.
- Controls or ablations, labeled separately: Before timing, a separate one-block, 32-thread marker kernel executes once; its per-thread map must contain 32 values equal to one. This proves that the loaded object and runtime executed, while agent logs must independently show recording of the exact target function, `kprobe_entry_stub` replacement, PTX compilation, module load, and successful attach. Target completion plus those exact target records are the return-only arm's target-engagement evidence. The representative handler's exact target counter is an engagement and losslessness control, not a performance baseline.
- Conclusion if each main baseline matches or wins: A statistically indistinguishable attached arm bounds detectable overhead for that grid; a faster native control quantifies the cost. No result is interpreted as mechanism speedup over native execution.
- Information, tuning, and compute fairness: All arms use the same binary, cell schedule, CUDA stream, integer work, launch count, hook-repeat count, GPU, driver, and power policy. Arms are randomized within each of ten paired blocks. CUDA events bracket only the kernel batch, excluding loader and PTX compilation time.
- Split or leakage rule when relevant: The matrix and interpretation are fixed before GPU execution. Failed correctness or engagement cells remain recorded and are excluded from performance estimates; performance values never determine validity.

## Workloads And Metrics
- Real workloads or tasks: A CUDA kernel with deterministic unsigned-integer work and one explicit hook site in a fixed loop. The block-footprint axis fixes 256 threads per block, 65,536 active threads (2,048 active warps), and all active work/hook calls while increasing the launched grid from 256 to 4,096 blocks; surplus threads take the same bounds-check-and-return path in every arm. The active-warp axis fixes the launch geometry at 4,096 blocks x 256 threads while increasing the active prefix from 65,536 to 1,048,576 threads (2,048 to 32,768 active warps). The first cell is shared, yielding nine unique cells.
- Primary metrics: Paired CUDA-event elapsed-time increment (milliseconds and percent) for each attached arm relative to native; paired-bootstrap 95% confidence interval across ten blocks. Descriptive diagnostics are incremental nanoseconds per dynamic warp and a linear fit over the active-warp axis.
- Correctness check or ground truth: Every arm checks all 1,048,576 output slots: active elements must match a host-computed integer oracle and inactive elements must retain a canary. Attached arms require the separate marker's 32 logical threads to each report exactly one callback, explicit target PTX-replacement evidence, clean detach, and no surviving owned process/shared-memory segment. The return-only target has no state update by definition, so its target execution gate is the conjunction of exact target replacement/load/attach records and successful completion of every target kernel. The representative arm uses a per-GPU-thread array: each active logical thread owns a distinct slot and updates it serially, so no two GPU threads race on a value. After CUDA synchronization and process exit, the complete run-length encoding of every key must equal the independently computed launch/hook-count oracle, including zero-valued inactive and unused regions.
- Repetitions, seeds, and uncertainty: One three-arm preflight, then ten randomized paired blocks using seed 1797. Bootstrap resampling uses seed 1797. No retry is selected by performance.
- Cost estimate when material: 30 finite application processes for the full experiment; nine scale-cell measurements per process. Expected wall time is minutes, with a one-hour hard campaign limit.

## Planned Runs
| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| preflight | dependency | one 256-block x 256-thread, 65,536-active-thread cell | native / return-only / counter | 1 paired block | Proves the real timing, attach, correctness, and readback path only |
| full | control | 9-cell scaling matrix | same binary, no agent | 10 | Native denominator for each paired block/cell |
| full | proposed | 9-cell scaling matrix | attached return-only handler | 10 | Measures minimum current handler/trampoline increment |
| full | proposed | 9-cell scaling matrix | attached per-thread counter | 10 | Measures representative state-update increment and exact callback coverage |

## Execution
- Authoritative command or workflow: From this directory, run `make BPFTIME_ROOT=/home/yunwei37/workspace/gpu/bpftime-table1-575 BPFTIME_BUILD=/home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575`, then `python3 run_scaling.py --phase preflight --output raw/preflight-<id>` or `python3 run_scaling.py --phase full --output raw/full-<id>` with the same two bpftime path options when non-default paths are needed.
- Real preflight case: All three arms at 256 blocks x 256 threads, one warmup launch, two timed launches, and one hook call per thread.
- Full completion rule: All 270 planned measurements (9 cells x 3 arms x 10 paired blocks) pass application correctness. The 20 attached application processes (2 attached arms x 10 paired blocks, each containing nine measurements) must all pass marker/transform/detach gates; the 10 representative application processes must all pass exact map-segment accounting for all nine cells. Telemetry and before/after kernel/GPU safety checks must pass.
- Raw-result path: The requested new output directory contains `result.json`, `summary.csv`, `summary.md`, per-run application/loader logs, and GPU telemetry.
- Checkpoint or recovery approach: `result.json` is rewritten after each completed arm. `--resume` runs only missing/invalid arms using the frozen schedule and parameters; it never overwrites prior logs or silently changes the matrix.

## Interpretation
- Positive result: Both attached arms remain approximately linear in dynamic active warps and the block-footprint series shows no material launched-footprint trend, with exact engagement and correctness throughout.
- Negative or contradictory result: Preserve and report nonlinear growth, an independent block penalty, or a failed scale point as a mechanism boundary; do not recast it as a favorable result.
- Mixed or inconclusive result: Report the cells and confidence intervals, name clock/variance or unsupported-scale limitations, and withhold a scaling claim.
- Target paper figure or table: One two-panel figure: fixed active work versus launched block footprint, and fixed launch geometry versus active warps, with native, return-only, and representative-counter curves or paired overheads.

## Reproducibility Notes
- Software and data versions: Record git revisions, CMake feature values, driver/GPU/CUDA properties, executable paths and ordinary file metadata. Do not record content hashes.
- Config and seed notes: Full runs use the source-declared nine-cell matrix, seed 1797, two warmups, eight timed launches, and two hook calls per active thread; preflight uses the single frozen cell above.
- Known deviations: This is a synthetic mechanism microbenchmark, not an application-throughput result. The selected Table 1 runtime has device verification disabled, so this run provides performance and execution evidence only, not verifier-enforcement evidence. The explicit stub is used to pin an unambiguous hook site; it does not establish transparent fallback-entry overhead. CUDA may execute the largest grids in multiple scheduling waves; that is expected scaling behavior, not a claim that all blocks or warps are simultaneously resident. Preflight records device limits and rejects any block size or grid dimension outside those limits.
