# Experiment Plan: RQ4 RTX 5090 Observability Overhead

**Review status:** Closed as rejected after the final permitted follow-up; see
`plan-review.md`. The two final implementation defects have been repaired, but
the runtime experiment requires a new independently reviewed proposal and an
installed 575.x driver before execution.

## Research Question
- RQ exactly as written in the paper: **RQ4 (Overhead): What is the overhead of gpubpf's core mechanisms and observability capabilities?**
- Specific uncertainty tested here: Whether three corrected, explicitly defined device-side observability tasks remain functional and lower-overhead than matched NVBit instrumentation on the RTX 5090, rather than only on the submitted P40 setup.
- Why the answer matters: The submitted author response commits to RTX 5090 measurements for the Table 1 device-side comparison, including NVBit. A result with non-engaged probes or an unmatched workload cannot satisfy that commitment.

## Paper-Value Admission
- Planned role: supporting
- Largest credible paper story this experiment could unlock: The device-side safety and SIMT execution design retains its low-overhead advantage over SASS rewriting on current Blackwell hardware.
- Strongest reviewer reject argument or load-bearing uncertainty addressed: Table 1 used an old P40 even though the rest of the evaluation uses an RTX 5090, so the reported advantage might be an artifact of hardware or unsupported tooling.
- Independent evidence added beyond existing runs and published results: A same-host, same-llama.cpp-build, same-prefill-workload comparison with verified probe engagement on sm_120. Existing July 2026 gpubpf smoke runs include zero-sample and incomplete cells; the existing NVBit evidence used a different workload or did not terminate.
- Why the result is not tautological, already settled, or dominated: Both systems inject real device-side instrumentation and can impose architecture-dependent overhead. Published NVBit results do not provide matched RTX 5090 numbers for these tools.
- Paper decision if positive: Replace the P40 Table 1 rows with RTX 5090 measurements and explicitly report the matched instrumentation and versions.
- Paper decision if contradictory, mixed, or inconclusive: Narrow the claim per tool; report any NVBit incompatibility separately from performance; do not count a failed or non-engaged baseline as a gpubpf win.
- Best alternative experiment and why this one has higher decision value: XSched Level-1 is the next-best runnable external baseline, but it is a voluntary addition; this experiment is a submitted hard commitment and directly addresses Reviewer A.

## Expected And Alternative Outcomes
- Current expected answer: Engaged gpubpf probes impose materially less prefill-throughput overhead than matched NVBit probes on RTX 5090.
- Strongest competing explanation: The P40 advantage came from unmatched instrumentation, probe non-engagement, or architecture/toolchain effects; on Blackwell, gpubpf may lose its advantage or fail to attach.
- Result that would contradict the expectation: A valid matched NVBit tool has equal or lower overhead than the corresponding gpubpf tool, or gpubpf cannot produce correct samples on the workload.

## Published Precedent And Real Assets
- Closest published protocol: NVBit MICRO'19 evaluates application slowdown under dynamic SASS instrumentation; the paper's submitted Table 1 measures llama.cpp prefill token/s degradation for three observability tasks.
- Official system/model/data/benchmark/tool and version: gpubpf/bpftime at the checked-out commits; llama.cpp build 7101; TinyLlama 1.1B Q4_K_M; official NVlabs NVBit v1.8 (latest as of 2026-08-31), identified by its release path and ordinary artifact metadata.
- What is reused: `run_observability_overhead.py`, the PTX-enabled llama-bench build, the real model, bpftime example tools, and official NVBit release examples/APIs.
- Necessary deviations or custom glue: Add three matched custom adapters using the official NVBit release and runner support; repair the current bpftime-agent build/path and the three gpubpf tool semantics below. No new experiment-control schema or result gate will be introduced.

## Comparison
- Proposed system or method: gpubpf device-side probes for (1) per-logical-thread exit timestamp records with block and thread coordinates (`kernelretsnoop`), (2) a final per-logical-thread exit-count histogram over the full configured map (`threadhist`), and (3) selected-kernel host-stub-to-device-entry latency (`launchlate`). The paper row descriptions must use these corrected definitions and must not call the first task per-block timing.
- Main baselines and the competing position each represents: Official NVBit, representing current SASS-level dynamic binary instrumentation without gpubpf's warp-uniform verified execution.
- Why each main baseline needs a matched run instead of citation alone: The reviewer asks for RTX 5090 values, and overhead depends on architecture, driver, CUDA version, workload, target kernel, and event volume.
- Controls or ablations, labeled separately: No-probe llama.cpp baseline; sample/event counts and output sanity checks for each tool; an uninstrumented NVBit load is not a scientific baseline.
- Conclusion if each main baseline matches or wins: The claimed overhead advantage does not generalize for that tool on Blackwell; report the boundary and retain only supported per-tool claims.
- Information, tuning, and compute fairness: Same binary, model, prompt size, exact target symbol, device hook point, event payload, aggregation/output volume, GPU clocks/default power policy, CUDA-graph setting, and repetition count. Only the selected kernel is enabled in NVBit (`apply_to_related=false`); unrelated kernels and related functions remain uninstrumented. For `kernelretsnoop`, gpubpf's ring buffer and NVBit's official device-to-host channel both transfer one 56-byte record per logical thread while the host only aggregates counts. For `launchlate`, gpubpf timestamps the exact host launch stub and consumes timestamps through a FIFO, whereas NVBit timestamps its native CUDA driver launch callback and passes the value through NVBit's per-launch argument. These host hook points are not identical and must be disclosed; the comparison is of each system's native implementation of the same host-submission-to-device-entry task, not identical injected host code.
- Split or leakage rule when relevant: Not applicable.

## Workloads And Metrics
- Real workloads or tasks: llama.cpp prefill of TinyLlama 1.1B Q4_K_M on the RTX 5090 using the existing paper harness, prompt length 512 and generation length 0.
- Primary metrics: Prefill throughput (tokens/s) and percent degradation versus interleaved no-probe baseline; geometric mean across completed repetitions.
- Correctness check or ground truth: Before timing, the no-probe control and all six instrumented configurations must produce exactly the same normalized stdout from an untimed deterministic `llama-cli` generation (`--seed 1797 --temp 0`, fixed prompt and eight generated tokens); the short normalized output is preserved verbatim, and every instrumented correctness run must also engage its probe. Each timed run must finish successfully, report the configured prompt length, produce non-zero samples for the exact selected symbol, and pass the checks the collectors can independently establish: all transferred exit records have nonzero timestamps; the final logical-thread histogram has at least one nonzero slot and a positive total; or the launch correlation has no overflow/underflow, matching host/device/sample counts, and non-negative latency bins. The payload includes block/thread coordinates, but the current collector does not independently reconstruct launch dimensions and therefore does not claim to validate their bounds. The histogram total is derived from its slots and is not described as an independent event count. The llama-bench JSON must contain one prompt result with finite positive throughput. Zero-sample or exact-output-mismatched runs are invalid, not zero-overhead wins.
- Repetitions, seeds, and uncertainty: 10 randomized/interleaved repetition blocks, each containing the no-probe control and the six instrumented cells. Report per-run values, geometric mean, paired per-block overhead, and a fixed-seed (`1797`) paired bootstrap 95% confidence interval for the primary effect `NVBit overhead - gpubpf overhead` for each task. Preserve failure and exclusion counts. The workload is deterministic and has no application random seed.
- Cost estimate when material: To be revised from real pp=32 preflight timing on the supported stack. The command timeout and any no-progress threshold will be set from those observations, not predeclared from unsupported-driver diagnostics.

## Planned Runs
| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| control | control | llama.cpp prefill, pp=512 | no probe | 10 interleaved | Throughput denominator and drift check |
| exit-time | proposed | same | gpubpf kernelretsnoop | 10 | Tests per-thread exit-record overhead |
| exit-time | baseline | same | matched custom NVBit exit-record adapter | 10 | Head-to-head result for Table 1 |
| activity | proposed | same | gpubpf threadhist | 10 | Tests thread activity histogram overhead |
| activity | baseline | same | matched custom NVBit activity adapter | 10 | Head-to-head result for Table 1 |
| launch | proposed | same | gpubpf launchlate | 10 | Tests launch latency overhead |
| launch | baseline | same | matched custom NVBit launch-latency adapter | 10 | Head-to-head result for Table 1 |

## Execution
- Authoritative command or workflow: From `gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4`, build checking is `make -C nvbit_adapters/observability CXX=g++ NVBIT_ROOT="$PWD/deps/nvbit_release_x86_64" ARCH=sm_120`; the real seven-cell preflight is `python run_revision_rq4.py --phase preflight --output-dir raw/preflight-<timestamp>`; and the paper-facing experiment is `python run_revision_rq4.py --phase full --output-dir raw/full-<timestamp>`. To resume missing or invalid cells without overwriting prior attempts, repeat the same command and output directory with `--resume`. The runner fixes preflight to pp=32 and one block, full execution to pp=512 and 10 blocks, schedule seed 1797, CUDA graphs off, the CUDA-enabled `bpftime/build-cuda-pr503`, and the pinned NVBit root.
- Real preflight case: On the supported 575.x stack, first pass the deterministic exact-output control for all seven paths, then run one pp=32 timing cell for every distinct path: no probe, all three corrected gpubpf tools, and all three matched custom NVBit adapters, on the actual llama.cpp binaries and model. The earlier diagnostic admission at `raw/preflight-20260831_013158/admission.json` observed driver 610 and two external SGLang processes, so it is not an official preflight attempt and no process was terminated. The current runner rejects driver 610 for both official preflight and full execution.
- Full completion rule: All seven planned configurations reach terminal status with 10 valid repetitions on the same RTX 5090 under an officially supported NVBit 575.x driver stack. Otherwise the experiment remains incomplete/inconclusive. Diagnostics on driver 610 cannot complete the paper comparison. No partial prefix is treated as the experiment result.
- Raw-result path: `workloads/llama.cpp/observability_overhead/revision-rq4/raw/<timestamp>/`.
- Checkpoint or recovery approach: Write logs and result JSON/CSV after every correctness attempt and repetition. A new run refuses a nonempty output directory. Resume requires the same phase, arguments, driver, model/llama binaries, source manifest, and hashes of the preserved per-run NVBit and gpubpf tool binaries; it skips only valid cells and writes retries to new attempt paths.

## Interpretation
- Positive result: Every compared probe engages correctly and gpubpf's claim-matched overhead is lower with confidence intervals that do not reverse the ordering.
- Negative or contradictory result: One or more valid NVBit cells match or beat gpubpf; narrow the Table 1 claim to the supported tools and explain the mechanism boundary.
- Mixed or inconclusive result: Tool semantics cannot be fairly matched, either implementation fails to engage, or variability/compatibility prevents ordering; report those cells as inconclusive and do not average them into a win.
- Target paper figure or table: Table 1 / `tab:obs-overhead`.

## Reproducibility Notes
- The exact five-file gpubpf example diff is preserved in
  `gpubpf-observability.patch`, against bpftime commit
  `d6316fa73edaac4fdfe21b89d4470da6cd9b8ae8`. Reverse-apply checking verifies
  that it matches the local experiment worktree. This artifact does not imply
  an upstream bpftime PR, runtime approval, or a completed experiment.
- Software and data versions: Record Git commits, NVBit release, supported 575.x driver, CUDA toolkit, llama.cpp build metadata, ordinary path/size/time metadata for required model and binaries, and GPU state in raw results.
- Config and seed notes: Preserve exact environment variables, target symbols, commands, repetition order, sample counts, and timeout status.
- Known deviations: The current host driver is 610.43.02, whereas NVBit's official README states driver `<=575.xx`. The gpubpf-enabled 575.57.08 open modules now build with BTF for the installed Linux 6.14.0-37 kernel, providing a concrete supported-stack path, but matching 575 userspace and a reboot have not been authorized. Implementation work and diagnostics may proceed, while paper-facing execution remains blocked until the same RTX 5090 is booted with that supported stack.
