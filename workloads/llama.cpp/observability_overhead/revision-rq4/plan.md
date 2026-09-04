# Experiment Plan: RQ4 RTX 5090 Observability Overhead

**Result, 2026-09-04:** The predeclared two-tool subset is complete. The
dependency preflight is `raw/preflight-575-noncross-clock-04`; the paper-value
run is `raw/full-575-noncross-clock-02`. Independent analysis accepted all five
correctness configurations and all 10 randomized five-cell blocks, with no
rejected or retried cell. The result is mixed: for exit records, mean overhead
was 99.6627% for gpubpf and 99.6208% for matched NVBit (paired effect
NVBit-minus-gpubpf -0.04185 percentage points, 95% CI [-0.04355, -0.04029]);
for the exit-count histogram it was 4.0071% and 10.3006%, respectively
(+6.29351 points [6.12507, 6.47076]). See `result-review.md`. This completes two
corrected RTX 5090 rows, not the cross-clock `launchlate` row, which remains
invalid. The runtime had GPU verification disabled, so this is functional
instrumentation evidence rather than verifier-overhead evidence.

**Predeclared non-cross-clock subset, 2026-09-03:** The executable
[`table1-noncross-clock-plan.md`](table1-noncross-clock-plan.md) selects only
`kernelretsnoop` and `threadhist` with `--tools kernelretsnoop threadhist`.
The failed `raw/preflight-575-noncross-clock-01` is preserved as failed: its
pp32 timing probe used 22,528 slots for 32,768 coordinates across 44 launches,
causing 450,560 OOB drops.
This is a narrower five-configuration Table 1 campaign, not a reclassification
of the retained `launchlate` failures and not completion of the original
seven-arm plan. Omitting `--tools` continues to select all three tools.

**Historical execution update, 2026-09-03:** The user requested completion of all
remaining revision commitments. The earlier closed review in `plan-review.md`
is retained as history, not a new approval loop. The host now runs Linux
6.15.11 and NVIDIA **575.57.08**. The experiment below retains its seven arms,
workload, exact-output/engagement gates and ten-block schedule. The driver
change and launch-lifecycle repairs did not themselves establish a valid
preflight or performance result; both were pending at this checkpoint.

The runner now acquires the two existing revision leases, rejects ambient
injection, fixes CUDA clients/loaders to CPUs 8–15, and starts continuous GPU
telemetry on CPU 16. **Do not pin the coordinator to CPUs 8–15**: the telemetry
CPU must remain available. Instrumentation is injected only after the affinity
wrapper. Each BPF cell uses a private segment, removed only after its owned
client/loader exit and its type, owner and inode checks pass. Before/after
kernel/GPU safety and fixed 400 W state are checked for every cell. A private
launchlate source-copy repair records GPU-before-host clock errors instead of
clamping them to zero; both systems require explicit zero-error counters.
Thirty-nine lightweight CPU tests pass, including a real interrupted CPU
child-group cleanup check. CPU build/diagnostic helpers now use bounded owned
teardown without changing the shared legacy harness. If CUDA-client cleanup
cannot confirm exit, the runner retains that client's loader and private
segment, records process identities and stops the campaign immediately;
post-safety errors cannot downgrade this terminal failure. Normal teardown is
client, loader, private segment, then telemetry and post-safety checks. No new
575 GPU result was claimed at that checkpoint.

The separate `bpftime-table1-575/build-table1-575` runtime now builds
successfully (108/108 steps). It carries the actual-thread-count metadata fix
from the strict R5 path and the controlled CUDA attachment repairs; the main
runtime and R5 build are unchanged. The histogram collector initializes the
entire configured array with a sentinel and requires complete readback,
including legitimate zero-valued tail entries. The runner requires all
1,048,576 entries / 8,388,608 bytes; mock truncated readbacks fail. This is
build/CPU evidence only, not real histogram engagement. Use both
`--bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575` and
`--bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575`
for the fresh 575 preflight and subsequent full run.

The formal runner is wired to the repaired lossless exit collector. It fixes
the exit channel at 22,528 thread slots,
requires exactly 256 records per slot for correctness and exactly 44 for timing,
and parses the collector's terminal
allocation, record-size, commit/readback, drop, dirty, pending, drain and
Cartesian-coverage fields. Deterministic correctness requires 720,896 records,
all with nonzero timestamps, across 220 launches and 22,528 unique coordinates.
The exact multiplicity oracle requires 1,024 coordinates observed 220 times,
1,024 observed 44 times and 20,480 observed 22 times, with no other
multiplicity or segment mismatch; the matching NVBit arm must report the same
total and 220 selected launches. This oracle is enabled only for the fixed
correctness cell. Timed
pp=32/pp=512 cells do not reuse those fixed correctness counts: each gpubpf cell
must instead be internally lossless and Cartesian-complete, then match its
NVBit partner's event and launch counts in that repetition block. The runner
also requires the known 47-byte normalized deterministic output and explicit
22,528-slot, 1,000 MiB shared-memory, source and build arguments, preventing an
accidental return to the unrelated default bpftime checkout.

Retained preflight 01 stopped before GPU execution because sudo's PATH lacked
`nvcc`; use an explicit `/usr/local/cuda-12.9/bin` PATH for the coordinator.
Preflight 02 built every tool but rejected the native correctness control's
empty stdout. In this llama-cli, generated tokens use `LOG()`, and
`--log-disable` suppresses them too. Remove only that flag for all correctness
arms, preserving the prompt, eight-token limit, seed, temperature and exact
nonempty-output requirement. No timing cell ran in either failed attempt.

The reused performance runtime currently has GPU verification disabled. This
study compares functional instrumentation cost; it does not establish enforced
SIMT verification, which is a separate strict-runtime experiment. Preserve all
earlier diagnostics, and use fresh 575 preflight/full directories.

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
- Official system/model/data/benchmark/tool and version: gpubpf/bpftime at the checked-out commits; the frozen plan named llama.cpp build 7101; TinyLlama 1.1B Q4_K_M; official NVlabs NVBit v1.8 (latest as of 2026-08-31), identified by its release path and ordinary artifact metadata. The accepted execution deviated to build 7102 (`26836b27`), but its preflight and every full-run arm consistently used the same binary, so the disclosed deviation does not create an H2H mismatch.
- What is reused: `run_observability_overhead.py`, the PTX-enabled llama-bench build, the real model, bpftime example tools, and official NVBit release examples/APIs.
- Necessary deviations or custom glue: Add three matched custom adapters using the official NVBit release and runner support; repair the current bpftime-agent build/path and the three gpubpf tool semantics below. No new experiment-control schema or result gate will be introduced.

## Comparison
- Proposed system or method: gpubpf device-side probes for (1) per-logical-thread exit timestamp records with global logical x/y/z coordinates (`kernelretsnoop`), (2) a final per-logical-thread exit-count histogram over the full configured map (`threadhist`), and (3) selected-kernel host-stub-to-device-entry latency (`launchlate`). The compact record does not preserve the original block/thread decomposition. The paper row descriptions must use these corrected definitions and must not call the first task per-block timing.
- Main baselines and the competing position each represents: Official NVBit, representing current SASS-level dynamic binary instrumentation without gpubpf's warp-uniform verified execution.
- Why each main baseline needs a matched run instead of citation alone: The reviewer asks for RTX 5090 values, and overhead depends on architecture, driver, CUDA version, workload, target kernel, and event volume.
- Controls or ablations, labeled separately: No-probe llama.cpp baseline; sample/event counts and output sanity checks for each tool; an uninstrumented NVBit load is not a scientific baseline.
- Conclusion if each main baseline matches or wins: The claimed overhead advantage does not generalize for that tool on Blackwell; report the boundary and retain only supported per-tool claims.
- Information, tuning, and compute fairness: Same binary, model, prompt size, exact target symbol, device hook point, 32-byte `(global_x, global_y, global_z, timestamp)` payload, coordinate aggregation/output gates, GPU clocks/default power policy, CUDA-graph setting, and repetition count. Only the selected kernel is enabled in NVBit (`apply_to_related=false`); unrelated kernels and related functions remain uninstrumented. The transports remain system-native (gpubpf per-thread rings versus NVBit `ChannelDev`), but both host collectors consume and validate the same records. For `kernelretsnoop`, both systems must report identical record size, timestamps, extents, complete coordinate multiplicities, events, collector status, and 44 selected launches: 1,441,792 events at pp32 and 23,068,672 at pp512. The gpubpf correctness ring remains 22,528×256; timing uses the source-derived `pp×1,024` coordinates and 44 entries per slot (892.000031 MiB at pp512), so the complete 44-launch burst fits even with no concurrent drain. For `launchlate`, gpubpf timestamps the exact host launch stub and consumes timestamps through a FIFO, whereas NVBit timestamps its native CUDA driver launch callback and passes the value through NVBit's per-launch argument. These host hook points are not identical and must be disclosed; the comparison is of each system's native implementation of the same host-submission-to-device-entry task, not identical injected host code.
- Split or leakage rule when relevant: Not applicable.

## Workloads And Metrics
- Real workloads or tasks: llama.cpp prefill of TinyLlama 1.1B Q4_K_M on the RTX 5090 using the existing paper harness, prompt length 512 and generation length 0.
- Primary metrics: Prefill throughput (tokens/s) and percent degradation versus interleaved no-probe baseline; geometric mean across completed repetitions.
- Correctness check or ground truth: Before timing, the no-probe control and all six instrumented configurations must produce the exact 47-byte normalized output `Deterministic tests are essential\n> EOF by user` from an untimed deterministic `llama-cli` generation (`--seed 1797 --temp 0`, fixed prompt and eight generated tokens); the output is preserved verbatim, and every instrumented correctness run must also engage its probe. The exit-record correctness arms additionally require 720,896 nonzero-timestamp 32-byte records, 220 selected launches, extent 88×256×1, and the exact 1,024-at-220, 1,024-at-44 and 20,480-at-22 coordinate segments in both systems. gpubpf must also report 22,528 requested/allocated slots, exactly 256 entries per slot, identical committed/runtime-collected/host-collected counts, and zero drop/error/dirty/pending/second-drain counters. Both collectors require no other multiplicity, no segment mismatch, no invalid coordinate, and a passing collector gate. Each timed run must finish successfully, report the configured prompt length, and pass the same lossless/Cartesian gates with the exact correctness oracle disabled. Timing is also exact: extent 128×256×1, 32,768 coordinates, 44 launches, and 1,441,792 events at pp32; extent 2,048×256×1, 524,288 coordinates, 44 launches, and 23,068,672 events at pp512; exactly 44 ring entries per coordinate; and equality of all recorded cross-arm semantics. The final logical-thread histogram must have at least one nonzero slot and a positive total; launch correlation must have no overflow/underflow, matching host/device/sample counts, and non-negative latency bins. The histogram total is derived from its slots and is not described as an independent event count. The llama-bench JSON must contain one prompt result with finite positive throughput. Zero-sample, count-mismatched, or exact-output-mismatched runs are invalid, not zero-overhead wins.
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
- Authoritative command or workflow: From `gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4`, build checking is `make -C nvbit_adapters/observability CXX=g++ NVBIT_ROOT="$PWD/deps/nvbit_release_x86_64" ARCH=sm_120`. Both experiment entries require `--bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 --gpu-thread-count 22528`: append those arguments to `python run_revision_rq4.py --phase preflight --output-dir raw/preflight-<timestamp>` or `python run_revision_rq4.py --phase full --output-dir raw/full-<timestamp>`. The runner passes `BPFTIME_SHM_MEMORY_MB=1000` to the private exit collector and enables `BPFTIME_KERNELRETSNOOP_EXACT_ORACLE=1` only for deterministic correctness; timed cells explicitly use `0`. The predeclared `kernelretsnoop-phase-capacity.patch` is the current overlay on the retained historical `gpubpf-observability.patch`: it compacts the observable and makes map entries configurable before load. The runner fixes correctness at 22,528×256 and timing at `pp×1,024`×44. To resume missing or invalid cells without overwriting prior attempts, repeat the same command and output directory with `--resume`. The runner fixes preflight to pp=32 and one block, full execution to pp=512 and 10 blocks, schedule seed 1797, CUDA graphs off, and the pinned NVBit root.
- Real preflight case: On the supported 575.x stack, first pass the deterministic exact-output control for all seven paths, then run one pp=32 timing cell for every distinct path: no probe, all three corrected gpubpf tools, and all three matched custom NVBit adapters, on the actual llama.cpp binaries and model. The earlier diagnostic admission at `raw/preflight-20260831_013158/admission.json` observed driver 610 and two external SGLang processes, so it is not an official preflight attempt and no process was terminated. The current runner rejects driver 610 for both official preflight and full execution.
- Full completion rule: All seven planned configurations reach terminal status with 10 valid repetitions on the same RTX 5090 under an officially supported NVBit 575.x driver stack. Otherwise the experiment remains incomplete/inconclusive. Diagnostics on driver 610 cannot complete the paper comparison. No partial prefix is treated as the experiment result.
- Raw-result path: `workloads/llama.cpp/observability_overhead/revision-rq4/raw/<timestamp>/`.
- Checkpoint or recovery approach: Write logs and result JSON/CSV after every correctness attempt and repetition. A new run refuses a nonempty output directory. Resume requires the same phase, arguments, boot, driver, source revisions and explicit source/binary inventories with ordinary file metadata; it skips only valid cells and writes retries to new attempt paths. Actual exact-output and engagement checks, not content-integrity fields, establish experiment validity.

## Interpretation
- Positive result: Every compared probe engages correctly and gpubpf's claim-matched overhead is lower with confidence intervals that do not reverse the ordering.
- Negative or contradictory result: One or more valid NVBit cells match or beat gpubpf; narrow the Table 1 claim to the supported tools and explain the mechanism boundary.
- Mixed or inconclusive result: Tool semantics cannot be fairly matched, either implementation fails to engage, or variability/compatibility prevents ordering; report those cells as inconclusive and do not average them into a win.
- Target paper figure or table: Table 1 / `tab:obs-overhead`.

## Reproducibility Notes
- The historical five-file gpubpf example diff is preserved in
  `gpubpf-observability.patch`, against bpftime commit
  `d6316fa73edaac4fdfe21b89d4470da6cd9b8ae8`. Reverse-apply checking verifies
  that it matches its original local experiment worktree. The current Table 1
  campaign additionally applies `kernelretsnoop-phase-capacity.patch`; read the
  two artifacts in order. Neither artifact implies an upstream bpftime PR,
  runtime approval, or a completed experiment.
- Software and data versions: Record Git commits, NVBit release, supported 575.x driver, CUDA toolkit, llama.cpp build metadata, ordinary path/size/time metadata for required model and binaries, and GPU state in raw results.
- Config and seed notes: Preserve exact environment variables, target symbols, commands, repetition order, sample counts, and timeout status.
- Historical stack deviation: Earlier diagnostics used 610.43.02, outside NVBit's documented `<=575.xx` range, and discussed a then-prepared Linux 6.14 build. Those records are unchanged. The new execution uses the already installed 575.57.08 userspace and Linux 6.15.11 runtime described in the dated update above; no installation or reboot is part of this runner.
