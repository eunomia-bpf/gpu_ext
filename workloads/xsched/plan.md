# Experiment Plan: RQ3 XSched Level-1 on sm_120

Status (2026-09-02): **authorized for automatic three-way performance execution**.
The earlier closed review is retained in `plan-review.md` as history. The
runner now measures the CUPTI/globaltimer offset in a separate 16-sample
bracketed probe before each cell, checks a second probe after the policy has
detached, and rejects combined drift/uncertainty above 1 ms. Result categories
are now mutually exclusive. No new academic review cycle is required.

## Full-workload continuation

The user subsequently requested completion of XSched, GPreempt, and MoE, not
only the short-budget pilot. The next XSched campaign restores **50 kernels
per stream, ten complete randomized blocks, and three isolated controls per
role**. Its four configurations are `native,xsched,gpubpf,bpftime_hpf`: the
original driver policy remains visible, and the same-frontend BPF HPF port is
reported separately rather than replacing it under the same name. The added
arm leaves each configuration's workload, original HPF settings, metrics,
correctness checks, and interpretation rule unchanged.

GPU execution follows the current MoE campaign. The new GSP propagation fix
must first pass a separate runtime canary; a host-field readback does not prove
firmware enforcement. All four arms then use the same newly recorded driver
image and a frozen native calibration. Preserve every existing negative and
inconclusive pilot result. Analysis now requires the six correctly sized
isolated controls and rejects a campaign containing a failed cell.

The command is `python3 -B workloads/xsched/run_xsched_rq3.py full --configs
native,xsched,gpubpf,bpftime_hpf --output workloads/xsched/raw/full-575-gsp
--reps FROZEN_REPS` (one shell line, new output directory). No full-workload
performance result is claimed yet. Historical “GPREEMPT-equivalent” labels
below denote the differentiated-timeslice/one-shot-preempt analogue, not the
complete GPreempt hint/blocking-kernel protocol; see the
[current source audit](../../docs/driver_docs/sched/gpreempt-analysis/feasibility-575-20260902.md).

## Fixed short-budget campaign

The user prioritizes obtaining native, original user-space-policy, and our
BPF-policy performance within the current session. `pilot` therefore runs
five complete seed-1797 randomized blocks, with five kernels per stream.
All other algorithm/workload settings below remain unchanged, including the
80 ms isolated kernel calibration, 2 LC + 4 BE processes, four streams each,
XSched HPF settings, gpubpf timeslices, correctness, and engagement checks.
The 40 LC samples per cell make nearest-rank P99 equal to the sample maximum;
the report must disclose that and also report P50/P95/mean and BE throughput.
There are no isolated role controls in this short-budget campaign. It is a
repeated three-way performance comparison, not the original 50-kernel/10-block
protocol, whose `full` command and completion rule remain unchanged below.
The saved `protocol.json` records this distinction before the first cell.

## Research Question

- RQ exactly as written in the paper: **RQ3 (Multi-Tenant Management): Does gpubpf improve tail latency, throughput, and resource fairness compared to user-space and global policies in multi-tenant settings?**
- Specific uncertainty tested here: On the same RTX 5090 compute-bound priority workload, how does gpubpf's GPREEMPT-equivalent driver policy compare with the Level-1 inter-kernel preemption that current public XSched provides for sm_120?
- Why the answer matters: R1 explicitly names XSched as the runnable research-system scheduling baseline, but also requires the revision to label the public implementation's actual preemption level rather than implying paper-level Level-3 support.

## Paper-Value Admission

- Planned role: supporting.
- Largest credible paper story this experiment could unlock: gpubpf provides lower LC tail latency at comparable BE throughput than a current user-space research scheduler even when XSched is given its native HPF policy and transparent CUDA interception.
- Strongest reviewer reject argument or load-bearing uncertainty addressed: The submitted scheduling section compares only native scheduling against a reimplemented GPREEMPT-like policy; a strong public user-space scheduling artifact might obtain the same result without gpubpf's driver integration.
- Independent evidence added beyond existing runs and published results: A same-host, finite, correctness-checked head-to-head with XSched commit `f49289f0220931df78de948ed841ecbaf960a919`, whose CUDA build already succeeds locally. Existing Fig. 12 does not include XSched.
- Why the result is not tautological, already settled, or dominated: XSched Level-1 blocks future command batches rather than preempting an executing kernel, so its result depends on kernel granularity and queue depth; it may match gpubpf for inter-kernel workloads or lose when a BE batch is already resident.
- Paper decision if positive: Add XSched Level-1 as a clearly scoped RQ3 baseline and keep the mechanism distinction explicit.
- Paper decision if contradictory, mixed, or inconclusive: Report XSched's win or parity and narrow the advantage claim; if engagement cannot be verified, report artifact incompatibility rather than a performance number.
- Best alternative experiment and why this one has higher decision value: Orion is the fallback scheduling artifact, but it requires per-kernel SM-demand profiling and is less directly aligned with the existing priority-preemption experiment.

## Expected And Alternative Outcomes

- Current expected answer: Both policies protect LC work, while gpubpf reacts at driver scheduling points and therefore has lower P99 latency when BE command batches are already submitted.
- Strongest competing explanation: The workload's inter-kernel boundaries are frequent enough that XSched Level-1 HPF matches or beats gpubpf with less driver-side overhead.
- Result that would contradict the expectation: Valid XSched Level-1 has equal or lower LC P99 latency and no worse BE throughput than gpubpf.

## Published Precedent And Real Assets

- Closest published protocol: The paper's Fig. 12 uses two LC and four BE processes, four CUDA streams per process, 50 compute kernels per stream, LC P99 submission-to-completion latency, and BE throughput. XSched's official transparent CUDA example runs concurrent priority-tagged vector-add processes under global HPF scheduling.
- Official system/model/data/benchmark/tool and version: gpubpf/bpftime at the checked-out commits; XSched upstream commit `f49289f0220931df78de948ed841ecbaf960a919`; CUDA 12.9; one RTX 5090.
- What is reused: The paper's process/stream/kernel counts and metrics, XSched's official CUDA shim and HPF server, and the gpubpf GPREEMPT-equivalent policy.
- Necessary deviations or custom glue: Replace XSched's effectively infinite random-sleep example with a finite deterministic vector-compute harness that exposes the same binary to all configurations, accepts LC/BE role and stream count, records every task latency, reports throughput, and checks output. No project-authored result gate or substitute baseline is introduced.

## Comparison

- Proposed system or method: gpubpf GPREEMPT-equivalent differentiated-timeslice/preemption policy.
- Main baseline and competing position: XSched global HPF with `XSCHED_AUTO_XQUEUE_LEVEL=1`, explicitly labeled **XSched Level-1 on sm_120**.
- Why the baseline needs a matched run instead of citation alone: XSched's paper evaluates higher levels on older GPUs, while sm_120 falls through to `CudaQueueLv1`; performance depends on finite command batches and kernel duration.
- Controls or ablations: Native CUDA scheduling with neither policy; single-process LC and BE controls to quantify isolated latency/throughput; XSched engagement logs and XQueue counts.
- Conclusion if the baseline matches or wins: Driver-level programmability is not needed for this inter-kernel workload; retain only broader extensibility/visibility claims and report the boundary.
- Information, tuning, and compute fairness: Same bit-identical harness binary (hard-linked as `bench_lc`/`bench_be` only to expose the process names required by gpubpf), process mix, streams, task count, kernel work, CPU affinity map, and two-phase release in every configuration. XSched uses the role-specific settings from its official transparent CUDA example: LC priority/threshold/batch `1/16/8`, BE `0/4/2`, Level-1, global HPF. `XSCHED_CUDA_LV3_IMPL` is explicitly unset. gpubpf uses the paper's LC/BE timeslices, 1,000,000/200 us, plus LC-launch-triggered BE preemption. Only the repetition count of the compute loop is tuned in an isolated native calibration to 80 ms (76--84 ms accepted), then frozen before the three-way preflight and full run. No memory oversubscription.
- Split or leakage rule: Not applicable.

## Workloads And Metrics

- Real workloads or tasks: Finite all-SM compute workload derived from XSched's official transparent-scheduling example, configured as two LC and four BE processes, four streams per process, and exactly 50 kernels per stream. Each kernel is calibrated to approximately 80 ms in isolation. This produces exactly 400 LC samples and 800 completed BE kernels per round, retaining the submitted experiment's unit of work.
- Primary metrics: Per-round P99 LC host-submission-to-device-start latency in microseconds and aggregate BE completed kernels/s, matching Fig. 12's launch-latency question. Immediately before every launch the host records `cuptiGetTimestamp`; every CTA leader contributes to an atomic minimum `%globaltimer` entry, and the last block leader records exit only after a block-wide barrier and the cross-block completion count. Every sample must satisfy `submit <= min_entry <= completed_exit`, establishing the common timestamp domain without a policy-active calibration kernel. Thus `entry - submit` includes CPU launch, XSched interception, XSched queueing, and native driver queueing. `exit - submit` is secondary only. BE throughput is exactly 800 divided by the common interval from the parent's BE release timestamp to the latest BE child's post-synchronization completion timestamp, all on `CLOCK_MONOTONIC_RAW`.
- Correctness check or ground truth: Every process must reach the start barrier without launching a kernel, create four streams, finish exactly 200 tasks, report a valid entry/exit pair and full block-completion count for every kernel, and exit zero. After copying the full sink to the host, the harness recomputes the 32 lane-specific recurrences and requires every output value to equal its expected lane value. XSched runs must show six processes, 24 unique audited XQueues at Level-1, their actual role-specific threshold/batch values, server logs for 8 priority-1 and 16 priority-0 queues, and at least one successful BE suspend and resume. Since no kernel runs before release, transition counts can only come from measurement. The exact small XSched source diff is compared with the reviewed patch, while source and runtime files use ordinary path/size/time metadata. gpubpf runs require at least six timeslice modifications, nonzero successful post-release preemptions, and zero errors. A run with no engagement is invalid.
- Repetitions, seeds, and uncertainty: Ten randomized, complete blocks after one full-path preflight. Within each block the order of native, XSched, and gpubpf is shuffled with seed 1797. Report per-round results, medians, fixed-seed 10,000-draw bootstrap 95% CIs, failure counts, and paired differences. BE non-inferiority is predeclared as a lower CI bound no worse than -5% relative throughput; it is not inferred after seeing results.
- Cost estimate: Set from the real finite preflight after the GPU becomes idle; target less than five minutes per six-process round.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| isolated-lc | control | one LC process | native CUDA | 3 | Establish uncontended latency |
| isolated-be | control | one BE process | native CUDA | 3 | Establish uncontended throughput |
| multitenant | control | 2 LC + 4 BE | native CUDA | 10 | Contention denominator |
| multitenant | baseline | same | XSched HPF, Level-1 | 10 | Public research-system H2H |
| multitenant | proposed | same | gpubpf priority preemption | 10 | Tests driver policy advantage |

## Execution

- Required stack before any runtime: use the same custom gpubpf-capable NVIDIA 575.57.08 driver on Linux 6.15.11 for all three configurations, expose `struct nv_gpu_sched_ops` and `bpf_nv_gpu_preempt_tsg` in `/sys/kernel/btf/nvidia`, provide non-interactive root permission for BPF loading, and leave the RTX 5090 idle. The runner refuses any mismatch and never kills unrelated processes. Every live phase holds the shared GPU and struct-ops leases and records per-cell before/after kernel, Xid, UVM, GPU-idle, 400 W power-limit, and struct-ops checks.
- Authoritative build: `python3 workloads/xsched/run_xsched_rq3.py build` from `gpu_ext/`. This applies the reviewed patch to an unmodified pinned XSched checkout, or verifies an already patched checkout byte-for-byte; any other diff is rejected. It then performs a clean CUDA rebuild using `/usr/bin/g++`, builds the `sm_120` harness, and rebuilds both gpubpf loaders from current source. The standalone harness command is `make -C workloads/xsched`.
- Admission: `python3 workloads/xsched/run_xsched_rq3.py admission`. It records driver, live compute processes, required BTF types/kfuncs, commits, the exact small XSched diff, and ordinary source/binary file metadata. A failed admission exits 2 and creates no run.
- Isolated calibration: `python3 workloads/xsched/run_xsched_rq3.py calibrate --output workloads/xsched/raw/calibration-<id> --reps 1000000`. Copy the emitted `frozen_reps` value verbatim; no policy or full result is examined while tuning.
- Real preflight: `python3 workloads/xsched/run_xsched_rq3.py preflight --output workloads/xsched/raw/preflight-<id> --reps <frozen_reps>`. It runs native, XSched Level-1, and gpubpf with two tasks per stream and requires all semantic-output, timestamp, policy, priority, and transition checks.
- Full run: `python3 workloads/xsched/run_xsched_rq3.py full --output workloads/xsched/raw/full-<id> --reps <frozen_reps>`. It executes ten seeded randomized three-configuration blocks with 50 tasks per stream.
- Analysis-only replay: `python3 workloads/xsched/run_xsched_rq3.py analyze --output workloads/xsched/raw/full-<id>`. No GPU or policy process is started.
- Cleanup: the runner starts each child in a private process group and signals only those recorded process groups (`SIGINT`, then `SIGTERM` on timeout). There is no `pkill`, global cleanup script, driver reload, or deletion of another run.
- Two-phase arrival: after all six processes report initialization, the parent records the common BE release time and releases all four BE processes. Each BE process submits task 0 on all streams and reports `running` only after stream-0 task-0 has entered the device and has not exited. After all four recorded running events, the parent waits a fixed 5 ms and records/releases both LC processes. The identical condition and delay apply to native, XSched, and gpubpf.
- CPU allocation: require at least ten allowed cores. XSched's server is pinned to slot 0 and gpubpf's two loaders to slots 0--1. Every workload process receives the identical shared eight-core mask spanning slots 2--9 in native, XSched, and gpubpf, so XSched's four launch-worker threads do not contend on one application core.
- Full completion rule: All controls finish and all three multi-tenant configurations have ten valid rounds on the same RTX 5090 and driver stack. Partial prefixes and non-engaged policies are incomplete, not results.
- Raw-result path: `workloads/xsched/raw/<timestamp>/`.
- Checkpoint or recovery approach: Preserve per-process stdout/stderr, server/policy logs, admission observations, release timestamps, and a result JSON after every round. A retry uses a new output directory; a nonempty output path is rejected by `mkdir(..., exist_ok=False)`.

## Interpretation

- Positive result: gpubpf lowers paired LC P99 while keeping BE throughput within the predeclared tolerance/CI.
- Negative or contradictory result: XSched matches or wins one or both primary metrics; report that boundary without averaging metrics into a single score.
- Mixed or inconclusive result: One system improves LC latency but materially reduces BE throughput, variability crosses both orderings, or engagement cannot be established.
- Frozen decision rule against XSched: **positive** only if the upper 95% CI of paired `gpubpf - XSched` LC launch-latency difference is below zero and the lower 95% CI of paired relative BE-throughput change is at least -5%. **Mixed** if LC improvement is established but the BE CI upper bound is below -5%. **Negative** if the LC CI lower bound is nonnegative, or BE inferiority is established without established LC improvement. All remaining combinations are **inconclusive**. The categories are mutually exclusive.
- Target paper figure or table: RQ3 scheduling subsection / Fig. 12 companion table or grouped plot.

## Reproducibility Notes

- Software and data versions: Record gpu_ext/bpftime commits and dirty state, XSched and all submodule commits, driver, CUDA toolkit, compiler, GPU clocks/power/temperature, and ordinary harness file metadata.
- Config and seed notes: Preserve exact process roles, streams, tasks, work repetitions, queue threshold/batch size, policy/timeslice settings, CPU affinity, launch order, and seed 1797.
- Known deviations: The public XSched implementation supplies only Level-1 on sm_120; the result must not be described as reproducing the XSched paper's Level-3 numbers. The machine has rebooted to Linux 6.15.11 and driver 575.57.08. The parent experiment coordinator is activating the matching custom gpubpf modules; current module readiness is determined from live BTF/admission, never assumed from this status prose.
