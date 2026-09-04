# Experiment Plan: RQ4 strict-admitted uniform map placement

## Research Question

- RQ exactly as written in the paper: "What is the overhead of gpubpf's core
  mechanisms and observability capabilities?"
- Specific uncertainty tested here: whether the current STRICT SIMT verifier
  admits a warp-uniform array-map operation through the real CUDA attach path,
  and whether device-global placement has lower callback-path latency than the
  otherwise operation-matched host-mapped placement.
- Why the answer matters: the completed per-lane placement experiment used
  WARNING mode and its map programs are rejected by STRICT. This separate
  experiment can establish a strict-admitted positive boundary without
  relabelling or changing that earlier workload.

## Paper-Value Admission

- Planned role: supporting.
- Largest credible paper story this experiment could unlock: gpubpf's current
  verifier and runtime compose end to end for uniform map policies, while map
  placement still has an operation-specific cost under strict admission.
- Strongest reviewer reject argument or load-bearing uncertainty addressed:
  the existing device-map numbers demonstrate runtime capability but no
  verifier-gated map execution.
- Independent evidence added beyond existing runs and published results: real
  STRICT acceptance, JIT, attach, GPU execution, deterministic host readback,
  and paired timing for map types 1503 and 1513 on RTX 5090.
- Why the result is not tautological, already settled, or dominated: the
  CPU-only admission audit does not execute a GPU program, and the completed
  Full6 run used lane-varying keys under WARNING mode.
- Paper decision if positive: report a strict-admissible uniform boundary next
  to the distinct WARNING-mode per-lane result.
- Paper decision if contradictory, mixed, or inconclusive: retain the strict
  execution result but do not claim a placement advantage for unresolved or
  unfavorable operations.
- Best alternative experiment and why this one has higher decision value: a
  new verifier rule for disjoint per-lane accesses could recover the original
  workload, but changes trusted safety logic. The uniform experiment tests the
  current unmodified verifier and directly closes the present evidence gap.

## Expected And Alternative Outcomes

- Current expected answer: all five attached programs are accepted in STRICT;
  direct device-global update and lookup are faster than their host-mapped
  counterparts.
- Strongest competing explanation: fixed trampoline/JIT work and cache effects
  dominate lookup, so only update may show a placement difference.
- Result that would contradict the expectation: a strict rejection, failed
  oracle, or a multiplicity-adjusted host/device interval wholly at or below
  1.0 for either operation.

## Published Precedent And Real Assets

- Closest protocol: the repository's completed operation-matched Full6 map
  placement protocol; CUDA-event elapsed time follows the CUDA runtime timing
  interface used by that microbenchmark.
- Official system/model/data/benchmark/tool and version: bpftime/gpubpf at the
  recorded Git revision, CUDA 12.9, driver 575.57.08, RTX 5090 (sm_120).
- What is reused: the same CUDA kernel, one explicit target hook, event timing,
  loader prime, fresh-process lifecycle, GPU leases, schedule construction,
  cleanup checks, and raw-log replay.
- Necessary deviations or custom glue: a separate five-program BPF object uses
  constant key 0 and constant values, and a dedicated loader/analyzer requires
  STRICT markers. No Full6 source or raw data are changed.

## Comparison

- Proposed system or method: type 1503 CUDA device-global array.
- Main baseline and competing position: type 1513 directly host-mapped array,
  representing the strongest operation-matched alternative placement.
- Why the main baseline needs a matched run instead of citation alone: strict
  admission, one-warp contention, trampoline work, driver, and cache behavior
  are specific to this runtime and device.
- Controls: native kernel and strict-admitted no-op callback. These are timing
  controls, not competing map placements.
- Conclusion if the main baseline matches or wins: no detected device-placement
  advantage for that operation under this uniform one-warp workload.
- Information, tuning, and compute fairness: both placements use one-entry
  arrays, the same key, value, callback count, launch schedule, application,
  source-level operation, sink for lookups, strict runtime, and process budget.

## Workloads And Metrics

- Real workload: one RTX 5090 CUDA block of 32 threads; each scalar callback
  performs the same map operation on key 0. Update lanes write the same nonzero
  value. Lookup lanes read the same initialized value and write it to the same
  device-resident observation map.
- Primary metrics: within-block log latency ratio
  `host-mapped/device-global`, separately for update and lookup, using CUDA
  event microseconds per launch.
- Correctness: exact application output; exactly one BPF program selected;
  one target transformation/module load/attach; exactly one target-PID STRICT
  acceptance with positive verifier timing; expected map descriptors; no
  reject/skip records; final nonzero map readback; full cleanup.
- Repetitions, seed, and uncertainty: 12 randomized balanced blocks, seed 1797;
  paired-median bootstrap with 10,000 resamples and 97.5% intervals for the two
  co-primary operations.
- Cost estimate: 72 fresh processes (12 blocks x 6 arms), 8 warmups and 64
  timed launches per process; expected under 20 minutes.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| main | proposed | uniform update/lookup | device-global map 1503 | 12 blocks | strict device placement effect |
| main | baseline | uniform update/lookup | host-mapped map 1513 | 12 blocks | strongest matched alternative |
| control | control | same kernel | STRICT no-op | 12 blocks | callback/trampoline reference |
| control | control | same kernel | native | 12 blocks | kernel-launch reference |

## Execution

- Authoritative workflow: build the dedicated object/loader, run
  `run_strict_uniform_map.py --phase preflight`, then a new output directory
  with `--phase full`, and replay it with `analyze_strict_uniform_map.py`.
- Real preflight: one complete six-arm block with one warmup and two timed
  launches through the exact strict runtime and object.
- Full completion rule: all 72 scheduled processes exist and independently
  pass environment, correctness, engagement, strict-admission, map-readback,
  and cleanup gates; no retries or optional stopping.
- Raw-result path: `raw/strict-uniform-map-{preflight,full}-575-*`.
- Checkpoint/recovery: raw cells are append-only; a failure invalidates that
  campaign. Fixes use a new directory and restart the affected campaign.

## Interpretation

- Positive result: both 97.5% ratio intervals lie wholly above 1.0.
- Negative or contradictory result: either interval lies wholly at or below
  1.0, or a strict/correctness gate fails.
- Mixed or inconclusive result: otherwise.
- Target paper figure or table: concise RQ4 device-map paragraph; no new panel
  unless it replaces weaker text within the page budget.

## Reproducibility Notes

- Software/data versions: recorded from the actual environment without file
  fingerprints; ordinary Git revisions and file sizes are retained.
- Config/seed notes: `ENABLE_EBPF_VERIFIER=ON`, CUDA attach and LLVM JIT on,
  `BPFTIME_VERIFIER_LEVEL=STRICT`, seed 1797.
- Known deviations: this is a same-key/same-value, scalar-per-thread one-warp
  workload, not the earlier per-lane map workload and not warp-leader
  aggregation. Idempotent readback proves the final effect, not invocation
  cardinality or verifier soundness.
