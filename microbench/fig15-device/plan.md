# Experiment plan: RTX 5090 device-map placement

## Research question and hypothesis

- Paper RQ: "What is the overhead of gpubpf's core mechanisms, including its
  observability capabilities and device-side extensions?"
- One tested hypothesis: for the current scalar per-thread gpubpf runtime on an
  RTX 5090, a device-resident array has lower callback-path latency than an
  otherwise operation-matched directly host-mapped array, for both lookup and
  update.
- Planned role: supporting mechanism evidence. This does not test warp
  aggregation, the SIMT verifier, policy performance, or full applications.

## Paper-value admission

The active Fig. 15(b) motivates hierarchical maps with an unmatched comparison
against a serialized helper RPC. The strongest runnable alternative placement
now in the runtime is `BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP`, which puts the same
array abstraction in mapped host memory. A prospective comparison can either
support a bounded locality claim, reveal that the old 6000x number was mostly
RPC protocol cost, or show no reliable placement difference. Each outcome
changes the paper text, and no retained raw data answer this matched question.

A new general observability workload would add breadth but would not repair the
specific evidence/label mismatch in the active map panel. The map experiment is
therefore the highest-value executable repair. The independent warp/eGPU
question is rejected from this experiment because no claim-matched
warp-aggregation implementation or official eGPU comparator is currently
available; `legacy-evidence-audit.md` gives its STOP rule.

## Expected and alternative outcomes

- Expected: device-resident lookup and update are faster than direct mapped
  host-memory versions because the latter traverse the discrete-GPU host link.
- Strongest competing explanation: caching, one-warp concurrency, fences, or
  trampoline cost dominates, leaving little or inconsistent placement effect.
- Contradiction: the multiplicity-adjusted interval for either operation lies
  wholly at or below a 1.0 host/device latency ratio.
- A positive result bounds this one-warp current-runtime mechanism. A negative
  or mixed result narrows the map-locality motivation; it does not challenge
  gpubpf's safety or policy expressibility thesis.

## Implementations and comparisons

All attached arms use one BPF object, the same explicit target hook, the same
current per-thread PTX pass, and maps with 32 `u64` entries. Programs differ
only in the selected map and operation.

- Proposed method: `BPF_MAP_TYPE_GPU_ARRAY_MAP` (1503), allocated with
  `cuMemAlloc` and reached by the device fast path.
- Main baseline: `BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP` (1513), directly mapped host
  memory. This is the strongest placement alternative and receives the same
  key, value, operation, callback count, and output sink.
- Secondary diagnostic: standard `BPF_MAP_TYPE_ARRAY`, reached by the legacy
  serialized host-helper RPC. It explains, but cannot stand in for, the main
  host-memory comparison.
- Controls: native same binary and an attached no-op program separate base
  kernel/trampoline cost from map work. They are not competing map systems.

The runtime must be the existing RTX 5090 performance build with CUDA attach
and LLVM JIT enabled and userspace verification disabled. Consequently this
experiment provides no verifier-enforcement evidence.

## Workload and correctness

The target is one block of 32 threads. Every thread executes one explicit hook
and writes a deterministic CUDA output. A CUDA event encloses one batch of
identical launches; loader setup, PTX compilation, map initialization, and map
readback are outside the event interval.

- Update: lane `i` writes a deterministic nonzero value to key `i`. The loader
  reads all 32 keys after the application finishes.
- Lookup: the loader initializes key `i` with a deterministic nonzero value;
  lane `i` looks it up and writes the returned value to the same device-resident
  observation map in every lookup arm. The loader reads all 32 observation
  keys. This common sink makes lookup results externally checkable.
- Native/no-op: the CUDA output must still match exactly. The no-op arm also
  requires exact target-stub replacement, patched-module load, and successful
  attach records.

Any CUDA error, nonzero process status, wrong output, missing transformation,
wrong map value, unknown map record, surviving owned process, or unreclaimed
private shared-memory segment invalidates that cell. Performance never decides
validity.

## Timing, repetitions, and metrics

- Preflight: one eight-arm block, one warmup and two timed launches. It proves
  only that every real path and readback works.
- Full run: exactly 16 paired blocks. Each arm starts in a fresh process. The
  eight arm positions are balanced twice using a seed-1797 randomized cyclic
  schedule, with the second cycle reversed. Each process performs 8 warmups
  then one CUDA-event interval over 64 timed launches.
- No performance-triggered retries, optional stopping, subset selection, or
  pooling with the 2025 Markdown summaries is allowed.

For update and lookup separately, the primary effect is the within-block log
latency ratio `log(host-mapped/device-resident)`. Report its median, exponentiated
ratio, and a paired-bootstrap 97.5% percentile interval (10,000 samples,
seed 1797 plus a fixed operation offset). The two 97.5% intervals give at least
95% family-wise coverage by Bonferroni. Also report paired absolute
microseconds-per-launch differences, device/no-op increments, RPC/device ratios,
and all raw arm medians descriptively.

- Supported: both main-baseline ratio intervals lie wholly above 1.0.
- Contradicted: either interval lies wholly at or below 1.0.
- Inconclusive: otherwise, including one positive operation and one unresolved
  operation.

There is no predeclared 6000x target. The measured ratios, including an
unfavorable or much smaller result, determine the replacement wording.

## Execution and artifacts

From this directory, after acquiring the existing GPU/struct-ops leases:

```sh
make BPFTIME_ROOT=/home/yunwei37/workspace/gpu/bpftime-table1-575 \
  BPFTIME_BUILD=/home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575
python3 run_map_tier.py --phase preflight --output raw/map-tier-preflight-575-01
python3 run_map_tier.py --phase full --output raw/map-tier-full-575-01
python3 analyze_map_tier.py raw/map-tier-full-575-01
```

The runner preserves the fixed `schedule.tsv`, build/environment records, and
per-arm application, loader, and agent logs. The analyzer ignores any derived
runner summary, reopens every raw log, recomputes correctness/engagement and
the paired estimates, and writes `analysis.md` plus `analysis.tsv`.

The full run is complete only after all 128 planned arm processes (16 blocks x
8 arms) and independent replay pass. A systematic implementation defect may be
fixed only before restarting the entire affected campaign under this same
design.

## Paper decision

- Positive: replace Fig. 15(b) with operation-matched ratios and explicitly
  distinguish direct host mapping from serialized CPU-map RPC.
- Contradicted or mixed: retain the measurements, remove the generic 6000x
  claim, and state the observed operation-specific boundary.
- Infrastructure failure: call the run incomplete and remove the quantitative
  panel unless the fixed path can be completed; never fall back to the legacy
  aggregate as proof.

No outcome licenses a warp-leader, block-count-independent, verifier-overhead,
or application-level performance claim.
