# POD full-575-01: independent raw audit

Result: **PASS for the frozen operator study; not a full-system or strict
verifier result.** The coordinator had exited before this audit read
[full-575-01](raw/full-575-01/manifest.json). All checks ran on CPU 17;
there was no GPU execution, build, tuning, raw-data edit or sample removal.

## Scope and method

Independently read all 25 execution records/logs and all 250 saved operator
cells; do not infer completion merely from the manifest. The audit rebuilt
the matrix/order, numerical summary consistency, actual CTA selection rule,
logical coverage and statistics directly from saved records. It did not call
the result analyzer to obtain those values. Existing read-only safety
validators were reused for the raw telemetry/safety interpretation, not
claimed as an independently implemented safety subsystem.

After the independent calculation, all **100 five-block means, 180 paired
geometric means/95% intervals and 900 individual block ratios** matched the
saved [analysis.json](raw/full-575-01/analysis.json) exactly at stored
precision (largest absolute difference zero). Per-shape values and every
comparison are in the [results report](results-575-20260903.md).

## Completeness and samples

- Exactly five blocks × five arms × ten unique shapes; 25 successful arm
  processes and 250 cells. The full manifest is complete, non-excluded,
  protocol `pod-fp16-upstream-match-v2`; no process reports an error.
- Rebuilt arm order from seed 20260903 and shape order from that seed plus
  the block; matched every report, log and actual block directory. Frozen
  shapes, FP16 dtype, input seeds, `fused_params=15`, ten warmups and
  numerical tolerances agree.
- Each cell has exactly 100 finite positive CUDA-event samples and 100
  finite positive host-wall samples: **25,000 per metric**. Recalculated
  arithmetic cell means from unrounded samples using `math.fsum`; all
  stored means and rounded log means agree.
- Recomputed all nine predeclared comparisons for both metrics and ten
  shapes using five block ratios, 10,000 whole-block bootstrap draws
  (seed 20260905), and linear percentile interpolation. No sample or block
  was excluded. Intervals are pointwise, not equivalence tests or adjusted
  simultaneous intervals.

## Numerical evidence and limitations

All 250 saved hard full-output FP16-arm-versus-original-FA validation
attestations pass with `atol=1e-3, rtol=1e-5`. The maximum recorded absolute
difference is **0.00006103515625**. This is not the zero from the one-shape
preflight. Full output tensors are not saved; this audit checks the real
execution attestations, not a new numerical GPU run.

| FA-versus-FP32 characterization | Prefill | Decode |
| --- | ---: | ---: |
| Complete phase scans | 250 | 250 |
| Total checked elements, including repeated inputs | 8,388,608,000 | 104,857,600 |
| Threshold exceedances, including repeats | 150 | 0 |
| Scans with an exceedance | 100 | 0 |
| Maximum absolute error | 0.0013279914855957031 | 0.000040277838706970215 |

Full scan shape/finite/mask/threshold fields and checkpoint summaries agree
in every cell. For identical model/batch seeds, reference statistics agree
across all arms/blocks. Per scan, Llama / 32 has one excess, Yi / 32 three,
Yi / 96 one and Yi / 192 one; these four shapes repeat 25 times each.
These scans characterize original FP16 FA against FP32; they do **not**
constitute a full-FP32 pass or a BPF-specific numerical failure.

All 100 saved worst query/head diagnostic directories have matching
model/batch/phase/protocol metadata and actual finite Q/K/V and output arrays,
with recorded file sizes and expected dtypes/shapes. Independently
recomputed attention in CPU FP64 directly from those Q/K/V arrays, without
calling the benchmark's saved-row recomputation helper. The largest
FP32-reference/FP64 difference is 5.426491115345655e-7. Every saved FP16 FA
row still has a threshold excess against FP64; maximum actual FP16-FA/FP64
error is 0.001327910378291719. This rechecks 100 saved rows, not complete
shapes in FP64 or model accuracy.

## Actual device selection and launch bridge

Traversed every saved diagnostic CTA context in all 150 POD cells,
independently validating the ABI/status, counter bounds and common device
pointer, actual SM ID, proportional/alternating first-operation rule,
SM-local ticket sequence, global atomic claims and exhaustion fallback.
Expanded slot/sub-CTA mappings, including tail handling, cover every logical
prefill/decode block exactly once. Nondeterministic assignment differences
between arms are retained, not rejected as a mismatch.

| Diagnostic evidence summed over 50 shape cells per arm | Inline | CUDA adapter | Device BPF |
| --- | ---: | ---: | ---: |
| Actual CTA contexts | 125,280 | 125,280 | 125,280 |
| Expanded prefill logical blocks | 102,400 | 102,400 | 102,400 |
| Expanded decode logical blocks | 43,520 | 43,520 | 43,520 |
| Valid exhausted-operation fallbacks | 5,970 | 5,992 | 5,989 |
| Actual selection engine | 1 | 1 | 2 |
| Checked bridge launches | N/A | 5,550 | 5,550 |

The different fallback totals follow nondeterministic scheduling; all
individual claims and mappings remain valid. BPF engine 2 is the actual
device selector, not a host-JIT, host replay or native C fallback. Detailed
contexts come from each cell's diagnostic launch; timed launches retain
launch-level count/error checks, not a second saved context array per launch.

Both adapter arms have 111 bridge calls per cell: one diagnostic, ten warmups,
100 timed launches. CUDA control also records 111 redirected launches.
All 150 POD diagnostics use 81,920 shared-memory bytes; every adapter bridge
records an opt-in readback covering the actual request within the device
limit. The inline and non-fused controls retain their original launch path.

## Execution, ownership and safety

- All 25 before/after runtime inventories agree with the 19-file frozen
  manifest. Actual command and injected environment paths bind lexically to
  that inventory and its exact PTX packet directory; no current binary is
  substituted to validate an archived command.
- Worker launch is CPU 8–15 `taskset` → `env` → the recorded Python/benchmark.
  Agent preload occurs only in the target environment, not the wrapper.
  The coordinator retains CPU 16 for telemetry.
- All five BPF loaders report one six-kernel READY and one orderly CLOSED.
  The five private segment names are unique; each execution records removal,
  and each exact segment is absent after completion. The client exits
  before owned loader/segment reclamation. No global process/SHM cleanup.
- All 25 safety windows are positive and non-overlapping in recorded order,
  spanning 11:05:35.821830–11:35:24.358002 UTC. Before/after checks report driver
  575.57.08, fixed 400 W power cap, no abnormal kernel/GPU events, no remaining
  compute application or struct-ops attachment. The raw continuous telemetry
  revalidates all stored summaries: **8,758 samples**, peak 63°C and
  19,880 MiB memory. Fixed-cap activity is not mistaken for a thermal fault.
- The Python interpreter path is fixed, but its binary/version was **not**
  among the 19 runtime inventory files. This audit does not invent that
  missing identity evidence.

## Adverse results and claim boundary

Nine of ten shapes have a small measurable device-BPF latency increase
versus the matched CUDA adapter (CUDA-event +0.51–1.18%) and inline selector
(+0.59–1.52%). Llama / 192 has small decreases, not a universal BPF advantage.
The host-wall metric agrees in direction. Three shapes lose to FA
two-stream overlap. Fusion gains relative to FA belong to the existing POD
algorithm/operator, not a new BPF policy; all comparisons are retained.

Median whole-client wall time is 295.63 s for BPF versus 12.01–14.01 s for
other arms. This includes initialization, FP32 scans, diagnostics, warmups,
ten shapes and exit; it is not isolated initialization or operator latency.

The runtime has strict GPU verification **OFF**. This audit supports actual
device execution and recorded numerical/coverage checks, not strict
verifier admission, general safety, arbitrary binary compatibility,
full-model accuracy or full Sarathi/POD serving-system reproduction.
Previous failed/interrupted attempts and the closed preflight remain
unchanged and excluded from formal statistics.
