# Expert Buffering analogue: historical timing raw audit

Date: 2026-09-03. Scope: the five retained four-arm timing blocks, their failed
attempt, and small supporting files. No GPU run, rebuild, paper edit, Git
operation, old-result modification or large-trace scan was performed.

## Outcome

The arithmetic in `timing-results.json` is reproducible from the 20 cell
results and their saved responses. The ten O/E hot-activation metrics and
engagement deltas also reconcile with the original small policy logs and
snapshots. This is **not an unconditional validation of the full historical
protocol**: the EOS protocol changed after block 1; request payloads, exact
clock alignment and independent post-run restoration records are missing;
the five large F route traces were deliberately not scanned in this audit.

- Run status: 20 completed historical cells; numerical/engagement checks below
  pass, but a homogeneous five-block protocol and all lifecycle claims remain
  incompletely evidenced.
- Tested hypothesis: no demonstrated reduction in repeated hot activations.
  The observed throughput change under protection is slightly negative.
- Research value: supporting evidence for a page-granular policy analogue.
- Paper impact: a mechanism/workload boundary, not reproduction of the original
  expert-atomic algorithm, an original-implementation comparison, or a general
  BPF advantage.
- Next decision: retain the negative historical result with these limitations;
  do not promote it to a fully reproduced Expert Buffering baseline.

The [read-only audit](audit_timing_raw.py) does not import the launchers or
`analyze_timing.py`. It recomputes the metrics from raw files and refuses any
individual content read above 2,000,000 bytes. From this workload directory:

```sh
taskset -c 17 python3 -B audit_timing_raw.py
taskset -c 17 python3 -B audit_timing_raw.py --inventory
```

The second command prints every retained timing-file path and size; neither
command writes a report or changes experiment state. The completed audit took
less than one second on CPU 17. Its successful exit covers the stated checks,
not the limitations below.

## What independently reconciles

All five `status.json` records, recorded orders and prompt filenames match
`timing-schedule.json`. Every completed cell has one saved warm-up response,
16 saved untimed responses and eight measured responses: **500 saved responses**
in the 20 completed cells. All have valid UTF-8 text, length termination,
512 reported prompt tokens, 64 reported completion tokens, 576 total tokens,
512 server timing prompt tokens, 64 predicted tokens and zero cached tokens.

The **160 measured responses / 10,240 reported output tokens** join exactly to
the corresponding `result.json` text and request order. Their monotonic-raw
start/end timestamps are ordered, nonoverlapping, inside the cell window, and
reproduce the saved E2E durations. All 20 throughput values independently equal
`512 / ((block_end_ns - block_start_ns) / 1e9)` and match both block status and
the tracked summary. All saved server logs contain 25 successful completion
requests, cleanup text, and no checked CUDA/load/OOM/traceback error pattern.

This is API/workload-shape verification, not a logits or tensor oracle. The
responses do not contain output token IDs. Across the measured set, O/U and
E/O text matches in all 40 matched prompt instances; F/U matches in 29/40.
The untimed repeated passes match only 19/40 for each U/O/E arm and 22/40 for F.
Thus the retained texts cannot support a general determinism or numerical
equivalence claim.

For each O/E cell, both complete 3,234-index hot-block snapshots match the class
table and the two saved result maps, with no duplicate/missing index or counter
regression. Recomputing `2 MiB * sum(max(0, after-before-1))` exactly reproduces
all ten repeated-hot-activation results. Full activation bytes also reconcile.
The immediately preceding raw `policy_stats` records reproduce every recorded
policy delta. All policy logs have one matching ready record with real map/
program IDs, positive required classification/decision counters, zero cold-head
placements and zero setter failures. O has no reorder decisions; E has positive
hot-tail, cold-native and hot-access-tail decisions. These counters establish
executed callbacks and accepted setter requests, not measured physical transfer
bytes or an independent count of driver-committed list moves.

All 8,080 telemetry rows reproduce per-cell sample counts, memory/temperature
peaks, power averages and SM-clock extrema; the three recorded thermal/brake
flags are inactive. All 160 server response creation seconds are compatible
with those telemetry intervals when the CSV wall-clock timestamps are interpreted
as America/Vancouver time, with one-second timestamp granularity. This is a
consistency check, not a saved wall-clock/monotonic-clock calibration.

The 15 small U/O/E layout traces independently contain 216 registrations each
(108 weights and 108 biases), one observed process, all 36 layers, and a zero-
drop final record. F's saved diagnostic reports consistently describe layers
0–31, 1,105 graphs/layer and zero incomplete graphs; their route counts match
the cell and aggregate summaries. **The original F trace events were not
recounted**, so this latter check is report consistency, not independent route
engagement verification during this audit. The runner stops F tracing before
measurement; its route evidence comes from warm-up/untimed requests.

## Recomputed effects

These retain all five historical blocks, including block 1. Intervals enumerate
all 3,125 size-five paired resamples with the same percentile definition as the
recorded analysis. Every aggregate estimate and interval in the tracked summary
matches the independent computation.

| Contrast | Throughput effect | Paired 95% interval |
| --- | ---: | ---: |
| O/U: observation mechanism / plain UVM | +0.5390% | −0.0676% to +1.6858% |
| E/O: protection / observation | −0.0730% | −0.1130% to −0.0355% |
| F/U: framework context / plain UVM | +25.5786% | +14.2021% to +39.6824% |

Repeated-hot E minus O averages **+10,066,329.6 bytes**, with interval
**−21,390,950.4 to +35,232,153.6 bytes**. Its secondary E/O ratio is 1.00011344,
with interval 0.99970392–1.00042647. Neither establishes improvement in the
intended allocation proxy. These are repeated 2 MiB activation counts, not
router misses, cache hits, completed evictions or PCIe traffic.

Because the EOS setting changed, an explicitly post-hoc sensitivity calculation
using only blocks 2–5 gives O/U +0.7010%, E/O −0.0565%, and F/U +20.4643%.
It does not reverse the qualitative O/E interpretation. It is four blocks,
not an invented fifth valid block or a replacement preregistered result.

The actual denominator is the runner's outer eight-request interval, including
request construction, JSON checking and synchronous raw-file writes. Across
cells it exceeds summed client HTTP durations by 0.0876–0.1897 seconds. These
small effects therefore do not isolate callback/JIT instruction cost. Requests
were nonstreaming; no independent first-token/first-visible-text timestamp was
saved for this timing campaign.

## Protocol and attribution corrections

1. **EOS change and incorrect failure description.** `timing-progress.md`
   states that block 2 failed on the sixth untimed request, prompt 3. However,
   [that saved sixth response](raw/timing/block-02-failed-attempt-01/gpubpf_observe/untimed-pass-1-request-06-prompt-3.json)
   reports 512+64 tokens and length termination at epoch second 1788247875.
   The [failed status](raw/timing/block-02-failed-attempt-01/status.json) records
   a later one-token stop at 1788247882; the server log confirms a subsequent
   one-token request. Six valid untimed responses are retained, with no seventh
   response or measured result. The next frozen slot is request 7 / prompt 7,
   but its submitted payload is not saved, so that identity is inferred from
   the schedule rather than independently recorded. The documented repair
   enables `ignore_eos` for blocks 2–5; block 1 was not rerun. Preserve both the
   failed attempt and the protocol deviation. Do not call this an unchanged
   five-block request protocol.
2. **Submitted payloads are missing.** The current source sets `ignore_eos=True`,
   but neither launch records nor response JSON store that option or the input
   token array. Current source cannot retrospectively prove every historical
   request option or exact prompt identity. Names, schedule and token counts
   associate requests, but do not replace saved inputs.
3. **Correct execution-path comparison.** U/O/E have identical saved server
   argv and environment within every block. The runner loads the same custom
   610 UVM module once before those cells; native UVM allocation/reclamation
   remains the actuator. O/E use Linux kernel `struct_ops` BPF callbacks and
   the typed PMM setter, not host uBPF JIT or device BPF. O/U measures the added
   observation path; E/O changes policy on that common path. No original
   user-space Expert Buffering implementation was run. Admission records show
   RTX 5090/610.43.02, but a complete loaded-module inventory is not saved per
   cell; do not identify these as the newer 575 campaign.
4. **F is context, not a matched actuator.** F changes the allocator/transfer
   path using `--n-cpu-moe 32`, removes the UVM override and enables the existing
   route marker. Its peak is 9,541 MiB versus 32,147 MiB for U/O/E. These are
   about 9.32 and 31.39 GiB, not the progress note's approximate 9.5/32.1 GiB.
   F/U cannot isolate the BPF mechanism or the protected-page policy.
5. **Only a profile-guided analogue.** Frozen top-ten expert selections become
   page/block classifications; hot/shared pages receive `USED/TAIL` requests,
   while cold pages preserve native ordering. This does not implement the
   original current-batch, inactive-first, expert-atomic buffer or whole-expert
   transfer overlap. Byte-identical input policies in native/BPF executors were
   not compared here.
6. **Clock and lifecycle evidence limits.** Policy events have no timestamps;
   snapshot ordinals and log order associate them with the measurement by the
   inspected runner, but no independent subsecond clock join is possible.
   Telemetry lacks a saved timezone/monotonic anchor and continuous foreign-
   process inventory, and does not record every possible throttle reason or
   power-limit setting. `post_block_stock_uvm` occurs only in the tracked
   summary, not a dedicated raw post-restore observation. The runner writes
   `status=passed` before `restore_stock_uvm()` in `finally`; that status alone
   cannot prove restoration succeeded. Server cleanup text and BPF-ready IDs
   are not recorded process exit codes or a final no-policy snapshot.

## Necessary release inventory

Retain all **685 timing files / 447,779,474 bytes**, including the failed
attempt. The table gives exact path patterns/counts/total sizes at this audit;
`--inventory` prints the complete individual list without reading trace bodies.
Do not publish only `timing-results.json`: it cannot independently reconstruct
the responses, snapshots or telemetry checks.

| Paths under `raw/timing/` | Files | Total bytes |
| --- | ---: | ---: |
| `block-01`…`block-05/admission.json` | 5 | 700 |
| `block-01`…`block-05/status.json` | 5 | 2,397 |
| completed cells' `launch.json` | 20 | 21,235 |
| completed cells' `result.json` | 20 | 1,302,133 |
| completed cells' `measured-request-*.json` | 160 | 161,456 |
| completed cells' `untimed-pass-*.json` | 320 | 323,029 |
| completed cells' `warmup.json` | 20 | 20,957 |
| completed cells' `gpu-telemetry.csv` | 20 | 683,428 |
| completed cells' `server.log` | 20 | 1,280,533 |
| completed cells' `class-table.txt` | 20 | 4,453,156 |
| completed cells' `layout-report.json` | 20 | 6,910 |
| O/E cells' `policy.jsonl` | 10 | 5,882,467 |
| completed cells' `trace.jsonl` | 20 | 431,890,611 |
| F cells' `route-diagnostic-report.json` | 5 | 1,400,699 |
| F cells' `route-diagnostic-hot-set.txt` | 5 | 9,608 |
| entire `block-02-failed-attempt-01/` | 15 | 340,155 |

The five large, **stat-only** `llama_ncmoe32/trace.jsonl` files are necessary for
independently reproducing the untimed route-engagement claim:

| Block | Bytes |
| --- | ---: |
| 01 | 86,230,716 |
| 02 | 86,266,539 |
| 03 | 86,227,872 |
| 04 | 86,231,724 |
| 05 | 86,240,295 |
| **Total** | **431,197,146** |

Also retain the existing plan/deviation notes, both runners, policy/marker
sources, `timing-schedule.json` (847 bytes), `timing-results.json` (11,566 bytes),
the common `../moe-infinity/prompts.json` (140,263 bytes),
`calibration-prompts.json` (133,571 bytes), `calibration-hot-set.txt` (2,156 bytes),
and the calibration run/layout/route reports (1,594 / 322 / 296,524 bytes).
Calibration raw traces and numerical correctness/preflight records remain
separate required inputs to their own claims; this timing audit did not reread
large calibration traces or certify those earlier gates.

No old CSV, JSON, JSONL, source summary or failure was changed. This audit does
not lower the original correctness/engagement requirements or fill a missing
record with a reconstructed one.
