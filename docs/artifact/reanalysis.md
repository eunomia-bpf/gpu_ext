# Artifact reanalysis scripts

CPU-only, read-only entrypoints that recompute statistics and figures from
existing completed raw evidence. No GPU, no builds, no new experiments, no
process control, no gates, no hashes, no `docs/paper` or raw-campaign edits.
Start with the repository's [artifact guide](../../ARTIFACT.md) for the
paper-to-evidence map and the separate runtime reproduction requirements.

All paths resolve relative to the repository root (default: two levels
above the `scripts/artifact/` location, override with `--repo-root`);
No model cache or original machine's temporary directory is needed. The
commands were tested from another working directory; full fresh-machine
GPU workload reproduction is a separate task.

## `scripts/artifact/reanalyze.py`

Recomputes cell metrics and paired statistics for the paper-selected
LMCache write-budget campaign, the two supplemental campaigns, the MoE
paper-v3-575 postboot timing campaign, and the XSched Level-2 matched
native route comparison, from their raw `result.json` records (90
published cell summaries across the five campaigns).

Campaigns and source maps:

- `storage` - LMCache GDS write-budget serving, five-block-02
  (25 cells: 5 blocks x
  fifo/native10/bpf10/native200/bpf200):
  `workloads/lmcache-disk/raw/gds-write-budget-575-20260907-five-block-02/
  block-00..04/position-0..4-<arm>/result.json`. The arm is read from the
  position directory name; native10/bpf10 and native200/bpf200 are the
  same policy programs with 10 ms and 200 ms cumulative write-delay
  budgets. Metrics: scheduled-arrival
  `read_scheduled_offer_to_completion_p99_ms`/`_p50_ms`,
  `write_completion_throughput_mib_s` and
  `total_storage_bandwidth_mib_s` from each cell's `metrics`; the
  budget-exhausted write count is retained from the campaign's
  `paired-analysis.json` row for the same block/position. All five arms
  are retained, including native10/bpf10 and the mixed-sign pairs
  (bpf200/native200 has two adverse blocks). Cross-checks per-arm medians
  (all four metrics) and per-pair p99/write-throughput medians against
  the campaign's existing `paired-analysis.json` (match/MISMATCH lines).
- `lm` - LMCache disk physical-reclaim serving, pXYN4F:
  `workloads/lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F/
  cells/block-*/position-*/result.json` (15 cells: 5 blocks x
  stock/native/bpf). Metric: warm generation throughput
  `warm_phase.output_tokens_per_s` from each cell's existing summary,
  plus `warm_ttft_median_ms`. Cross-checks medians and paired medians
  against the campaign's existing `analysis.json` (match/MISMATCH lines).
- `xsched` - XSched Level-2 device policy pair, buBjns (15 cells:
  5 global blocks x baseline/native_port/bpf_port):
  block 0 `native_port` is the reused sample from
  `workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5/
  cells/block-01-native_port/result.json` (publication/coordination timing
  gap retained as a limitation); block 0 `baseline`/`bpf_port` from
  `workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns/
  block0-missing/block-01-*/result.json`; global blocks 1..4 from
  `.../buBjns/remaining/block-0{1..4}-*/result.json`. Metrics:
  `lc_service.p99_us` converted to ms and `be_kernels_per_s`;
   `be_service.p99_us` and the host elapsed times are retained in the
   per-cell table. Cross-checks the per-arm `lc_service_p99_median_us`
   medians against the two existing sub-summaries
   (`block0-missing/summary.json`, `remaining/summary.json`).
- `xsched-route` - XSched Level-2 matched native route comparison,
  level2-native-route-20260910-082400 (20 cells: 5 blocks x
  baseline/l1_native/l2_cuxtra/l2_bpfhost):
  `workloads/xsched/raw/level2-native-route-20260910-082400/
  block-0{1..5}-<arm>/result.json`. This is the original-actuator cuXtra
  sm_120 campaign, separate from the `xsched` policy-port campaign
  (`xsched-route` runs the original-actuator cuXtra route on the
  reproducible nine-patch source install; the `xsched` campaign runs the
  measured reference's executable-logic policy port). Arms: `baseline`
  has no shim and no server; `l1_native` is the same HAL with upstream
  Level-1 queue actuation; `l2_cuxtra` and `l2_bpfhost` are the original
  cuXtra/SASS Level-2 actuator (captured SASS guardian and resume blobs,
  binary surgery) with native and BPF host HPF respectively. Metrics:
  `lc_service.p99_us`/`mean_us` converted to ms and `be_kernels_per_s`;
  `be_service.p99_us` and the host elapsed times are retained in the
  per-cell table. Engagement gates are recorded per arm (level-2 queues
  audited on the cuXtra arms, level-1 on `l1_native`, none on the
  baseline); gate-failed cells are excluded from the medians and listed.
  Cross-checks the per-arm `lc_service_p99_median_us` medians against the
  campaign's `summary.json` (match/MISMATCH lines).
- `moe` - MoE paper-v3-575 postboot timing,
  `workloads/moe-infinity/raw/paper-v3-575/
  timing-849ea75d-02-postboot/` (15 cells: 5 blocks x
  native-off/paper-native/paper-bpf). Source: the top-level
  `block-0{1..5}-attempt-01/result.json` files only; each holds a
  `cells` list with one entry per mode. The per-mode subdirectory
  records (SSE dumps, telemetry, launch/admission files) are not
  read. Primary throughput: `verified_output_tokens/duration_s` over
  the full eight-request window including final drain (512 tokens
  per arm here); TTFT: per-cell `first_text_ttft_median_ms`, first
  visible text, not first model token. Arm roles: native-off is the
  baseline (dispatcher count-cache eviction), paper-native the
  native arm, paper-bpf the BPF arm (userspace bpftime JIT
  selectors) - not the earlier native-UVM or kernel stride-LFU
  campaigns. Statistics: per-arm marginal medians and, for each of
  the three pairs (paper-bpf/paper-native,
  paper-native/native-off, paper-bpf/native-off), the geometric-mean
  ratio `exp(mean(log(candidate/reference)))` per block for
  throughput and TTFT, with per-block ratios retained; blocks with
  error/incomplete status keep their numeric ratios and are listed,
  never filled in or dropped. The paired bootstrap CIs in the
  audited analysis are retained but not recomputed. Cross-checks arm
  medians and all three geometric-mean ratios against the
  campaign's existing `audited-analysis-final.json`
  (`.analysis.modes[mode]`, `.analysis.paired[pair]`,
  `.secondary.first_visible_text_ttft.paired[pair]`;
  match/MISMATCH lines). This is a cell-summary statistical
  reanalysis, not a fresh SSE/correctness audit and not
  original-hardware or full-artifact reproduction.

Statistics vocabulary (kept explicit in the report):

- per-cell value: the raw per-cell metric.
- marginal median: median of a single arm's available numeric values over
  its blocks. Missing values are omitted; incomplete status is reported
  separately and does not automatically exclude an available value.
- paired ratio (per block): `100*(candidate/reference - 1)` for that block.
- median of paired ratios: median over blocks of the per-block paired
  ratios.
- ratio of medians: `100*(median(candidate blocks)/median(reference
  blocks) - 1)`.
- geometric-mean paired ratio (used by `moe`):
  `exp(mean over blocks of log(candidate/reference))`. It is not the
  median of paired ratios; the two differ in general, and only the
  geometric mean is reported for `moe`.

For `lm`, `xsched`, `xsched-route` and `storage`, the median of paired
ratios and ratio of medians are reported separately; `moe` uses the
geometric-mean ratio above.
Pairs use available numeric values for both arms; missing or incomplete
status is reported separately, never filled in. The tool never
stops running jobs or imposes gates.

What this recomputes: statistics only, from the per-cell summary metrics
each runner already recorded in `result.json`. The per-request records
(offer/submitted/completion times) retained in the same files are not
reprocessed, so this is a cell-summary statistical reanalysis, not a
raw-request percentile reconstruction, not a new GPU measurement, and not
evidence for GPU-direct P2P.

Usage:

```
python3 scripts/artifact/reanalyze.py                   # all four campaigns, stdout
python3 scripts/artifact/reanalyze.py --campaign storage  # one campaign
python3 scripts/artifact/reanalyze.py --campaign lm     # another campaign
python3 scripts/artifact/reanalyze.py --campaign moe    # MoE timing campaign
python3 scripts/artifact/reanalyze.py --output PATH     # exclusive creation
```

`--output` creates the report exclusively: an existing file aborts the run
with an error; nothing is deleted or overwritten, and the path is rejected
under `docs/paper` or inside any raw campaign directory.

Verified run (`--campaign all`, 2026-09-09): all 70 cells present
(25 + 15 + 15 + 15), no MISMATCH lines against the existing
summaries. Storage
scheduled-arrival read-p99 medians (ms)
fifo/native10/bpf10/native200/bpf200 =
323.706609 / 245.705683 / 239.091153 / 123.141164 / 118.096937 and
write-throughput medians (MiB/s)
5043.566001 / 5105.571055 / 4988.735170 / 5732.372957 / 5786.561373;
median-of-paired-ratios bpf200/fifo p99 -61.904198% and
bpf200/fifo write throughput +17.989185%, matching the published campaign
analysis. LM medians
stock/native/bpf = 73.574518 / 68.608032 / 65.422796 token/s and
median-of-paired-ratios BPF/native -4.047522%, BPF/stock -11.241969%,
native/stock -5.436485%. MoE medians
native-off/paper-native/paper-bpf =
11.896381 / 11.223299 / 11.190012 token/s and
1975.941635 / 1513.4446535 / 1535.957289 ms TTFT; geometric-mean
ratios paper-bpf/paper-native 0.996540 (throughput) / 0.995129
(TTFT) and paper-bpf/native-off 0.930786 / 0.775152, matching the
audited analysis.

## `scripts/artifact/reproduce_figures.py`

Reproduces the paper-selected observability figure without touching the
paper. `plot_obs_with_array.py` (in
`docs/paper/tex-revision/img/results-raw/revision/`) expects the
GPU-array campaign to hold exactly the five blocks the paper selected, but
the raw `cells.json` was later extended to ten paired blocks. The first
ten entries (blocks 1..5, both arms) are unchanged, and the ten-block
extension is retained in the raw. The wrapper:

1. reads the current raw `cells.json` (read-only) and selects blocks 1..5
   (five baseline + five `gpubpf_kernelretsnoop` cells, storage
   `gpu-array-onevalue`; it refuses to run if that selection is not
   complete);
2. writes the selected cells and the summary fields those cells imply
   (baseline mean, per-block paired overhead `100*(baseline-tool)/
   baseline`, and its mean) into a fresh output directory - no invented
   measurements;
3. invokes the unchanged obs figure script with `--new-cells`,
   `--new-summary`, `--data-output`, `--output-prefix` all inside that
   directory;
4. invokes the current seven-panel policy plot script
   (`plot_port_panels.py`, `matched_port_panels_v2`) with its
   `--output-prefix` in the same directory. The obsolete
   `workloads/matched-ports` five-panel script is not called.

Default output directory: `<repo-root>/reproduction`. All outputs are
created exclusively and never overwrite anything; when a name is taken,
choose a new `--out-dir` instead of deleting previous outputs. The
wrapper records a `reproduction-note.json` stating that the five-block
subset is the paper-selected subset of the current ten-block `cells.json`,
not an independent run, and reports the retained ten-block extension mean
(5.400619% over 10 cells) alongside the derived five-block mean
(5.572554%, baseline 37979.256081 tok/s).

Verified run: exit 0; `reproduction/` contains
`obs-blocks1-5-cells.json`, `obs-blocks1-5-summary.json`,
`reproduction-note.json`, `obs-with-array-data.json`
(`rtx5090_gpu_array.n_pairs = 5`, mean 5.572554166810581;
`rtx5090_table1.n_pairs_per_arm = 10` retained),
`obs-overhead-with-array.pdf/.png`, and
`matched-policy-panels.pdf/.png`. Re-running into the same directory
fails fast at the first exclusive write, as intended. No `docs/paper`
file and no raw record was modified.
