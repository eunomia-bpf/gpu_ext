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

Recomputes cell metrics and paired statistics for the two latest
supplemental campaigns from their raw `result.json` records.

Campaigns and source maps:

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

Statistics vocabulary (kept explicit in the report):

- per-cell value: the raw per-cell metric.
- marginal median: median of a single arm's values across complete blocks.
- paired ratio (per block): `100*(candidate/reference - 1)` for that block.
- median of paired ratios: median over blocks of the per-block paired
  ratios.
- ratio of medians: `100*(median(candidate blocks)/median(reference
  blocks) - 1)`.

The two differ in general; both are reported and labeled. Pairs use
available numeric values for both arms; missing or incomplete status is
reported separately, never filled in. The tool never
stops running jobs or imposes gates.

Usage:

```
python3 scripts/artifact/reanalyze.py                 # both campaigns, stdout
python3 scripts/artifact/reanalyze.py --campaign lm   # one campaign
python3 scripts/artifact/reanalyze.py --output PATH   # exclusive creation
```

`--output` creates the report exclusively: an existing file aborts the run
with an error; nothing is deleted or overwritten, and the path is rejected
under `docs/paper` or inside any raw campaign directory.

Verified run (`--campaign all`): all 30 cells present (15 + 15), no
MISMATCH lines against the existing summaries; LM medians
stock/native/bpf = 73.574518 / 68.608032 / 65.422796 token/s and
median-of-paired-ratios BPF/native -4.047522%, BPF/stock -11.241969%,
native/stock -5.436485%, matching the published campaign analysis.

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
