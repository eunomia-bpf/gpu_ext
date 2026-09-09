# Artifact: results, sources, and reproduction

This is the repository entry point, not a manuscript edit. Updated 2026-09-09.
The artifact is intentionally more detailed than the paper: it retains all
historical results, supplementary measurements, failed attempts and limitations.
Newer measurements do not silently replace the paper's selected cohort.

## Start here

| Goal | Entry |
| --- | --- |
| Find the measurements used by the current paper | [Paper-to-evidence map](#paper-to-evidence-map) |
| Recalculate or render published measurements without a GPU | [CPU-only reproduction](#cpu-only-reproduction) |
| Find additional experiments and negative results | [Supplementary and historical results](#supplementary-and-historical-results) |
| Build and run a workload again | [Runtime reproduction](#runtime-reproduction) |
| Understand remaining release gaps | [Open items](#open-items) |

The organization is logical; existing source and raw-data paths stay in place
so old reports and commands retain their meaning:

```text
ARTIFACT.md                 current navigation and scope
workloads/<system>/         source, workload instructions, dated reports
  raw/ or results*/        immutable run records and derived summaries
microbench/                mechanism and device microbenchmarks
docs/experiment/            design notes, safety evidence, deployment records
docs/eval/agent/            agent-study reports and current prompt templates
docs/paper/                manuscript and its selected plot inputs (read-only here)
```

Read the dated report before using a raw directory. A historical `run.sh` is
an execution record, not necessarily a portable launcher: it may contain the
original machine's build paths, module locations or process IDs. Failed runs
are retained as failures, not counted as successful performance measurements.

## Paper-to-evidence map

The current manuscript entry is [main.tex](docs/paper/main.tex), which includes
`tex-revision/`; the older `tex/` tree is historical. Labels below are stable
source labels rather than page or figure numbers that change with layout.
These links locate evidence; they do not imply every GPU experiment has been
rerun from a fresh checkout.

| Paper item | Selected source / report | Important scope |
| --- | --- | --- |
| `fig:matched-ports` | [Seven-panel data](docs/paper/tex-revision/img/results-raw/revision/matched-policy-panels.json), [plot source](docs/paper/tex-revision/img/results-raw/revision/plot_port_panels.py) | The older [five-panel data](workloads/matched-ports/port-panels.json) is not the current figure. |
| MoE-Infinity | [Protected five-pair report](workloads/moe-infinity/results-paper-v3-protected-575.md), [workload](workloads/moe-infinity/README.md) | Figure uses mode medians; prose reductions use geometric means of within-pair TTFT ratios. |
| Expert Buffering | [Section-VI comparison](workloads/expert-buffering-policy/section-vi/results-performance-575-20260903.md) | Current panel uses whole-expert demand loads; throughput is a different retained metric. |
| FineMoE | [Performance report](workloads/finemoe/results-performance.md) | Figure baseline is all-positive speculative prefetch. Demand-only is also measured and must not be omitted when discussing all controls. |
| Hummingbird | [Host-policy comparison](workloads/hummingbird/results-575-20260903.md) | Periodic and BurstGPT arrivals; supplementary host/device experiment is separate. |
| POD-Attention | [Operator report](workloads/pod-attention/results-575-20260903.md) | Llama decode batch 128, two-stream FlashAttention baseline; operator averages are not five-run medians. |
| XSched | [Scheduling point data](workloads/gpreempt/figures/scheduling-comparison-2x2.points.json), [workload](workloads/xsched/README.md) | Paper's Level-1 metric is submission-to-first-block latency, not the supplemental Level-2 service p99. |
| GPREEMPT | [Load study](workloads/gpreempt/results-load-study-575-20260903.md), [LC-knee study](workloads/gpreempt/results-lc-knee-575-20260903.md) | Identify arrival rates and continuous-load groups before comparing numbers. |
| LMCache local-disk paragraph | [Write-budget report](workloads/lmcache-disk/results-575-gds-write-budget-20260907.md), [25 raw cells and paired analysis](workloads/lmcache-disk/raw/gds-write-budget-575-20260907-five-block-02/) | FIFO / native200 / bpf200 read p99: 323.707 / 123.141 / 118.097 ms. Scheduled-arrival storage latency, not vLLM TTFT. |
| `fig:obs-overhead` (originally Table 1) | [Selected plot data](docs/paper/tex-revision/img/results-raw/revision/obs-with-array-data.json), [original 5090 campaign](workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md), [GPU-array extension](workloads/llama.cpp/observability_overhead/revision-rq4/results-onevalue-array-bootstrap-575-20260907/README.md) | Prefill token/s loss. Current 5.57% bar is the first five GPU-array pairs; the ten-pair extension is 5.40%. Both remain. |
| `fig:all-kernels-priority` | [Current memory/scheduling observation CSV](docs/paper/tex-revision/img/results-raw/revision/memory-oversubscription-cells.csv), [notes](docs/paper/tex-revision/img/results-raw/revision/memory-composition-notes.md) | Do not substitute the earlier single-workload `fig13-fast` campaign for the current multi-kernel figure. |
| `fig:clc-policies` | [CLC data and plots](docs/paper/img/results-raw/clc/) | Historical cluster-launch-control experiment. |
| `fig:llama-expert-offload`, `fig:vllm-kv-offload` | [llama.cpp figure sources](docs/paper/img/results-raw/llama.cpp/), [vLLM figure sources](docs/paper/img/results-raw/vllm/) | Original case studies, not the new matched-policy ports. |
| `fig:gnn-epoch`, `fig:faiss-perf` | [GNN source](workloads/pytorch/benchmark_gnn_uvm.py), [GNN selected plot](docs/paper/tex-revision/img/results-raw/revision/plot_gnn_compact.py), [FAISS workload](workloads/faiss/README.md) | Retain application-prefetch and allocation controls. |
| `fig:two-tenant`, `fig:microbench` | [Co-location sources](docs/paper/img/results-raw/multi-tenant/), [runtime microbench sources](docs/paper/img/results-raw/runtime/) | Different workloads and denominators from policy-port figures. |
| Safety and agent workflow | [Safety evidence](docs/experiment/revision-safety/), [agent materials](docs/eval/agent/README.md), [original transcript inventory](docs/experiment/revision-artifact-inventory.md) | Current prompt templates are not the missing original study transcripts. |

## CPU-only reproduction

For the supplemental LMCache physical-reclaim and XSched Level-2 campaigns,
the following standard-library-only command reads all 30 published cell
summaries and recomputes per-arm medians and within-block comparisons:

```sh
python3 -B scripts/artifact/reanalyze.py --campaign all
```

It also works from another working directory when called by its absolute
script path. This was executed successfully on 2026-09-09; the numbers match
the recorded summaries. `--campaign lm` or `--campaign xsched` selects one,
and `--output NEW_FILE` saves the report without overwriting an existing file.
This recomputes statistics from cell summaries, not GPU measurements or every
per-request timestamp. Missing records are reported explicitly.

Run from the repository root with Python 3 and Matplotlib available. These
commands write only to a newly allocated output directory, never the paper:

```sh
artifact_out=$(mktemp -d /tmp/gpubpf-artifact.XXXXXX)
python3 -B docs/paper/tex-revision/img/results-raw/revision/plot_port_panels.py \
  --output-prefix "$artifact_out/matched-seven"
```

This seven-panel command was executed successfully on 2026-09-09. It renders
the published panel data; it is not a fresh GPU run or a full reanalysis of
every worker log. The panel inputs name their underlying result reports.

The observability plot's default command currently needs an explicit historic
five-pair input: its source expects five GPU-array pairs, while the live input
file now contains the completed ten-pair extension. The default invocation
was tested and fails with `expected 5 baseline blocks`. A portable wrapper
outside the paper is being prepared; neither the paper nor measurements are
to be changed to conceal the cohort mismatch.

Use the estimator named in each report. A ratio of medians, median of paired
ratios, geometric mean of paired ratios, and mean per-pair overhead are
different quantities. Retain units, timing boundaries, cohort sizes and all
arms; lower latency and higher throughput have opposite improvement signs.

## Supplementary and historical results

These are additional evidence, not automatic replacements for the selected
paper cohort. Each linked report includes source/raw references and limits.

| Experiment | Evidence | Relationship to paper |
| --- | --- | --- |
| LMCache five-arm serving | [Report](workloads/lmcache-disk/results-575-lmcache-gds-five-arm-20260906.md) | Recompute, CPU, disk FIFO, native and BPF controls. |
| Disk/UVM physical reclaim | [15-cell results](workloads/lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F/RESULTS.md) | Warm generation throughput: stock/native/BPF 73.575/68.608/65.423 token/s. Negative policy result; CPU-staged disk restore, not demonstrated GPU-direct P2P. |
| XSched Level-2 host/device port | [Five-block results](workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns/README.md) | 15 cells; BPF/native service-p99 paired median +13.51%. Same NVBit actuator, not original cuXtra artifact reproduction on sm_120. |
| Hummingbird host/device | [Report](workloads/hummingbird/hostdev/results-performance-20260908.md) | Additional device path; keep negative and uncertain effects. |
| Full-record device-buffer layout | [Warp-contiguous SoA report](workloads/llama.cpp/observability_overhead/revision-rq4/results-full-record-soa-warp-20260909.md) | Different full-record payload experiment; does not replace the Table-1 cohort. |
| Trampoline plus map scaling | [Off-mode results](microbench/fig15-device/strict-warp-map-scaling/raw/off-scalability-20260908.d8sa7gog/RESULTS.md) | 240 timing samples plus hook counts; total callback/map cost must not be described as block-independent. |
| Stale-state | [Live results](workloads/stale-state-575/results-performance-gds-20260907.md) | Measured state delay and throughput, not merely the earlier harness. |
| P40 observability | [Submitted values](workloads/llama.cpp/observability_overhead/p40-submitted-table1.json) | Original 8/85%, 3/87%, 14/93% remain historical; no invented per-run variance. |

## Runtime reproduction

Start with the workload README/report above, then its build and run entrypoint.
Source code, small input files, logs and results belong in Git; downloaded
models, compiled libraries and regenerable disk-cache payloads do not. Cache
removal does not remove the run's result; the workload must regenerate it.

Most revision runs use RTX 5090, Linux 6.15.11, NVIDIA 575.57.08 and CUDA 12.9,
but workload-specific runtime revisions and patches still matter. Follow each
record's source revisions; stock 575 alone does not provide all gpubpf hooks.
Resolve local model, bpftime, driver and workload-build locations anew. Never
signal a historical PID or load a module solely because an old log names it.

On the shared experiment host, GPU runs and heavy builds coordinate with
`/tmp/gpubpf-revision-gpu0.lock` and `/tmp/gpubpf-revision-struct-ops.lock`.
Reanalysis above does not need a GPU or driver changes. Do not run all live
experiments concurrently on the same device or repeat completed cells merely
to fill optional diagnostics.

## Open items

- Extend the portable CPU reanalysis beyond the two supplemental campaigns
  above and complete the historic-cohort figure entrypoint.
- Check fresh-checkout build/run instructions per workload; a successful
  recorded run is not yet proof that all local build dependencies are published.
- Recover the original agent-study transcripts from their author; present
  reports and new prompt templates do not replace them.
- Record manuscript/evidence discrepancies here without editing the paper:
  the current seven-panel English caption says medians generally, while POD's
  selected source reports operator means. Detailed artifact statistics must
  preserve the source estimator.
- Original cuXtra Level-2 on sm_120 remains under implementation; the completed
  shared-actuator policy-port comparison is a separate result.
