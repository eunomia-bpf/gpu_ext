# FineMoE dynamic-set comparison — completed 2026-09-03

All **20 cells / five paired blocks** completed and passed independent raw
analysis. Dynamic sets reduced completed, evicted-unused expert payload versus
the all-positive prefetch ablation, but **demand-only remained faster**. The
matched BPF/C throughput difference is unresolved; this is neither a BPF
superiority nor a formal equivalence result.

## Scope and measurements

The four arms share the original Qwen1.5-MoE-A2.7B-Chat BF16 model, official
FineMoE execution path, repaired executor, offline 1,000-entry history and
16,834,658,304-byte pool. Demand-only disables speculation; all-positive admits
every positive predicted probability; native C and actual host uBPF JIT execute
the same corrected Eq. 6–8 dynamic-set selector. The ablation is not a separate
SOTA competitor, and BPF does not replace CUDA transfers or the inference engine.

Each fresh-process cell has one excluded warmup and eight held-out public
MT-Bench first-turn inputs, truncated to at most 16 input tokens, with exactly
16 generated tokens each. Five complete arm orders were randomized with seed
20260903. All cells ran on the same RTX 5090 / NVIDIA 575.57.08 / Linux 6.15.11
stack, with the fixed 400 W cap. Recorded safety windows span
**08:31:45–08:59:10 UTC**. The later host reboot did not interrupt this campaign.

Primary throughput is 128 verified tokens divided by the application window
from first measured submission through final verified completion. It includes
policy/search and input-materialization overhead, but excludes loading, warmup
and the final drain. TTFT and TPOT are per-cell request medians. The following
values are medians of five cells, not pooled-request estimates.

| Arm | Throughput, token/s ↑ | TTFT, ms ↓ | TPOT, ms ↓ | CPU time, s |
| --- | ---: | ---: | ---: | ---: |
| Demand-only | 5.1713 | 689.34 | 141.14 | 142.61 |
| All-positive ablation | 3.6228 | 924.20 | 227.01 | 215.02 |
| FineMoE native C | 4.4994 | 663.26 | 187.70 | 167.30 |
| FineMoE BPF | 4.5144 | 659.17 | 187.64 | 166.76 |

CPU time is the recorded process CPU-time delta, not elapsed wall time or an
independently reconstructed OS counter. Median throughput including drain is
respectively 5.1710 / 3.6219 / 4.4988 / 4.5138 token/s; median drain duration is
1.271 / 8.488 / 4.099 / 4.029 ms.

## Completed copies and paired effects

These are medians of **logical completed payload GB (10^9 bytes)** through
drain. They are not measured PCIe traffic, bandwidth, or transfer time saved.
First-use, evicted-unused and resident-unused classes conserve speculative
bytes in every cell; separately computed column medians need not sum exactly.

| Arm | Demand | Speculative total | First demand use | Evicted unused | Resident unused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demand-only | 145.108 | 0 | 0 | 0 | 0 |
| All-positive | 158.170 | 582.109 | 24.447 | 555.378 | 2.266 |
| Native C | 125.955 | 252.048 | 30.070 | 220.975 | 0.969 |
| BPF | 125.920 | 251.097 | 30.053 | 220.352 | 0.917 |

Resident-unused copies may still be used later: they are right-censored, **not
classified as waste**. Canceled queued requests are not completed copies. No
cell had copies completing after its application deadline; raw in-window,
post-window and drained partitions are nevertheless retained separately.

Ratios below use paired geometric means and 10,000 whole-block bootstrap
resamples, seed 20260903. Intervals are 95% percentile intervals for paired
effects, not error bars around the medians above.

| Comparison | Throughput ratio [95% CI] | Evicted-unused payload ratio [95% CI] |
| --- | --- | --- |
| Native C / all-positive | 1.243773 [1.239066, 1.248646] | 0.397272 [0.395179, 0.399003] |
| BPF / all-positive | 1.246394 [1.243217, 1.249578] | 0.395883 [0.393413, 0.398297] |
| BPF / demand-only | 0.873779 [0.871567, 0.876416] | Undefined: demand-only has zero speculative bytes |
| BPF / native C | 1.002107 [0.998266, 1.005226] | 0.996504 [0.992531, 1.000036] |

The BPF dynamic-set arm admits fewer unnecessary copies than all-positive:
evicted-unused payload is **60.41% lower [60.17%, 60.66%]**, with **24.64% higher
throughput [24.32%, 24.96%]**. Native C shows the same policy effect. These gains
are attributable to the dynamic-set policy versus an aggressive ablation, not
to BPF execution. Against demand-only, BPF throughput is **12.62% lower
[12.36%, 12.84%]**. Its secondary TTFT is 5.25% lower, but TPOT is 32.87% higher;
speculation does not pay for aggregate throughput on this frozen workload.

The remaining roughly 220 GB of evicted-unused payload per dynamic-set cell
and higher CPU cost are consistent with excessive speculation and shared
execution overhead. This experiment does not isolate their individual latency
contributions. The BPF/C throughput estimate is +0.21% [−0.17%, +0.52%], so it
does not establish an execution-substrate advantage, disadvantage or equivalence.

Every formal throughput cell, in token/s (raw block IDs retained):

The [two-panel result plot](figures/dynamic-set-comparison.pdf) shows all five
cells per arm for throughput and completed speculative-payload classification.
It makes the tradeoff visible: less unused speculation than all-positive,
but lower throughput than demand-only. The [caption](figures/dynamic-set-comparison.caption.md)
defines its windows and censoring; the [20-row CSV](figures/dynamic-set-comparison.csv)
preserves exact plotted values. The table below is an exact-value lookup.

| Block | Demand-only | All-positive | Native C | BPF |
| --- | ---: | ---: | ---: | ---: |
| 00 | 5.1713 | 3.6070 | 4.4723 | 4.5023 |
| 01 | 5.1420 | 3.6228 | 4.4994 | 4.5190 |
| 02 | 5.1728 | 3.6331 | 4.5364 | 4.5144 |
| 03 | 5.1905 | 3.6146 | 4.5241 | 4.5257 |
| 04 | 5.1568 | 3.6329 | 4.4931 | 4.5112 |

## Verification, limits and reproduction

All **160 measured requests / 2,560 tokens** exactly matched the original HF
reference; 20 warmups are excluded. The prior four-arm preflight independently
checked all 36 saved actual arrays: 87,515,136 finite FP32 values, zero difference
at the original fixed tolerance 0.0. Its complete C/BPF input, mask and downstream
enqueue traces matched. Formal timing disables full-logit transfer, event
capture and shadow checks, retaining exact token checks and real counters.
BPF executed **105,240 JIT decisions** across its five measured cells.

Independent analysis reconstructs all copy lifetimes and request metrics, checks
all 20 cells and the 36 actual preflight arrays, and rejects missing/extra attempts.
The 42-file runtime inventory was unchanged across every cell. There were no
copy or compute-release errors, pool overruns, disallowed throttling, or cleanup
failures. The 7,907 telemetry samples peaked at 19,017 MiB and 52 C; all cell
teardowns returned to no compute processes, UVM reference count zero and no
reported Xid/kernel abnormality. Earlier preparation failures remain separate.

This is valid **supporting, mixed** evidence for the dynamic-set component:
unused-transfer reduction is supported against the all-positive ablation;
net throughput benefit over demand-only is contradicted. It is not full
FineMoE/EuroSys reproduction, a new BPF-discovered policy, or kernel-UVM/device-BPF
execution. The public MT-Bench substitution, short demo inputs, offline history,
d=6, common executor repairs and binary64 prefix arithmetic limit generalization;
the [plan](plan.md) and [preparation history](results-preparation.md) preserve them.
Next: incorporate this bounded policy/mechanism result, including the demand-only
loss, into the revision synthesis; no favorable-result rerun is required.

The planned five-block plot is now available above. CPU-only regeneration to
a fresh prefix (no GPU imports, numerical preflight rerun, or new measurements):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 17 python3 -B plot_results.py --output-prefix /tmp/finemoe-new-figure
```

Plot verification (2026-09-03): all three focused projection tests and eight
existing scheduling-plot tests pass. The exported vector PDF is 504 × 205.2 pt
(7 × 2.85 inches); its PNG was visually inspected. The plot reuses the scheduling
figure's unchanged style, and no paper float or completed measurement was edited.

The [complete campaign](raw/full-v1/campaign.json), all per-cell raw files in
[raw/full-v1](raw/full-v1/), and the
[independent analysis](raw/full-v1/independent-analysis.json) retain every metric
and all six prespecified comparisons. CPU-only recomputation from this directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 17 .venv/bin/python -B analyze_results.py raw/full-v1
```

The completed GPU command and frozen preparation dependencies are in
[the execution record](plan.md#execution); use fresh output directories for any
separately justified future campaign, never overwrite this one.
