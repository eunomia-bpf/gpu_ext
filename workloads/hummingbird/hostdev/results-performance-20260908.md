# Hummingbird host/device mapping: five-block results

All **20 cells across five blocks completed with exit zero**. The existing
60-second client executes periodic 100 Hz VGG foreground requests alongside
continuous ResNet-152 background requests, with the same frozen idle-policy
profile, tile budgets and input/output-small settings. No completed cell
was repeated. These are measurements of the existing idle algorithm and
its device coordinate mapping, not a new scheduling algorithm.

## Performance

Values below are medians across five cells per arm. Foreground response
latency includes time from scheduled arrival to completion. Background
goodput counts only requests completed within the 60-second window.

| Implementation | BE goodput, requests/s | HP response p99, ms |
|---|---:|---:|
| Original host C + inline device mapping | 133.150 | 1.884592 |
| Host BPF + inline device mapping | 132.950 | 1.913771 |
| Host BPF + callable native mapping adapter | 129.017 | 1.871289 |
| Host BPF + callable device BPF mapping | 128.967 | 1.876349 |

The native callable arm is an **adapter control**, not the unchanged
original implementation. The historical no-policy baseline remains in
the [existing host/pipeline study](../pipeline/results-575-20260903.md);
it is not relabeled as a contemporaneous baseline in this comparison.

Paired background-goodput effects use the geometric mean of within-block
ratios, with 95% paired block-bootstrap intervals. These are not differences
between the table's marginal medians.

| Comparison | Goodput change | 95% interval |
|---|---:|---:|
| Host BPF / original, both inline | -0.063% | [-0.163%, +0.043%] |
| Callable native adapter / inline, both host BPF | -2.921% | [-3.221%, -2.588%] |
| Device BPF / callable native adapter | -0.072% | [-0.302%, +0.173%] |
| Host+device BPF / original inline implementation | -3.052% | [-3.198%, -2.934%] |

Thus this implementation reproduces the policy with a measured total
throughput cost of about 3%, rather than outperforming the original.
The incremental device-BPF comparison is small and mixed: two of five
paired throughput differences are positive and three negative, spanning
-0.334% to +0.350%. Its interval includes no difference; this is not a
formal equivalence result. The adapter comparison accounts for most of
the observed throughput gap, but does not isolate individual instruction,
stack, register or host-execution costs.

Foreground p99 changes are also mixed. Device BPF versus callable native
has geometric-mean change +0.798%, interval [-0.624%, +2.395%]; the total
BPF/original comparison is +0.056%, interval [-1.424%, +1.642%]. These
measurements do not establish better foreground latency.

## Retained first cell and sensitivity

Block 0 retains the first successful device-BPF run at 20:48:56--20:49:56
PDT. Its three controls ran at 21:21--21:25 PDT after a module reload.
This block is not fully adjacent/randomized. Blocks 1--4 use fresh processes
and a seeded within-block arm order under one compatible-core interval;
their exact sequence appears in `run-remaining.sh`.

Excluding block 0, total BPF/original background goodput changes by -3.076%
(interval [-3.247%, -2.933%]); incremental device BPF/native-adapter
goodput changes by -0.009% ([-0.264%, +0.246%]). The throughput conclusion
is unchanged. The four-block foreground device-BPF/native-adapter p99
change is +1.346% ([+0.016%, +3.069%]); this adverse sensitivity must not
be hidden by the five-block interval that includes zero.

## Scope and implementation evidence

The original host client is reused. The two callable variants use opt-in
rebuilt ResNet cubins: 43 actual specialized native mapping call targets
are replaced by the eBPF-derived callable function for the BPF arm.
The mapper's results feed the original kernel's coordinate-dependent
computations. This is not transparent instrumentation of an unchanged
model binary, nor a measurement of dynamic policy replacement.

Read-only `cuobjdump --dump-resource-usage` outputs show zero stack bytes
in all 44 original kernels, versus 48 stack bytes in 43 kernels for both
callable variants. Some register allocations also differ. These observations
offer a plausible explanation for adapter overhead, not a separate causal
experiment. They do not establish that the entire 3% cost is unavoidable
for every possible BPF interface or compiler implementation.

All 120000 foreground requests complete inside their measurement windows.
The background arms complete 39909 / 39884 / 38719 / 38691 requests inside
the windows in table order, plus one final background completion per cell
after its window. Existing client output comparisons report zero maximum
absolute error throughout; no additional validation campaign was added.

The initial stock-core ioctl failures remain in the first-run directory.
Loading the saved compatible scheduler core resolves the startup failure.
The final batch ends at 21:45:36 PDT with `PAIRED_EXIT=0 RESTORATION_OK=1`.
The previous stock core and saved GDS UVM are restored, both services are
active, loaders 4135269/4135270 report `attached`, and GPU usage returns to
0% / 1 MiB. This scoped batch is complete; LMCache KV integration and
XSched Level-2 remain separate unfinished work.

## Records and analysis

Raw directory: [raw-paired-20260908.uDB4jK](raw-paired-20260908.uDB4jK/README.md).
It contains 19 fresh logs, exact invocation/order scripts, lifecycle logs,
the three resource listings, `analysis.json` and `effects.json`. The retained
first BPF log remains in `raw-first-real-20260908.08lUpN/`; no large cubin,
model payload or cache is committed. Prior/adverse measurements are unchanged.

The local-model analysis script uses the existing request parser and
`analyze_three_way.estimate_ratios` (10000 bootstrap draws, seed 20260903).
Root first executed with explicit import directories, then reran the
completed standalone script successfully without an external `PYTHONPATH`:

```sh
python3 -B workloads/hummingbird/hostdev/analyze_paired.py \
  --out workloads/hummingbird/hostdev/raw-paired-20260908.uDB4jK/analysis.json
```

`analysis.json` preserves all 20 per-cell metrics. `effects.json` additionally
applies that same estimator to the four named comparisons, both including
and excluding block 0, and retains each block ratio, median change and range.
The reusable script now recomputes all four comparisons, medians, ranges
and bootstrap intervals directly from the 20 logs. The executed client
commands, completed data and estimates above do not depend on the unused
`run_hostdev.py` launcher.
