# Fresh memory/scheduling comparison on RTX 5090

Five rotated blocks complete all 20 cells and all 40 tenant processes with
return code zero. All policy loaders also exit zero. There are no retries,
discarded cells, missing performance values, or tenant timeouts. The measured
campaign runs from 07:59:58 to 08:08:21 UTC on 2026-09-07. The original core,
saved GDS UVM, GDM, persistence service and owned GDS loader are restored.

## Main result

Per-tenant wall time starts at that tenant's SIGCONT and ends at its own
process exit, including initialization, warmup, timed work and cleanup.
These are independent process durations, not the historical sequential-wait
proxy, per-request p99, or kernel-only time. Values below are medians over
five cells, in seconds; lower is better.

| Policy | High priority | Low priority |
| --- | ---: | ---: |
| Baseline, no policy | 56.547662 | 56.595119 |
| Memory only | 25.159860 | 26.852055 |
| Scheduling only | 3.938445 | 6.076045 |
| Combined | 3.546399 | 6.455460 |

Paired percent changes use `100*(numerator/denominator-1)` within each block,
then report the median and full range, not a ratio of descriptive medians.

| Comparison | High-priority change, median [range] | Low-priority change, median [range] |
| --- | ---: | ---: |
| Memory / baseline | -55.673% [-56.390%, -53.077%] | -52.554% [-53.171%, -50.339%] |
| Scheduling / baseline | -93.072% [-93.110%, -92.957%] | -89.268% [-89.412%, -89.202%] |
| Combined / baseline | -93.711% [-93.777%, -93.699%] | -88.592% [-88.656%, -88.530%] |
| Combined / memory | -85.930% [-86.681%, -85.554%] | -75.955% [-77.157%, -75.506%] |
| Combined / scheduling | -10.166% [-10.535%, -8.613%] | +6.299% [+5.442%, +7.147%] |

Every block shows both policies individually improving both tenants over
baseline. Scheduling supplies most of the improvement. Adding memory policy
to scheduling consistently improves the high-priority tenant but slows the
low-priority tenant: this is a priority tradeoff, not an all-metrics win.

## Internal timing and interpretation

| Policy | High internal time, ms | Low internal time, ms |
| --- | ---: | ---: |
| Baseline | 15652.000 | 15601.700 |
| Memory only | 13581.000 | 413.498 |
| Scheduling only | 413.441 | 413.520 |
| Combined | 413.408 | 413.505 |

These are medians across the five cells of the benchmark-reported time. With
`--iterations=1`, each cell has one timed HotSpot trial after the benchmark's
warmup; each trial itself executes ten stencil steps. Scheduling and combined
have nearly identical internal times, despite their different end-to-end
process durations. Do not attribute the combined/high-priority gain solely
to steady-state kernel execution. Reported bandwidth is based on the
benchmark's nominal bytes accessed, not measured PCIe migration traffic.

This result contradicts a general claim that timeslicing is ineffective for
memory-bound oversubscribed workloads. One plausible explanation is improved
residency/locality when the long high-priority slice reduces interleaving;
this campaign does not isolate migration causes or record driver-classified
thrashing. It demonstrates the measured scheduling-policy effect on this
HotSpot workload, not a universal optimum or a new hardware-preemption result.

The [historical evidence analysis](../../docs/eval/rq3-revision-audit.md)
retains the old single-round numbers, including sub-1% scheduling differences.
Those runs attached the initialization-time policy after workload launch,
did not retain policy hits, and did not measure both tenant exits independently.
They cannot establish scheduling-policy ineffectiveness. No old CSV, figure
data, failed launch, or reported number is overwritten by this campaign.

## Configuration and raw evidence

Runner source: `ad2e5748`, implemented by local Qwen 27B OpenCode and reviewed
by root. The root ran the existing runner after a temporary core-module switch;
it did not wait for, or run, the still-in-development lifecycle wrapper.
Core source `a2b40efd` supplies scheduling hooks; the UVM is the saved
61,945,872-byte GDS module at
`/var/tmp/gds-restore-before-stale-20260907.AWgdmi/nvidia-uvm.ko`.
Kernel: 6.15.11-061511-generic. Driver: 575.57.08. GPU: RTX 5090.

Two processes run `uvmbench --kernel=hotspot --size_factor=0.6 --mode=uvm
--iterations=1 --output=PATH`. Each reports a 41024-by-41024 grid and roughly
19260 MiB of allocated arrays, so the combined working set exceeds VRAM.
High is resumed 20 ms before low, identically in all arms. Memory policy
uses high/low parameters 20/80; scheduling uses 1000000/200 us. Both attach
before the stopped tenants execute their CUDA initialization. The same
candidate core and saved UVM serve every arm, including baseline.

```
python3 -u workloads/fig13-fast/run_fig13_fast.py --blocks 5 --timeout 0 \
  --results-dir /home/yunwei37/workspace/gpu/gpu_ext/workloads/fig13-fast/results
```

Block order rotates modulo four; the fifth repeats the first ordering, so
this is not a perfectly position-balanced five-block design. It is one
workload, two identical tenants and fixed priorities, not a broad workload
sweep or a matched native-implementation mechanism-overhead comparison.

All ten scheduling-bearing cells record `policy_hit=12`, `policy_miss=0`,
`timeslice_mod=12`, `setter_error=0`, and `control_override=2`. Memory logs
contain periodic per-PID and aggregate counters. The runner CSV's memory
metadata captures the first matching entry, not the final aggregate; use
the final `=== Summary ===` block for aggregate interpretation. No counters
are used to reject performance cells.

[Complete raw directory](results/fig13_fast_20260907_005958/) contains all
20 `meta.json` files, 40 tenant logs and native benchmark CSVs, every policy
log, run configuration, event log, full terminal/module-restoration log,
and [paired analysis](results/fig13_fast_20260907_005958/paired-analysis.json).
The root computed the arithmetic directly from `fig13_fast.csv`. The local
OpenCode/GLM [reusable analyzer](analyze_results.py) has now also run on these
same 20 completed cells. Its [JSON](results/fig13_fast_20260907_005958/reanalysis/fig13_fast_analysis.json)
and [readable report](results/fig13_fast_20260907_005958/reanalysis/fig13_fast_analysis.md)
match every published per-arm and block-paired median, minimum and maximum.
All rows are retained; all 40 tenant return codes are zero. The analyzer
keeps the CSV's early memory metadata separately from the last aggregate
summary in each policy log. These callback counters are not migration-byte
measurements. No new GPU measurements were needed for this reanalysis.

Reproduce from the repository root:

```bash
python3 workloads/fig13-fast/analyze_results.py \
  workloads/fig13-fast/results/fig13_fast_20260907_005958 \
  --output-dir workloads/fig13-fast/results/fig13_fast_20260907_005958/reanalysis
```

No new correctness, clock, admission, or review experiment was added.
