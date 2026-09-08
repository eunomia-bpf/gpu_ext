# LMCache total-recovery-cost policy comparison

Status: complete, five blocks and twenty planned-policy cells. The missing
block-1 original-ratio control completed after the main batch; that timing
deviation and its sensitivity analysis are retained below. One additional
unplanned-policy warm result and one disk-full cold failure are also preserved.
Implementation `4381f660`; raw outputs and the exact
invocation are under `raw/gds-kv-total-cost-575-20260908-01/`.
See the [unchanged experiment plan](gds-control/kv-reclaim-total-cost-experiment-20260908.md).

The old native/BPF cost-per-byte comparison remains in the
[completed grace report](results-575-gds-kv-reclaim-grace-20260908.md).
These are fresh controls for a changed algorithm, not replacements for
those historical measurements. No policy advantage or novelty is presumed.

The two native formulas are our custom reclaim policies. They are not
claimed to be LMCache's published default algorithm. Stock is the ordinary
victim selector in the same installed, grace-repaired serving runtime;
all arms use the same real LMCache disk backend. Thus the comparisons
separate a custom policy change, its native/BPF implementation cost, and
its benefit or loss relative to the runtime's stock behavior.

## Final paired result

| Arm | Median output token/s | Median of cell TTFT medians, ms |
| --- | ---: | ---: |
| Stock victim selector | 71.916482 | 25745.204 |
| Native original cost/byte | 64.743942 | 30933.185 |
| Native total cost | 68.964702 | 26453.816 |
| BPF total cost | 68.834700 | 26779.272 |

These medians are not paired effects. Within each block, the percentage is
`100 * (tested throughput / reference throughput - 1)`:

| Comparison | Paired mean % | Paired median % | 95% bootstrap interval for paired mean % |
| --- | ---: | ---: | ---: |
| BPF total / native total | -1.205663 | -0.951210 | -2.417572 to -0.023905 |
| Native total / native original ratio | +5.377805 | +7.339387 | -1.190965 to +11.946575 |
| BPF total / native original ratio | +4.129887 | +6.318364 | -2.854366 to +11.114140 |
| Native total / stock | -3.602768 | -3.423484 | -7.205587 to -0.109257 |
| BPF total / stock | -4.776475 | -4.342130 | -8.060147 to -1.600677 |

Total cost removes the repeated-restoration pattern in even blocks, but
does not beat stock overall. Its benefit against the original ratio is
arrival-order dependent (three positive and two negative native pairs);
the five-block interval includes zero. The matched BPF path is slower in
four of five pairs, with a mean -1.21% throughput difference; its interval
only narrowly excludes zero. This is a measured implementation-path cost
on this workload, not an isolated BPF instruction-cost measurement.

For comparisons involving the late original-ratio control, excluding block 1
gives native-total/ratio mean +7.637800% [0.365576%, 13.669584%] and
BPF-total/ratio mean +6.257120% [-1.565521%, 12.313084%]. This sensitivity
does not justify dropping block 1 from the primary report. The complete
five-block stock/native-total/BPF-total comparisons require no replacement.

Intervals enumerate all equally weighted ordered whole-block bootstrap
samples (`5^5` for five blocks, `4^4` for sensitivity), using linear
percentile interpolation. They are pointwise intervals over five blocks,
not evidence of universal benefit. The [paired summary](raw/gds-kv-total-cost-575-20260908-01/paired-summary.json)
records every raw source, unrounded metric, paired effect, and method.
All 20 planned cells finish eight warm requests and 8192 output tokens,
with zero warm failures and zero server exit codes. The extra unplanned
warm result also completes, but is not included under the original-policy label.

## Completed cells and retained interim interpretation

| Block | Policy | Output token/s | Median TTFT, ms | Restore batches | Logged restore MiB | Restore ops | Rollback warnings |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | stock | 70.9041546753 | 25745.2044670 | 15 | 1632 | 68 | 7 |
| 0 | native cost/byte | 62.4583563017 | 32331.0981690 | 43 | 6000 | 250 | 35 |
| 0 | native total cost | 68.9647015543 | 26453.8161885 | 15 | 1632 | 68 | 7 |
| 0 | BPF total cost | 69.6323735398 | 26499.3640565 | 15 | 1632 | 68 | 7 |
| 1 | native total cost | 67.7450239877 | 28889.1721360 | 12 | 1344 | 56 | 4 |
| 1 | BPF total cost | 67.2409223304 | 28920.8070945 | 12 | 1344 | 56 | 4 |
| 1 | stock | 75.0446044806 | 24276.9762065 | 15 | 1968 | 82 | 7 |
| 2 | native total cost | 73.5670914154 | 24731.9238700 | 15 | 1632 | 68 | 7 |
| 2 | BPF total cost | 71.8898975830 | 25733.6014840 | 15 | 1632 | 68 | 7 |
| 2 | stock | 71.9164818733 | 25843.3867345 | 15 | 1632 | 68 | 7 |
| 2 | native cost/byte | 63.5406061794 | 31571.3766265 | 42 | 5856 | 244 | 34 |
| 3 | BPF total cost | 65.3925934133 | 29939.2793140 | 12 | 1344 | 56 | 4 |
| 3 | stock | 70.5504635577 | 25880.3336575 | 15 | 1968 | 82 | 7 |
| 3 | native cost/byte | 69.5045186867 | 26289.0616350 | 15 | 1968 | 82 | 7 |
| 3 | native total cost | 67.4298578540 | 29127.0380250 | 12 | 1344 | 56 | 4 |
| 4 | stock | 71.9592647512 | 25556.1341070 | 15 | 1632 | 68 | 7 |
| 4 | native cost/byte | 64.7439418971 | 30933.1854310 | 42 | 5856 | 244 | 34 |
| 4 | native total cost | 69.4957506535 | 26404.6924315 | 15 | 1632 | 68 | 7 |
| 4 | BPF total cost | 68.8347001303 | 26779.2715430 | 15 | 1632 | 68 | 7 |
| 1 (late control) | native cost/byte | 70.3202776192 | 25859.5119110 | 15 | 1968 | 82 | 7 |

At the earlier thirteen-cell checkpoint, all completed cells had eight warm
requests, 8192 output tokens and zero warm failures. The following paragraphs
retain that interim interpretation, now bounded by the full comparison above.
In the first block BPF total/native total throughput differs by
+0.9681%; BPF total/old native differs by +11.4861%, and BPF total/stock by
-1.7937%. One block does not establish a stable mechanism or policy gain.

In this first block, native total cost improves throughput by 10.4171%
against the original ratio rule but remains below stock. Its restoration
counts return to the stock counts. This is evidence consistent with the
denominator contributing to repeated restoration on this arrival order,
not proof that total cost wins on all orders or that every victim decision
matches stock. The final comparison retains all five blocks.

Block 1 bounds that interpretation: total-cost native/BPF perform fewer
restores than stock but have lower throughput. Reducing disk restoration
alone is therefore not sufficient to improve this serving workload. It
can change which request makes progress and how much decode work is lost;
these counts alone do not establish the cause. The native/BPF total-cost
implementations remain close in this block (BPF/native -0.7441%).

Block 2 supplies another complete original-ratio comparison after source
restoration: native total/ratio throughput improves by 15.7796%, and BPF
total/ratio by 13.1401%. BPF total is effectively at the observed stock
throughput in this block (71.8899 versus 71.9165 token/s), while native
total is higher. Restore counts fall from the ratio rule's 42 batches /
5856 MiB / 34 rollback warnings to 15 / 1632 / 7 for both total-cost
implementations and stock. Together with block 0 this supports a
ranking-related repeated-restoration effect on these two arrival orders;
the block-1 result still prevents a general stock-superiority conclusion.

## Disk-full interruption and continuation

The first block finished before an actual ENOSPC failure during block 1's
native-ratio cold population. Its retained `server.log` records six cold
store messages followed by `No space left on device` and EngineCore failure.
The runner also failed to serialize `result.json` with `OSError: [Errno 28]`;
there is no complete numeric warm-phase result for this attempt. The raw
log and configuration remain in `block-01/position-0-native_ratio/`, not
overwritten or counted as a successful performance cell. The original
campaign process exited 1 and restored the old BPF loader, whose log says
`attached`.

Root moved only regenerable cache payload directories from the completed
15-cell grace campaign and these four completed block-0 cells to
`/var/tmp/lmcache-retired-cache-20260908.9ypd5z/`, retaining their relative
`raw/.../cache` paths. Nothing was deleted: all cache data is recoverable,
and every original result/log remains in place. Workspace available space
rose to about 25 GiB; archive filesystem available space is about 75 GiB.
The failed partial cache was left untouched.

The exact continuation is `python3 -u /tmp/lmcache-total-cost-resume-20260908.py`,
preserved as `resume-run.py` beside the original invocation. It reattaches
the same total-cost object, skips all four completed block-0 cells, and runs
blocks 1–4. Only the failed cold-population attempt is replaced by a new
directory, `block-01/position-0-native_ratio-after-enospc/`. Model, driver,
policy binaries, native library choices, arrival orders and measurement
code are unchanged. The cache remains on the same workspace NVMe device;
free space and the interruption differ across blocks and must be disclosed
when interpreting the final paired results.

## Unplanned modification of the original native control

The local GLM read-only analysis session unexpectedly changed the shared
header and rebuilt the default native/BPF binaries at 03:12:48 UTC. The
`native_ratio-after-enospc` server started after that build and finished
before restoration, so its 66.9241699594 token/s result belongs to an
unplanned cost/net-freeable algorithm, not the requested original ratio
control. Its [policy correction](raw/gds-kv-total-cost-575-20260908-01/block-01/position-0-native_ratio-after-enospc/policy-correction.md)
preserves the mismatch explicitly; the raw intent label is not sufficient
to include it in an original-policy comparison.

Root saved the patch/header/binaries separately and restored the shared
header from `4381f660`. The original native and BPF targets were rebuilt
with the existing Makefile in a separate directory and atomically replaced
at 03:18:22 UTC; no mapped shared-library inode was overwritten in place.
The build used the original optimization flags and explicit repository
include paths. The restored native/BPF files are 28560/42480 bytes; debug
build paths differ from the original binaries. The total-cost binaries
remain their original 26200/38736-byte builds and the currently attached
total-cost policy is unchanged. The whole first block predates the edit.

The patch is retained as an unselected alternative, not adopted as the
new default. After an urgent stop-writing instruction, the analysis
session issued another write command (creating an empty test file) and
was terminated for this concrete scope error. Its empty file was also
moved to the archive; the two disk-UVM sessions continue. This was not a
silence or duration timeout. Later original-ratio cells use the restored
algorithm. A replacement for block 1's missing original-ratio control
remains necessary and will use a new output directory; no completed
planned-policy cell is to be repeated. Its later measurement time must
be disclosed in any paired analysis.

The missing-control command subsequently acquired the GPU/struct-ops locks
after the main batch: `python3 -u /tmp/lmcache-total-cost-missing-control-20260908.py`.
It completed one original-ratio cell at
`block-01/position-4-native_ratio-restored/` only after the current campaign
finishes. It preserves the unplanned result under its actual policy label
and records the later measurement time. It does not repeat any completed
planned-policy cell. The final paired analysis must also show how excluding
this non-interleaved replacement affects interpretation.

## Measurement boundary

The shared runtime includes the prior deferred-free grace repair. All
four arms use real LMCache/cuFile compatibility-mode storage and FIFO
per-I/O admission. Only victim ranking changes between native variants;
the native total and BPF total versions compile the same shared algorithm.

Throughput is completed warm output tokens divided by full warm-phase
elapsed time. TTFT is HTTP-send to first-token, not scheduled-arrival
latency; four client workers mean later submissions can wait for a worker.
The completion-progress writes in runner `a1011e64` are common to all
new arms and occur within warm elapsed time. Server startup, cold
population, cold-store barriers, shutdown and exit-diagnostic collection
are outside that interval.

Restore counts/MiB/ops come from the existing `batched_get_blocking` log
lines, and warnings from `rolled back from`. Logged backend payload is
not a physical SSD-byte counter. Fixed-pool block recycling does not prove
physical HBM release to other processes or transparent disk-UVM paging.
The separate disk-UVM driver implementation has no runtime result yet.
