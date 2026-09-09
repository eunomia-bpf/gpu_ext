# Off-only hook scalability: completed

User explicitly selected automatic warp OFF. Root ran 240 timings and six
separate actual-call observations. RTX 5090, NVIDIA 575.57.08, CUDA 12.9,
bpftime-auto-warp 886b4ca. No On configuration was run.

Each grid uses 128 threads per CTA and ten alternating Native/Off pairs.
Work denotes additional per-thread integer arithmetic iterations (0/128/1024).
Eight warmup launches precede 128 timed launches. The original shared_update
BPF object performs the same constant GPU-array map update in every thread.
The measurement includes trampoline plus this callback/map access, not an
isolated empty-trampoline latency. The workload is a synthetic arithmetic
kernel, not an application-level scalability result.

The only benchmark change is the host-side CTA argument cap, raised from 64
to 8192 in an isolated source/build copy. The default workload math, target
hook, BPF, full-output comparison, map readback and strict checks are reused.
All 240 applications and 120 loaders completed successfully. Every Off run
has map readback and reports no automatic-warp admission. The isolated
counts use two launches without warmup and are excluded from performance.

## Timing results

Native/Off columns are medians in milliseconds for all 128 launches.
Overhead is the median of ten within-pair (Off/Native - 1) percentages.
Intervals use 10,000 paired bootstrap resamples with seed 1797. Added time
is the median within-pair difference divided by 128, in microseconds per
launch; it must not be interpreted as single-callback latency.

| CTAs | Work | Native ms | Off ms | Overhead, 95% CI | Added us/launch |
| ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 0 | 0.259904 | 0.524112 | 101.607% [99.756%, 110.861%] | 2.070 |
| 512 | 0 | 0.260000 | 0.523952 | 101.397% [98.306%, 108.950%] | 2.060 |
| 2048 | 0 | 0.278816 | 1.056480 | 279.293% [276.427%, 282.980%] | 6.085 |
| 8192 | 0 | 0.788464 | 3.545840 | 349.524% [345.489%, 352.168%] | 21.549 |
| 128 | 128 | 0.524448 | 0.544848 | 3.868% [3.332%, 4.048%] | 0.159 |
| 512 | 128 | 0.534608 | 0.792160 | 48.356% [47.757%, 48.611%] | 2.020 |
| 2048 | 128 | 1.316992 | 2.099968 | 59.386% [59.341%, 59.502%] | 6.112 |
| 8192 | 128 | 4.454672 | 8.391216 | 88.351% [88.321%, 88.406%] | 30.755 |
| 128 | 1024 | 1.577232 | 1.839504 | 16.629% [16.609%, 16.664%] | 2.049 |
| 512 | 1024 | 3.145744 | 3.387184 | 7.672% [7.527%, 7.812%] | 1.885 |
| 2048 | 1024 | 8.919392 | 9.882000 | 10.782% [10.492%, 11.043%] | 7.514 |
| 8192 | 1024 | 32.253681 | 33.539823 | 3.985% [3.976%, 4.003%] | 10.042 |

## Actual invocation counts

| CTAs | Work | Final count over two launches |
| ---: | ---: | ---: |
| 128 | 0 | 32768 |
| 512 | 0 | 131072 |
| 2048 | 0 | 524288 |
| 8192 | 0 | 2097152 |
| 8192 | 128 | 2097152 |
| 8192 | 1024 | 2097152 |

## Interpretation

The range reaches 8192 CTAs / 1048576 threads per launch. Observed callback
counts grow proportionally to CTA count, and do not change when arithmetic
increases at fixed geometry. Off's added execution time grows at larger
grids in the low-work settings, so block-count-independent cost is not a
supported claim. At 8192 CTAs the highest-compute setting has 3.985% overhead,
versus 349.524% and 88.351% in the lower-compute settings. These values do
not establish universal low overhead; the complete work/grid matrix is
retained. Overhead is not monotonic in work at every smaller grid, so no
simple single-variable scaling law is claimed.

## Files and completion

run.py reuses the existing runner, run-counts.py reuses its actual-counter
path, analyze.py reads original CUDA event logs and preserves all pairs.
Raw logs, three 80-row timing CSVs, summary.csv and counts.log are retained.
Both scripts returned zero. GPU was at 0% / 1 MiB with both shared locks
released and explicitly handed back to the experiment Codex. No driver
reload, old-data edit or paper edit was performed. At handoff the execution
owner had not committed or pushed. The experiment-coordination session
subsequently collected the completed data for publication: 240 timing rows
(120 Native and 120 Off), zero nonzero process-exit rows, and six separate
count records. The isolated benchmark source and scripts are included;
the compiled benchmark and generated PTX remain local. No GPU cell was
repeated during collection.
