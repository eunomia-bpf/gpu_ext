# Larger-grid automatic warp and hook results

Root completed 180 timing measurements and 12 separate actual-counter observations
on RTX 5090, driver 575.57.08. The experiment Codex yielded the GPU and
shared runtime; both experiment locks were held for each sequential sweep.
No driver reload or paper change. Reused the original shared_update BPF object
and existing runner. Runtime revision is recorded in blocks/environment.txt.

Each setting has ten rotating native/off/on groups, 128 threads per CTA,
eight warmup launches and 128 timed launches. All 180 applications and
120 attached loaders exit zero, shared map readback succeeds, and all 60
on configurations report compiler automatic warp admission.

Times below are medians in milliseconds for all 128 launches. The percentage
is the median of ten within-group on/off changes, with the existing
10,000-resample confidence interval. Positive means slower.

| CTAs / work iterations | Native ms | Off ms | On ms | On/off change, 95% CI |
| --- | ---: | ---: | ---: | --- |
| blocks-b16-w0 | 0.259952 | 0.338480 | 0.523984 | +54.762% [+52.716%, +55.320%] |
| blocks-b32-w0 | 0.259584 | 0.339296 | 0.523872 | +54.560% [+52.488%, +55.806%] |
| blocks-b64-w0 | 0.257792 | 0.375824 | 0.524112 | +39.501% [+37.848%, +42.244%] |
| work-b64-w8 | 0.263440 | 0.523808 | 0.522720 | +0.006% [-0.507%, +0.196%] |
| work-b64-w32 | 0.272688 | 0.524304 | 0.524496 | +0.012% [-0.159%, +0.174%] |
| work-b64-w128 | 0.524816 | 0.532304 | 0.533120 | +0.157% [-0.263%, +0.794%] |

## Actual calls (separate two-launch diagnostic)

| CTAs / work | Off | On |
| --- | ---: | ---: |
| 16 / 0 | 4096 | 128 |
| 32 / 0 | 8192 | 256 |
| 64 / 0 | 16384 | 512 |
| 64 / 8 | 16384 | 512 |
| 64 / 32 | 16384 | 512 |
| 64 / 128 | 16384 | 512 |

Calls increase with CTA count and fall by 32x with automatic warp execution.
Increasing arithmetic at fixed geometry leaves the observed call count unchanged.
The zero-added-work settings are slower with automatic execution; with added
arithmetic the intervals include zero. This establishes call reduction, not
an execution speedup or block-count-independent total overhead. The measurements
do not isolate which transformation cost offsets the reduced calls.

The counter run contains diagnostic atomics and synchronization and is not used
for timing. Final cumulative observations are used once, not summed. The timing
CSV milliseconds were checked against original FIG15_MEASUREMENT log values.
All old measurements remain untouched. Run scripts, raw logs, CSV and analysis
are retained in this directory. The execution owner had not committed or
pushed at handoff. The experiment-coordination session subsequently collected
this completed directory for publication without rerunning any GPU cell:
180 timing rows (60 per arm), zero nonzero process-exit rows, and 12 count
records. Only scripts, reports and small text records are included; no
runtime binaries, model files, or caches are added.
