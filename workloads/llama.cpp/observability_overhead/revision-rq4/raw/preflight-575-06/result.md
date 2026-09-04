# RTX 5090 Table 1 preflight attempt 06

Status: **4/7 correctness arms valid; no performance cell started**.

All seven arms ran against the same deterministic TinyLlama output. The
baseline, NVBit kernel-return, NVBit thread histogram, and gpubpf thread
histogram arms passed. Both thread-histogram implementations accounted for
720,896 events and 22,528 nonzero thread slots; gpubpf also completed its full
1,048,576-entry readback. NVBit kernel-return likewise recorded 720,896
nonzero timestamped events across 220 selected launches.

Three fail-closed gates stopped the campaign:

- gpubpf kernel-return committed and collected all 720,896 records with zero
  reported drops, but its exact coordinate-multiplicity oracle found 3,808
  per-coordinate mismatches, so the collector exited nonzero.
- The repaired gpubpf launch-latency host path engaged correctly: 220 target
  launches were enqueued, paired with 220 device entries, and had no queue or
  target-filter errors. Its start and end clock intervals nevertheless drifted
  apart during the long instrumented run; 219 samples became clock errors and
  the endpoint-overlap gate rejected the arm.
- NVBit launch latency had a valid overlapping calibration interval, zero
  clock errors, and accounted for all 220 selected launches as 212 classified
  plus eight explicitly uncertain samples. The current zero-uncertainty gate
  rejected it rather than silently placing boundary-straddling intervals into
  histogram bins.

Because correctness was incomplete, the runner created zero timed blocks and
zero performance comparisons. The immediate post-run audit observed Linux
6.15.11, NVIDIA 575.57.08, 15 MiB GPU memory use, 0% utilization, a 400 W
power limit, UVM reference count zero, no compute processes, and no attached
struct_ops. This attempt is diagnostic evidence only and must not be pooled
with a successful Table 1 campaign.
