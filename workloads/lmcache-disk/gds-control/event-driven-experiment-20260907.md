# Event-driven LMCache feedback experiment

## Question and evidence

RQ2, policy versus mechanism: can BPF execute the same storage-admission
policy with competitive performance to the native implementation?
The completed polling-feedback campaign shows a paired BPF/native read-p99
increase of 23.601% at the median, adverse in all five blocks. Almost every
deferred write reaches its 10 ms budget; thousands of repeated decisions
therefore seldom enable earlier submission. Existing records cannot isolate
ioctl, Python scheduling and storage-service costs.

This supporting experiment replaces repeated polling with event-driven
wakeup. It adds a new execution path, not another copy of completed cells.
The strongest runnable competing implementation is the same native policy;
FIFO is the no-deferral control. Repeating the old static-input workload would
not answer this feedback-path overhead question.

## Implementation and comparison

Local OpenCode implements `live-event-driven` as an opt-in variant. The same
native/BPF executor waits until pending demand reaches zero or the cumulative
10 ms write budget expires, then asks the existing decider again. Native and
BPF receive the same actual pending-read metadata and use identical buffers,
I/O paths and recording. FIFO still immediately submits every operation.

The pending-demand predicate and admission budget are retained. Wakeup timing
is deliberately different from the historical <=1 ms re-evaluation schedule;
this is a new executor variant, not an assertion of identical execution traces.
The original `fixed-delay` and `live-feedback` paths remain available.

Reuse LMCache 0.5.4 GdsBackend and real cuFile compatibility-mode direct I/O,
64 reads plus 96 writes of 24 MiB objects, 2/4 ms scheduled spacing, 4096 MiB
GPU buffer, and five rotated FIFO/native/BPF blocks in fresh child processes.
This is storage-backend request performance, not vLLM token throughput and
not demonstrated hardware NVMe/GPU P2P. No additional preflight or admission
gate is introduced. Wait for local archive I/O to finish before measuring.

```sh
workloads/lmcache-disk/current-venv/bin/python \
  workloads/lmcache-disk/run_gds_mixed_backend.py \
  --policy-variant live-event-driven --blocks 5 --reads 64 --writes 96 \
  --gds-buffer-size-mib 4096 \
  --output workloads/lmcache-disk/raw/gds-mixed-live-event-driven-575-20260907-five-block
```

## Interpretation and retention

Record all 15 attempts, scheduled-arrival read p50/p99, completed-write
throughput, total bandwidth, decision counts, and feedback inputs/actions.
Report within-block BPF/native and native/FIFO changes separately, with medians
and all five paired values; p99 with 64 reads is the maximum, and observed
ranges are not confidence intervals. Better tail latency at lower write
throughput is a tradeoff, not an unconditional win.

Lower BPF/native cost would support this changed executor on this workload.
An adverse result would motivate examining actual scheduling/service intervals
instead of assuming all cost comes from BPF calls. Historical FIFO performance
changed substantially between campaigns, so do not infer a causal event-driven
versus polling speedup from non-interleaved historical arm medians. A matched
executor ablation would be needed for that separate claim.

Preserve the complete old polling report and raw results. Keep every new
failure or unfavorable result; no performance-threshold selection or retries.
Source, command, raw records, analysis and report are committed and pushed by
the root after implementation and the completed run.
