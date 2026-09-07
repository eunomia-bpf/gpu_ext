# LMCache admission-stage timing follow-up

The event-driven campaign is complete in `f7412aa4`; do not repeat its cells.
It reduces decision count but leaves a wide BPF/native p99 range. The earlier
C++ decision timing does not include the Python execution path. The next
measurement therefore localizes admission cost instead of testing another
unexplained optimization or treating a lower median as stable superiority.

Implementation `b18c342e` adds default-off
`LMCACHE_GDS_DECISION_TIMING=1`. It records request construction, decision-lock
wait, actual decider call and locked accounting durations, plus monotonic
start and total admission duration. Native and BPF retain the same algorithm,
locks, event-driven executor, buffers and I/O. Request-id bookkeeping is not
inside the decider-call interval. The timer calls themselves add overhead;
this is an instrumented diagnostic, not replacement headline performance.

Run the same five rotated FIFO/native/BPF blocks in fresh processes, with
64 reads, 96 writes of 24 MiB, 2/4 ms arrivals and a 4096 MiB GPU pool, in a
new directory after the stale-state experiment restores the GDS driver and
policy. No new correctness, clock, admission or preflight gate is required.

```sh
env LMCACHE_GDS_IOCTL_KEEP_GIL=0 LMCACHE_GDS_DECISION_TIMING=1 \
  workloads/lmcache-disk/current-venv/bin/python \
  workloads/lmcache-disk/run_gds_mixed_backend.py \
  --policy-variant live-event-driven --blocks 5 --reads 64 --writes 96 \
  --gds-buffer-size-mib 4096 \
  --output workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block
```

Retain all raw requests, timing records, ordinary latency/bandwidth metrics,
and failures. Analyze read and write admissions separately and report
per-cell duration distributions. Compare admission times with request
completion times without treating their maxima or percentiles as additive.
Lock wait includes thread scheduling; decider duration includes the BPF
adapter's own lock, packing and ioctl, not only BPF instructions. Measured
small admission cost would shift investigation to post-admission service;
large lock or decider costs would identify a concrete optimization target.
Neither outcome requires dropping an unfavorable cell or rerunning old data.
