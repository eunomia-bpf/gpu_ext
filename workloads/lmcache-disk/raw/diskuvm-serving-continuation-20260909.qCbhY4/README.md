# Disk/UVM serving continuation: second rotated block complete

2026-09-09 PDT, RTX 5090 / CUDA 12.9 / 575.57.08. This continuation ran
02:32:21–02:39:57 PDT. It adopted all three completed block-0 `result.json`
records unchanged, then ran only block 1 in native / BPF / stock order.
The existing measured 62502 ns/token calibration was reused, not rerun.

| Block | Stock token/s | Native token/s | BPF token/s | BPF/native |
| --- | ---: | ---: | ---: | ---: |
| 0, previously published | 73.199047 | 66.625019 | 64.477420 | -3.2234% |
| 1, new | 74.594413 | 74.382520 | 69.010868 | -7.2217% |
| Median of the two | 73.896730 | 70.503769 | 66.744144 | not a paired effect |

Each new cell completes eight warm requests, 8192 generated tokens, and
returns zero with no recorded HTTP request failures. Across both blocks,
all six cells complete 48 warm requests and 49152 generated tokens. Two
paired blocks are not the planned five-block comparison; no confidence
interval or stable mechanism-overhead claim is made. Both observed pairs
are unfavorable to BPF and remain included. The warm-burst metric excludes
startup, cold population, barriers, and shutdown, as in block 0.

The transport and policy hot paths are unchanged. Main adapter `5d726d06`
adds a process-specific counter dump at GDS backend close and suppresses
empty auxiliary-process dumps. Syntax compilation passed. However, the
actual engine shutdown did not call that close function: no corresponding
`GDS backend closed` log or per-PID disk/UVM diagnostic file appears in
the new cells. Thus the counter-collection problem is **not resolved** by
this change. These remain real opt-in serving measurements without exact
restore/fallback attribution; lack of counters does not erase the measured
performance. The installed EngineCore shutdown path is the next bounded
integration target, not an additional experiment gate.

Records are in the original campaign directory:
`../diskuvm-serving-20260909.9uBMtq/cells/block-01/`. Its `raw.jsonl` was
appended and its campaign/summary now cover both blocks. Block-0 request
records and the first-block report remain unchanged. The later collector
helper was still under development when this runner imported its code;
original per-cell diagnostic fields are not rewritten after the run.

`run-serving.sh` preserves the exact `--resume --blocks 2` command and
module lifecycle. Both shared leases cover the run. Compared with the first
lifecycle, restoration waits for the UVM reference count to drain after
the owned policy loaders exit. It then restores the exact saved original
UVM and restarts GDS/KV loaders 389223/389224; both say `attached`.
`lifecycle.log` records `DISK_RUN_EXIT=0 RESTORATION_OK=1`, with no separate
restoration retry or reboot needed. Post-run GPU is idle at 0%, 1 MiB.
No manuscript was edited and no new paper or baseline was introduced.
