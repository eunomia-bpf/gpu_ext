# Disk/UVM serving with valid descriptor and ioctl handling

## First completed block; five-block campaign still running

Started 2026-09-09 03:36:41 PDT. Sources 8a352587 and a26c57a5 fix
the descriptor mode/flag and ctypes ioctl readback, and permit vLLM
worker drain on shutdown. This is a separate treatment from the
earlier fallback-only runs; none of their completed cells is reused.

| Block 0 arm | Warm token/s | Prepared | Restored | Restore errors / fallback |
| --- | ---: | ---: | ---: | ---: |
| Stock reclaim policy | 69.150960 | 48 | 16 | 52 / 52 |
| Native reclaim policy | 62.769598 | 48 | 5 | 15 / 15 |
| BPF reclaim policy | 61.733256 | 48 | 8 | 12 / 12 |

Each cell completes eight warm requests, 8192 generated tokens and
server exit zero, with no recorded HTTP failures. BPF/native throughput
changes by -1.6510% in this first block.
One block does not establish a stable advantage or overhead estimate.

For the first time these serving cells report successful backing
preparation and completed UVM restorations, rather than only a fallback
path. However, substantial restore errors still invoke the original
GDS read. Thus this is a **hybrid UVM-restoration / GDS-fallback
performance observation**, not a claim that every read uses UVM.
The exact restore exception is currently swallowed by the pre-existing
read wrapper; it is the next implementation issue to expose and repair.
Counters are existing observations, not a gate used to discard timings.

All three arms opt into the same disk/UVM transport. Stock disables
KV reclaim decisions; native and BPF run the same reclaim policy via
their respective implementation paths. This comparison does not
isolate UVM transport versus an untouched GDS transport. Warm-burst
throughput excludes startup, cold population, barriers and shutdown;
it is not the earlier disk-read p99 or pure GPU prefill rate.

run-serving.sh contains the exact five-block command. It reuses the
fixed workload, 62502 ns/token calibration, original fault helper and
previously built disk/UVM driver. Both shared leases cover the entire
campaign. Active GDS/KV loaders are 482025/482026; the existing EXIT trap
will restore the saved original UVM when the run completes.
Restoration remains pending during this live run.

The first-block raw JSON, responses and per-PID counter dumps are
retained under cells/block-00. Later blocks are still running and will
be appended, not treated as completed by this report. Old Table 1,
trampoline, LMCache and failed bring-up records remain unchanged.
No new paper baseline or manuscript edit is introduced.
