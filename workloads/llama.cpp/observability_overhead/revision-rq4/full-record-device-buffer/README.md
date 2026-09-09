# GPU-local full-record device buffer

GPU-local storage for the kernelretsnoop per-thread kernel-return probe.
Instead of writing each 80-byte record to host-mapped ring memory on the hot
path (the original `BPF_MAP_TYPE_GPU_RINGBUF_MAP` path), the probe keeps the
entire event stream in GPU memory through the existing
`BPF_MAP_TYPE_GPU_ARRAY_MAP` (type 1503) and copies it out in one bounded,
post-run bulk collection. This is bounded capture with post-run collection,
not an unbounded concurrently drained streaming ring; the old full-record
ring remains the comparison and fallback.

## Capacity and layout

All of the original per-thread observation is preserved, unchanged:

- All ten u64 fields per record (block xyz, thread xyz, block-dimension xyz,
  per-thread timestamp) — 80 bytes, `struct frdb_record`.
- 524288 thread slots and 256 records per slot (the full original capacity;
  the observed 23068672-record pp512 stream does not shrink it).
- Every thread is recorded: no leader-only filtering, timestamp
  substitution, count-only aggregation, sampling, or deduplication.

The first eight-bank build declared 1342701584 bytes per value but actually
emitted a BTF size of 268959760 bytes, consistent with an intermediate
size-in-bits truncation. BTF's serialized struct size itself is in bytes.
The unchanged total capacity is now split into **32 equal banks of 16384 thread slots**
(`FRDB_NUM_BANKS` x `FRDB_SLOTS_PER_BANK` = 524288). Each bank value is
`335675408` bytes (about 320 MiB), whose size-in-bits stays below 2^32. The
record-major physical arrangement stores a slot's record k at
`records[k * FRDB_SLOTS_PER_BANK + slot]`, so simultaneous appends by
adjacent threads land 80 bytes apart rather than one 256-record slot apart.

`full_record_device_buffer_value.h` is shared byte-for-byte by the BPF
program and the host collector:

- `total_overflow`: bank-local appends rejected because a slot's 256-record
  capacity was reached.
- `total_out_of_range`: sink for threads whose linear coordinate exceeds the
  524288-slot capacity, counted in bank 0 (an out-of-range coordinate maps to
  no bank).
- `slot_counters[16384]`: one 64-bit append counter per thread slot.
- `records[16384 * 256]`: record-major storage, 80 bytes each.

## Opt-in field-major SoA layout (default OFF)

`make LAYOUT=soa` builds a field-major SoA variant of the same bank value.
It is opt-in and off by default; the default build keeps the measured
record-major AoS layout, the `.output/` object tree, and the
`full-record-device-buffer` binary, and the measured AoS results in
[results-full-record-device-buffer-20260908.md](../results-full-record-device-buffer-20260908.md)
are untouched.

For the SoA build, within each record index k the ten fields become ten
contiguous field planes of 16384 u64 thread slots: `plane[k * 16384 +
slot]`. Simultaneous appends by adjacent lanes then write 8 bytes apart
instead of 80 bytes apart. The BPF writer and the host collector select
the layout from the same `FRDB_SOA_LAYOUT` compile flag through the shared
`full_record_device_buffer_value.h`, so both sides always match.

Unchanged between layouts: all ten u64 fields and per-thread timestamps,
the per-slot counters, 256 records per slot, 32 banks x 16384 slots, the
coordinate mapping and bank ids, the single host drain buffer, and the
post-client 32 whole-value drains. No sampling, deduplication, or
leader-only filtering is introduced. Each bank value stays 335675408
bytes in both layouts, so the observed BTF size truncation remains
avoided.

The SoA build goes to a separate object tree (`.output-soa/`) and a
distinct binary (`full-record-device-buffer-soa`), so toggling `LAYOUT`
cannot reuse objects from the other layout. The collector prints
`Full-record layout: field-major SoA` (the default prints `record-major
AoS`). SoA measurements live in their own raw directories and results
report and do not alter the measured AoS numbers above.

## Opt-in 32-slot grouped SoA (AoSoA) layout (default OFF)

`make LAYOUT=aosoa` builds a 32-slot grouped SoA (AoSoA) variant of the
same bank value. It is opt-in and off by default; the default build keeps
the measured record-major AoS layout, `LAYOUT=soa` keeps the measured
field-major SoA layout, and neither alters the measured results above.

For bank-local slot `s`, record index `k`, and field `f` in [0, 10), the
grouped build places the u64 at `fields[k][s / 32][f][s % 32]`, i.e. u64
index `(((k * (16384 / 32) + s / 32) * 10 + f) * 32 + s % 32)`. All ten
fields of the 32 slots of one group then occupy 2,560 contiguous bytes at
a fixed `k`: the 32 lanes of a group write 8 bytes apart within each
field, and a slot's ten fields stay inside that 2,560-byte window instead
of ten widely separated field planes. These logical 32-slot groups are
not an assumption that every launch geometry maps them to exactly one
physical warp.

The BPF writer and the host collector select the layout from the same
`FRDB_AOSOA_LAYOUT` compile flag through the shared
`full_record_device_buffer_value.h`, so both sides always match.

Unchanged across all three layouts: all ten u64 fields and per-thread
timestamps (80 bytes/event), the per-slot counters, 256 records per slot,
32 banks x 16384 slots, the coordinate mapping and bank ids, the single
host drain buffer, and the post-client 32 whole-value drains. No
sampling, deduplication, or leader-only filtering is introduced. Each
bank value stays 335675408 bytes in every layout, so the observed BTF
size truncation remains avoided.

The AoSoA build goes to a separate object tree (`.output-aosoa/`) and a
distinct binary (`full-record-device-buffer-aosoa`), so toggling `LAYOUT`
cannot reuse objects from the other layouts. The collector prints
`Full-record layout: 32-slot grouped SoA (AoSoA)`. AoSoA measurements
live in their own raw directories and results report and do not alter
the measured AoS and SoA numbers above.

The five-block grouped-SoA versus current-SoA versus uninstrumented
baseline comparison reuses the existing runner pattern in
`paired-aosoa.py`: copy it into the new campaign raw directory next to
the copied `full-record-device-buffer-aosoa` binary (renamed
`kernelretsnoop`), then run it from that directory. The SoA arm reuses
the frozen `raw/full-record-soa-20260908.HWfuRS/kernelretsnoop` binary.

## Coordinate mapping

The BPF program reuses the existing per-thread linear coordinate
(`getGlobalThreadId`: `z*W*H + y*W + x` over the grid/block/thread extents),
computed in u64 before any bank/slot indexing. Bank = `thread_id / 16384`,
slot = `thread_id % 16384`. Out-of-range coordinates (`thread_id >= 524288`)
and per-slot overflow are reported in their own counters, not dropped
silently; the supported geometry is not generalized.

## Collector

The collector owns the skeleton, so the map (and its device buffer) is
created in this process at skeleton load, with a CUDA context initialized
first and kept alive through the drain. After the CUDA client finishes and the
collector receives SIGINT, it performs exactly **32 whole-value
`bpf_map_lookup_elem(bank)` drains**, reusing one host destination allocation
(`sizeof(struct frdb_value)`), and reports the bulk-drain bytes and wall time
separately from llama.cpp prefill token/s:

```
Full-record bank count: 32
Full-record slots per bank: 16384
Full-record total slots: 524288
Full-record records per slot: 256
Full-record record bytes: 80
Full-record value bytes (per bank): 335675408
Full-record drain bytes: 10741613056
Full-record drain time ns: <bulk-copy wall time>
Full-record committed records: <total>
Full-record active slots: <count>
Full-record overflow records: <capacity overflow, reported not hidden>
Full-record out-of-range threads: <outside 524288-slot geometry>
Full-record nonzero timestamps: <count>
```

For the pp512 TinyLlama workload the expected steady state is 23068672
committed records across 524288 slots with 0 overflow and 0 out-of-range and
all timestamps nonzero. A successful collection returns 0; the diagnostic
counts are reported without gating.

## Files

- `full-record-device-buffer.bpf.c` — BPF kretprobe program (map type 1503,
  32 banks, per-thread records).
- `full_record_device_buffer_value.h` — shared bank value layout.
- `full-record-device-buffer.c` — host collector (32 drains, one reused
  buffer).
- `Makefile` — builds `full-record-device-buffer` (BPF object, skeleton,
  collector) and the opt-in `LAYOUT=soa` / `LAYOUT=aosoa` variants.
- `paired-aosoa.py` — five-block AoSoA/SoA/baseline paired runner template
  for the new grouped-SoA comparison (copied into the campaign raw
  directory at run time).
- `.gitignore` — ignores the local `.output*` build trees and binaries.

## Build

The Makefile has been built successfully in the required staged layout;
see the [build record](../raw/full-record-make-build-20260908.KbBhPH/README.md).

The Makefile uses the same relative layout as `example/gpu/*`, so stage this
directory three levels below the bpftime root, e.g.
`/home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-device-buffer`,
then:

```bash
cd /home/yunwei37/workspace/gpu/bpftime-auto-warp
make -C example/gpu/full-record-device-buffer
```

Requirements: clang with BPF target support, the bpftime tree with
`third_party/{libbpf,bpftool,vmlinux}`, and the CUDA toolkit
(`CUDA_HOME=/usr/local/cuda` by default; the collector links `-lcuda` and
includes `<cuda.h>`). The probe already targets the pp512 workload kernel
exit symbol
`_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`.

## Run

Terminal 1 — collector. It initializes its own CUDA context before the map is
created and keeps that context alive through the drain:

```bash
cd /home/yunwei37/workspace/gpu/bpftime-auto-warp
BPFTIME_SHM_MEMORY_MB=<explicit> BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/full-record-device-buffer/full-record-device-buffer
```

The 32 banks allocate 10741613056 bytes in GPU memory. Host shared memory
contains one staging value (335675408 bytes) plus runtime/agent overhead,
not a second copy of the entire GPU arena. The measured runner conservatively
uses `BPFTIME_SHM_MEMORY_MB=12301`, inherited from its ring comparison;
this is a tested setting, not a measured minimum requirement.

Terminal 2 — the existing pp512/tg0 TinyLlama workload with the bpftime agent
preloaded (same agent/launch setup as the kernelretsnoop arms, e.g.
`llama-bench -pp 512 -tg 0 ...` under
`LD_PRELOAD=build/runtime/agent/libbpftime-agent.so`). Prefill throughput
comes from llama-bench itself; the collector does not measure it.

Collector startup order: the first statement in `main` is a read-only
`bpf_map_get_next_id(0, ...)` whose return value is ignored. The preloaded
syscall server starts lazily on its first bpf call; that startup reaches
`cuInit`, and if the application `cuInit` runs first, CUDA's internal
`fopen` re-enters the still-initializing server's own `cuInit` and deadlocks.
Issuing one bpf call first completes the server's lazy startup before the
collector's `cuInit`. This is an initialization-order workaround, not a gate
and not part of the measurement path.

After the pp512 run completes, send SIGINT (Ctrl-C) to the collector; it then
performs the 32 whole-value drains and prints the labels above. The drain
bytes/time are the bulk-collection cost, reported separately from prefill
token/s.

## Limits

The 32-bank x 16384-slot x 256-record capacity is the frozen original
per-thread geometry. A thread coordinate beyond 524288 slots is reported
through `Full-record out-of-range threads`, and a slot beyond 256 records
through `Full-record overflow records`; neither is dropped silently. The
current 335675408-byte banks stay below 512 MiB and avoid the observed
compiler size truncation. No new runtime
bulk-copy API is used.

The [completed five-block report](../results-full-record-device-buffer-20260908.md)
records 23267.229 token/s median GPU-local prefill throughput, 39.099%
paired baseline-relative loss, and 1490.922 ms median post-client copy.
It preserves the older ring results and does not replace Table 1.
