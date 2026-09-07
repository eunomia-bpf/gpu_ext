# onevalue-array candidate

Single-value GPU_ARRAY map replacement for the kernelretsnoop GPU ring buffer.
It uses the existing `BPF_MAP_TYPE_GPU_ARRAY_MAP` (type 1503,
`nv_gpu_shared_array_map_impl`) with `max_entries=1`: the one map value holds
the entire record set, so the whole result comes back as one host lookup of
the single value — no runtime bulk-copy API is needed.

## Value layout (frozen)

`onevalue_array_value.h` is shared byte-for-byte by the BPF program and the
host collector:

- 3 reserved u64 header words: `total_committed` (never written by the
  device; committed total is derived host-side as the sum of the per-warp
  counters), `total_overflow` (appends rejected because the per-warp capacity
  was reached), `total_out_of_range` (warp coordinates outside the
  1D 16384-warp address space).
- `warp_counters[16384]`: one 64-bit append counter per warp.
- `events[16384 * 44]`: one 32-byte
  `(coordinate_x, coordinate_y, coordinate_z, timestamp)` event per warp,
  capacity 44 launches per warp.

Frozen capacity: 16384 warps x 44 launches = 720896 events, matching the
existing Table 1 pp512 geometry (extent 16384 x 1 x 1, 44 launches,
720896 events). Counters plus events are 22.125 MiB; the full value is
23199768 bytes.

The BPF program keeps the current warp-leader semantics from
`kernelretsnoop-phase-capacity.patch` (only `(linear_thread & 31) == 0`
records; `coordinate_x = block_x * warps_per_block + (linear_thread >> 5)`),
looks up key 0 to obtain the device pointer, and does all bounded indexing
and stores on the GPU. It never copies the value onto the BPF stack and never
uses `bpf_map_update_elem` on the arena. There is no sampling, no
count-only replacement of records, and no dropped record presented as
throughput: capacity overflow and out-of-range warps are reported in their
own counters.

## Files

- `onevalue-array.bpf.c` — BPF kretprobe program (map type 1503, key 0).
- `onevalue_array_value.h` — shared record layout.
- `onevalue-array.c` — host collector.
- `Makefile` — builds `onevalue-array` (BPF object, skeleton, collector).
- `.gitignore` — ignores the local `.output/` build tree and binary.

## Build

The Makefile uses the same relative layout as `example/gpu/*`, so stage this
directory three levels below the bpftime root, e.g.
`/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt/example/gpu/onevalue-array`,
then:

```bash
cd /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt
make -C example/gpu/onevalue-array
```

Requirements: clang with BPF target support, bpftime tree with
`third_party/{libbpf,bpftool,vmlinux}`, CUDA 12.9 toolkit
(`CUDA_HOME=/usr/local/cuda` by default; the collector links `-lcuda` and
includes `<cuda.h>`).

The shipped probe annotation targets `vectorAdd`
(`SEC("kretprobe/_Z9vectorAddPKfS0_Pf")`). For the original Table 1 pp512
llama.cpp workload, rewrite it to the exact target symbol used by the
harness:

```
SEC("kretprobe/_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli")
```

(the mangled `rope_norm<true,0,float,...>` kernel exit symbol).

## Run

Terminal 1 — collector. It initializes its own CUDA context before the map is
created (skeleton load) and keeps that context alive through the drain:

```bash
cd /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt
BPFTIME_SHM_MEMORY_MB=256 BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/onevalue-array/onevalue-array
```

Terminal 2 — the existing Table 1 llama.cpp pp512 workload with the bpftime
agent preloaded (same agent/launch setup as the kernelretsnoop arm, e.g.
`llama-bench -pp 512 -tg 0 ...` under
`LD_PRELOAD=build/runtime/agent/libbpftime-agent.so`). Prefill throughput
comes from llama-bench itself; the collector does not measure it.

Collector startup order: the first statement in `main` is a read-only
`bpf_map_get_next_id(0, ...)` whose return value is ignored. The preloaded
syscall server lazily starts up on its first bpf call; that startup reaches
`cuInit`, and if the application `cuInit` runs first, CUDA's internal
`fopen` re-enters the still-initializing server's own `cuInit` and deadlocks.
Issuing one bpf call first completes the server's lazy startup before the
collector's `cuInit`. This is an initialization-order workaround, not a gate
and not part of the measurement path.

After the pp512 run completes, send SIGINT (Ctrl-C) to the collector. It then
performs exactly one whole-value `bpf_map_lookup_elem(key=0)` drain and
prints, among stable labels:

```
One-value map type: 1503
One-value map value bytes: 23199768
One-value drain bytes: 23199768
One-value drain time ns: <bulk-copy wall time>
Warp append counters total: <derived committed total>
Stored events: <derived committed total>
Coordinate extent x: <max warp id + 1>
Event coordinate mismatches: <count>
Reserved total committed (device): <must stay 0>
Overflow events: <capacity overflow, reported not hidden>
Out-of-range events: <outside 1D warp space, reported not hidden>
Nonzero timestamps: <count>
```

The drain bytes/time are the bulk-collection cost, reported separately from
prefill token/s. A successful collection returns 0; the diagnostic counts
above are reported without gating. For the pp512 workload the expected steady
state is 720896 stored events, 16384 active warps, extent x 16384, and 0
overflow / 0 out-of-range / 0 mismatches.

## Limits

The 16384-warp x 44-launch capacity is frozen for the measured pp512
geometry. A larger warp space or more launches than 44 per warp is reported
through `Out-of-range events` / `Overflow events`, not dropped silently.
