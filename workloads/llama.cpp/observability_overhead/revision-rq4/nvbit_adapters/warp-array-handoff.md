# Handoff: NVBit kernelretsnoop warp-array variant (matched granularity)

New explicit opt-in NVBit mode for the RTX5090 Table1 comparison: one record
per warp at kernel exit, stored directly in a frozen device-local array,
matching the gpubpf `onevalue-array-candidate` layout and finish semantics.
The original default `OBS_MODE=kernelretsnoop` per-thread channel mode is
unchanged; all old results are preserved and remain the old (unmatched,
23068672 per-thread-record) baseline. Do not claim the old 99.62% as a new
baseline.

## Files changed (project source only)

- `observability/common.h` — adds `OBS_KERNELRETSNOOP_WARP_ARRAY = 4` and
  `struct warp_array_value_t`: 3 reserved u64 header words
  (`total_committed` never written by the device, `total_overflow`,
  `total_out_of_range`) + `warp_counters[16384]` + 32-byte
  `(x, y, z, globaltimer)` `events[16384 * 44]` = 23199768 bytes, enforced by
  static_asserts. Byte-for-byte identical to `onevalue_array_value.h`.
- `observability/inject_funcs.cu` — new `observe_exit` branch: records only
  when `linear_thread % 32 == 0` (actual-exit predicate guard kept),
  `coordinate_x = blockIdx.x * ceil(blockDim.x*blockDim.y*blockDim.z/32) +
  linear_thread/32`, `y = blockIdx.y`, `z = blockIdx.z`; per-warp append via
  `warp_counters[x]` (no per-event channel, no shared global committed
  atomic, no sampling/count-only replacement); out-of-range warp id and
  per-warp capacity (44) go to their own counters.
- `observability/observability.cu` — parses the mode, allocates the buffer
  with `cudaMalloc` (device-local, zeroed), injects the same `observe_exit`
  call with the array pointer, and after the timed target work performs one
  full-buffer `cudaMemcpy` readback whose wall time is reported separately.
  Reuses existing target selection/instrumentation and collector lifecycle;
  no new runtime or framework.
- `observability/Makefile` — restored to the original; no build-mode change.
  Build from a source copy, as the existing Table1 runner already does.

## Opt-in environment

Same as the other NVBit arms, with the new mode value:

```
OBS_MODE=kernelretsnoop_warp_array
OBS_TARGET_SYMBOL=<exact mangled rope_norm symbol>
```

(`OBS_GPU_THREAD_COUNT` is not used by this mode. The startup log prints
`NVBIT_OBS mode_name=kernelretsnoop_warp_array value_bytes=23199768
warp_capacity=16384 events_per_warp=44` so the mode is explicit.)

## Build (distinct scratch tree; historical in-source .so/objects untouched)

Mirror of `run_revision_rq4.py build_nvbit`: copy the source dir (ignoring
`*.o`, `*.so`, `*.fatbin`, `flush_channel.c`), then run the original Makefile
inside the copy with an absolute `NVBIT_ROOT`.

The already-built scratch tree:

```
/tmp/nvbit-warp-array-build-GlqYJ8/observability/
```

Compiled library:

```
/tmp/nvbit-warp-array-build-GlqYJ8/observability/observability.so
```

Rebuild command (same shape, new dir if the scratch dir is gone):

```bash
SCRATCH=$(mktemp -d /tmp/nvbit-warp-array-build-XXXXXX)
mkdir -p "$SCRATCH"
cp -r workloads/llama.cpp/observability_overhead/revision-rq4/nvbit_adapters/observability \
  "$SCRATCH/observability"
find "$SCRATCH" \( -name '*.o' -o -name '*.so' -o -name '*.fatbin' \
  -o -name 'flush_channel.c' -o -name 'clock_domain_test' \) -delete
make -C "$SCRATCH/observability" CXX=g++ \
  NVBIT_ROOT="$PWD/workloads/llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64" \
  ARCH=sm_120
```

(The build above completed with no warnings, CUDA 12.9, sm_120.)

## Diagnostics (stderr, context termination)

`NVBIT warp_array ...` labels: `value_bytes`, `readback_bytes` +
`readback_time_ns` (full-buffer host copy, reported separately from
throughput), `warp_append_counters_total` / `stored_events` (host-side sum of
`warp_counters`), `active_warps`, `coordinate_extent_x`,
`event_coordinate_mismatches`, `reserved_total_committed` (must stay 0),
`overflow_events`, `out_of_range_events`, `nonzero_timestamps`,
`collection_complete` (0/1 diagnostic; the gpubpf onevalue collector likewise
reports without gating).

## Limitations

- Frozen capacity 16384 warps x 44 launches (pp512 geometry, 720896 events);
  larger warp space or more launches surface as `out_of_range_events` /
  `overflow_events`, never silently.
- No device-side atomic on the per-warp counter: same assumption as the BPF
  candidate — each (block, warp) appends at its own index, one launch at a
  time; concurrent multi-stream launches of the same kernel are not covered.
- `coordinate_y != 0` / `coordinate_z != 0` are counted out-of-range, matching
  the BPF candidate's 1D warp address space.
- Buffer is plain `cudaMalloc` device memory (23199768 bytes); the readback
  happens once, after the timed target work, and is excluded from the
  throughput measurement.
- No GPU runs, driver reload, or benchmark were performed here; root owns
  the fresh baseline/gpubpf/NVBit comparison runs and the queued runner
  (unchanged). The scratch dir under /tmp may be cleared on reboot; rerun the
  rebuild command if so.
