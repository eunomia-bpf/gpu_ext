# Runtime facts for the two kernelretsnoop candidates

These are implementation pointers, not new experiment gates. No optimized
performance is claimed here. Both local-model tasks preserve existing sources
and results and produce separately selectable candidates.

## Existing measurement geometry

The first gpubpf kernelretsnoop record in
`results-table1-warp-plt-575-06/cells.json` reports 720896 committed/collected
32-byte events, 16384 coordinates, 44 launches, extent (16384,1,1), and no
drops. Thus its actual payload is 22 MiB. The current ring allocates 524288
physical-thread slots, each with a 24-byte header and 44 aligned 40-byte
records: 935329824 bytes including its 32-byte error-counter block, about
892 MiB. This does not mean capacity can simply be reduced in the current
runtime: its slot indexing and the normalized event coordinate are distinct.

## Existing device-array support

In `/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt`:

- `runtime/src/handler/map_handler.cpp:1019` constructs
  `nv_gpu_shared_array_map_impl` via the managed-memory constructor. The ring
  constructor is in the same file around line 1075, not a `make_unique`
  factory in `bpftime_shm_internal.cpp`.
- `runtime/src/bpf_map/gpu/nv_gpu_shared_array_map.cpp` implements map type
  1503 with `cuMemAlloc`, zeroing and a CUDA IPC handle. The owner CUDA context
  must exist when the map is created.
- Its `elem_lookup` copies **one entire map value** to host staging. Therefore
  a single-entry map can hold an event arena and provide one bulk host lookup,
  without a new runtime bulk-copy API. A 64-bit count plus 44 events per warp
  would require 22.125 MiB before any additional metadata/padding.
- A map-value pointer must be used directly in BPF; never create that arena
  on the BPF stack. Preserve each coordinate and timestamp, and report any
  capacity overflow without hiding it or creating a performance-admission gate.

## Collection boundary

Keep the owner/collector alive through target completion and final lookup.
Report prefill token/s using the existing workload and separately record the
bulk-copy/collection wall time and bytes. Moving collection outside prefill
does not make its cost disappear. No host/device clock-alignment experiment
is needed. Existing P40 and RTX 5090 results stay intact.

The generic candidate belongs in `device-array-candidate/`; the single-value
candidate belongs in `onevalue-array-candidate/`. Shared runtime/runner files
are not owned by either task. Do not launch nested agents for further searching.

## Local-model execution record

Qwen Next session `ses_f8698e3ebffeLAt5A6TpaHM9i5` returned CLI exit 1 before
producing source. Its stored assistant error is `APIError`, HTTP 524. The
root did not cancel it or impose a short timeout. A fresh Qwen 27B session,
titled `kernelretsnoop-onevalue-implementation`, takes over this same candidate
after that confirmed terminal error. The original GLM candidate and LMCache
Qwen 27B session remain running; no fourth root OpenCode run is started.
