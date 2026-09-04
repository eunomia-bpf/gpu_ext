# Device-map implementation and evidence audit

Date: 2026-09-04

This bounded audit answers Reviewer A's question about where cross-layer map
data resides, how device programs access it, and how the CPU reads it back. It
separates source support, CPU-only tests, and retained RTX 5090 evidence. No GPU
work was run for this audit.

## Short answer

The implementation supports three relevant placement classes:

1. CUDA device-global allocations, exposed to an instrumented kernel through
   the trampoline (`1501`, `1502`, and `1503`);
2. host allocations registered and mapped into the CUDA address space
   (`1512`, `1513`, and the ring storage used by `1527`); and
3. a Linux kernel BPF array whose mmap pages are registered and mapped into the
   CUDA address space (`1504`).

The third class is **not CUDA block shared SRAM**. The name
`BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP` means a map shared among the Linux
kernel, host process, and GPU. Its storage is mapped host memory. No audited map
type allocates values in per-block on-chip CUDA shared memory. Paper text should
therefore say "local/warp aggregation" only where the actual program performs
it, and should not describe map type 1504 as an on-chip shared-memory shard.

The strongest retained device result is the cross-layer raw-record campaign:
on RTX 5090 / driver 575.57.08, type 1527 carried 34,560 distinct bounded raw
tuples to the host, while type 1502 supplied a separate aggregate control. The
overflow arm reported all 2,560 omitted records and was rejected as incomplete.
This demonstrates bounded non-composable device-to-host state, not arbitrary
objects, coherent snapshots, automatic placement, or map performance.

For this audit, the existing CPU-only ring test passed 53 assertions in six
cases, and the GPU-map thread-count tests passed 23 assertions in two cases,
with both processes pinned to CPUs 16--23. These results validate only the
host-side boundaries described below; they do not consume or validate a GPU.

## Evidence levels

- **Source support:** allocation, device helper dispatch, loader publication,
  and host lookup/readback paths exist in the current source tree.
- **CPU-only test:** a retained test exercises parsing, layout, draining, or
  lowering without executing a CUDA workload. This does not validate device
  addressability or device/host visibility.
- **RTX 5090 result:** a retained, analyzed run proves device execution and
  host readback on the named hardware and driver. Examples and old console
  output are not counted as formal evidence.

## Map-by-map audit

| ID / map type | Physical placement and device path | Host path | Source support | CPU-only test | Retained RTX 5090 result |
| --- | --- | --- | --- | --- | --- |
| 1501 `GPU_HASH_MAP` | Values use the device array allocator; keys and occupancy metadata remain in runtime shared memory. The current trampoline has no direct 1501 lookup/update branch, so a device BPF helper falls through to the serialized host helper path. | Runtime open addressing plus device-array value copies. | Present, but the split key/value design and fallback path need a dedicated semantics test. | None located. | None accepted. Do not call this a fully device-resident direct-access map. |
| 1502 `PERGPUTD_ARRAY_MAP` | CUDA device-global allocation; direct helper computes `key × thread_count + global_thread_id`. | Whole per-thread slice copied with GDRCopy when available, otherwise CUDA device-to-host copy. | Present. | Thread-count parsing and bounds only. | Yes: aggregate control in `cross-layer-raw-map`; also exercised by the current device observability harness. |
| 1503 `GPU_ARRAY_MAP` | Single CUDA device-global allocation; direct lookup returns a key offset and direct update copies into it. | One value copied with GDRCopy when available, otherwise CUDA device-to-host copy. | Present. | No map-semantics test located. | No standalone formal placement/readback result located. Several examples exist, but their output was not retained as a controlled result. |
| 1504 `GPU_KERNEL_SHARED_ARRAY_MAP` | Linux BPF array mmap pages are registered with `cuMemHostRegister` and mapped with `cuMemHostGetDevicePointer`; the trampoline uses the same direct array branch as 1503. A runtime-shared-memory fallback exists when no kernel map is supplied. | Direct mmap access, with a Linux BPF syscall fallback. | Present. | No end-to-end semantics test located. | No formal result located. It must not be described as CUDA on-chip shared memory. |
| 1512 `PERGPUTD_ARRAY_HOST_MAP` | Per-thread array in runtime-managed host memory registered/mapped for CUDA; direct helper uses the per-thread offset and a system fence. | Direct host-memory lookup. | Present. | No map-semantics test located. | No formal result located. |
| 1513 `GPU_ARRAY_HOST_MAP` | Single-copy array in runtime-managed host memory registered/mapped for CUDA; direct array lookup/update plus a system fence. | Direct host-memory lookup. | Present. | No map-semantics test located. | No formal result located. |
| 1527 `GPU_RINGBUF_MAP` | Per-thread bounded rings in mapped host memory; device reserve/submit logic uses system-scope atomics and explicit drop counters. | `bpftime_poll_gpu_ringbuf_map` drains records; a stats API exposes pending, collected, and drop counts. | Present. | Ring layout, drain, alignment, corrupt-record, overflow-state, and unsupported-operation tests use host shared memory only. | Yes: 15/15 cross-layer raw-record cells, including deliberate overflow detection. |

## Lowering and publication path

This is helper dispatch, not a separate map-specific LLVM lowering pass:

1. the PTX pass registers the normal eBPF map helpers, including lookup helper
   1 and update helper 2;
2. the generated PTX calls trampoline functions such as
   `_bpf_helper_ext_0001` and `_bpf_helper_ext_0002`;
3. the CUDA attach context builds one `MapBasicInfo` record per map from the
   runtime handler, including map type, geometry, thread bound, and the
   device-visible buffer pointer;
4. the loader copies that table into the patched module's constant
   `map_info` array; and
5. the trampoline selects a direct device/mapped-memory path for array maps or
   the device ring, and otherwise uses the serialized host helper channel.

This implementation supports raw records as well as reducible summaries. It
does not provide a general coherent distributed object: concurrent single-copy
array writes are last-writer-wins, host observations may be stale, and readers
must choose synchronization appropriate to the selected map and data format.

## Audited source and runnable entry points

The implementation inventory is in the sibling `bpftime-table1-575` checkout:

- declarations: `runtime/include/bpftime_shm.hpp`;
- runtime construction/publication: `runtime/src/handler/map_handler.cpp` and
  `runtime/src/attach/bpf_attach_ctx_cuda.cpp`;
- device helper dispatch: `attach/nv_attach_impl/trampoline/default_trampoline.cu`;
- device/global and host-backed storage: `runtime/src/bpf_map/gpu/`;
- Linux-kernel-array bridge: `runtime/src/bpf_map/gpu_kernel_shared/`;
- CPU-only tests: `runtime/unit-test/cuda/test_gpu_ringbuf.cpp` and
  `runtime/unit-test/maps/test_gpu_thread_count.cpp`;
- executable examples: `example/gpu/gpu_shared_map/` (1501/1502/1503),
  `example/gpu/host_map_test/` (1512/1513), and
  `example/gpu/threadhist-gpu-kernel-shared-map/` (1504).

The `gpu_shared_map` README currently calls 1503 "UVA Zero-Copy," but its
audited implementation allocates device memory and copies values for host
lookup. That README is not publication evidence and should be corrected in a
separate bpftime change before reuse.

The formal raw-record result and analyzer are:

- `workloads/cross-layer-raw-map/results-full-575-02.md`;
- `workloads/cross-layer-raw-map/analyze_raw_map.py`; and
- `workloads/cross-layer-raw-map/raw/full-575-02/`.

## Smallest useful next experiment

The next device-map experiment should be a matched placement comparison, not a
new broad hierarchy claim:

1. compare 1503 (CUDA device-global array) against 1513 (host-mapped array)
   under identical lookup and update work;
2. include native and no-op controls, deterministic written values, full host
   readback, randomized complete blocks, and independent analysis;
3. report lookup/update cost and exact correctness separately; and
4. label the result as a two-placement comparison on RTX 5090, not universal
   hierarchy optimization.

The in-progress plan under `microbench/fig15-device/` targets this question,
but it is not counted here as completed evidence. After that experiment, a
correctness-only 1504 round trip can establish Linux-kernel-array → GPU update
→ host/kernel readback. A true per-block CUDA shared-SRAM map would require a
new ABI, lifetime rules, lowering, and verifier treatment; it should not be
promised as a quick follow-up.

## Safe paper wording

> gpubpf exposes map implementations backed by GPU global memory and by host
> memory mapped into the GPU address space. Device hooks can publish both
> per-thread summaries and bounded raw records for host consumption. The
> current prototype does not provide a general coherent snapshot across these
> domains or a map backed by CUDA block shared memory; synchronization and
> aggregation are policy- and map-specific.
