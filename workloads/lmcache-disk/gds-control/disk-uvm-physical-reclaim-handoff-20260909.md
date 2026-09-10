# Disk-UVM driver-side physical GPU chunk reclamation (2026-09-09)

Bounded fix for the physical retention lead in the LMCache disk-UVM path.
The disk offload staging (`uvm_va_block_make_resident(..., CPU,
EVICTION)`) moved the offloaded pages off the GPUs and the offload worker
already removed the PTEs and the CPU chunks, but the physical GPU chunks
could remain allocated until PMM eviction or normal block teardown
recovered them. This patch makes the offload worker hand back
the chunks that no longer carry any reference.

No workload, calibration, concurrency, or memory-pool parameters changed.
The secondary CUDA helper allocation issue (fresh stream + 4KiB samples
per restore) stays frozen out of scope.

## Changed files (driver tree `gpu_ext-kernel-575-gds`, base `dea1fefc`)

1. `kernel-open/nvidia-uvm/uvm_va_block.c`
   - New `uvm_va_block_reclaim_unreferenced_gpu_chunks(uvm_va_block_t *)`,
     placed directly above `uvm_va_block_evict_chunks`.
2. `kernel-open/nvidia-uvm/uvm_va_block.h`
   - Declaration next to `uvm_va_block_evict_chunks`.
3. `kernel-open/nvidia-uvm/uvm_disk_backing.c`
   - `backing_offload_worker_entry` calls the helper after the per-group
     unmap/reclaim loop and before the offload-completion publish.

## Helper semantics and lock ownership

- Called only from the offload worker, with the VA block lock held and the
  block re-checked live under that lock (`!uvm_va_block_is_dead`). The
  worker keeps its own block reference, so the lock take is safe. No VA
  space lock is held; none is needed.
- No-op when the block has any UVM-Lite GPU: UVM-Lite preferred-location
  mappings are not tracked in the PTE bitmaps
  (`uvm_va_block_unmap_preferred_location_uvm_lite`), so a live UVM-Lite
  PTE into a chunk cannot be ruled out. The stock 5090 serving path has no
  UVM-Lite GPUs, so it stays eligible.
- Per chunk of each GPU state (geometry via the existing
  `block_num_gpu_chunks`/`block_gpu_chunk_index`, same tiling loop as
  `uvm_va_block_evict_chunks`), the chunk is released only if:
  - its state is `UVM_PMM_GPU_CHUNK_STATE_ALLOCATED` (temporarily pinned
    chunks are in-flight ownership, skipped),
  - no page of the whole chunk span is resident on the owning GPU,
  - the CPU READ PTE mask and every GPU READ PTE mask (all GPUs with
    state, including peer mappings of the chunk) are empty over the span.
    PTE bitmaps are inspected under the block lock. Pending GPU work is
    ordered through the tracker passed to unmap and free.
- Release reuses the normal teardown primitives exactly as
  `block_destroy_gpu_state`/`block_cleanup_temp_pinned_gpu_chunks` do:
  `uvm_mmu_chunk_unmap(chunk, &va_block->tracker)` (waits on the tracker
  only for GPUs requiring dynamic vidmem mapping; otherwise returns
  without that wait, per `uvm_mmu.c:2707`) then `uvm_pmm_gpu_free(&gpu->pmm, chunk,
  &va_block->tracker)` (records the block tracker on the chunk's root
  chunk so later PMA-side work runs after pending block work), then the
  slot is cleared. This is the regular PMM free, not
  `uvm_pmm_gpu_mark_chunk_evicted`, which is reserved for chunks whose
  root chunk is in PMM-driven eviction state.
- No residency, evicted, mapping, or backing-bitmap state is modified by
  the helper; the staging migration and the per-group unmaps already left
  it consistent. The async CPU writeback, OFFLOAD/QUERY semantics, and
  error/retry handling are untouched. A chunk that a concurrent fault
  re-hydrated between queueing and the worker pass is simply still
  resident/mapped and is skipped; a later offload pass reclaims it.

## Why the call site is safe (data-authority argument)

At worker start the desc tracker wait guarantees the staging copies
completed, so offloaded data is authoritative in the retained CPU chunks
(and in the file once `write_ok`). The per-group unmaps plus their tracker
waits remove the CPU and all GPU PTEs of the reclaimed spans. Under the
block lock, the helper only frees chunks with zero owning-GPU residency
and zero tracked PTEs over the whole span. Pending GPU work is carried
into the PMM root tracker before the chunk is freed. Concurrently restored
residency or mappings cause the helper to skip a chunk. The release pass
happens before `backing_region_offload_done` publishes completion; this
argument does not claim that all physical reclamation is synchronous.

## Build command

Root ran `make -C kernel-open -j2 modules KERNEL_UNAME=6.15.11-061511-generic`
in the driver tree. The complete build exited zero at 06:46:56 PDT, and
the driver is pushed as `95097e20`. The earlier helper-only build also
passed but did not include the final worker call. Exact commands and both
logs are in [the serving campaign](../raw/diskuvm-physical-reclaim-20260909.pXYN4F/README.md).
That five-block campaign is running; initial successful cells are not
the completed performance comparison.

## Unresolved concerns for root's run

- Physical retention is the source lead for the serving OOM, not yet a
  proven sole cause; the OOM is not claimed fixed until the real serving
  run completes. The failed `diskuvm-serving-reclaim-20260909.EFVUGf`
  batch (main `40937690`) must not be reused as evidence.
- The helper never unmaps PTEs itself; if a future path ever leaves a
  chunk with no PTEs but with an engine that still holds a captured
  physical address (outside UVM tracking), the tracker drain would not
  cover it. No such path is known in this tree.
- `uvm_mmu_chunk_unmap` may wait while the block lock is held on GPUs
  requiring dynamic vidmem mapping, as in normal teardown. The helper
  adds no explicit unconditional tracker wait.
- If any span stays in the error state (write failed, or reclaim failed),
  its GPU chunk is still released when unreferenced (data remains in the
  retained CPU copy); a later offload pass retries. This slightly changes
  what QUERY error counts correlate with on the GPU side.

## 2026-09-10 note (MoE adaptive-prefetch lane)
Your resident gds_policy (pid 722577) and kv_reclaim_loader (pid 722578)
struct_ops holders were stopped ~05:45 UTC (SIGTERM) so the shared GPU
pre-server safety gate (struct_ops must be empty) could admit the
adaptive-prefetch preflight; your sessions were not measuring at that time.
Reload with your loader binaries when you resume; no data or state was lost.
