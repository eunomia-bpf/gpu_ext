# Can disk become an automatically offloaded UVM tier? 575 source boundary

Date: 2026-09-07. Read-only source inspection. No driver changes, no GPU runs,
no edits to files owned by root or the two local sessions.

Source inspected (driver 575.57.08, branch `test-sched`, gpubpf-enabled tree,
per `gpu_ext-kernel-575-gds/GPUBPF-RUNTIME-575.md` and `version.mk`
`NVIDIA_VERSION = 575.57.08`):

- `gpu_ext-kernel-575-gds/kernel-open/nvidia-uvm/uvm_hmm.c` (3790 lines)
- `gpu_ext-kernel-575-gds/kernel-open/nvidia-uvm/uvm_va_block.c` (13711 lines)
- `gpu_ext-kernel-575-gds/kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c` (663 lines), `uvm_bpf_struct_ops.h`
- call sites: `uvm.c`, `uvm_perf_prefetch.c`, `uvm_pmm_gpu.c`,
  `uvm_stale_state_v1.c`, `uvm_gpu_replayable_faults.c`,
  `uvm_gpu_non_replayable_faults.c`
- existing adapters in `gpu_ext/workloads/lmcache-disk/gds-control/`:
  `gds_policy.bpf.c`, `gds_policy.c`, `gds_executor.cu`,
  `lmcache_gds_backend_adapter.py`, `lmcache_gds_policy_adapter.py`

Question: in this 575 implementation, does file-backed HMM memory
automatically migrate to the GPU on access? Short answer: no. File-backed
(non-anonymous) HMM VMAs are forced to CPU residency in this HMM path; a
GPU fault is served by remote-mapping the CPU-resident page, never by
copying data into device memory. The restriction is a hard-coded check,
not BPF-reachable policy state.

## 1. Where the restriction is enforced (source-supported fact)

- Policy gate: `uvm_hmm_must_use_sysmem()` at `uvm_hmm.c:3768-3785`.
  Returns true when `!vma_is_anonymous(vma) || (vm_flags & VM_SPECIAL) ||
  vma_is_dax(vma) || is_vm_hugetlb_page(vma)`. A file-backed VMA is
  non-anonymous, so it is always sysmem-only. In-source comments mark the
  gap as unimplemented: "TODO: Bug 3660968: Remove this hack as soon as HMM
  migration is implemented for VMAs other than anonymous private memory"
  and "TODO: Bug 3660968: add support for file-backed migrations"
  (`uvm_hmm.c:3771-3782`); related TODOs: THP migration (Bug 3368756,
  `uvm_hmm.c:3781`), swap-cached pages (Bug 4050579,
  `uvm_va_block.c:11578-11579` and `uvm_hmm.c:2752-2758`), atomics on
  MAP_SHARED files (Bug 4014681, `uvm_hmm.c:2514-2521`).

- Residency selection: `block_select_processor_residency()` at
  `uvm_va_block.c:11559-11590`. `if (is_uvm_fault_force_sysmem_set() ||
  !hmm_migratable || uvm_hmm_must_use_sysmem(va_block,
  va_block_context->hmm.vma)) { *read_duplicate = false; return UVM_ID_CPU; }`
  (`uvm_va_block.c:11581-11586`). So even when the faulting processor is a
  GPU, the new residency for a file-backed HMM page is the CPU. Reached via
  `uvm_va_block_select_residency()` (`uvm_va_block.c:11762`) from the
  replayable-fault service (`uvm_gpu_replayable_faults.c:1549`) and the
  non-replayable-fault service (`uvm_gpu_non_replayable_faults.c:408`).

- Mapping: `map_get_allowed_destinations()` at `uvm_va_block.c:8543-8580`.
  The branch that would give the faulting GPU a local (device) copy
  explicitly excludes the `uvm_hmm_must_use_sysmem()` case
  (`uvm_va_block.c:8565-8572`); file-backed pages fall to "Common case: Just
  map wherever the memory happens to reside" (`uvm_va_block.c:8573-8580`),
  i.e. the GPU gets a remote sysmem mapping of the CPU page.

- Fault service: `uvm_hmm_va_block_service_locked()` at `uvm_hmm.c:2951`.
  When `new_residency` is the CPU it returns
  `hmm_block_cpu_fault_locked()` (`uvm_hmm.c:2618`), which uses the kernel's
  `hmm_range_fault()` (`uvm_hmm.c:2407`, `uvm_hmm.c:3648`) to populate the
  VMA's page — for a file-backed VMA the kernel VFS/page cache supplies the
  bytes from the backing store — then `populate_region()` and
  `uvm_va_block_service_copy(processor_id, UVM_ID_CPU, ...)`
  (`uvm_hmm.c:2671-2682`; `uvm_va_block_service_copy` at
  `uvm_va_block.c:11883` calls `uvm_va_block_make_resident_copy(...,
  UVM_ID_CPU, ...)`). No device-private copy is created.

- Atomic GPU faults: `hmm_block_atomic_fault_locked()` at
  `uvm_hmm.c:2474`. For `vma->vm_flags & (VM_SHARED | VM_HUGETLB)` the fault
  returns `NV_ERR_NOT_SUPPORTED` (`uvm_hmm.c:2518-2521`, Bug 4014681 comment
  at `uvm_hmm.c:2514-2517`). Otherwise it keeps the data in sysmem by
  making the page device-exclusive
  (`hmm_make_device_exclusive_range()` at `uvm_hmm.c:2441`, via the kernel's
  `make_device_exclusive*`), populates the CPU page
  (`hmm_va_block_cpu_page_populate`), and services with
  `uvm_va_block_service_copy(processor_id, UVM_ID_CPU, ...)`
  (`uvm_hmm.c:2579-2581`). Even atomic HMM traffic in this tree stays
  CPU-resident; only anonymous private memory ever gets device-private
  pages.

- Manual migration: `uvm_hmm_va_block_migrate_locked()`
  (`uvm_hmm.c:3169`) takes `dest_id` and records it in
  `uvm_hmm_migrate_event.dest_id` (`uvm_hmm.c:3201`); the `UVM_MIGRATE`
  ioctl path passes through the requested destination, which may be a GPU
  (`uvm_migrate.c:243`, cause `UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE`). For
  file-backed or VM_SPECIAL VMAs the kernel's `migrate_vma_setup()` does
  not handle the VMA, so `migrate_vma_setup_locked()` fails with -EINVAL
  and UVM falls back to making the pages CPU-resident: "Note that
  migrate_vma_setup() doesn't handle file backed or VM_SPECIAL VMAs so if
  UvmMigrate() tries to migrate such a region, -EINVAL will be returned
  and we will only try to make the pages be CPU resident"
  (`uvm_hmm.c:3214-3225`, fallback `hmm_make_resident_cpu()`). When a GPU
  destination does proceed, `dmamap_src_sysmem_pages()`
  (`uvm_hmm.c:2727`) still skips any page whose `src_pfn` is not
  `MIGRATE_PFN_MIGRATE`: "HMM currently has some limitations on what pages
  can be migrated. For example, no file backed pages, device private pages
  owned by a different device, device exclusive or swapped out pages"
  (`uvm_hmm.c:2746-2749`). Net effect: a file-backed HMM page never becomes
  GPU-resident in this source. (The eviction path is a different function:
  `hmm_va_block_evict_chunks()` at `uvm_hmm.c:3359` does initialize
  `.dest_id = UVM_ID_CPU` (`uvm_hmm.c:3376`), because HMM eviction moves
  device-private pages back to the CPU.)

- HMM admission gates (context): `uvm_hmm_is_enabled_system_wide()`
  (`uvm_hmm.c:132-149`: disabled by `uvm_disable_hmm`, by ATS, by
  confidential computing, and requires the VA space mm enabled) and
  `uvm_hmm_vma_is_valid()` (`uvm_hmm.c:565-580`: rejects
  `userfaultfd_armed(vma)`, VM_IO/VM_PFNMAP, UVM-managed VMAs, and VMAs
  without VM_READ unless explicitly allowed). An mmap-anonymous plus
  userfaultfd VMA is therefore not a drop-in HMM shortcut in this source.

Conclusion (source-supported fact): in 575.57.08 there is no automatic or
on-access migration of file-backed HMM data to the GPU. The upstream source
itself marks file-backed HMM migration as an open bug (3660968). This is
consistent with, but not derived from, the current public CUDA guide's
statement that file-backed memory stays CPU-resident by default and does not
migrate on access.

## 2. Writeback / completion / mapping ownership (source-supported fact)

- File-backed HMM page: the kernel's page cache (the file's address_space)
  owns the page, its dirtiness, and writeback to the backing store. UVM
  never writes file data to disk. When Linux invalidates the range, the
  mmu_interval notifier callback `hmm_invalidate()`
  (`uvm_hmm.c:408-508`) does only: unmap the GPUs via
  `uvm_va_block_unmap()` ("We only need to unmap GPUs since Linux handles
  the CPUs", `uvm_hmm.c:480`), `uvm_va_block_munmap_region()` on
  MMU_NOTIFY_UNMAP/CLEAR (`uvm_hmm.c:495-496`), and
  `uvm_va_block_remove_cpu_chunks()` (`uvm_hmm.c:501`). UVM's role for a
  file-backed page is mapping ownership (GPU PTEs) and fault-in
  orchestration; data ownership and writeback are the kernel's.

- Device-private (anonymous HMM) page: UVM owns the vidmem<->sysmem copy
  (`hmm_copy_devmem_page()` at `uvm_hmm.c:157`,
  `uvm_hmm_migrate_alloc_and_copy()` at `uvm_hmm.c:3120`,
  `uvm_hmm_va_block_migrate_locked()` at `uvm_hmm.c:3169`,
  `uvm_hmm_va_block_evict_chunks()` /
  `uvm_hmm_va_block_evict_pages_from_gpu()` at `uvm_hmm.c:3440` /
  `uvm_hmm.c:3454`). Writeback of that data to any backing store is not a
  UVM concept; once data is in sysmem, the kernel owns it, and
  swap-cached pages are explicitly rejected in the HMM migration path
  (`nv_PageSwapCache` -> `NV_WARN_MISMATCHED_TARGET`, `uvm_hmm.c:2752-2758`,
  Bug 4050579).

- Fault completion: replayable/non-replayable UVM fault service completes
  the GPU fault after residency is made correct; for file-backed HMM the
  completion is "page faulted in on CPU by the kernel + remote GPU PTE
  installed", never "data copied to the GPU".

## 3. Which existing gpubpf hooks actually cover those blocks

The 575.57.08 `test-sched` tree exposes two struct_ops interfaces
(`uvm_bpf_struct_ops.c:28-79`): `gpu_mem_ops` with
`gpu_test_trigger`, `gpu_page_prefetch`, `gpu_page_prefetch_iter`,
`gpu_block_activate`, `gpu_block_access`, `gpu_evict_prepare`,
`gpu_stale_state_prefetch_v1`, and `gpu_storage_ops` with
`gpu_storage_decide`; plus kfuncs `bpf_gpu_set_prefetch_region`,
`bpf_gpu_request_reorder`, `bpf_gpu_stale_state_v1_request`,
`bpf_gpu_storage_record` (`uvm_bpf_struct_ops.c:177-262`).

Call sites:

- `gpu_page_prefetch` / `gpu_page_prefetch_iter`:
  `uvm_perf_prefetch.c:131` / `uvm_perf_prefetch.c:192`, inside
  `compute_prefetch_region()` (consumed via
  `uvm_perf_prefetch_get_hint_va_block()`, `uvm_va_block.c:11846`). These
  shape how many extra pages are prefetched within the already-selected
  residency; they cannot change residency or page ownership. For a
  file-backed block the hint applies to CPU-resident pages only.
- `gpu_block_activate` / `gpu_block_access` / `gpu_evict_prepare`:
  `uvm_pmm_gpu.c:831` / `uvm_pmm_gpu.c:1661` / `uvm_pmm_gpu.c:1719` — PMM
  GPU-chunk eviction-list ordering. This governs eviction of UVM-allocated
  GPU chunks, not HMM file-backed residency.
- `gpu_stale_state_prefetch_v1`: `uvm_stale_state_v1.c:514`, called from
  `compute_prefetch_region()` (`uvm_perf_prefetch.c:123`) as the
  versioned read-only decision before the prefetch hooks.
- `gpu_storage_decide`: `uvm_api_gpu_storage_decide()`
  (`uvm.c:1020`), BPF consulted at `uvm.c:1058` via
  `uvm_bpf_call_gpu_storage_decide()`. It is a userspace-initiated ioctl
  (`UVM_GPU_STORAGE_DECIDE`, registered in the ioctl table at
  `uvm.c:1102`), one call per object-level storage request (op read/write,
  flags DEMAND/SPECULATIVE/RECOMPUTABLE/SAFE_TO_DEFER, hbm pressure,
  deadline/slack, recompute estimate, queue depth). BPF may only record a
  decision via `bpf_gpu_storage_record()` (`uvm_bpf_struct_ops.c:216-262`):
  SUBMIT_NOW / DEFER (defer_ns clamped to 10 ms max) / RECOMPUTE, with
  priority <= 7 and batch_target 1..64. "The kernel handler never sees or
  stores an fd, file offset, GPU pointer, CUDA stream, or completion"
  (`uvm_bpf_struct_ops.c:71-75`).

Conclusion: none of the existing gpubpf hooks sits on the file-backed HMM
restriction path (`uvm_hmm_must_use_sysmem` -> residency -> mapping). The
hard-coded `!vma_is_anonymous(vma)` check has no policy data, no hook, and
no BPF-visible state in this tree. The closest existing surface to disk
offload is the object-level `UVM_GPU_STORAGE_DECIDE` + BPF policy, which is
exactly what the gds-control workloads already use:
`gds_policy.bpf.c` implements `gpu_storage_decide` with the SUBMIT_NOW /
DEFER / RECOMPUTE precedence rules; `lmcache_gds_policy_adapter.py` packs
the 136-byte command-82 ABI and drives fifo/native/BPF deciders;
`lmcache_gds_backend_adapter.py` decides per cache key before
`GdsBackend._save_gds` / `_load_gds` (LMCache owns the CuFile objects, fds,
GPU addresses, streams, and I/O completion); `gds_executor.cu` exercises
the cuFile compatibility path
(`CUFILE_PARAM_FORCE_COMPAT_MODE=true`, `gds_executor.cu:423`), consistent
with the GDS overview statement that direct P2P does not support migratable
managed memory and compatibility buffers are used.

## 4. Integration options: object-level disk KV offload vs
transparent same-address SSD paging

(a) Object-level disk-backed KV offload (stays inside the measured
boundary; no UVM source change required — source-supported):

- Keep KV objects in explicitly allocated buffers (cudaMalloc / registered
  memory), not in file-backed HMM VMAs. In this driver a file-backed HMM
  VMA is forced to CPU residency and can only give the GPU a remote
  sysmem mapping (section 1); it cannot be promoted to the GPU.
- Keep per-object admission on the existing
  `UVM_GPU_STORAGE_DECIDE` ioctl + BPF policy (SUBMIT_NOW/DEFER/RECOMPUTE,
  defer <= 10 ms, batch 1..64); the kernel returns a scheduling decision
  only, I/O execution stays in cuFile.
- Ownership model already in place: LMCache owns handle/fd/GPU pointer/
  stream/completion; BPF records decisions; UVM performs no storage I/O.
- The current async LMCache serving work (async backend/bootstrap and
  runner, owned by root and the two local sessions; this note does not
  touch those files) extends the retrieval side: per-object admission
  decisions plus cuFile execution and completion. A complete object-level
  unified offload lifecycle is not yet covered by that work and would
  additionally need the following proposed application-layer integration,
  reusing existing LMCache responsibilities where available:
  - object registration/versions: a stable object_id and versioning
    across save/restore so policy decisions and on-disk objects agree on
    what a cache key points to;
  - dirty-vs-backed state: tracking which object versions exist on disk
    versus only in HBM, so admission can choose between SUBMIT_NOW and
    RECOMPUTE correctly;
  - completed-write-before-release ordering: a deferred write must not
    release or reuse its source buffer until the write completion is
    observed;
  - restore-before-consumer ordering: a demand read must land in the
    buffer before the consuming kernel reads it;
  - reclaim coordination: HBM copy eviction/reclaim must respect
    in-flight read/write decisions from the policy path.
- No driver work is required for object-level offload while it stays
  inside the boundary above.

(b) Truly transparent same-address fault-driven SSD paging (SSD as an
automatically offloaded UVM tier) — not supported by the inspected path.
These are design requirements and options, not a proven minimal or exhaustive
driver patch set:

1. A file-backed HMM approach needs migration support before changing
   `uvm_hmm_must_use_sysmem()`. Removing the restriction alone does not make
   `migrate_vma_setup()` support these pages; see section 1 and Bug 3660968.
2. Define backing-data ownership, dirty tracking, writeback, GPU mapping
   invalidation, and fault completion together. Cooperation with Linux's
   page cache or a separately managed backing store are design alternatives;
   this source does not prove that UVM itself must own every I/O.
3. Either support shared/atomic accesses or explicitly constrain the initial
   abstraction. Existing MAP_SHARED atomic restrictions cannot be silently
   removed by a placement policy.
4. Define interaction with swapped and swap-cached pages, given the migration
   limitations. Bypassing swap is one possible choice, not a requirement
   established by this source inspection.

These proposed extensions have not been implemented or measured by this
task. The restrictions rule out a configuration-only solution, not every
possible future driver or runtime design.

## 5. Classification

- Source-supported fact: sections 1-3 (file:line references above), plus the
  fact that this tree is driver 575.57.08 `test-sched` with the
  `gpu_mem_ops`/`gpu_storage_ops` interfaces.
- Proposed changes: the automatic object-lifecycle integration in section
  4(a) and design options in section 4(b), not completed implementations.
- Runtime capability not measured: (i) latency/bandwidth of GPU remote
  sysmem access over file-backed HMM ranges on this host; (ii) cuFile
  compatibility-path throughput under concurrent serving load
  (`gds_executor.cu` is a benchmark harness, its numbers are not part of
  this note); (iii) HMM enablement state on the live host
  (`uvm_disable_hmm` module parameter, ATS/CC off, VA-space mm enabled —
  `uvm_hmm_is_enabled_system_wide()`, `uvm_hmm.c:132-149`; not verified
  here).

## 6. Recommendation

- Do not treat "disk as an automatically offloaded UVM tier" as an available
  setting: the boundary is a hard-coded `!vma_is_anonymous()` check with no
  BPF- or userspace-reachable policy, and file-backed HMM can at best
  remote-map CPU pages. The official CUDA guide and GDS overview statements
  are consistent with this source; the source is the authority for this
  host.
- Keep the current object-level async LMCache serving as the disk offload
  path: per-object `UVM_GPU_STORAGE_DECIDE` policy (existing cmd-82 ABI and
  BPF program) plus cuFile execution in compatibility mode. That is inside
  the verified boundary and matches the ownership model the adapters
  already implement.
- Treat section 4(b) as a future driver-development work item (Bug
  3660968-class work), not as a configuration or BPF policy option. Until
  such a driver exists, any "transparent same-address SSD paging" claim
  would be unsupported by this 575 implementation.

## 7. Reuse existing LMCache write ownership; distinguish staging from KV residency

Additional read-only inspection on September 7 uses the installed LMCache
source under `gpu_ext/workloads/lmcache-disk/current-venv/lib/python3.12/site-packages/lmcache/`.
These are implementation findings, not new performance measurements.

- `v1/cache_engine.py:488-568`: `store()` allocates separate MemoryObjs,
  copies the selected GPU KV ranges with `batched_from_gpu`, then calls
  `StorageManager.batched_put`. Thus the disk writer's buffer is an offload
  copy; releasing it does not itself select or reclaim vLLM's original KV
  blocks.
- `v1/storage_backend/storage_manager.py:386-435`: `batched_put()` submits
  objects to the selected storage backends, copying between allocators where
  needed, then drops its references.
- `v1/storage_backend/gds_backend.py:591-612`: `submit_put_task()` takes an
  additional MemoryObj reference before scheduling the asynchronous save.
  On the normal save path, `_async_save_bytes_to_disk()` awaits `_save_gds`,
  registers the key, schedules the separate metadata-file write and releases
  that reference in `finally` (636-710). The existing writer therefore already
  provides ordinary asynchronous I/O buffer ownership; a new placement policy
  should reuse it rather than introduce a second writer/lifetime subsystem.
- The completion callback at 713-718 follows data-write completion but does
  not await the separately scheduled metadata-file task (684-695). Treat this
  as a runtime data-availability event, not proof of restart recovery or
  crash-durable persistence. No new durability test or measurement gate is
  proposed here.
- `gds_backend.py:1104-1115`: `pin` and `unpin` return false because GDS has no
  eviction implementation; `remove` raises `NotImplementedError`.
  `allocate` (1128-1178) allocates from the staging pool and explicitly warns
  that GDS eviction is unsupported. These methods are not an existing
  disk-aware KV-residency policy.

Implication for the next implementation: completing the async read adapter
closes a retrieval opportunity, not the full automatic-offload lifecycle.
To unify placement, associate existing LMCache keys and I/O-completion events
with the serving engine's actual KV block ownership/reuse decisions. BPF can
then influence which backed KV is retained or reclaimed and which missing KV
is prefetched, while LMCache continues to transport the bytes. Reclaiming only
temporary GDS staging buffers is not equivalent to evicting the engine's KV.
The inspected call chain does not yet establish that association or a working
end-to-end reclaim policy. This is the remaining integration question, not a
claim that arbitrary GPU pointers already support transparent SSD paging.
