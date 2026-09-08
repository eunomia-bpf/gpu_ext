# GPU promotion patch (opt-in) for disk-backed UVM ranges

Deliverable of the current session: `gpu-promotion.patch` only. No driver
install/reload/build/GPU runs were performed here; root owns
integration, build, real runs, and commit/push.

## Baseline behavior (driver tree at c8e2831d, unpatched)

A GPU fault on a sealed disk-backed range page that is durably on disk but
not resident anywhere is forced to CPU residency:
`uvm_va_block_select_residency()` in `kernel-open/nvidia-uvm/uvm_va_block.c`
(~line 11915) overrides the selected residency to `UVM_ID_CPU`. The CPU
populate path then hydrates the content from the backing file via
`block_hydrate_sealed_range()` (`uvm_va_block.c:1915`, called from
`block_populate_pages_cpu()` at `uvm_va_block.c:2121`). The GPU is mapped to
that CPU page through the IOMMU
(`uvm_va_block_service_finish()` -> `block_get_processor_to_map()`, closest
resident processor = CPU), so the data stays CPU-resident ("sticky CPU"):
the 2026-09-08 restore experiment measured 5.408 ms repeated GPU reads vs
0.337 ms initial-GPU baseline
(`workloads/lmcache-disk/results-disk-uvm-restore-20260908.md`).

## What the patch adds

A per-registered-range opt-in flag, `gpu_promote`, stored in
`uvm_disk_backing_shared_struct` (shared by all views, so it survives range
splits). Default: off. While off, every path below is byte-for-byte the
baseline behavior.

### New ioctl (ABI)

`UVM_DISK_BACKING_SET_GPU_PROMOTION = UVM_IOCTL_BASE(87)`
(`kernel-open/nvidia-uvm/uvm_ioctl.h`), params
`UVM_DISK_BACKING_SET_GPU_PROMOTION_PARAMS`:
`abiVersion(=1)`, `pad0(=0)`, `rangeStart` (page aligned), `rangeEnd`
(inclusive), `enable` (0/1), `pad1(=0)`, `rmStatus` (OUT).
Errors: `NV_ERR_INVALID_ARGUMENT` (bad version/pads/alignment/range or not a
managed range), `NV_ERR_INVALID_STATE` (managed range without an attached
backing). Handler: `uvm_api_disk_backing_set_gpu_promotion()` in
`kernel-open/nvidia-uvm/uvm.c`, routed through
`UVM_ROUTE_CMD_STACK_INIT_CHECK` next to the existing disk-backing ioctls;
takes the VA space write lock, mirroring `uvm_api_disk_backing_register()`
and `uvm_api_disk_backing_query()`.

### New backing API

`kernel-open/nvidia-uvm/uvm_disk_backing.h/.c`:
- `NV_STATUS uvm_disk_backing_set_gpu_promote(uvm_va_range_managed_t *, bool)`
  (VA space write lock; sets `backing->shared->gpu_promote`),
- `bool uvm_disk_backing_gpu_promote(uvm_disk_backing_t *)` (plain read;
  false for NULL; safe on the fault path).

## New behavior while the flag is set

A GPU fault on a page that is durably on disk and not resident anywhere now
keeps the GPU as the target residency:

1. `uvm_va_block_select_residency()` (`uvm_va_block.c`, ~line 11915): the
   forced-CPU override is skipped when `uvm_disk_backing_gpu_promote()` is
   set.
2. `block_populate_pages()` (`uvm_va_block.c`, ~line 3292, GPU destination
   branch): on-disk pages with no resident copy anywhere are added to
   `make_resident.pages_staged`. `block_populate_pages_cpu()` then allocates
   a fresh CPU staging chunk and restores the durable bytes into it via
   `block_hydrate_sealed_range()` (file read, block lock dropped, same
   claim/done/fail hydration protocol as the baseline CPU path).
3. `block_copy_resident_pages()` (`uvm_va_block.c`, ~line 4783): the staged
   CPU chunks are moved to the destination with UVM's normal CPU->GPU copy
   engine pass (the same staged-migration plumbing used for cross-GPU
   migration without P2P). The final staging invariant
   (`UVM_ASSERT(pages_copied == pages_copied_to_cpu)` at ~line 4959) is
   extended to count disk-hydrated staged pages, which reach the CPU staging
   chunks by file read instead of by a copy from a resident source.
4. `uvm_va_block_make_resident_finish()`: the page ends up GPU-resident;
   `uvm_va_block_service_finish()` maps the faulting GPU to its own GPU
   chunk. The transient CPU staging chunk is left allocated but
   non-resident, exactly the state vanilla cross-GPU staging produces.

Net effect: first GPU fault after offload = one file read into a CPU staging
chunk + one CPU->GPU CE copy; the page is then GPU-resident, so repeated
GPU faults are served locally.

## Unchanged by the patch

- Flag off (default): identical to baseline in every path.
- CPU fault restoration: the override only fires when the selected
  residency is a GPU; CPU populate/hydration is untouched.
- Pages resident somewhere: never staged or re-hydrated; normal
  migration paths unchanged.
- Error pages (`io_error` bit): excluded from staging by the
  `uvm_disk_backing_page_on_disk()` predicate, same as the baseline
  override; a fault that reaches `block_hydrate_sealed_range()` with a
  recorded I/O error still fails with `NV_ERR_INVALID_STATE`.
- Never-resident, never-on-disk pages: authoritative zero content, as
  before.
- Offload (ioctl 85), query (ioctl 86), register (ioctl 84), write
  rejection, and the offload worker are untouched.

## Exact files (10 hunks)

- `kernel-open/nvidia-uvm/uvm_disk_backing.h` (1 hunk: new API declarations)
- `kernel-open/nvidia-uvm/uvm_disk_backing.c` (2 hunks: `gpu_promote` field
  in the shared struct; setter/getter)
- `kernel-open/nvidia-uvm/uvm_va_block.c` (4 hunks: residency-override gate
  in `uvm_va_block_select_residency()`; staging extension in
  `block_populate_pages()`; staging-invariant extension in
  `block_copy_resident_pages()`; comment)
- `kernel-open/nvidia-uvm/uvm_ioctl.h` (1 hunk: ioctl 87 + params)
- `kernel-open/nvidia-uvm/uvm.c` (2 hunks: handler; routing entry)

## Applying

From the driver tree root (state: commit c8e2831d):

    git apply path/to/gpu-promotion.patch

`git apply --check` was verified clean against c8e2831d; no tree was
modified. Build, driver load, and the restore-vs-repeat benchmark (enable
promotion via ioctl 87 between the offload and the first GPU read, then
compare repeated GPU read against the 5.408 ms / 0.337 ms baselines) are
root tasks.
