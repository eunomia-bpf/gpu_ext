# CPU fault integration notes for disk UVM

## Managed VMA split correction, 2026-09-08

Root checked the current 575 source while the direct Qwen 27B implementation
session was designing range lifetime handling. `uvm.c:390` explicitly states
that `uvm_vm_open` cannot return an error. The split caller at lines 501–508
invokes `uvm_vm_open_failure` on a non-OK result; that helper destroys/disables
both managed VMA associations. Therefore, returning `NV_ERR_INVALID_STATE`
from `uvm_va_range_split` does **not** reject `munmap` with a clean error while
preserving the original mapping. The proposed description of that behavior
was incorrect and was sent back to the implementation owner for correction.

The implementation must account for split/close and backing lifetime under
the actual callback semantics, including disk-only data belonging to addresses
that remain mapped. A retained block/backing reference alone does not restore
the VMA association or make the split reversible. This source correction is
not a completed paging implementation or a new runtime test. The four-file
disk-backing work in progress is still not wired into ioctl/fault/Kbuild paths.

Local GLM completed a read-only source investigation; root checked the core
retry/population paths against the current 575 worktree. This is implementation
guidance, not a runnable disk-UVM result or a new validation campaign.

- `uvm_va_space_cpu_fault` is in `uvm_va_space.c`, not `uvm.c`. Its outer
  loop uses `NV_WARN_MORE_PROCESSING_REQUIRED`. It drops the VA-space lock
  and only sleeps when a thrashing wake timestamp is set; otherwise retry
  is not a disk-completion wait.
- The separate `UVM_VA_BLOCK_LOCK_RETRY` macro loops on
  `NV_ERR_MORE_PROCESSING_REQUIRED`. These status constants must not be
  conflated when implementing fault suspension/restart.
- CPU chunk population sets `UVM_CPU_CHUNK_ALLOC_FLAGS_ZERO` if some pages
  in the allocation region are not resident on any processor. A disk-only
  page must be hydrated before normal resident/mapping success; a read error
  must not be treated as no backing or allowed to fall through to zero-fill.
- CPU mapping permissions are checked later in `block_map_cpu_page`;
  `uvm_va_block_check_logical_permissions` is not by itself proof that CPU
  write permissions or a sealing contract are enforced.
- The existing CPU-fault path records mmap-lock ownership without dropping
  the actual mmap lock in its retry loop. Simply returning `VM_FAULT_RETRY`
  is not sufficient: any new path must obey the kernel's retry/lock contract.
  The model's suggestion that generic mm automatically releases it was not
  accepted as implementation guidance.
- The backing and chunk lifetime must outlive I/O and waits. Teardown must
  not wait for a worker while holding locks that worker needs. Revalidate
  state after a lock is released; a retained object does not prove that its
  old range or installed chunk still represents the same mapping.

The writer received these concrete findings. Its current source-only backing
interface and range associations are incomplete; no driver install or GPU
fault/offload experiment has been performed for this extension.

## Implementation handoff after provider error

The Qwen 27B implementation session subsequently ended with a terminal
`APIError`, HTTP 524. Its CLI exited; root did not impose a timeout or stop
it for inactivity. The three partial source files remain in the existing
575 worktree. A fresh local GLM session continues from those files and these
notes, with permission for ordinary CPU builds now that the LMCache campaign
and missing control have both finished. Driver installation/reload and GPU
execution remain root-coordinated. No completion claim is attached to this
handoff, and no fourth concurrent local session was launched.
