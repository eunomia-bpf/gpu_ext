# gpu_ext UVM Call Path

Source was verified against the readable 575.57.08 tree at `/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module`, submodule commit `921567e16857811de57d57aef111782df70847e2` with local gpu_ext modifications. The public checkout's kernel submodule is not initialized, so these line references intentionally identify the actual matching source tree rather than guessing from the README.

## Replayable Fault Service

In `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`:

- The top-level `service_fault_batch()` starts at line 2232 and calls `service_fault_batch_dispatch()` at line 2325.
- `service_fault_batch_dispatch()` starts at line 1946 and dispatches a VA block to `service_fault_batch_block()` at line 1983.
- `service_fault_batch_block()` starts at line 1606 and enters `service_fault_batch_block_locked()` through `UVM_VA_BLOCK_RETRY_LOCKED` around line 1628.
- `service_fault_batch_block_locked()` starts at line 1375, computes per-page residency, and calls `uvm_va_block_service_locked()` at line 1586 when faults need service.

In `kernel-open/nvidia-uvm/uvm_va_block.c`, `uvm_va_block_service_locked()` starts at line 12307 and calls its local prefetch-hint helper at line 12332 before performing migrations and mappings.

The verified replayable-fault path is therefore:

```text
service_fault_batch()
  -> service_fault_batch_dispatch()
  -> service_fault_batch_block()
  -> service_fault_batch_block_locked()
  -> uvm_va_block_service_locked()
```

## Prefetch Policy Hook

The actual chain is:

```text
uvm_va_block_service_locked()
  -> uvm_va_block_get_prefetch_hint()
  -> uvm_perf_prefetch_get_hint_va_block()
  -> uvm_perf_prefetch_prenotify_fault_migrations()
  -> compute_prefetch_mask()
  -> compute_prefetch_region()
  -> uvm_bpf_call_gpu_page_prefetch()
  -> rcu_dereference(uvm_ops)
  -> gpu_mem_ops.gpu_page_prefetch
  -> attached eBPF struct_ops policy
```

Evidence:

- `uvm_va_block.c`: `uvm_va_block_get_prefetch_hint()` line 11828 and call to `uvm_perf_prefetch_get_hint_va_block()` line 11846.
- `uvm_perf_prefetch.c`: `uvm_perf_prefetch_get_hint_va_block()` line 472, prenotify call line 494, `uvm_perf_prefetch_prenotify_fault_migrations()` line 352, mask call line 408, `compute_prefetch_mask()` line 324, region call line 336, and `compute_prefetch_region()` line 103.
- The branch inserts `uvm_bpf_call_gpu_page_prefetch()` directly at lines 113-114 of `compute_prefetch_region()`, before the original bitmap-tree computation. A policy can return DEFAULT, BYPASS, or ENTER_LOOP; ENTER_LOOP also invokes `uvm_bpf_call_gpu_page_prefetch_iter()`.
- `uvm_bpf_struct_ops.c`: wrapper starts at line 381 and calls `ops->gpu_page_prefetch` at lines 392-395 under RCU.

This differs from an upstream-only UVM tree because `uvm_bpf_call_gpu_page_prefetch*` and the `gpu_mem_ops` struct_ops dispatch are gpu_ext additions.

## Excluded Hook

This experiment does not depend on `gpu_block_access`. The extension documentation records that its current call site can skip pinned chunks, so its absence is not interpreted as absence of GPU memory access. The first policies tested are prefetch-only policies and do not call `bpf_gpu_block_move_head()`.
