# Eviction Policy Safety Audit

Evidence class: `STATIC_SOURCE_AUDIT`. No policy was attached or executed.

| Policy | Hooks | move_head | move_tail | Direct list | Sleepable/WQ | printk | Bounded state | Initial suitable |
|---|---|---|---|---|---|---|---|---|
| eviction_fifo | gpu_block_access, gpu_block_activate, gpu_evict_prepare | False | False | False | False | True | False | False |
| eviction_cycle_moe | gpu_block_access, gpu_block_activate, gpu_evict_prepare | False | True | False | False | False | True | True |
| prefetch_always_max_cycle_moe | gpu_block_access, gpu_block_activate, gpu_evict_prepare, gpu_page_prefetch, gpu_page_prefetch_iter, gpu_test_trigger | False | True | False | False | False | True | True |
| prefetch_cooperative | gpu_block_access, gpu_block_activate, gpu_evict_prepare, gpu_page_prefetch, gpu_page_prefetch_iter, gpu_test_trigger | False | True | False | True | False | True | False |
| eviction_lfu | gpu_block_access, gpu_block_activate, gpu_evict_prepare | True | True | False | False | True | True | False |
| eviction_mru | gpu_block_access, gpu_block_activate, gpu_evict_prepare | True | True | False | False | True | False | False |

## Gate

- Any `move_head`, direct linked-list operation, or sleepable migration/WQ use is excluded from the first oversubscription run.
- `eviction_fifo` is also excluded until its high-frequency `bpf_printk` calls and implementation/comment mismatch are fixed.
- `prefetch_always_max_cycle_moe` uses only `move_tail` plus bounded per-CPU state and is the sole current joint-policy candidate, but it remains gated on stable Stage 3C results.
- `prefetch_cooperative` is excluded from the initial matrix because it schedules sleepable cross-block migration with `bpf_wq`.
