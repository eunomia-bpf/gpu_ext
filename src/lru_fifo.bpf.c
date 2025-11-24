/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* FIFO eviction policy for GPU memory management
 *
 * In FIFO (First-In-First-Out), the chunk that has been in memory the longest
 * is evicted first. This is implemented by:
 * - Moving newly populated chunks to the head of the list (highest priority for eviction)
 * - Keeping the default LRU behavior for activate and depopulate
 */

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* For FIFO, we use default behavior when a chunk becomes evictable */
    bpf_printk("BPF FIFO: chunk_activate (using default behavior)\n");
    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* FIFO policy: Move newly populated chunk to head of list
     * This makes it the FIRST candidate for eviction (oldest first)
     *
     * The kernel already moved the chunk to tail (lowest priority),
     * so we move it to head to implement FIFO.
     */
    bpf_printk("BPF FIFO: chunk_populate - moving to head (FIFO order)\n");
    bpf_uvm_pmm_chunk_move_head(chunk, list);
    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_depopulate")
int BPF_PROG(uvm_pmm_chunk_depopulate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* For FIFO, we use default behavior when chunk is depopulated */
    bpf_printk("BPF FIFO: chunk_depopulate (using default behavior)\n");
    return 0;
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    /* FIFO doesn't need special preparation before eviction
     * The list is already ordered by populate time due to move_head in populate
     */
    bpf_printk("BPF FIFO: eviction_prepare (no reordering needed)\n");
    return 0;
}

/* Dummy implementations for prefetch hooks - not used in LRU policy */
SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_fifo = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
    .uvm_pmm_chunk_activate = (void *)uvm_pmm_chunk_activate,
    .uvm_pmm_chunk_populate = (void *)uvm_pmm_chunk_populate,
    .uvm_pmm_chunk_depopulate = (void *)uvm_pmm_chunk_depopulate,
    .uvm_pmm_eviction_prepare = (void *)uvm_pmm_eviction_prepare,
};
