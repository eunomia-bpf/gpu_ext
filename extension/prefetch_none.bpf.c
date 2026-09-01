/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* No prefetch policy - disables all prefetching
 * This policy sets the prefetch region to empty, effectively disabling prefetch
 */
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    bpf_printk("BPF prefetch_none: Disabling prefetch for page_index=%u\n", page_index);

    /* Set result_region to empty (first == outer means empty region) */
    bpf_gpu_set_prefetch_region(decision_ctx, 0, 0);

    /* Return BYPASS to skip default kernel computation */
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

/* This hook is called on each tree iteration - not used in none policy */
SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    /* Not used in none policy */
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

/* Dummy implementation for the old test trigger */
SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_none = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
};
