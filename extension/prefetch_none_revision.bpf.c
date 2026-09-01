/* SPDX-License-Identifier: GPL-2.0 */
/* Minimal no-prefetch policy for the revision mechanism-cost comparison. */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    (void)page_index;
    (void)bitmap_tree;
    (void)max_prefetch_region;

    bpf_gpu_set_prefetch_region(result_region, 0, 0);
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_none_revision = {
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
};
