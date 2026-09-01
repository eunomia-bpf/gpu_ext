/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Combined Always-Max Prefetch + Cycle-Aware MoE Eviction Policy
 *
 * Prefetch: Always prefetch the entire max_prefetch_region (aggressive)
 * Eviction: T1 (frequently accessed) chunk protection via per-CPU counters
 *
 * This combines the best of both:
 *   - always_max prefetch: brings in the full VA block on any fault
 *   - cycle_moe eviction: protects attention weights from being evicted
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* ========================================================================
 * Eviction Configuration (from cycle_moe)
 * ======================================================================== */

#define T1_FREQ_THRESHOLD 3
#define COUNTER_SLOTS 16384
#define COUNTER_MASK (COUNTER_SLOTS - 1)

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, COUNTER_SLOTS);
    __type(key, u32);
    __type(value, u8);
} access_counts SEC(".maps");

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
    u64 ptr = 0;
    bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
    return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

/* ========================================================================
 * Prefetch Hooks (always_max)
 * ======================================================================== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

/* ========================================================================
 * Eviction Hooks (cycle_moe)
 * ======================================================================== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u32 idx = chunk_hash(chunk);
    u8 *count;

    count = bpf_map_lookup_elem(&access_counts, &idx);
    if (count) {
        u8 c = *count;
        if (c < 255)
            *count = c + 1;

        if (c + 1 >= T1_FREQ_THRESHOLD) {
            bpf_gpu_block_move_tail(chunk, list);
            return 1; /* BYPASS: T1 protected */
        }
    }

    return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    return 0;
}

SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    return 0;
}

/* ========================================================================
 * Struct ops registration — all 6 hooks
 * ======================================================================== */

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_always_max_cycle_moe = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
