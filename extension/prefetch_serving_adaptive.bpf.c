/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Serving-Adaptive Prefetch: fault-rate-gated always_max
 *
 * For vLLM serving workloads:
 *   - High fault rate (prefill/loading): always_max prefetch
 *   - Low fault rate (decode): skip to kernel default (protect P99 TPOT)
 *
 * Fault rate is tracked as a simple counter over a sliding window.
 * The threshold is configurable via the config map.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* ===== Configuration ===== */

/*
 * Config map:
 *   key 0 = fault_threshold (faults per window to trigger always_max, default 50)
 *   key 1 = window_ns (window duration in nanoseconds, default 10ms = 10000000)
 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, u32);
    __type(value, u64);
} sa_config SEC(".maps");

/* ===== Stats ===== */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} stat_prefetch SEC(".maps");   /* times always_max triggered */

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} stat_skip SEC(".maps");       /* times skipped to default */

/* ===== Per-CPU fault rate tracking ===== */
struct fault_window {
    u64 window_start_ns;
    u64 fault_count;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct fault_window);
} fault_tracker SEC(".maps");

static __always_inline u64 get_config(u32 key, u64 default_val)
{
    u64 *val = bpf_map_lookup_elem(&sa_config, &key);
    if (val && *val != 0)
        return *val;
    return default_val;
}

static __always_inline void inc_stat(void *map)
{
    u32 key = 0;
    u64 *val = bpf_map_lookup_elem(map, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

/*
 * Check if current fault rate exceeds threshold.
 * Returns 1 if in "high fault" mode (should prefetch aggressively).
 */
static __always_inline int is_high_fault_rate(void)
{
    u32 key = 0;
    struct fault_window *fw = bpf_map_lookup_elem(&fault_tracker, &key);
    if (!fw)
        return 1; /* fallback to aggressive */

    u64 now = bpf_ktime_get_ns();
    u64 window_ns = get_config(1, 10000000ULL); /* default 10ms */
    u64 threshold = get_config(0, 50ULL);        /* default 50 faults/window */

    /* Check if we're still in the current window */
    if (now - fw->window_start_ns > window_ns) {
        /* Window expired — check rate then reset */
        int high = (fw->fault_count >= threshold);
        fw->window_start_ns = now;
        fw->fault_count = 1;
        return high;
    }

    fw->fault_count++;
    return (fw->fault_count >= threshold);
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    if (is_high_fault_rate()) {
        /* High fault rate: always_max */
        uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
        uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
        bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);
        inc_stat(&stat_prefetch);
        return 1; /* UVM_BPF_ACTION_BYPASS */
    }

    /* Low fault rate: let kernel decide (conservative, protect P99) */
    inc_stat(&stat_skip);
    return 0; /* UVM_BPF_ACTION_DEFAULT */
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

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_serving_adaptive = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
};
