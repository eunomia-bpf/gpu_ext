/* SPDX-License-Identifier: GPL-2.0 */
/*
 * PID-based Prefetch Policy
 *
 * Allocates prefetch bandwidth based on process priority.
 * High priority PID gets lower threshold (easier to prefetch, more bandwidth).
 * Low priority PID gets higher threshold (harder to prefetch, less bandwidth).
 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "eviction_common.h"
#include "trace_helper.h"

char _license[] SEC("license") = "GPL";

/* Configuration map - same format as eviction policies */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} config SEC(".maps");

/* Per-PID statistics */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);  /* PID */
    __type(value, struct pid_chunk_stats);
} pid_stats SEC(".maps");

/* Per-CPU cache for current VA block's owner PID */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);  /* owner_tgid */
} va_block_pid_cache SEC(".maps");

/* Helper: Get threshold for a specific PID */
static __always_inline u32 get_threshold_for_pid(u32 pid)
{
    u64 *high_pid_ptr, *high_param_ptr;
    u64 *low_pid_ptr, *low_param_ptr;
    u64 *default_param_ptr;
    u32 key;

    /* Get high priority PID and its threshold */
    key = CONFIG_PRIORITY_PID;
    high_pid_ptr = bpf_map_lookup_elem(&config, &key);
    key = CONFIG_PRIORITY_PARAM;
    high_param_ptr = bpf_map_lookup_elem(&config, &key);

    if (high_pid_ptr && high_param_ptr && *high_pid_ptr == pid) {
        return (u32)*high_param_ptr;
    }

    /* Get low priority PID and its threshold */
    key = CONFIG_LOW_PRIORITY_PID;
    low_pid_ptr = bpf_map_lookup_elem(&config, &key);
    key = CONFIG_LOW_PRIORITY_PARAM;
    low_param_ptr = bpf_map_lookup_elem(&config, &key);

    if (low_pid_ptr && low_param_ptr && *low_pid_ptr == pid) {
        return (u32)*low_param_ptr;
    }

    /* Default threshold for other PIDs */
    key = CONFIG_DEFAULT_PARAM;
    default_param_ptr = bpf_map_lookup_elem(&config, &key);
    if (default_param_ptr) {
        return (u32)*default_param_ptr;
    }

    return 50;  /* Default 50% if not configured */
}

/* Helper: Update PID stats */
static __always_inline void update_pid_stats(u32 pid, bool allowed)
{
    struct pid_chunk_stats *stats, new_stats = {};

    stats = bpf_map_lookup_elem(&pid_stats, &pid);
    if (!stats) {
        new_stats.total_activate = 1;
        if (allowed)
            new_stats.policy_allow = 1;
        else
            new_stats.policy_deny = 1;
        bpf_map_update_elem(&pid_stats, &pid, &new_stats, BPF_ANY);
        return;
    }

    __sync_fetch_and_add(&stats->total_activate, 1);
    if (allowed)
        __sync_fetch_and_add(&stats->policy_allow, 1);
    else
        __sync_fetch_and_add(&stats->policy_deny, 1);
}

/*
 * Hook: uvm_perf_prefetch_get_hint_va_block (via kprobe)
 *
 * Called BEFORE before_compute. Captures owner PID and stores in per-CPU cache.
 */
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(prefetch_get_hint_va_block,
               uvm_va_block_t *va_block,
               void *va_block_context,
               u32 new_residency,
               void *faulted_pages,
               u32 faulted_region_packed,
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
    u32 key = 0;
    u32 *cached_pid = bpf_map_lookup_elem(&va_block_pid_cache, &key);
    if (!cached_pid)
        return 0;

    /* Use trace_helper.h to get owner PID */
    *cached_pid = get_owner_pid_from_va_block(va_block);
    return 0;
}

/* Helper: Get cached owner PID */
static __always_inline u32 get_cached_owner_pid(void)
{
    u32 key = 0;
    u32 *cached_pid = bpf_map_lookup_elem(&va_block_pid_cache, &key);
    return cached_pid ? *cached_pid : 0;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    u32 owner_tgid = get_cached_owner_pid();
    u32 threshold = get_threshold_for_pid(owner_tgid);

    bpf_printk("PID-Prefetch: pid=%u, page=%u, threshold=%u%%\n",
               owner_tgid, page_index, threshold);

    /* Initialize result_region to empty */
    bpf_uvm_set_va_block_region(result_region, 0, 0);

    /* Return ENTER_LOOP to let driver iterate tree and call on_tree_iter */
    return 2; // UVM_BPF_ACTION_ENTER_LOOP
}

SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    u32 owner_tgid = get_cached_owner_pid();
    u32 threshold = get_threshold_for_pid(owner_tgid);

    /* Calculate subregion_pages from current_region */
    uvm_page_index_t first = BPF_CORE_READ(current_region, first);
    uvm_page_index_t outer = BPF_CORE_READ(current_region, outer);
    unsigned int subregion_pages = outer - first;

    /* Apply PID-based threshold: counter * 100 > subregion_pages * threshold
     *
     * Lower threshold -> easier to pass -> more prefetch -> more bandwidth
     * Higher threshold -> harder to pass -> less prefetch -> less bandwidth
     */
    bool allowed = (counter * 100 > subregion_pages * threshold);

    /* Update stats for this PID */
    if (owner_tgid > 0) {
        update_pid_stats(owner_tgid, allowed);
    }

    if (allowed) {
        bpf_uvm_set_va_block_region(prefetch_region, first, outer);
        return 1; // Selected this region
    }

    return 0; // Region doesn't meet threshold
}

/* Dummy implementation for test trigger */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_prefetch_pid_tree = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
