/* SPDX-License-Identifier: GPL-2.0 */
/*
 * PID-based Quota Eviction Policy for GPU Memory Management
 *
 * Strategy:
 * - Each process has a chunk quota (percentage of total active chunks)
 * - Within quota: move_tail (LRU, protected)
 * - Over quota: don't move (easier to evict)
 *
 * This creates differentiated memory residency based on quota allocation.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "trace_helper.h"

#include "eviction_common.h"

char _license[] SEC("license") = "GPL";

/* Configuration map */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} config SEC(".maps");

/* Per-PID chunk counter: owner_pid -> stats */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, u32);
    __type(value, struct pid_chunk_stats);
} pid_chunk_count SEC(".maps");

/* Active chunk tracking: chunk_ptr -> owner_pid */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, u64);    /* chunk pointer */
    __type(value, u32);  /* owner_pid */
} active_chunks SEC(".maps");

static __always_inline u64 get_config_u64(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&config, &key);
    return val ? *val : 0;
}

/* Get total active chunks across all PIDs */
static __always_inline u64 get_total_active_chunks(void)
{
    u64 total = 0;
    u32 priority_pid = (u32)get_config_u64(CONFIG_PRIORITY_PID);
    u32 low_priority_pid = (u32)get_config_u64(CONFIG_LOW_PRIORITY_PID);
    struct pid_chunk_stats *stats;

    if (priority_pid != 0) {
        stats = bpf_map_lookup_elem(&pid_chunk_count, &priority_pid);
        if (stats)
            total += stats->current_count;
    }

    if (low_priority_pid != 0) {
        stats = bpf_map_lookup_elem(&pid_chunk_count, &low_priority_pid);
        if (stats)
            total += stats->current_count;
    }

    return total > 0 ? total : 1;
}

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u32 owner_pid;
    struct pid_chunk_stats *stats;
    struct pid_chunk_stats new_stats = {0};
    u64 chunk_ptr = (u64)chunk;

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    /* Check if this chunk was already tracked (avoid double counting) */
    if (bpf_map_lookup_elem(&active_chunks, &chunk_ptr))
        return 0;

    /* Track this chunk as active */
    bpf_map_update_elem(&active_chunks, &chunk_ptr, &owner_pid, BPF_ANY);

    /* Update per-PID stats */
    stats = bpf_map_lookup_elem(&pid_chunk_count, &owner_pid);
    if (stats) {
        __sync_fetch_and_add(&stats->current_count, 1);
        __sync_fetch_and_add(&stats->total_activate, 1);
    } else {
        new_stats.current_count = 1;
        new_stats.total_activate = 1;
        bpf_map_update_elem(&pid_chunk_count, &owner_pid, &new_stats, BPF_ANY);
    }

    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u32 owner_pid;
    u64 priority_pid;
    u64 low_priority_pid;
    u64 quota_percent;
    struct pid_chunk_stats *pid_stats;
    u64 current_count;
    u64 total_chunks;
    u64 quota_chunks;

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    priority_pid = get_config_u64(CONFIG_PRIORITY_PID);
    low_priority_pid = get_config_u64(CONFIG_LOW_PRIORITY_PID);

    /* Get current active chunk count for this PID */
    pid_stats = bpf_map_lookup_elem(&pid_chunk_count, &owner_pid);
    current_count = pid_stats ? pid_stats->current_count : 0;

    /* Update total_used for this PID */
    if (pid_stats) {
        __sync_fetch_and_add(&pid_stats->total_used, 1);
    }

    /* Get total active chunks for percentage calculation */
    total_chunks = get_total_active_chunks();

    /* Determine quota percentage based on PID (0-100) */
    if (priority_pid != 0 && owner_pid == (u32)priority_pid) {
        quota_percent = get_config_u64(CONFIG_PRIORITY_PARAM);
    } else if (low_priority_pid != 0 && owner_pid == (u32)low_priority_pid) {
        quota_percent = get_config_u64(CONFIG_LOW_PRIORITY_PARAM);
    } else {
        quota_percent = get_config_u64(CONFIG_DEFAULT_PARAM);
    }

    /* If quota is 0, treat as unlimited (default LRU) */
    if (quota_percent == 0) {
        return 0; /* Default LRU behavior */
    }

    /* Calculate quota in chunks: quota_chunks = total_chunks * quota_percent / 100 */
    quota_chunks = (total_chunks * quota_percent) / 100;
    if (quota_chunks == 0)
        quota_chunks = 1;

    /* Within quota: move_tail (LRU, protected) */
    if (current_count <= quota_chunks) {
        bpf_uvm_pmm_chunk_move_tail(chunk, list);
        if (pid_stats) {
            __sync_fetch_and_add(&pid_stats->policy_allow, 1);
        }
        return 1; /* BYPASS */
    }

    /* Over quota: don't move (easier to evict) */
    if (pid_stats) {
        __sync_fetch_and_add(&pid_stats->policy_deny, 1);
    }
    return 1; /* BYPASS - don't let kernel do LRU move */
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    struct list_head *first;
    uvm_gpu_chunk_t *chunk;
    u32 *tracked_pid;
    struct pid_chunk_stats *stats;
    u64 chunk_ptr;

    if (!va_block_used)
        return 0;

    /* Get the first entry in the va_block_used list (head of eviction) */
    first = BPF_CORE_READ(va_block_used, next);
    if (!first || first == va_block_used)
        return 0;

    /*
     * The list entry is embedded in uvm_gpu_chunk_t.list
     * uvm_gpu_root_chunk_t has chunk as first member (offset 0)
     * So: container_of(first, uvm_gpu_chunk_t, list)
     */
    chunk = (uvm_gpu_chunk_t *)((char *)first -
              __builtin_offsetof(struct uvm_gpu_chunk_struct, list));
    chunk_ptr = (u64)chunk;

    /* Only decrement if we tracked this chunk in activate */
    tracked_pid = bpf_map_lookup_elem(&active_chunks, &chunk_ptr);
    if (!tracked_pid)
        return 0;  /* Chunk was not tracked by us, don't decrement */

    /* Decrement current_count for the tracked PID */
    stats = bpf_map_lookup_elem(&pid_chunk_count, tracked_pid);
    if (stats && stats->current_count > 0) {
        __sync_fetch_and_sub(&stats->current_count, 1);
    }

    /* Remove from active_chunks */
    bpf_map_delete_elem(&active_chunks, &chunk_ptr);

    return 0;
}

SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_pid_lfu = {
    .uvm_bpf_test_trigger_kfunc = (void *)NULL,
    .uvm_prefetch_before_compute = (void *)NULL,
    .uvm_prefetch_on_tree_iter = (void *)NULL,
    .uvm_pmm_chunk_activate = (void *)uvm_pmm_chunk_activate,
    .uvm_pmm_chunk_used = (void *)uvm_pmm_chunk_used,
    .uvm_pmm_eviction_prepare = (void *)uvm_pmm_eviction_prepare,
};
