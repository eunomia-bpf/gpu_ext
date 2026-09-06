/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Migration-Debt Eviction Policy (LMCache/gpubpf prototype)
 *
 * Debt signal:
 * - gpu_evict_prepare walks the head of the USED list; each tracked chunk
 *   that is an eviction candidate increments its at-risk debt.
 * - gpu_block_access on a chunk with debt > 0 observes a later reuse: the
 *   debt is cleared and the chunk is saved to the tail (second chance),
 *   except disk-durable low-reuse chunks, which are held as preferred
 *   eviction candidates.
 *
 * High debt:
 * - When aggregate debt pressure reaches the configured threshold,
 *   speculative prefetch is suppressed (empty region, BYPASS).
 * - Disk-durable low-reuse chunks at the debt cap are preferred eviction
 *   candidates: tracking is dropped and they are left at the head of the
 *   USED list so the kernel evicts them (cheap to restore from local NVMe
 *   once the LMCache warm phase has made the pool disk-durable).
 *
 * Warm-phase disk-durable flag (KV-range scoped):
 * - The loader attaches a uprobe/uretprobe pair on uvm_kv_malloc in the
 *   workload's allocator shared object.  The uprobe saves the enter args
 *   (size) in a HASH map keyed by pid_tgid; the uretprobe records a
 *   successful (non-NULL) allocation as {start, end, tgid} in a single
 *   ARRAY-map slot, keeping only the largest successful allocation
 *   (documented limitation: a KV pool split across several smaller
 *   allocations is only partially covered, and a freed pool stays
 *   recorded until a larger allocation replaces it).
 * - gpu_block_activate reads the chunk's va_block start and owner pid:
 *   the chunk is marked KV only when the start lies inside the recorded
 *   range for the same tgid.  The warm-phase disk-durable flag
 *   (debt_config key DEBT_CONFIG_DISK_DURABLE, set by the loader once
 *   the LMCache warm phase has durably written the KV pool to local
 *   NVMe) is sampled at activation only for KV chunks; UVM memory
 *   outside the recorded range is never claimed durable.  The loader's
 *   'w' command also marks currently tracked chunks retroactively,
 *   again only entries already marked KV.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "trace_helper.h"
#include "eviction_common.h"
#include "eviction_debt_model.h"

char _license[] SEC("license") = "GPL";

/* Configuration map (warm flag, debt cap, pressure threshold). */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} debt_config SEC(".maps");

/* Per-chunk migration-debt state: chunk_ptr -> state. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, u64);
    __type(value, struct debt_chunk_state);
} chunk_debt SEC(".maps");

/* Aggregate migration debt pressure (single u64 gauge, key 0). */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} debt_pressure SEC(".maps");

/* Per-PID stats. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, u32);
    __type(value, struct pid_chunk_stats);
} pid_chunk_count SEC(".maps");

/* Enter args for in-flight uvm_kv_malloc calls: pid_tgid -> size. */
struct kv_alloc_args {
    u64 size;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);
    __type(value, struct kv_alloc_args);
} kv_alloc_args SEC(".maps");

/* Single-KV-pool range: largest successful uvm_kv_malloc (key 0). */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct debt_kv_range);
} kv_pool_range SEC(".maps");

SEC("uprobe")
int BPF_UPROBE(uvm_kv_malloc_enter, u64 size, int device, void *stream)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct kv_alloc_args args = {};

    args.size = size;
    bpf_map_update_elem(&kv_alloc_args, &pid_tgid, &args, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_URETPROBE(uvm_kv_malloc_ret, void *ret)
{
    u32 zero = 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct kv_alloc_args *args;
    struct debt_kv_range *cur;
    struct debt_kv_range range = {};

    args = bpf_map_lookup_elem(&kv_alloc_args, &pid_tgid);
    if (!args)
        return 0;

    cur = bpf_map_lookup_elem(&kv_pool_range, &zero);
    if (ret && debt_kv_range_replace(cur, (u64)ret, args->size)) {
        range.start = (u64)ret;
        range.end = (u64)ret + args->size;
        range.tgid = pid_tgid >> 32;
        range._pad = 0;
        bpf_map_update_elem(&kv_pool_range, &zero, &range, BPF_ANY);
    }

    bpf_map_delete_elem(&kv_alloc_args, &pid_tgid);
    return 0;
}

static __always_inline u64 get_debt_config_u64(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&debt_config, &key);
    return val ? *val : 0;
}

static __always_inline u64 *pressure_ptr(void)
{
    u32 key = 0;
    return bpf_map_lookup_elem(&debt_pressure, &key);
}

static __always_inline void pressure_add(u64 *p, u64 v)
{
    if (p && v)
        __sync_fetch_and_add(p, v);
}

/* Soft gauge: clamp instead of wrapping under concurrent updates. */
static __always_inline void pressure_sub(u64 *p, u64 v)
{
    u64 cur;

    if (!p || !v)
        return;
    cur = *p;
    if (cur >= v)
        __sync_fetch_and_sub(p, v);
    else
        *p = 0;
}

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u64 chunk_ptr = (u64)chunk;
    u32 owner_pid;
    u32 zero = 0;
    u64 va_start = 0;
    int is_kv = 0;
    uvm_va_block_t *va_block;
    struct debt_kv_range *kv;
    struct debt_chunk_state state;
    struct pid_chunk_stats *stats;
    struct pid_chunk_stats new_stats = {0};

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    /* Check if already tracked */
    if (bpf_map_lookup_elem(&chunk_debt, &chunk_ptr))
        return 0;

    /* KV only when the chunk's VA block starts inside the recorded
     * single-KV-pool range for the same owner tgid. */
    va_block = BPF_CORE_READ(chunk, va_block);
    if (va_block)
        va_start = BPF_CORE_READ(va_block, start);
    kv = bpf_map_lookup_elem(&kv_pool_range, &zero);
    is_kv = debt_kv_range_matches(kv, owner_pid, va_start);

    /* Sample the warm-phase disk-durable flag at activation time; the
     * model gates it on is_kv (non-KV chunks are never durable). */
    debt_activate(&state, owner_pid, is_kv,
                  get_debt_config_u64(DEBT_CONFIG_DISK_DURABLE) != 0);

    bpf_map_update_elem(&chunk_debt, &chunk_ptr, &state, BPF_ANY);

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

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u64 chunk_ptr = (u64)chunk;
    u64 debt_max;
    u64 delta;
    enum debt_access_action action;
    struct debt_chunk_state *state;
    struct pid_chunk_stats *stats;

    state = bpf_map_lookup_elem(&chunk_debt, &chunk_ptr);
    if (!state)
        return 0;

    debt_max = get_debt_config_u64(DEBT_CONFIG_DEBT_MAX);
    action = debt_access(state, debt_max, &delta);
    pressure_sub(pressure_ptr(), delta);

    stats = bpf_map_lookup_elem(&pid_chunk_count, &state->owner_pid);
    if (stats)
        __sync_fetch_and_add(&stats->total_used, 1);

    if (action == DEBT_ACCESS_SAVE) {
        /* Reuse observed while at risk: clear the debt and save the
         * chunk with a second chance (move to tail). */
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED,
                                NV_GPU_PMM_POSITION_TAIL);
        if (stats)
            __sync_fetch_and_add(&stats->policy_allow, 1);
    }

    /* KEEP and HOLD deliberately leave the chunk in place: no reorder
     * request.  HOLD marks a disk-durable low-reuse chunk as the
     * preferred eviction candidate. */
    return 1;
}

SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    struct list_head *cur;
    u64 chunk_ptr;
    u64 debt_max;
    u64 delta;
    int i;

    if (!va_block_used)
        return 0;

    debt_max = get_debt_config_u64(DEBT_CONFIG_DEBT_MAX);

    /*
     * Walk up to 8 chunks from HEAD. For each tracked chunk:
     * - Below the cap: increment debt (mark "at risk").  A later
     *   gpu_block_access clears it and saves the chunk.
     * - At/above the cap (low reuse): disk-durable chunks are dropped
     *   from tracking and left at HEAD as preferred eviction victims;
     *   non-durable ones stay pending their next reuse save.
     *
     * We cannot request a reorder here because the chunk pointer
     * derived from container_of is not a trusted pointer for the
     * verifier; saves happen in gpu_block_access instead.
     */
    cur = va_block_used;
    #pragma unroll
    for (i = 0; i < 8; i++) {
        struct debt_chunk_state *state;
        enum debt_prepare_action action;
        struct pid_chunk_stats *stats;

        cur = BPF_CORE_READ(cur, next);
        if (!cur || cur == va_block_used)
            break;

        /* container_of: list_head -> uvm_gpu_chunk_struct */
        chunk_ptr = (u64)((char *)cur -
                    __builtin_offsetof(struct uvm_gpu_chunk_struct, list));

        state = bpf_map_lookup_elem(&chunk_debt, &chunk_ptr);
        if (!state)
            break;

        action = debt_prepare(state, debt_max, &delta);
        pressure_add(pressure_ptr(), delta);

        if (action == DEBT_PREPARE_VICTIM) {
            /* Preferred eviction candidate: release its debt from the
             * aggregate pressure, stop tracking, leave it at HEAD. */
            u64 *p = pressure_ptr();
            if (p)
                pressure_sub(p, debt_cleanup_delta(state));

            stats = bpf_map_lookup_elem(&pid_chunk_count, &state->owner_pid);
            if (stats) {
                __sync_fetch_and_add(&stats->policy_deny, 1);
                if (stats->current_count > 0)
                    __sync_fetch_and_sub(&stats->current_count, 1);
            }
            bpf_map_delete_elem(&chunk_debt, &chunk_ptr);
            break; /* Found victim, stop */
        }
        /* MARK and PENDING: keep walking. */
    }

    return 0;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    u64 threshold;
    u64 pressure = 0;
    u64 *p;

    threshold = get_debt_config_u64(DEBT_CONFIG_PRESSURE_THRESHOLD);
    if (threshold == 0)
        return 0; /* gate disabled: default kernel prefetch */

    p = pressure_ptr();
    if (p)
        pressure = *p;

    if (!debt_suppress_prefetch(pressure, threshold))
        return 0; /* default kernel prefetch */

    /* High migration debt: suppress speculative prefetch. */
    bpf_gpu_set_prefetch_region(decision_ctx, 0, 0);
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_debt = {
    .gpu_test_trigger = (void *)NULL,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)NULL,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
