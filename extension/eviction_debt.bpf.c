/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Migration-Debt Eviction Policy (LMCache/gpubpf prototype)
 *
 * Debt signal (KV-scoped):
 * - gpu_evict_prepare walks the head of the USED list; each tracked KV
 *   chunk that is an eviction candidate increments its at-risk debt.
 *   Non-KV chunks are ignored: they never increment debt, reorder, or
 *   contribute aggregate pressure, and are left to native eviction.
 * - gpu_block_access on a chunk with debt > 0 observes a later reuse: the
 *   debt is cleared and the chunk is saved to the tail (second chance),
 *   except disk-durable low-reuse chunks, which are held as preferred
 *   eviction candidates.
 *
 * High debt:
 * - When aggregate (KV) debt pressure reaches the configured threshold,
 *   speculative prefetch is suppressed (empty region, BYPASS).
 * - Disk-durable KV chunks are immediate preferred eviction candidates:
 *   tracking is released on the first eviction-candidate observation,
 *   without waiting for the debt cap, at the chunk's current native
 *   list position; it then becomes eligible for native eviction when
 *   the kernel's walker reaches it (cheap to restore from local NVMe
 *   once the LMCache warm phase has made the pool disk-durable).
 *
 * Warm-phase disk-durable flag (KV-range scoped):
 * - The loader attaches a uprobe/uretprobe pair on uvm_kv_malloc in the
 *   workload's allocator shared object, plus an entry uprobe on
 *   uvm_kv_free.  The malloc uprobe saves the enter args (size) in a
 *   HASH map keyed by pid_tgid; the uretprobe records each successful
 *   (non-NULL) allocation as {start, end, owner_tgid, active} in the
 *   bounded 64-slot ARRAY range table (kv_pool_table): a live duplicate
 *   re-records into its own slot, otherwise the first inactive slot is
 *   taken.  When all slots are live, further allocations are not
 *   recorded (documented limitation).  The uvm_kv_free uprobe retires
 *   the live entry whose bounds and owner tgid match; if the loader
 *   cannot attach that probe, entries live for the process run
 *   lifetime.
 * - gpu_block_activate reads the chunk's va_block start and owner pid
 *   and scans the bounded table: the chunk enters the debt ledger only
 *   when the start lies inside a live entry recorded for the same
 *   tgid; non-KV chunks stay untracked (no debt entry, no PID stats).
 *   The warm-phase disk-durable flag (debt_config key
 *   DEBT_CONFIG_DISK_DURABLE, set by the loader once the LMCache warm
 *   phase has durably written the KV pool to local NVMe) is sampled at
 *   activation only for KV chunks; UVM memory outside any live entry is
 *   never claimed durable.  The loader's 'w' command also marks
 *   currently tracked chunks retroactively, again only entries already
 *   marked KV.  Disk-durable KV chunks are immediate eviction victims
 *   in gpu_evict_prepare: they are left for native eviction without
 *   waiting for the debt cap.
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

/* Per-PID stats for tracked (KV) chunks.  policy_deny counts
 * policy-release / preferred-candidate observations, not proof that
 * the exact chunk was evicted. */
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

/*
 * Bounded KV pool range table: DEBT_KV_RANGE_MAX ARRAY slots, slot i
 * holding the i-th recorded successful uvm_kv_malloc range
 * {start, end, owner_tgid, active} (0 = free slot).  Scans are bounded
 * by the constant DEBT_KV_RANGE_MAX, so they are verifier-friendly.
 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, DEBT_KV_RANGE_MAX);
    __type(key, u32);
    __type(value, struct debt_kv_entry);
} kv_pool_table SEC(".maps");

SEC("uprobe")
int BPF_UPROBE(uvm_kv_malloc_enter, u64 size, int device, void *stream)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct kv_alloc_args args = {};

    args.size = size;
    bpf_map_update_elem(&kv_alloc_args, &pid_tgid, &args, BPF_ANY);
    return 0;
}

/*
 * Successful uvm_kv_malloc: record [ret, ret+size) in the bounded range
 * table for the allocating tgid.  Mirrors the bounded scan of
 * debt_kv_table_slot_for(): a live duplicate of the same [start, end)
 * and owner_tgid re-records into its own slot (idempotent); otherwise
 * the first inactive slot is taken; a full table is left alone.  A
 * concurrent free of the same range between the scan and the write
 * could re-record it, but the allocator serializes allocations against
 * frees, so the window is not observable in practice.
 */
SEC("uretprobe")
int BPF_URETPROBE(uvm_kv_malloc_ret, void *ret)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct kv_alloc_args *args;
    u32 tgid, i, slot;
    u64 start, end;
    int free_slot = -1;

    args = bpf_map_lookup_elem(&kv_alloc_args, &pid_tgid);
    if (!args)
        return 0;

    if (ret) {
        start = (u64)ret;
        end = start + args->size;

        if (start && args->size && end > start) {
            tgid = pid_tgid >> 32;
            for (i = 0; i < DEBT_KV_RANGE_MAX; i++) {
                struct debt_kv_entry *e;

                e = bpf_map_lookup_elem(&kv_pool_table, &i);
                if (!e)
                    continue;
                if (e->active && e->start == start && e->end == end &&
                    e->owner_tgid == tgid) {
                    free_slot = (int)i;
                    break;
                }
                if (!e->active && free_slot < 0)
                    free_slot = (int)i;
            }
            if (free_slot >= 0) {
                struct debt_kv_entry *e;

                slot = (u32)free_slot;
                e = bpf_map_lookup_elem(&kv_pool_table, &slot);
                if (e) {
                    e->start = start;
                    e->end = end;
                    e->owner_tgid = tgid;
                    e->active = 1;
                }
            }
        }
    }

    bpf_map_delete_elem(&kv_alloc_args, &pid_tgid);
    return 0;
}

/*
 * uvm_kv_free(ptr, size, ...): retire the live table entry recorded by
 * the same tgid whose bounds match [ptr, ptr+size).  Mirrors the
 * bounded scan of debt_kv_table_retire_slot().  Frees without a
 * matching live entry (e.g. a range that was never recorded because
 * the table was full) are ignored.
 */
SEC("uprobe")
int BPF_UPROBE(uvm_kv_free_enter, void *ptr, u64 size, int device, void *stream)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid, i;
    u64 start, end;

    start = (u64)ptr;
    if (!start || !size)
        return 0;
    end = start + size;
    if (end <= start)
        return 0;

    tgid = pid_tgid >> 32;
    for (i = 0; i < DEBT_KV_RANGE_MAX; i++) {
        struct debt_kv_entry *e;

        e = bpf_map_lookup_elem(&kv_pool_table, &i);
        if (e && e->active && e->owner_tgid == tgid &&
            e->start == start && e->end == end) {
            e->active = 0;
            break;
        }
    }
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
    u32 i;
    u64 va_start = 0;
    int is_kv = 0;
    uvm_va_block_t *va_block;
    struct debt_chunk_state state;
    struct pid_chunk_stats *stats;
    struct pid_chunk_stats new_stats = {0};

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    /* Check if already tracked */
    if (bpf_map_lookup_elem(&chunk_debt, &chunk_ptr))
        return 0;

    /*
     * Bounded KV-range scan: track the chunk only when its va_block
     * start lies inside a live table entry recorded for the same
     * owner tgid.  Mirrors debt_kv_table_contains() over the
     * DEBT_KV_RANGE_MAX ARRAY slots; a non-KV chunk stays out of
     * chunk_debt and the PID stats (no map/stat overhead) and is
     * left to native eviction.
     */
    va_block = BPF_CORE_READ(chunk, va_block);
    if (va_block)
        va_start = BPF_CORE_READ(va_block, start);
    for (i = 0; i < DEBT_KV_RANGE_MAX; i++) {
        struct debt_kv_entry *e;

        e = bpf_map_lookup_elem(&kv_pool_table, &i);
        if (debt_kv_entry_contains(e, owner_pid, va_start)) {
            is_kv = 1;
            break;
        }
    }

    if (!is_kv)
        return 0; /* non-KV: untracked, default decision */

    /* Sample the warm-phase disk-durable flag at activation time. */
    debt_activate(&state, owner_pid, 1,
                  get_debt_config_u64(DEBT_CONFIG_DISK_DURABLE) != 0);

    bpf_map_update_elem(&chunk_debt, &chunk_ptr, &state, BPF_ANY);

    /* Update per-PID stats (tracked = KV chunks only). */
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
     * Walk up to 8 chunks from HEAD.  Non-KV chunks never enter the
     * debt ledger (gpu_block_activate leaves them untracked), so their
     * lookup misses; the scan continues past those entries so tracked
     * KV chunks later in the bounded window are still reached.  For
     * each tracked (KV) chunk:
     * - Disk-durable KV: immediate preferred eviction victim (no
     *   debt-cap wait); tracking is released at its current native
     *   list position, and it becomes eligible for native eviction
     *   when reached.
     * - KV below the cap: increment debt (mark "at risk").  A later
     *   gpu_block_access clears it and saves the chunk.
     * - KV at/above the cap (low reuse, not durable): stays pending its
     *   next reuse save.
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
            continue; /* untracked entry (non-KV): keep walking */

        action = debt_prepare(state, debt_max, &delta);
        pressure_add(pressure_ptr(), delta);

        if (action == DEBT_PREPARE_VICTIM) {
            /* Preferred eviction candidate: release its debt from the
             * aggregate pressure and stop tracking at its current
             * native list position; it becomes eligible for native
             * eviction when reached.  The counters observe the policy
             * release, not proof the exact chunk was evicted. */
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
