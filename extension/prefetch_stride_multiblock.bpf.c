/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Stride-Predictive Multi-Block Prefetch
 *
 * Based on cross_block_v2 structure (kprobe + struct_ops + bpf_wq),
 * but replaces 1-block direction-aware prefetch with stride-predictive
 * multi-block lookahead (up to K=6 blocks ahead).
 *
 * Algorithm:
 *   - Track last 4 block-to-block strides per CPU
 *   - Build confidence when strides are consistent
 *   - Prefetch K = 1 + confidence/2 blocks ahead (max 6)
 *   - Uses per-CPU wq_map slots: cpu * MAX_LOOKAHEAD + i
 *
 * Eviction: cycle_moe (hardcoded, best for GNN)
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

#define T1_FREQ_THRESHOLD 3
#define COUNTER_SLOTS 16384
#define COUNTER_MASK (COUNTER_SLOTS - 1)

/* VA block size = 2MB */
#define VA_BLOCK_SIZE (2ULL * 1024 * 1024)

/* Maximum lookahead blocks (compile-time upper bound for map sizing) */
#define MAX_LOOKAHEAD 6

/* Runtime-configurable max lookahead (set via rodata before load) */
const volatile int max_lookahead = 6;

/* Max CPUs for wq_map sizing: 64 CPUs * 6 lookahead = 384 */
#define MAX_CPUS 64
#define WQ_MAP_SIZE (MAX_CPUS * MAX_LOOKAHEAD)

/* ===== Per-CPU context from kprobe ===== */

struct va_block_ctx {
    u64 va_start;
    u64 va_end;
    u64 va_space;   /* opaque handle for bpf_gpu_migrate_range() */
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct va_block_ctx);
} va_block_cache SEC(".maps");

/* ===== Stride prediction state (per-CPU) ===== */

struct stride_state {
    s64 stride_hist[4];    /* last 4 block-to-block strides (signed, in bytes) */
    u64 last_block;        /* last fault's block VA (va_end) */
    u64 pending_target;    /* last predicted next block VA */
    u32 confidence;        /* 0-10, how many consecutive correct predictions */
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct stride_state);
} stride_cache SEC(".maps");

/* ===== Per-CPU access frequency counters for eviction ===== */

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, COUNTER_SLOTS);
    __type(key, u32);
    __type(value, u8);
} access_counts SEC(".maps");

/* ===== Cross-block prefetch request data + embedded bpf_wq ===== */

struct prefetch_data {
    u64 va_space;
    u64 addr;
    u64 length;
    struct bpf_wq work;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 384);  /* MAX_CPUS * MAX_LOOKAHEAD */
    __type(key, int);
    __type(value, struct prefetch_data);
} wq_map SEC(".maps");

/* ===== Stats counters ===== */
/*
 * key  0: kprobe fires
 * key  1: prefetch hook fires
 * key  2: wq scheduled
 * key  3: wq callback ran
 * key  4: migrate success (ret == 0)
 * key  5: migrate failed (ret != 0)
 * key  6: same-block skip (rate-limit)
 * key  7: wq callback skipped (invalid data)
 * key  8: stride match (all 4 strides equal)
 * key  9: stride miss (strides inconsistent)
 * key 10: prediction hit (current_block == pending_target)
 * key 11: total blocks prefetched
 */
#define STAT_KPROBE         0
#define STAT_PREFETCH       1
#define STAT_WQ_SCHED       2
#define STAT_WQ_CALLBACK    3
#define STAT_MIGRATE_OK     4
#define STAT_MIGRATE_FAIL   5
#define STAT_RATELIMIT      6
#define STAT_WQ_SKIP        7
#define STAT_STRIDE_MATCH   8
#define STAT_STRIDE_MISS    9
#define STAT_PRED_HIT       10
#define STAT_BLOCKS_PREFETCHED 11
#define NUM_STATS           12

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 12);
    __type(key, u32);
    __type(value, u64);
} sm_stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&sm_stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline void stat_add(u32 key, u64 delta)
{
    u64 *val = bpf_map_lookup_elem(&sm_stats, &key);
    if (val)
        __sync_fetch_and_add(val, delta);
}

/* ===== Helpers ===== */

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
    u64 ptr = 0;
    bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
    return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

/* ===== kprobe: capture va_block context ===== */

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block,
               uvm_va_block_t *va_block)
{
    stat_inc(STAT_KPROBE);

    u32 key = 0;
    struct va_block_ctx *info = bpf_map_lookup_elem(&va_block_cache, &key);
    if (!info)
        return 0;

    if (va_block) {
        info->va_start = BPF_CORE_READ(va_block, start);
        info->va_end = BPF_CORE_READ(va_block, end);

        /* Navigate: va_block -> managed_range -> va_range.va_space */
        uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
        if (managed) {
            uvm_va_space_t *vs = BPF_CORE_READ(managed, va_range.va_space);
            u64 vs_val = 0;
            bpf_probe_read_kernel(&vs_val, sizeof(vs_val), &vs);
            info->va_space = vs_val;
        } else {
            info->va_space = 0;
        }
    } else {
        info->va_start = 0;
        info->va_end = 0;
        info->va_space = 0;
    }

    return 0;
}

/* ===== bpf_wq callback: process context, sleepable ===== */

static int do_prefetch(void *map, int *key, void *value)
{
    stat_inc(STAT_WQ_CALLBACK);

    struct prefetch_data *data = value;
    if (!data || !data->va_space || !data->length) {
        stat_inc(STAT_WQ_SKIP);
        return 0;
    }

    int ret = bpf_gpu_migrate_range(data->va_space, data->addr, data->length);
    if (ret == 0)
        stat_inc(STAT_MIGRATE_OK);
    else
        stat_inc(STAT_MIGRATE_FAIL);

    return 0;
}

/* ===== PREFETCH: always_max + stride-predictive multi-block ===== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    stat_inc(STAT_PREFETCH);

    /* 1) Intra-block: always_max -- prefetch entire VA block */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);

    /* 2) Cross-block: stride-predictive multi-block prefetch via bpf_wq */
    u32 zero = 0;
    struct va_block_ctx *blk = bpf_map_lookup_elem(&va_block_cache, &zero);
    if (!blk || !blk->va_space || !blk->va_end)
        return 1; /* BYPASS (always_max only) */

    u64 va_space = blk->va_space;
    u64 current_block = blk->va_end;

    /* Get per-CPU stride state */
    struct stride_state *ss = bpf_map_lookup_elem(&stride_cache, &zero);
    if (!ss)
        return 1;

    /* Rate-limit: skip if same block as last */
    if (current_block == ss->last_block) {
        stat_inc(STAT_RATELIMIT);
        return 1;
    }

    /* Compute stride (signed, can be negative for backward scan) */
    s64 stride = 0;
    if (ss->last_block != 0)
        stride = (s64)(current_block - ss->last_block);

    /* Update stride history (shift window) */
    ss->stride_hist[0] = ss->stride_hist[1];
    ss->stride_hist[1] = ss->stride_hist[2];
    ss->stride_hist[2] = ss->stride_hist[3];
    ss->stride_hist[3] = stride;

    /* Check stride consistency: all 4 strides equal and non-zero */
    u32 confidence = ss->confidence;
    if (stride != 0 &&
        ss->stride_hist[0] == ss->stride_hist[1] &&
        ss->stride_hist[1] == ss->stride_hist[2] &&
        ss->stride_hist[2] == ss->stride_hist[3]) {
        /* All 4 strides match */
        stat_inc(STAT_STRIDE_MATCH);
        if (confidence < 10)
            confidence = confidence + 1;
    } else {
        stat_inc(STAT_STRIDE_MISS);
        if (confidence >= 2)
            confidence = confidence - 2;
        else
            confidence = 0;
    }

    /* Prediction hit check: did we land where we expected? */
    if (ss->pending_target != 0 && current_block == ss->pending_target) {
        stat_inc(STAT_PRED_HIT);
        if (confidence <= 8)
            confidence = confidence + 2;
        else
            confidence = 10;
    }

    ss->confidence = confidence;

    /* Decide prefetch depth K based on confidence */
    u32 K = 1 + confidence / 2;  /* 0->1, 2->2, 4->3, 6->4, 8->5, 10->6 */
    u32 effective_max = (u32)max_lookahead;
    if (effective_max > MAX_LOOKAHEAD)
        effective_max = MAX_LOOKAHEAD;
    if (K > effective_max)
        K = effective_max;

    /* Need a valid stride to prefetch */
    if (stride == 0) {
        ss->last_block = current_block;
        ss->pending_target = 0;
        return 1;
    }

    /* Prefetch K blocks ahead using multiple wq_map entries */
    u32 cpu = bpf_get_smp_processor_id();
    if (cpu >= MAX_CPUS)
        cpu = MAX_CPUS - 1;

    u32 base_idx = cpu * MAX_LOOKAHEAD;
    u32 scheduled = 0;

    /* Bounded loop: verifier requires static bounds */
    for (int i = 1; i <= 6; i++) {
        if ((u32)i > K)
            break;

        u64 target_addr = current_block + (u64)((s64)i * stride);

        /* Compute wq_map key */
        int wq_key = (int)(base_idx + (u32)(i - 1));
        if (wq_key < 0 || wq_key >= WQ_MAP_SIZE)
            break;

        struct prefetch_data *data = bpf_map_lookup_elem(&wq_map, &wq_key);
        if (!data)
            continue;

        data->va_space = va_space;
        data->addr = target_addr;
        data->length = VA_BLOCK_SIZE;

        stat_inc(STAT_WQ_SCHED);
        bpf_wq_init(&data->work, &wq_map, 0);
        bpf_wq_set_callback(&data->work, do_prefetch, 0);
        bpf_wq_start(&data->work, 0);
        scheduled++;
    }

    if (scheduled > 0)
        stat_add(STAT_BLOCKS_PREFETCHED, scheduled);

    /* Update state */
    ss->pending_target = current_block + (u64)stride;
    ss->last_block = current_block;

    return 1; /* BYPASS */
}

SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    return 0;
}

/* ===== EVICTION: cycle_moe (hardcoded) ===== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* cycle_moe: T1 protect + DEFAULT for non-T1 (kernel LRU refresh) */
    u32 idx = chunk_hash(chunk);
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (!count)
        return 0;

    u8 c = *count;
    if (c < 255)
        *count = c + 1;

    if (c + 1 >= T1_FREQ_THRESHOLD) {
        /* T1: protect by moving to tail */
        bpf_gpu_block_move_tail(chunk, list);
        return 1; /* BYPASS */
    }

    /* Non-T1: DEFAULT, let kernel LRU refresh */
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

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_stride_multiblock = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
