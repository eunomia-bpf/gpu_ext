/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Proactive Layer Migration + Always-Max Prefetch + Cycle-MoE Eviction
 *
 * Novel algorithm: When detecting GPU processing of layer L, proactively
 * pre-migrate chunks from layer L+1 via bpf_wq BEFORE the GPU faults on them.
 * This overlaps DMA transfer with GPU compute, potentially reducing fault
 * handling latency.
 *
 * Key difference from cross-block prefetch (which was neutral/harmful):
 *   - Cross-block prefetched spatially adjacent VA blocks (guessing)
 *   - This prefetches the NEXT LAYER's chunks (semantically correct)
 *   - Layer access order is deterministic: 0→1→...→N-1→0→1→...
 *
 * Components:
 *   1. always_max prefetch (proven +57-70% for MoE)
 *   2. cycle_moe T1 protection (proven safe eviction)
 *   3. Layer detection via VA boundary array (from equal-count analysis)
 *   4. bpf_wq proactive migration for next layer's VA range
 *
 * Config (via config_map):
 *   - num_layers: number of model layers (default 36)
 *   - prefetch_bytes: how many bytes to pre-migrate per layer transition
 *   - prefetch_ahead: how many layers ahead to pre-migrate (default 1)
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Configuration */
#define MAX_LAYERS 40
#define COUNTER_SLOTS 16384
#define COUNTER_MASK (COUNTER_SLOTS - 1)
#define T1_FREQ_THRESHOLD 3
#define VA_BLOCK_SIZE (2ULL * 1024 * 1024)

/* ===== Config ===== */
struct config {
    u32 num_layers;
    u32 prefetch_ahead;     /* How many layers ahead to pre-migrate (1-3) */
    u64 prefetch_bytes;     /* Bytes to pre-migrate per layer transition */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct config);
} config_map SEC(".maps");

/* Layer VA boundaries from equal-count analysis */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_LAYERS);
    __type(key, u32);
    __type(value, u64);
} layer_boundaries SEC(".maps");

/* Runtime state */
struct runtime_state {
    u32 current_layer;
    u32 decode_step;
    u64 last_fault_va;
    u32 faults_this_step;
    u32 last_prefetched_layer;  /* Avoid duplicate proactive prefetches */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct runtime_state);
} state_map SEC(".maps");

/* Per-CPU va_block context from kprobe */
struct va_block_ctx {
    u64 va_start;
    u64 va_end;
    u64 va_space;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct va_block_ctx);
} va_block_cache SEC(".maps");

/* T1 frequency counter */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, COUNTER_SLOTS);
    __type(key, u32);
    __type(value, u8);
} access_counts SEC(".maps");

/* bpf_wq work items for proactive migration */
struct prefetch_data {
    u64 va_space;
    u64 addr;
    u64 length;
    struct bpf_wq work;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 64);
    __type(key, int);
    __type(value, struct prefetch_data);
} wq_map SEC(".maps");

/* Stats counters */
#define STAT_KPROBE        0
#define STAT_PREFETCH      1
#define STAT_LAYER_DETECT  2
#define STAT_WQ_SCHED      3
#define STAT_WQ_CALLBACK   4
#define STAT_MIGRATE_OK    5
#define STAT_MIGRATE_FAIL  6
#define STAT_TOKEN_BOUNDARY 7
#define STAT_DEDUP_SKIP    8
#define NUM_STATS          10

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, NUM_STATS);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
    u64 ptr = 0;
    bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
    return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

/* O(1) layer transition check: given current_layer and fault VA,
 * check if we've moved to the next layer by comparing against
 * layer_boundaries[current_layer + 1].
 * Returns new layer index, or current_layer if no transition.
 * Layer access is sequential (0→1→...→N-1→0→...), so +1 check suffices. */
static __always_inline u32 check_layer_transition(u64 va, u32 current_layer,
                                                    u32 num_layers)
{
    if (num_layers == 0)
        return 0xFFFFFFFF;

    u32 next = current_layer + 1;
    if (next >= num_layers)
        return current_layer;  /* At last layer, wait for token boundary */

    u64 *next_boundary = bpf_map_lookup_elem(&layer_boundaries, &next);
    if (!next_boundary)
        return current_layer;

    if (va >= *next_boundary)
        return next;  /* Crossed into next layer */

    return current_layer;  /* Still in current layer */
}

/* ===== kprobe: capture va_block + va_space context ===== */

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block, uvm_va_block_t *va_block)
{
    stat_inc(STAT_KPROBE);

    u32 key = 0;
    struct va_block_ctx *info = bpf_map_lookup_elem(&va_block_cache, &key);
    if (!info)
        return 0;

    if (va_block) {
        info->va_start = BPF_CORE_READ(va_block, start);
        info->va_end = BPF_CORE_READ(va_block, end);

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

/* ===== bpf_wq callback: proactive migration ===== */

static int do_proactive_migrate(void *map, int *key, void *value)
{
    stat_inc(STAT_WQ_CALLBACK);

    struct prefetch_data *data = value;
    if (!data || !data->va_space || !data->length)
        return 0;

    int ret = bpf_gpu_migrate_range(data->va_space, data->addr, data->length);
    if (ret == 0)
        stat_inc(STAT_MIGRATE_OK);
    else
        stat_inc(STAT_MIGRATE_FAIL);

    return 0;
}

/* Schedule proactive migration for target layer */
static __always_inline void schedule_proactive_prefetch(u64 va_space,
                                                         u32 target_layer,
                                                         u64 prefetch_bytes)
{
    if (!va_space || target_layer == 0xFFFFFFFF)
        return;

    /* Get target layer's start VA */
    u64 *target_start = bpf_map_lookup_elem(&layer_boundaries, &target_layer);
    if (!target_start || *target_start == 0)
        return;

    /* Use CPU-indexed wq_map entry */
    int wq_key = bpf_get_smp_processor_id() % 64;
    struct prefetch_data *data = bpf_map_lookup_elem(&wq_map, &wq_key);
    if (!data)
        return;

    data->va_space = va_space;
    data->addr = *target_start;
    data->length = prefetch_bytes;

    stat_inc(STAT_WQ_SCHED);
    bpf_wq_init(&data->work, &wq_map, 0);
    bpf_wq_set_callback(&data->work, do_proactive_migrate, 0);
    bpf_wq_start(&data->work, 0);
}

/* ===== PREFETCH: always_max + proactive layer migration ===== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    stat_inc(STAT_PREFETCH);

    /* 1) Intra-block: always_max */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);

    /* 2) Proactive layer migration: detect current layer from fault VA,
     *    schedule pre-migration for next layer */
    u32 zero = 0;
    struct va_block_ctx *blk = bpf_map_lookup_elem(&va_block_cache, &zero);
    if (!blk || !blk->va_space || !blk->va_start)
        return 1; /* BYPASS */

    struct config *cfg = bpf_map_lookup_elem(&config_map, &zero);
    struct runtime_state *st = bpf_map_lookup_elem(&state_map, &zero);
    if (!cfg || !st || cfg->num_layers == 0)
        return 1;

    u64 fault_va = blk->va_start;

    /* Detect token boundary (VA regression) */
    if (fault_va < st->last_fault_va && st->faults_this_step > 10) {
        st->decode_step++;
        st->current_layer = 0;
        st->faults_this_step = 0;
        st->last_prefetched_layer = 0xFFFFFFFF;
        stat_inc(STAT_TOKEN_BOUNDARY);
    }

    st->last_fault_va = fault_va;
    st->faults_this_step++;

    /* O(1) layer transition check (no loop — verifier-safe) */
    u32 new_layer = check_layer_transition(fault_va, st->current_layer,
                                            cfg->num_layers);

    /* Layer transition detected? */
    if (new_layer != st->current_layer) {
        st->current_layer = new_layer;
        stat_inc(STAT_LAYER_DETECT);

        /* Schedule proactive prefetch for next layer */
        u32 target = (new_layer + 1) % cfg->num_layers;

        /* Dedup: don't re-prefetch same layer in same token */
        if (target != st->last_prefetched_layer) {
            schedule_proactive_prefetch(blk->va_space, target,
                                        cfg->prefetch_bytes);
            st->last_prefetched_layer = target;
        } else {
            stat_inc(STAT_DEDUP_SKIP);
        }

        /* Optionally prefetch layer+2 if ahead > 1 */
        if (cfg->prefetch_ahead > 1 && cfg->num_layers > 2) {
            u32 target2 = (new_layer + 2) % cfg->num_layers;
            if (target2 != st->last_prefetched_layer) {
                schedule_proactive_prefetch(blk->va_space, target2,
                                            cfg->prefetch_bytes);
            }
        }
    }

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

/* ===== EVICTION: cycle_moe T1 protection ===== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u32 idx = chunk_hash(chunk);
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (!count)
        return 0;

    u8 c = *count;
    if (c < 255)
        *count = c + 1;

    if (c + 1 >= T1_FREQ_THRESHOLD) {
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
        return 1; /* BYPASS: T1 protected */
    }

    return 0; /* Default LRU for non-T1 */
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
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
struct gpu_mem_ops uvm_ops_proactive_layer = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
