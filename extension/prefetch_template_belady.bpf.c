/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Template-Aware Prefetch + Belady OPT Eviction
 *
 * Combines NVBit-derived (or analytical) per-layer VA range knowledge with
 * cycle-distance-based Belady eviction. The core idea from MSched:
 *
 * Prefetch:
 *   - always_max within each VA block (proven +57-70% for MoE)
 *   - Per-region awareness: T1 regions always_max, T3 regions can be conservative
 *
 * Eviction:
 *   - Use chunk VA → layer mapping to compute Belady cycle distance
 *   - Chunks belonging to layers far in the future → evict first (move_head)
 *   - Chunks belonging to layers coming soon → protect (move_tail)
 *   - T1 chunks (attention/embeddings, freq >= threshold) → always protect
 *
 * Userspace loader populates:
 *   - va_to_layer_map: chunk VA >> 21 → layer_id (from NVBit or analytical model)
 *   - config_map: num_layers, t1_freq_threshold
 *
 * Runtime learning:
 *   - Tracks current layer from fault VA pattern (monotonically ascending within decode step)
 *   - Detects decode step boundaries from VA address regression
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Configuration */
#define MAX_LAYERS 40  /* Keep small to avoid BPF verifier jump complexity limit */
#define COUNTER_SLOTS 16384
#define COUNTER_MASK (COUNTER_SLOTS - 1)
#define T1_FREQ_THRESHOLD 3
#define VA_SHIFT 21  /* 2MB pages for VA → layer lookup */

/* Config map: loaded by userspace */
struct config {
    u32 num_layers;       /* Total number of model layers (e.g., 36 for 120B) */
    u32 t1_freq_threshold;/* Frequency threshold for T1 classification */
    u32 protect_distance; /* Layers within this distance are protected */
    u64 model_va_start;   /* Start of model weight VA range */
    u64 model_va_end;     /* End of model weight VA range */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct config);
} config_map SEC(".maps");

/* Layer VA boundaries: boundary_va[i] = start VA of layer i.
 * For a chunk with VA v: find largest i where boundary_va[i] <= v → layer = i.
 * Populated by userspace from equal-count chunk_trace analysis. */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_LAYERS);
    __type(key, u32);
    __type(value, u64);
} layer_boundaries SEC(".maps");

/* VA → layer mapping: optional hash map for specific chunk mappings.
 * Used as fast-path before falling back to boundary search. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);  /* ~128 GB VA space / 2MB = 65536 entries */
    __type(key, u32);            /* VA >> VA_SHIFT */
    __type(value, u32);          /* layer_id */
} va_to_layer_map SEC(".maps");

/* Runtime state: tracked per-CPU for lockless access */
struct runtime_state {
    u32 current_layer;    /* Estimated current layer being processed */
    u32 decode_step;      /* Decode step counter */
    u64 last_fault_va;    /* Last fault VA (for regression detection) */
    u64 faults_this_step; /* Faults in current decode step */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct runtime_state);
} state_map SEC(".maps");

/* T1 frequency counter (same pattern as passive MRU) */
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

static __always_inline u32 va_to_layer(u64 va, u32 num_layers)
{
    /* Fast path: check hash map first */
    u32 page_num = (u32)(va >> VA_SHIFT);
    u32 *layer_id = bpf_map_lookup_elem(&va_to_layer_map, &page_num);
    if (layer_id)
        return *layer_id;

    /* Slow path: binary-ish search through boundary array.
     * BPF verifier requires bounded loops, so we do a linear scan
     * (MAX_LAYERS=64 iterations is acceptable). */
    u32 result = 0xFFFFFFFF;
    for (u32 i = 0; i < MAX_LAYERS; i++) {
        if (i >= num_layers)
            break;
        u64 *boundary = bpf_map_lookup_elem(&layer_boundaries, &i);
        if (!boundary)
            break;
        if (va >= *boundary)
            result = i;
        else
            break;  /* Boundaries are sorted, no need to check further */
    }
    return result;
}

static __always_inline u32 belady_distance(u32 chunk_layer, u32 current_layer, u32 num_layers)
{
    if (chunk_layer == 0xFFFFFFFF || num_layers == 0)
        return 0;  /* Unknown → don't modify ordering */

    /* Cycle distance: how many layers until this chunk is needed again.
     * In LLM decode, layers are accessed 0 → N-1 → 0 → N-1 → ...
     * Distance = (chunk_layer - current_layer) mod num_layers */
    return (chunk_layer - current_layer + num_layers) % num_layers;
}

/* ===== PREFETCH: always_max (proven best for MoE) ===== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    /* always_max: prefetch entire VA block on any fault */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);
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

/* ===== EVICTION: T1-protect + Belady distance ===== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    /* Update runtime state: track fault VA for layer detection. */
    uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
    if (!va_block)
        return 0;
    u64 chunk_va = BPF_CORE_READ(va_block, start);
    if (chunk_va == 0)
        return 0;

    u32 zero = 0;
    struct runtime_state *st = bpf_map_lookup_elem(&state_map, &zero);
    if (!st)
        return 0;

    /* Detect decode step boundary: VA regression means new decode step */
    if (chunk_va < st->last_fault_va && st->faults_this_step > 10) {
        st->decode_step++;
        st->current_layer = 0;
        st->faults_this_step = 0;
    }

    st->last_fault_va = chunk_va;
    st->faults_this_step++;

    /* Quick layer update via hash map only (no boundary loop) */
    u32 page_num = (u32)(chunk_va >> VA_SHIFT);
    u32 *layer_id = bpf_map_lookup_elem(&va_to_layer_map, &page_num);
    if (layer_id)
        st->current_layer = *layer_id;

    /* T1 frequency check (moved from gpu_block_access) */
    u32 idx = chunk_hash(chunk);
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (!count)
        return 0;

    u8 c = *count;
    if (c < 255)
        *count = c + 1;

    if (c + 1 >= T1_FREQ_THRESHOLD) {
        /* T1 chunk: always protect */
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
        return 1; /* BYPASS */
    }

    /* Non-T1: use Belady distance for eviction ordering */
    struct config *cfg = bpf_map_lookup_elem(&config_map, &zero);
    if (!cfg)
        return 1;

    u32 chunk_layer = va_to_layer(chunk_va, cfg->num_layers);
    if (chunk_layer == 0xFFFFFFFF)
        return 1; /* Unknown layer: passive MRU fallback */

    u32 distance = belady_distance(chunk_layer, st->current_layer, cfg->num_layers);
    u32 protect_dist = cfg->protect_distance;
    if (protect_dist == 0)
        protect_dist = 3;  /* Default: protect layers within 3 of current */

    if (distance <= protect_dist) {
        /* Coming soon → protect */
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
    } else if (distance > cfg->num_layers / 2) {
        /* Far away → prioritize for eviction */
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_HEAD);
    }
    /* Middle distance: no move (passive ordering) */

    return 1; /* BYPASS */
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
struct gpu_mem_ops uvm_ops_template_belady = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
