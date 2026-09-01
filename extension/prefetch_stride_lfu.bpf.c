/* SPDX-License-Identifier: GPL-2.0 */
/* Measurement build: one struct_ops object combining host stride + LFU. */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

#define CONFIG_CONFIDENCE_THRESHOLD 0
#define CONFIG_PREFETCH_PAGES 1
#define CONFIG_MAX_STRIDE 2
#define MAX_FREQ 255
#define LFU_ACCESS_SAMPLE_MASK 255

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} policy_config SEC(".maps");

struct stride_state {
    s32 last_page;
    s32 stride;
    s32 confidence;
    u32 total_faults;
    u32 prefetch_count;
    u32 stride_hits;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, u64);
    __type(value, struct stride_state);
} stride_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} va_block_cache SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 100000);
    __type(key, u64);
    __type(value, u32);
} chunk_freq SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_FREQ + 1);
    __type(key, u32);
    __type(value, u64);
} freq_to_chunk SEC(".maps");

struct lfu_state {
    u32 min_freq;
    u32 total_chunks;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct lfu_state);
} lfu_global SEC(".maps");

struct engagement_stats {
    u64 page_fault_calls;
    u64 stride_detections;
    u64 prefetches_issued;
    u64 lfu_activations;
    u64 lfu_accesses;
    u64 lfu_sampled_updates;
    u64 lfu_reorder_requests;
    u64 eviction_prepares;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct engagement_stats);
} engagement SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} lfu_access_clock SEC(".maps");

static __always_inline struct engagement_stats *get_engagement(void)
{
    u32 key = 0;
    return bpf_map_lookup_elem(&engagement, &key);
}

static __always_inline u64 get_config(u32 key, u64 fallback)
{
    u64 *value = bpf_map_lookup_elem(&policy_config, &key);
    return value ? *value : fallback;
}

static __always_inline s32 abs_s32(s32 value)
{
    return value < 0 ? -value : value;
}

static __always_inline u64 get_cached_va_block(void)
{
    u32 key = 0;
    u64 *cached = bpf_map_lookup_elem(&va_block_cache, &key);
    return cached ? *cached : 0;
}

static __always_inline struct lfu_state *get_lfu_state(void)
{
    u32 key = 0;
    struct lfu_state *state = bpf_map_lookup_elem(&lfu_global, &key);
    if (!state) {
        struct lfu_state initial = { .min_freq = 1, .total_chunks = 0 };
        bpf_map_update_elem(&lfu_global, &key, &initial, BPF_ANY);
        state = bpf_map_lookup_elem(&lfu_global, &key);
    }
    return state;
}

static __always_inline void clean_old_freq_bucket(u64 address, u32 old_freq)
{
    u64 *representative = bpf_map_lookup_elem(&freq_to_chunk, &old_freq);
    if (representative && *representative == address) {
        u64 zero = 0;
        bpf_map_update_elem(&freq_to_chunk, &old_freq, &zero, BPF_ANY);
    }
}

static __always_inline bool increase_freq(u64 address,
                                          uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u32 *frequency = bpf_map_lookup_elem(&chunk_freq, &address);
    if (!frequency)
        return false;

    u32 old_freq = *frequency;
    u32 new_freq = old_freq < MAX_FREQ ? old_freq + 1 : MAX_FREQ;
    clean_old_freq_bucket(address, old_freq);
    bpf_map_update_elem(&chunk_freq, &address, &new_freq, BPF_ANY);
    bpf_map_update_elem(&freq_to_chunk, &new_freq, &address, BPF_ANY);

    struct lfu_state *state = get_lfu_state();
    if (state && old_freq == state->min_freq) {
        u64 *old_bucket = bpf_map_lookup_elem(&freq_to_chunk, &old_freq);
        if (!old_bucket || *old_bucket == 0)
            state->min_freq = new_freq;
    }
    if (new_freq > old_freq)
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
    return new_freq > old_freq;
}

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
    u64 *cached = bpf_map_lookup_elem(&va_block_cache, &key);
    if (cached)
        *cached = (u64)va_block;
    return 0;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    struct engagement_stats *stats = get_engagement();
    if (stats)
        stats->page_fault_calls++;

    bpf_gpu_set_prefetch_region(decision_ctx, 0, 0);
    u64 va_block_ptr = get_cached_va_block();
    if (!va_block_ptr)
        return 1;

    struct stride_state *state = bpf_map_lookup_elem(&stride_map, &va_block_ptr);
    if (!state) {
        struct stride_state initial = {
            .last_page = (s32)page_index,
            .stride = 0,
            .confidence = 0,
            .total_faults = 1,
            .prefetch_count = 0,
            .stride_hits = 0,
        };
        bpf_map_update_elem(&stride_map, &va_block_ptr, &initial, BPF_ANY);
        return 1;
    }

    __sync_fetch_and_add(&state->total_faults, 1);
    if (state->last_page < 0) {
        state->last_page = (s32)page_index;
        return 1;
    }

    s32 current_stride = (s32)page_index - state->last_page;
    state->last_page = (s32)page_index;
    if (current_stride == 0)
        return 1;

    s32 max_stride = (s32)get_config(CONFIG_MAX_STRIDE, 128);
    if (abs_s32(current_stride) > max_stride) {
        if (state->confidence > 0)
            state->confidence--;
        return 1;
    }

    if (current_stride == state->stride) {
        state->confidence++;
        __sync_fetch_and_add(&state->stride_hits, 1);
        if (stats)
            stats->stride_detections++;
    } else {
        if (state->confidence > 0)
            state->confidence--;
        state->stride = current_stride;
    }

    s32 threshold = (s32)get_config(CONFIG_CONFIDENCE_THRESHOLD, 2);
    if (state->confidence < threshold)
        return 1;

    u32 prefetch_pages = (u32)get_config(CONFIG_PREFETCH_PAGES, 2);
    s32 predicted = (s32)page_index + state->stride;
    s32 first;
    s32 outer;
    if (state->stride > 0) {
        first = predicted;
        outer = predicted + (s32)prefetch_pages;
    } else {
        first = predicted - (s32)prefetch_pages + 1;
        outer = predicted + 1;
    }

    s32 max_first = (s32)BPF_CORE_READ(max_prefetch_region, first);
    s32 max_outer = (s32)BPF_CORE_READ(max_prefetch_region, outer);
    if (first < max_first)
        first = max_first;
    if (outer > max_outer)
        outer = max_outer;
    if (first < 0)
        first = 0;
    if (first >= outer)
        return 1;

    bpf_gpu_set_prefetch_region(decision_ctx,
                                (uvm_page_index_t)first,
                                (uvm_page_index_t)outer);
    __sync_fetch_and_add(&state->prefetch_count, 1);
    if (stats)
        stats->prefetches_issued++;
    return 1;
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

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    struct engagement_stats *stats = get_engagement();
    if (stats)
        stats->lfu_activations++;

    u64 address = (u64)chunk;
    u32 frequency = 1;
    bpf_map_update_elem(&chunk_freq, &address, &frequency, BPF_ANY);
    bpf_map_update_elem(&freq_to_chunk, &frequency, &address, BPF_ANY);
    struct lfu_state *state = get_lfu_state();
    if (state) {
        state->min_freq = 1;
        state->total_chunks++;
    }
    bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_HEAD);
    return 1;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    struct engagement_stats *stats = get_engagement();
    if (stats)
        stats->lfu_accesses++;
    u32 key = 0;
    u64 *clock = bpf_map_lookup_elem(&lfu_access_clock, &key);
    if (!clock)
        return 1;
    (*clock)++;
    if ((*clock & LFU_ACCESS_SAMPLE_MASK) != 0)
        return 1;
    if (stats)
        stats->lfu_sampled_updates++;
    if (increase_freq((u64)chunk, decision_ctx) && stats)
        stats->lfu_reorder_requests++;
    return 1;
}

SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    struct engagement_stats *stats = get_engagement();
    if (stats)
        stats->eviction_prepares++;
    return 0;
}

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buffer, int length)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_stride_lfu = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
