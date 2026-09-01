/* SPDX-License-Identifier: GPL-2.0 */
/*
 * PID-aware FIFO with Second Chance Eviction Policy
 *
 * Combines FIFO simplicity with access-awareness and PID-based differentiation:
 * - Each chunk has a "chance count" based on owner PID priority
 * - High priority: more chances (harder to evict)
 * - Low priority: fewer chances (easier to evict)
 *
 * Algorithm:
 * - activate: set initial chance_count based on PID, tail position (FIFO)
 * - gpu_block_access: if chunk was "warned" (chance decremented by eviction),
 *     move to tail (second chance save) and restore chance. Otherwise FIFO bypass.
 * - eviction_prepare: walk HEAD chunks, decrement chances via map.
 *     Chunks with 0 chances are left at HEAD for kernel to evict.
 *     Chunks with remaining chances will be saved on next access.
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
} policy_config SEC(".maps");

/* Per-chunk state */
struct chunk_state {
    u32 owner_pid;
    u8  chance_count;
    u8  initial_chance;
};

/* Chunk state tracking: chunk_ptr -> state */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, u64);
    __type(value, struct chunk_state);
} chunk_states SEC(".maps");

/* Per-PID stats */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, u32);
    __type(value, struct pid_chunk_stats);
} pid_chunk_count SEC(".maps");

static __always_inline u64 get_config_u64(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&policy_config, &key);
    return val ? *val : 0;
}

static __always_inline u8 get_initial_chance(u32 owner_pid)
{
    u64 priority_pid = get_config_u64(CONFIG_PRIORITY_PID);
    u64 low_priority_pid = get_config_u64(CONFIG_LOW_PRIORITY_PID);

    if (priority_pid != 0 && owner_pid == (u32)priority_pid) {
        u64 chance = get_config_u64(CONFIG_PRIORITY_PARAM);
        return chance > 255 ? 255 : (u8)chance;
    } else if (low_priority_pid != 0 && owner_pid == (u32)low_priority_pid) {
        u64 chance = get_config_u64(CONFIG_LOW_PRIORITY_PARAM);
        return chance > 255 ? 255 : (u8)chance;
    } else {
        u64 chance = get_config_u64(CONFIG_DEFAULT_PARAM);
        return chance > 255 ? 255 : (u8)chance;
    }
}

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u64 chunk_ptr = (u64)chunk;
    u32 owner_pid;
    struct chunk_state state = {0};
    struct pid_chunk_stats *stats;
    struct pid_chunk_stats new_stats = {0};

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    /* Check if already tracked */
    if (bpf_map_lookup_elem(&chunk_states, &chunk_ptr))
        return 0;

    /* Set initial state based on PID */
    state.owner_pid = owner_pid;
    state.initial_chance = get_initial_chance(owner_pid);
    state.chance_count = state.initial_chance;

    bpf_map_update_elem(&chunk_states, &chunk_ptr, &state, BPF_ANY);

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

    /* FIFO: new chunks go to tail (will be evicted after older ones) */
    /* Default behavior is fine, no need to move */

    return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u64 chunk_ptr = (u64)chunk;
    struct chunk_state *state;
    struct pid_chunk_stats *stats;
    u32 owner_pid;

    state = bpf_map_lookup_elem(&chunk_states, &chunk_ptr);
    if (!state)
        return 0;

    owner_pid = state->owner_pid;

    /* Update total_used */
    stats = bpf_map_lookup_elem(&pid_chunk_count, &owner_pid);
    if (stats) {
        __sync_fetch_and_add(&stats->total_used, 1);
    }

    if (state->chance_count < state->initial_chance) {
        /* Chunk was "warned" by evict_prepare (chance decremented).
         * This access saves it — move to tail (second chance!) */
        state->chance_count = state->initial_chance;
        bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);

        if (stats)
            __sync_fetch_and_add(&stats->policy_allow, 1);
        return 1;
    }

    /* Not at risk — maintain FIFO order, don't move */
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
    struct chunk_state *state;
    struct pid_chunk_stats *stats;
    int i;

    if (!va_block_used)
        return 0;

    /*
     * Walk up to 8 chunks from HEAD. For each:
     * - If chance > 0: decrement (mark "at risk"). Next gpu_block_access
     *   will see reduced chance and move it to tail (second chance save).
     * - If chance == 0: allow eviction, clean up tracking.
     *
     * We cannot request a reorder here because the chunk pointer
     * derived from container_of is not a trusted pointer for the verifier.
     * The actual move happens in gpu_block_access instead.
     */
    cur = va_block_used;
    #pragma unroll
    for (i = 0; i < 8; i++) {
        cur = BPF_CORE_READ(cur, next);
        if (!cur || cur == va_block_used)
            break;

        /* container_of: list_head -> uvm_gpu_chunk_struct */
        bpf_probe_read_kernel(&chunk_ptr, sizeof(chunk_ptr),
                              (void *)((char *)cur -
                              __builtin_offsetof(struct uvm_gpu_chunk_struct, list)));
        /* chunk_ptr is now the address of the chunk struct itself
         * (since list is at some offset, we read the struct start) */
        chunk_ptr = (u64)((char *)cur -
                    __builtin_offsetof(struct uvm_gpu_chunk_struct, list));

        state = bpf_map_lookup_elem(&chunk_states, &chunk_ptr);
        if (!state)
            break;

        if (state->chance_count > 0) {
            /* Decrement chance — chunk is "warned".
             * If accessed before next eviction, gpu_block_access will
             * see chance < initial and move it to tail. */
            state->chance_count--;
        } else {
            /* No more chances, allow eviction */
            stats = bpf_map_lookup_elem(&pid_chunk_count, &state->owner_pid);
            if (stats) {
                __sync_fetch_and_add(&stats->policy_deny, 1);
                if (stats->current_count > 0)
                    __sync_fetch_and_sub(&stats->current_count, 1);
            }
            bpf_map_delete_elem(&chunk_states, &chunk_ptr);
            break; /* Found victim, stop */
        }
    }

    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_fifo_chance = {
    .gpu_test_trigger = (void *)NULL,
    .gpu_page_prefetch = (void *)NULL,
    .gpu_page_prefetch_iter = (void *)NULL,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
