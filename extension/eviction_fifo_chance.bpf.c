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
 * - activate: set initial chance_count based on PID
 * - chunk_used: reset chance_count (access = valuable)
 * - eviction_prepare: check HEAD chunk's chance:
 *     - chance > 0: decrement, move_tail (give another chance)
 *     - chance = 0: allow eviction
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
    u64 *val = bpf_map_lookup_elem(&config, &key);
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

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
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

SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
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

    /* Reset chance count on access (this chunk is valuable) */
    state->chance_count = state->initial_chance;

    /* FIFO: don't move on access, maintain arrival order */
    /* Return BYPASS to skip kernel's default LRU move */
    return 1;
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    struct list_head *first;
    uvm_gpu_chunk_t *chunk;
    struct chunk_state *state;
    struct pid_chunk_stats *stats;
    u64 chunk_ptr;
    int i;

    if (!va_block_used)
        return 0;

    /* Process up to 8 chunks at HEAD, giving second chances */
    #pragma unroll
    for (i = 0; i < 8; i++) {
        first = BPF_CORE_READ(va_block_used, next);
        if (!first || first == va_block_used)
            break;

        chunk = (uvm_gpu_chunk_t *)((char *)first -
                  __builtin_offsetof(struct uvm_gpu_chunk_struct, list));
        chunk_ptr = (u64)chunk;

        state = bpf_map_lookup_elem(&chunk_states, &chunk_ptr);
        if (!state) {
            /* Unknown chunk, let it be evicted */
            break;
        }

        if (state->chance_count > 0) {
            /* Give another chance: decrement and move to tail */
            state->chance_count--;
            bpf_uvm_pmm_chunk_move_tail(chunk, va_block_used);

            /* Record as policy_allow (saved from eviction) */
            stats = bpf_map_lookup_elem(&pid_chunk_count, &state->owner_pid);
            if (stats) {
                __sync_fetch_and_add(&stats->policy_allow, 1);
            }
            /* Continue to check next HEAD */
        } else {
            /* No more chances, allow eviction */
            /* Record as policy_deny (will be evicted) */
            stats = bpf_map_lookup_elem(&pid_chunk_count, &state->owner_pid);
            if (stats) {
                __sync_fetch_and_add(&stats->policy_deny, 1);
                if (stats->current_count > 0) {
                    __sync_fetch_and_sub(&stats->current_count, 1);
                }
            }
            /* Remove from tracking */
            bpf_map_delete_elem(&chunk_states, &chunk_ptr);
            break; /* Found victim, stop */
        }
    }

    return 0;
}

SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_fifo_chance = {
    .uvm_bpf_test_trigger_kfunc = (void *)NULL,
    .uvm_prefetch_before_compute = (void *)NULL,
    .uvm_prefetch_on_tree_iter = (void *)NULL,
    .uvm_pmm_chunk_activate = (void *)uvm_pmm_chunk_activate,
    .uvm_pmm_chunk_used = (void *)uvm_pmm_chunk_used,
    .uvm_pmm_eviction_prepare = (void *)uvm_pmm_eviction_prepare,
};
