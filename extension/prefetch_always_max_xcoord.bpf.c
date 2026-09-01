/* SPDX-License-Identifier: GPL-2.0 */
/*
 * xCoord GPU-side: Always-Max Prefetch + Cycle-MoE Eviction + Shared State
 *
 * Same as prefetch_always_max_cycle_moe but with xCoord shared maps:
 *   - gpu_state_map: per-PID fault_rate, eviction_count, is_thrashing
 *   - uvm_worker_pids: tracks UVM BH kthread PIDs for sched_ext boosting
 *
 * This is the recommended GPU-side policy for xCoord experiments.
 * Pinned maps at /sys/fs/bpf/xcoord_gpu_state, /sys/fs/bpf/xcoord_uvm_workers
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "trace_helper.h"
#include "shared_maps.h"

char _license[] SEC("license") = "GPL";

/* ========================================================================
 * Eviction Configuration (from cycle_moe)
 * ======================================================================== */

#define T1_FREQ_THRESHOLD 3
#define COUNTER_SLOTS 16384
#define COUNTER_MASK (COUNTER_SLOTS - 1)

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

/* ========================================================================
 * xCoord Shared Maps (pinned for sched_ext to read)
 * ======================================================================== */

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, XCOORD_MAX_PIDS);
    __type(key, u32);
    __type(value, struct gpu_pid_state);
} gpu_state_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, XCOORD_MAX_WORKERS);
    __type(key, u32);
    __type(value, u64);
} uvm_worker_pids SEC(".maps");

static __always_inline void track_uvm_worker(void)
{
    u32 worker_pid = bpf_get_current_pid_tgid() >> 32;
    u64 now = bpf_ktime_get_ns();
    bpf_map_update_elem(&uvm_worker_pids, &worker_pid, &now, BPF_ANY);
}

static __always_inline void update_gpu_state_fault(u32 pid)
{
    struct gpu_pid_state *state;
    struct gpu_pid_state new_state = {};
    u64 now = bpf_ktime_get_ns();

    state = bpf_map_lookup_elem(&gpu_state_map, &pid);
    if (state) {
        __sync_fetch_and_add(&state->fault_count, 1);
        u64 elapsed = now - state->last_update_ns;
        if (elapsed > 1000000000ULL) {
            state->fault_rate = state->fault_count * 1000000000ULL / elapsed;
            state->is_thrashing = (state->fault_rate > XCOORD_THRASHING_THRESHOLD) ? 1 : 0;
            state->fault_count = 0;
            state->last_update_ns = now;
        }
    } else {
        new_state.fault_count = 1;
        new_state.fault_rate = 0;
        new_state.last_update_ns = now;
        bpf_map_update_elem(&gpu_state_map, &pid, &new_state, BPF_ANY);
    }
}

static __always_inline void update_gpu_state_eviction(u32 pid)
{
    struct gpu_pid_state *state;
    state = bpf_map_lookup_elem(&gpu_state_map, &pid);
    if (state)
        __sync_fetch_and_add(&state->eviction_count, 1);
}

static __always_inline void update_gpu_state_used(u32 pid)
{
    struct gpu_pid_state *state;
    state = bpf_map_lookup_elem(&gpu_state_map, &pid);
    if (state)
        __sync_fetch_and_add(&state->used_count, 1);
}

/* ========================================================================
 * Prefetch Hooks (always_max)
 * ======================================================================== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

/* ========================================================================
 * Eviction Hooks (cycle_moe + xCoord tracking)
 * ======================================================================== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u32 owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid) {
        update_gpu_state_fault(owner_pid);
        update_gpu_state_used(owner_pid);
        track_uvm_worker();
    }

    /* cycle_moe: T1 frequency-based protection */
    u32 idx = chunk_hash(chunk);
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (count) {
        u8 c = *count;
        if (c < 255)
            *count = c + 1;

        if (c + 1 >= T1_FREQ_THRESHOLD) {
            bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
            return 1; /* BYPASS: T1 protected */
        }
    }

    return 0; /* kernel default for non-T1 */
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
    /* xCoord: track eviction event */
    track_uvm_worker();
    return 0;
}

/* ========================================================================
 * Struct ops registration — all 6 hooks
 * ======================================================================== */

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_always_max_xcoord = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
