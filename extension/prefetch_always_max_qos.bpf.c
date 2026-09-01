/* SPDX-License-Identifier: GPL-2.0 */
/*
 * QoS-Driven Eviction: Always-Max Prefetch + Cycle-MoE + Feedback-Controlled
 * LC Page Protection
 *
 * Dual-actuator approach:
 *   SENSOR:     LC fault_rate (per-PID, from gpu_state_map)
 *   CONTROLLER: Integral controller (regulate_eviction, lazy in chunk_activate)
 *   ACTUATOR:   eviction_bias → chunk_used protects LC pages via move_tail
 *
 * When LC fault_rate > target:
 *   - LC chunks unconditionally move_tail (LRU protected, won't be evicted)
 *   - BE chunks follow normal cycle_moe (subject to eviction)
 * When LC fault_rate < target:
 *   - eviction_bias decays → BE pages resume normal protection
 *
 * Also publishes xCoord shared maps for optional CPU-side sched_ext.
 *
 * Based on prefetch_always_max_xcoord.bpf.c
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

/* ========================================================================
 * LC PID Registration (populated by loader via -l PID)
 * ======================================================================== */

#define MAX_LC_PIDS 16

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_LC_PIDS);
    __type(key, u32);
    __type(value, u32);
} lc_pid_array SEC(".maps");

const volatile u32 n_lc_pids = 0;

/* ========================================================================
 * Feedback Controller Parameters (rodata, set by loader)
 * ======================================================================== */

const volatile u64 target_fault_rate = 200;       /* LC target faults/sec */
const volatile u64 ki_gain = 10;                   /* integral gain */
const volatile u32 decay_shift = 2;                /* decay rate: >>2 = /4 per interval */
const volatile u64 max_integral = 10000;           /* anti-windup cap */
const volatile u64 regulate_interval_ns = 100000000ULL; /* 100ms */

/* ========================================================================
 * Controller State (BPF global variables, readable from userspace)
 * ======================================================================== */

volatile u64 eviction_integral = 0;
volatile u32 eviction_bias = 0;       /* 0-1000: LC protection strength */
volatile u64 last_regulate_ns = 0;
volatile u64 lc_fault_rate_observed = 0;

/* ========================================================================
 * Statistics
 * ======================================================================== */

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u64));
    __uint(max_entries, 8);
} stats SEC(".maps");

enum stat_idx {
    STAT_ACTIVATE       = 0,  /* chunk_activate calls */
    STAT_USED           = 1,  /* chunk_used calls (gpu_block_access) */
    STAT_T1_PROTECTED   = 2,  /* cycle_moe T1 frequency protection */
    STAT_LC_PROTECTED   = 3,  /* LC chunk moved to tail in activate (protected) */
    STAT_EVICT          = 4,  /* evict_prepare calls */
    STAT_REGULATE       = 5,  /* regulate_eviction calls */
    STAT_BE_DEMOTED     = 6,  /* BE chunk moved to head in activate (penalized) */
    STAT_UNUSED7        = 7,  /* reserved */
};

static __always_inline void stat_inc(u32 idx)
{
    u64 *cnt_p = bpf_map_lookup_elem(&stats, &idx);
    if (cnt_p)
        (*cnt_p)++;
}

/* ========================================================================
 * Helper: UVM Worker Tracking
 * ======================================================================== */

static __always_inline void track_uvm_worker(void)
{
    u32 worker_pid = bpf_get_current_pid_tgid() >> 32;
    u64 now = bpf_ktime_get_ns();
    bpf_map_update_elem(&uvm_worker_pids, &worker_pid, &now, BPF_ANY);
}

/* ========================================================================
 * Helper: GPU State Tracking (same as xcoord)
 * ======================================================================== */

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
 * Helper: LC PID Check
 * ======================================================================== */

static __always_inline bool is_lc_process(u32 pid)
{
    /*
     * Loop bound is compile-time constant (MAX_LC_PIDS=16)
     * so BPF verifier accepts it.
     */
    for (u32 i = 0; i < MAX_LC_PIDS; i++) {
        if (i >= n_lc_pids)
            return false;
        u32 *lc_pid = bpf_map_lookup_elem(&lc_pid_array, &i);
        if (lc_pid && *lc_pid == pid)
            return true;
    }
    return false;
}

/* ========================================================================
 * Feedback Controller: regulate_eviction()
 *
 * Called lazily in chunk_activate (every page fault).
 * Reads LC fault_rate from gpu_state_map, computes eviction_bias.
 * ======================================================================== */

static __always_inline void regulate_eviction(void)
{
    u64 now = bpf_ktime_get_ns();

    if (now - last_regulate_ns < regulate_interval_ns)
        return;
    last_regulate_ns = now;

    stat_inc(STAT_REGULATE);

    if (n_lc_pids == 0)
        return;

    /* 1. SENSOR: read max LC fault_rate */
    u64 max_lc_fr = 0;

    for (u32 i = 0; i < MAX_LC_PIDS; i++) {
        if (i >= n_lc_pids)
            break;
        u32 *lc_pid_p = bpf_map_lookup_elem(&lc_pid_array, &i);
        if (!lc_pid_p || *lc_pid_p == 0)
            continue;

        struct gpu_pid_state *state = bpf_map_lookup_elem(&gpu_state_map, lc_pid_p);
        if (!state)
            continue;

        /* Staleness check: ignore if not updated in 2s */
        if (now - state->last_update_ns > 2000000000ULL)
            continue;

        if (state->fault_rate > max_lc_fr)
            max_lc_fr = state->fault_rate;
    }

    lc_fault_rate_observed = max_lc_fr;

    /* 2. CONTROLLER: integral with decay */
    u64 integral = eviction_integral;

    if (max_lc_fr > target_fault_rate) {
        u64 error = max_lc_fr - target_fault_rate;
        integral += error * ki_gain / 1000;
        if (integral > max_integral)
            integral = max_integral;
    } else {
        /* LC OK → decay: let BE recover */
        integral >>= decay_shift;
    }

    eviction_integral = integral;

    /* 3. ACTUATOR: convert to eviction bias (0-1000) */
    u32 bias = 0;
    if (max_integral > 0) {
        bias = (u32)(integral * 1000 / max_integral);
        if (bias > 1000)
            bias = 1000;
    }
    eviction_bias = bias;
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
 * Eviction Hooks (cycle_moe + QoS eviction protection)
 * ======================================================================== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    stat_inc(STAT_ACTIVATE);

    u32 owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid) {
        update_gpu_state_fault(owner_pid);
        track_uvm_worker();
    }

    /* Run feedback controller (lazy, every regulate_interval) */
    regulate_eviction();

    /*
     * PRIMARY ACTUATOR: QoS eviction bias via LRU position at activate time.
     *
     * chunk_activate fires on EVERY page fault (verified: activate=1.8M).
     *
     * When eviction_bias > 0:
     *   LC chunk → move_tail (MRU = last to evict = protected)
     *   BE chunk → move_head (LRU = first to evict = penalized)
     * When eviction_bias == 0:
     *   fall through to cycle_moe T1 check below
     */
    if (eviction_bias > 0 && owner_pid) {
        if (is_lc_process(owner_pid)) {
            stat_inc(STAT_LC_PROTECTED);
            bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
            return 1; /* BYPASS: LC at MRU tail */
        } else {
            stat_inc(STAT_BE_DEMOTED);
            bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_HEAD);
            return 1; /* BYPASS: BE at LRU head → evict first */
        }
    }

    /* cycle_moe: T1 frequency-based protection (moved from gpu_block_access) */
    u32 idx = chunk_hash(chunk);
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (count) {
        u8 c = *count;
        if (c < 255)
            *count = c + 1;

        if (c + 1 >= T1_FREQ_THRESHOLD) {
            stat_inc(STAT_T1_PROTECTED);
            bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
            return 1; /* BYPASS: T1 protected */
        }
    }

    return 0; /* default: kernel moves to tail */
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
    stat_inc(STAT_EVICT);
    track_uvm_worker();
    return 0;
}

/* ========================================================================
 * Struct ops registration
 * ======================================================================== */

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_always_max_qos = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
