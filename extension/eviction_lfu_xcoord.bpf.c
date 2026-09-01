/* SPDX-License-Identifier: GPL-2.0 */
/*
 * xCoord GPU-side: LFU Eviction + Shared GPU State Map
 *
 * Based on eviction_freq_pid_decay.bpf.c, with the addition of writing
 * per-PID GPU state (fault_rate, eviction_count) to a shared BPF map
 * that sched_ext can read for GPU-aware CPU scheduling.
 *
 * This is the gpu_ext half of xCoord's cross-subsystem coordination.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "trace_helper.h"
#include "eviction_common.h"
#include "shared_maps.h"

char _license[] SEC("license") = "GPL";

/* ========== Configuration ========== */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 8);
	__type(key, u32);
	__type(value, u64);
} policy_config SEC(".maps");

/* ========== Per-PID chunk stats (internal, same as freq_pid_decay) ========== */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, u32);
	__type(value, struct pid_chunk_stats);
} pid_chunk_count SEC(".maps");

/* ========== Active chunk tracking ========== */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 65536);
	__type(key, u64);    /* chunk pointer */
	__type(value, u32);  /* owner_pid */
} active_chunks SEC(".maps");

/* ========== Per-chunk access counter ========== */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 65536);
	__type(key, u64);    /* chunk pointer */
	__type(value, u64);  /* access count */
} chunk_access_count SEC(".maps");

/*
 * ========== xCoord Shared GPU State Map ==========
 *
 * This is the key addition for xCoord. This map will be pinned to
 * /sys/fs/bpf/xcoord_gpu_state by the userspace loader, allowing
 * sched_ext to read GPU state for scheduling decisions.
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, XCOORD_MAX_PIDS);
	__type(key, u32);                /* PID */
	__type(value, struct gpu_pid_state);
} gpu_state_map SEC(".maps");

/*
 * ========== xCoord UVM Worker Tracking ==========
 *
 * Tracks kernel worker thread PIDs that are actively handling UVM operations.
 * sched_ext reads this to boost the correct threads (kworkers, not user PIDs).
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, XCOORD_MAX_WORKERS);
	__type(key, u32);    /* worker thread PID (tgid) */
	__type(value, u64);  /* last activity timestamp (ns) */
} uvm_worker_pids SEC(".maps");

static __always_inline void track_uvm_worker(void)
{
	u32 worker_pid = bpf_get_current_pid_tgid() >> 32;
	u64 now = bpf_ktime_get_ns();
	bpf_map_update_elem(&uvm_worker_pids, &worker_pid, &now, BPF_ANY);
}

static __always_inline u64 get_config_u64(u32 key)
{
	u64 *val = bpf_map_lookup_elem(&policy_config, &key);
	return val ? *val : 0;
}

/*
 * update_gpu_state - Update shared GPU state for a PID
 *
 * Called on every chunk_activate (page fault). Maintains a sliding-window
 * fault rate that sched_ext reads for priority decisions.
 */
static __always_inline void update_gpu_state_fault(u32 pid)
{
	struct gpu_pid_state *state;
	struct gpu_pid_state new_state = {};
	u64 now = bpf_ktime_get_ns();

	state = bpf_map_lookup_elem(&gpu_state_map, &pid);
	if (state) {
		__sync_fetch_and_add(&state->fault_count, 1);

		/* Update fault_rate every ~1 second */
		u64 elapsed = now - state->last_update_ns;
		if (elapsed > 1000000000ULL) { /* 1 second */
			/* fault_rate = faults in window / elapsed seconds */
			state->fault_rate = state->fault_count * 1000000000ULL / elapsed;
			state->is_thrashing = (state->fault_rate > XCOORD_THRASHING_THRESHOLD) ? 1 : 0;
			state->fault_count = 0;
			state->last_update_ns = now;
		}
	} else {
		/* First fault for this PID */
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
	if (state) {
		__sync_fetch_and_add(&state->eviction_count, 1);
	}
}

static __always_inline void update_gpu_state_used(u32 pid)
{
	struct gpu_pid_state *state;

	state = bpf_map_lookup_elem(&gpu_state_map, &pid);
	if (state) {
		__sync_fetch_and_add(&state->used_count, 1);
	}
}

/* ========== Hook: chunk_activate (page fault) ========== */
SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
	     uvm_pmm_gpu_t *pmm,
	     uvm_gpu_chunk_t *chunk,
	     uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
	u32 owner_pid;
	struct pid_chunk_stats *stats;
	struct pid_chunk_stats new_stats = {0};
	u64 chunk_ptr = (u64)chunk;

	owner_pid = get_owner_pid_from_chunk(chunk);
	if (owner_pid == 0)
		return 0;

	/* xCoord: update shared GPU state + track worker thread */
	update_gpu_state_fault(owner_pid);
	track_uvm_worker();

	/* Check if this chunk was already tracked */
	if (bpf_map_lookup_elem(&active_chunks, &chunk_ptr))
		return 0;

	/* Track this chunk as active */
	bpf_map_update_elem(&active_chunks, &chunk_ptr, &owner_pid, BPF_ANY);

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

/* ========== Hook: chunk_used (access tracking) ========== */
SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
	     uvm_pmm_gpu_t *pmm,
	     uvm_gpu_chunk_t *chunk,
	     uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
	u32 owner_pid;
	u64 priority_pid;
	u64 low_priority_pid;
	u64 decay_factor;
	struct pid_chunk_stats *pid_stats;
	u64 chunk_ptr = (u64)chunk;
	u64 *access_count;
	u64 count;

	owner_pid = get_owner_pid_from_chunk(chunk);
	if (owner_pid == 0)
		return 0;

	/* xCoord: update shared GPU state + track worker thread */
	update_gpu_state_used(owner_pid);
	track_uvm_worker();

	priority_pid = get_config_u64(CONFIG_PRIORITY_PID);
	low_priority_pid = get_config_u64(CONFIG_LOW_PRIORITY_PID);

	pid_stats = bpf_map_lookup_elem(&pid_chunk_count, &owner_pid);
	if (pid_stats) {
		__sync_fetch_and_add(&pid_stats->total_used, 1);
	}

	/* Determine decay factor based on PID priority */
	if (priority_pid != 0 && owner_pid == (u32)priority_pid) {
		decay_factor = get_config_u64(CONFIG_PRIORITY_PARAM);
		if (decay_factor == 0) decay_factor = 1;
	} else if (low_priority_pid != 0 && owner_pid == (u32)low_priority_pid) {
		decay_factor = get_config_u64(CONFIG_LOW_PRIORITY_PARAM);
		if (decay_factor == 0) decay_factor = 10;
	} else {
		decay_factor = get_config_u64(CONFIG_DEFAULT_PARAM);
		if (decay_factor == 0) decay_factor = 5;
	}

	/* Get and increment access count for this chunk */
	access_count = bpf_map_lookup_elem(&chunk_access_count, &chunk_ptr);
	if (access_count) {
		count = __sync_fetch_and_add(access_count, 1) + 1;
	} else {
		u64 one = 1;
		bpf_map_update_elem(&chunk_access_count, &chunk_ptr, &one, BPF_ANY);
		count = 1;
	}

	/* Move tail only when access count reaches decay threshold */
	if (count % decay_factor == 0) {
		bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
		if (pid_stats) {
			__sync_fetch_and_add(&pid_stats->policy_allow, 1);
		}
	} else {
		if (pid_stats) {
			__sync_fetch_and_add(&pid_stats->policy_deny, 1);
		}
	}

	return 1; /* BYPASS kernel LRU */
}

/* ========== Hook: eviction_prepare ========== */
SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
	     uvm_pmm_gpu_t *pmm,
	     struct list_head *va_block_used,
	     struct list_head *va_block_unused)
{
	struct list_head *first;
	uvm_gpu_chunk_t *chunk;
	u32 *tracked_pid;
	struct pid_chunk_stats *stats;
	u64 chunk_ptr;

	if (!va_block_used)
		return 0;

	first = BPF_CORE_READ(va_block_used, next);
	if (!first || first == va_block_used)
		return 0;

	chunk = (uvm_gpu_chunk_t *)((char *)first -
		  __builtin_offsetof(struct uvm_gpu_chunk_struct, list));
	chunk_ptr = (u64)chunk;

	tracked_pid = bpf_map_lookup_elem(&active_chunks, &chunk_ptr);
	if (!tracked_pid)
		return 0;

	/* xCoord: update shared GPU state with eviction + track worker */
	update_gpu_state_eviction(*tracked_pid);
	track_uvm_worker();

	/* Decrement current_count for the tracked PID */
	stats = bpf_map_lookup_elem(&pid_chunk_count, tracked_pid);
	if (stats && stats->current_count > 0) {
		__sync_fetch_and_sub(&stats->current_count, 1);
	}

	/* Remove from tracking maps */
	bpf_map_delete_elem(&active_chunks, &chunk_ptr);
	bpf_map_delete_elem(&chunk_access_count, &chunk_ptr);

	return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_lfu_xcoord = {
	.gpu_test_trigger = (void *)NULL,
	.gpu_page_prefetch = (void *)NULL,
	.gpu_page_prefetch_iter = (void *)NULL,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
