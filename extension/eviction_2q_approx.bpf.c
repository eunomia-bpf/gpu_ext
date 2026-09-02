/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Approximate 2Q / segmented-LRU eviction policy.
 *
 * The current PMM ABI can move only the callback's chunk to a list endpoint.
 * We therefore map 2Q's A1in and Am queues onto one UVM used list:
 *
 *   probationary (A1in-like) -> USED/HEAD
 *   protected    (Am-like)   -> USED/TAIL
 *
 * A second distinct list-generation episode promotes a chunk by default.
 * Same-generation activate/access callbacks count once. This is deliberately
 * an approximation: there is no separately sized A1out ghost queue and no ABI
 * to demote an arbitrary protected chunk. Direct-mapped per-CPU metadata keeps
 * the hot callback allocation-free; slot collisions or CPU migration cause a
 * conservative re-admission to probation.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "safe_policy_models.h"

char _license[] SEC("license") = "GPL";

#define CONFIG_PROMOTE_AFTER 0
#define CONFIG_MAX_GENERATION_GAP 1

#define TWOQ_SLOTS 16384
#define TWOQ_SLOT_MASK (TWOQ_SLOTS - 1)

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 2);
	__type(key, u32);
	__type(value, u64);
} policy_config SEC(".maps");

struct twoq_slot {
	u64 owner_id;
	u64 root_id;
	struct safe_twoq_state state;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, TWOQ_SLOTS);
	__type(key, u32);
	__type(value, struct twoq_slot);
} chunk_slots SEC(".maps");

struct twoq_stats {
	u64 activate_events;
	u64 access_events;
	u64 admissions;
	u64 identity_resets;
	u64 generation_resets;
	u64 same_episode_events;
	u64 probation_head_requests;
	u64 promotions;
	u64 protected_tail_requests;
	u64 reorder_errors;
	u64 eviction_prepares;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct twoq_stats);
} metrics SEC(".maps");

static __always_inline struct twoq_stats *get_metrics(void)
{
	u32 zero = 0;

	return bpf_map_lookup_elem(&metrics, &zero);
}

static __always_inline u64 get_config(u32 key, u64 fallback)
{
	u64 *value = bpf_map_lookup_elem(&policy_config, &key);

	return value && *value ? *value : fallback;
}

static __always_inline u32 slot_index(u64 owner_id, u64 root_id)
{
	u64 mixed = (root_id >> 6) ^ (root_id >> 18) ^
		    (owner_id >> 7) ^ (owner_id >> 23);

	return (u32)mixed & TWOQ_SLOT_MASK;
}

static __always_inline int request_position(
	uvm_bpf_pmm_decision_ctx_t *decision_ctx,
	enum safe_twoq_action action,
	struct twoq_stats *stats)
{
	u64 position;
	int err;

	if (action == SAFE_TWOQ_MOVE_HEAD)
		position = NV_GPU_PMM_POSITION_HEAD;
	else if (action == SAFE_TWOQ_MOVE_TAIL)
		position = NV_GPU_PMM_POSITION_TAIL;
	else
		return 0;

	err = bpf_gpu_request_reorder(decision_ctx,
				      NV_GPU_PMM_DESTINATION_USED,
				      position);
	if (err != 0) {
		if (stats)
			stats->reorder_errors++;
		return 0;
	}

	return 1;
}

static __always_inline int observe_chunk(
	uvm_bpf_pmm_decision_ctx_t *decision_ctx,
	struct twoq_stats *stats)
{
	u64 owner_id = BPF_CORE_READ(decision_ctx, observed.owner_id);
	u64 root_id = BPF_CORE_READ(decision_ctx, observed.root_id);
	u64 generation = BPF_CORE_READ(decision_ctx, observed.generation);
	u32 index;
	u32 promote_after;
	u64 maximum_gap;
	struct twoq_slot *slot;
	enum safe_twoq_action action;
	int new_identity;
	int generation_reset = 0;
	int same_episode = 0;
	u8 old_segment;

	if (!owner_id || !root_id)
		return 0;

	index = slot_index(owner_id, root_id);
	slot = bpf_map_lookup_elem(&chunk_slots, &index);
	if (!slot)
		return 0;

	new_identity = slot->owner_id != owner_id || slot->root_id != root_id;
	if (!new_identity && slot->state.segment != SAFE_TWOQ_EMPTY) {
		maximum_gap = get_config(CONFIG_MAX_GENERATION_GAP, 2);
		generation_reset = safe_twoq_generation_is_new(
			slot->state.generation, generation, maximum_gap);
		same_episode = !generation_reset &&
			       slot->state.generation == generation;
	}

	if (new_identity) {
		slot->owner_id = owner_id;
		slot->root_id = root_id;
		if (stats) {
			stats->admissions++;
			if (slot->state.segment != SAFE_TWOQ_EMPTY)
				stats->identity_resets++;
		}
	}
	else if (generation_reset && stats) {
		stats->generation_resets++;
		stats->admissions++;
	}
	else if (same_episode && stats) {
		stats->same_episode_events++;
	}

	old_segment = slot->state.segment;
	promote_after = (u32)get_config(CONFIG_PROMOTE_AFTER, 2);
	action = safe_twoq_observe(&slot->state,
				   generation,
				   promote_after,
				   new_identity || generation_reset);

	if (stats) {
		if (action == SAFE_TWOQ_MOVE_HEAD)
			stats->probation_head_requests++;
		else if (old_segment == SAFE_TWOQ_PROBATION &&
			 slot->state.segment == SAFE_TWOQ_PROTECTED)
			stats->promotions++;
		else if (action == SAFE_TWOQ_MOVE_TAIL)
			stats->protected_tail_requests++;
	}

	return request_position(decision_ctx, action, stats);
}

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
	     uvm_pmm_gpu_t *pmm,
	     uvm_gpu_chunk_t *chunk,
	     uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
	struct twoq_stats *stats = get_metrics();

	if (stats)
		stats->activate_events++;
	observe_chunk(decision_ctx, stats);
	/* Activate requests are applied independently of the return action. */
	return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
	     uvm_pmm_gpu_t *pmm,
	     uvm_gpu_chunk_t *chunk,
	     uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
	struct twoq_stats *stats = get_metrics();

	if (stats)
		stats->access_events++;
	/* BYPASS only when our callback-local reorder request was accepted. */
	return observe_chunk(decision_ctx, stats) ? 1 : 0;
}

SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
	     uvm_pmm_gpu_t *pmm,
	     struct list_head *va_block_used,
	     struct list_head *va_block_unused)
{
	struct twoq_stats *stats = get_metrics();

	if (stats)
		stats->eviction_prepares++;
	return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_2q_approx = {
	.gpu_test_trigger = (void *)0,
	.gpu_page_prefetch = (void *)0,
	.gpu_page_prefetch_iter = (void *)0,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
