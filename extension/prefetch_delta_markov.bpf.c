/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Block-local first-order delta/Markov prefetch policy.
 *
 * A read-only kprobe associates the current bitmap-tree callback context with
 * the VA block being serviced.  For each VA block, the policy learns one
 * likely successor for every observed page delta: delta[n] -> delta[n + 1].
 * It then requests a bounded intra-block region through the validated
 * bpf_gpu_set_prefetch_region() ABI.  It never invokes a raw migration helper
 * and never stores a callback decision pointer.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "safe_policy_models.h"

char _license[] SEC("license") = "GPL";

#define CONFIG_CONFIDENCE_THRESHOLD 0
#define CONFIG_PREFETCH_PAGES 1
#define CONFIG_MAX_DELTA 2

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 3);
	__type(key, u32);
	__type(value, u64);
} policy_config SEC(".maps");

struct block_key {
	u64 block_pointer;
	u64 block_start;
};

/* bitmap_tree is a temporary service context, so resolve it to a VA block. */
struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 256);
	__type(key, u64);
	__type(value, struct block_key);
} tree_to_block SEC(".maps");

struct block_state {
	s32 last_page;
	s32 last_delta;
	u8 initialized;
	u8 reserved[7];
};

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 4096);
	__type(key, struct block_key);
	__type(value, struct block_state);
} block_states SEC(".maps");

struct transition_key {
	struct block_key block;
	s32 predecessor;
	u32 reserved;
};

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 16384);
	__type(key, struct transition_key);
	__type(value, struct safe_delta_transition);
} transitions SEC(".maps");

struct delta_markov_stats {
	u64 context_captures;
	u64 callbacks;
	u64 blocks_initialized;
	u64 deltas_observed;
	u64 invalid_deltas;
	u64 transitions_created;
	u64 transition_matches;
	u64 transition_decays;
	u64 transition_replacements;
	u64 confident_predictions;
	u64 prefetch_requests;
	u64 empty_requests;
	u64 map_errors;
	u64 request_errors;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct delta_markov_stats);
} metrics SEC(".maps");

static __always_inline struct delta_markov_stats *get_metrics(void)
{
	u32 zero = 0;

	return bpf_map_lookup_elem(&metrics, &zero);
}

static __always_inline u64 get_config(u32 key, u64 fallback)
{
	u64 *value = bpf_map_lookup_elem(&policy_config, &key);

	return value && *value ? *value : fallback;
}

static __always_inline u64 bitmap_tree_id(
	uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
	u64 pointer = 0;

	/* Convert the trusted callback pointer to an opaque scalar map key. */
	bpf_probe_read_kernel(&pointer, sizeof(pointer), &bitmap_tree);
	return pointer;
}

static __always_inline int request_region(
	uvm_bpf_prefetch_decision_t *decision_ctx,
	u32 first,
	u32 outer,
	struct delta_markov_stats *stats)
{
	int err = bpf_gpu_set_prefetch_region(decision_ctx, first, outer);

	if (err != 0) {
		if (stats)
			stats->request_errors++;
		return 0;
	}

	if (stats) {
		if (first < outer)
			stats->prefetch_requests++;
		else
			stats->empty_requests++;
	}
	return 1;
}

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block,
	       uvm_va_block_t *va_block,
	       void *va_block_context,
	       u32 new_residency,
	       void *faulted_pages,
	       u32 faulted_region_packed,
	       uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
	struct delta_markov_stats *stats = get_metrics();
	struct block_key block = {};
	u64 tree_id;

	if (!va_block || !bitmap_tree)
		return 0;

	tree_id = (u64)bitmap_tree;
	block.block_pointer = (u64)va_block;
	block.block_start = BPF_CORE_READ(va_block, start);
	if (!tree_id || !block.block_pointer)
		return 0;

	if (bpf_map_update_elem(&tree_to_block, &tree_id, &block, BPF_ANY) != 0) {
		if (stats)
			stats->map_errors++;
		return 0;
	}
	if (stats)
		stats->context_captures++;
	return 0;
}

static __always_inline void account_transition(
	enum safe_delta_update update,
	struct delta_markov_stats *stats)
{
	if (!stats)
		return;
	if (update == SAFE_DELTA_NEW)
		stats->transitions_created++;
	else if (update == SAFE_DELTA_MATCH)
		stats->transition_matches++;
	else if (update == SAFE_DELTA_DECAY)
		stats->transition_decays++;
	else if (update == SAFE_DELTA_REPLACE)
		stats->transition_replacements++;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
	     uvm_page_index_t page_index,
	     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	     uvm_va_block_region_t *maximum_region,
	     uvm_bpf_prefetch_decision_t *decision_ctx)
{
	struct delta_markov_stats *stats = get_metrics();
	struct block_key *block;
	struct block_key block_value = {};
	struct block_state *state;
	struct block_state initial = {};
	struct transition_key learn_key = {};
	struct transition_key predict_key = {};
	struct safe_delta_transition *transition;
	struct safe_delta_transition new_transition = {};
	enum safe_delta_update update;
	u64 tree_id;
	u32 maximum_first;
	u32 maximum_outer;
	u32 maximum_delta;
	u32 confidence_threshold;
	u32 prefetch_pages;
	u32 first = 0;
	u32 outer = 0;
	s32 current_delta;
	s32 previous_delta;
	s32 predicted_delta;
	s64 magnitude;

	if (stats)
		stats->callbacks++;

	maximum_first = BPF_CORE_READ(maximum_region, first);
	maximum_outer = BPF_CORE_READ(maximum_region, outer);
	tree_id = bitmap_tree_id(bitmap_tree);
	block = bpf_map_lookup_elem(&tree_to_block, &tree_id);
	if (!block)
		return request_region(decision_ctx, 0, 0, stats) ? 1 : 0;
	block_value = *block;

	state = bpf_map_lookup_elem(&block_states, &block_value);
	if (!state) {
		initial.last_page = (s32)page_index;
		initial.last_delta = 0;
		initial.initialized = 1;
		if (bpf_map_update_elem(&block_states, &block_value, &initial,
					BPF_ANY) != 0) {
			if (stats)
				stats->map_errors++;
		}
		else if (stats) {
			stats->blocks_initialized++;
		}
		return request_region(decision_ctx, 0, 0, stats) ? 1 : 0;
	}

	current_delta = (s32)page_index - state->last_page;
	state->last_page = (s32)page_index;
	maximum_delta = (u32)get_config(CONFIG_MAX_DELTA, 128);
	magnitude = current_delta < 0 ? -(s64)current_delta : (s64)current_delta;
	if (current_delta == 0 || magnitude > (s64)maximum_delta) {
		state->last_delta = 0;
		if (stats)
			stats->invalid_deltas++;
		return request_region(decision_ctx, 0, 0, stats) ? 1 : 0;
	}

	if (stats)
		stats->deltas_observed++;

	previous_delta = state->last_delta;
	state->last_delta = current_delta;
	if (previous_delta != 0) {
		learn_key.block = block_value;
		learn_key.predecessor = previous_delta;
		transition = bpf_map_lookup_elem(&transitions, &learn_key);
		if (transition) {
			update = safe_delta_learn(transition, current_delta, 1);
			account_transition(update, stats);
		}
		else {
			update = safe_delta_learn(&new_transition,
						 current_delta, 0);
			if (bpf_map_update_elem(&transitions, &learn_key,
						&new_transition, BPF_ANY) != 0) {
				if (stats)
					stats->map_errors++;
			}
			else {
				account_transition(update, stats);
			}
		}
	}

	predict_key.block = block_value;
	predict_key.predecessor = current_delta;
	transition = bpf_map_lookup_elem(&transitions, &predict_key);
	confidence_threshold = (u32)get_config(CONFIG_CONFIDENCE_THRESHOLD, 2);
	if (!transition ||
	    !safe_delta_predict(transition, confidence_threshold,
				&predicted_delta))
		return request_region(decision_ctx, 0, 0, stats) ? 1 : 0;

	if (stats)
		stats->confident_predictions++;
	prefetch_pages = (u32)get_config(CONFIG_PREFETCH_PAGES, 2);
	if (!safe_delta_region((u32)page_index, predicted_delta,
			       prefetch_pages, maximum_first, maximum_outer,
			       &first, &outer))
		return request_region(decision_ctx, 0, 0, stats) ? 1 : 0;

	return request_region(decision_ctx, first, outer, stats) ? 1 : 0;
}

SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
	     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	     uvm_va_block_region_t *maximum_region,
	     uvm_va_block_region_t *current_region,
	     unsigned int counter,
	     uvm_bpf_prefetch_decision_t *decision_ctx)
{
	return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_delta_markov = {
	.gpu_test_trigger = (void *)0,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)0,
	.gpu_block_access = (void *)0,
	.gpu_evict_prepare = (void *)0,
};
