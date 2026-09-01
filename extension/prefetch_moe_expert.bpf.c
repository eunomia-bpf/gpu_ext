/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Transparent MoE expert proactive prefetch via bitmap fault replay.
 *
 * Tracks the largest cudaMallocManaged() allocation as the model buffer.
 * With cycle_moe eviction, frequently reused attention pages stay resident,
 * while expert pages fault once per routing decision. Replaying the previous
 * token's model-fault bitmap therefore acts as transparent expert prefetch.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

#define T1_FREQ_THRESHOLD 3
#define COUNTER_SLOTS 16384
#define COUNTER_MASK (COUNTER_SLOTS - 1)

#define VA_BLOCK_SIZE (2ULL * 1024 * 1024)
#define BITMAP_WORDS 512
#define BITMAP_BITS (BITMAP_WORDS * 64)

#ifndef TASK_COMM_LEN
#define TASK_COMM_LEN 16
#endif

const volatile __u64 min_alloc_size = 10ULL * 1024 * 1024 * 1024;
const volatile __u64 min_sync_gap_ns = 5000000ULL;
const volatile char target_comm[TASK_COMM_LEN] = "";

struct model_buffer_info {
	u64 base_addr;
	u64 size;
	u32 tgid;
	u32 reserved;
};

struct token_stats {
	u64 sync_count;
	u64 prefetch_count;
	u64 faults_recorded;
	u64 prefetch_blocks;
};

enum debug_stat {
	DBG_RECORD_CALLED = 0,
	DBG_NO_MODEL,
	DBG_NO_MODEL_ADDR,
	DBG_OVERFLOW,
	DBG_NO_VA_BLOCK,
	DBG_NO_BLOCK_START,
	DBG_OUT_OF_RANGE,
	DBG_BITMAP_SET,
	DBG_BITMAP_DUP,
	DBG_MAX_STATS,
};

struct alloc_args {
	u64 size;
	u64 dev_ptr_addr;
};

struct pending_prefetch_state {
	u64 generation;
	u64 drained_generation;
};

struct prefetch_work_item {
	u64 va_space;
	u64 generation;
	struct bpf_wq work;
};

struct bitmap_swap_ctx {
	u64 has_bits;
};

struct bitmap_prefetch_ctx {
	u64 va_space;
	u64 base_addr;
	u64 size;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct model_buffer_info);
} model_buffer_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, BITMAP_WORDS);
	__type(key, u32);
	__type(value, u64);
} fault_bitmap_current SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, BITMAP_WORDS);
	__type(key, u32);
	__type(value, u64);
} fault_bitmap_prev SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} va_space_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct token_stats);
} token_stats_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 16);
	__type(key, u32);
	__type(value, u64);
} debug_stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} first_block_start SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, COUNTER_SLOTS);
	__type(key, u32);
	__type(value, u8);
} access_counts SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct prefetch_work_item);
} wq_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u32);
} target_tgid_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u64);
	__type(value, struct alloc_args);
} alloc_args_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct pending_prefetch_state);
} pending_prefetch_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} last_sync_ns_map SEC(".maps");

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
	u64 ptr = 0;

	bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
	return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

static __always_inline int comm_matches_target(void)
{
	char comm[TASK_COMM_LEN] = {};
	int i;

	if (target_comm[0] == '\0')
		return 1;

	if (bpf_get_current_comm(comm, sizeof(comm)) != 0)
		return 0;

#pragma unroll
	for (i = 0; i < TASK_COMM_LEN; i++) {
		if (comm[i] != target_comm[i])
			return 0;
		if (target_comm[i] == '\0')
			return 1;
	}

	return 1;
}

static long reset_bitmaps_cb(u32 word, void *ctx)
{
	if (word >= BITMAP_WORDS)
		return 0;

	u64 *current = bpf_map_lookup_elem(&fault_bitmap_current, &word);
	u64 *prev = bpf_map_lookup_elem(&fault_bitmap_prev, &word);

	if (current)
		*current = 0;
	if (prev)
		*prev = 0;

	return 0;
}

static __always_inline void bump_debug_stat(enum debug_stat stat)
{
	u32 key = stat;
	u64 *value;

	if (stat >= DBG_MAX_STATS)
		return;

	value = bpf_map_lookup_elem(&debug_stats, &key);
	if (value)
		__sync_fetch_and_add(value, 1);
}

static __always_inline void remember_first_block_start(u64 block_start)
{
	u32 zero = 0;
	u64 *value;

	value = bpf_map_lookup_elem(&first_block_start, &zero);
	if (value && *value == 0)
		*value = block_start;
}

static long reset_debug_stats_cb(u32 stat, void *ctx)
{
	u64 *value;

	if (stat >= DBG_MAX_STATS)
		return 0;

	value = bpf_map_lookup_elem(&debug_stats, &stat);
	if (value)
		*value = 0;

	return 0;
}

static long swap_bitmap_cb(u32 word, void *ctx)
{
	struct bitmap_swap_ctx *swap = ctx;
	u64 *current;
	u64 *prev;
	u64 bits = 0;

	if (word >= BITMAP_WORDS)
		return 0;

	current = bpf_map_lookup_elem(&fault_bitmap_current, &word);
	prev = bpf_map_lookup_elem(&fault_bitmap_prev, &word);
	if (!current || !prev)
		return 0;

	bits = *current;
	*prev = bits;
	*current = 0;
	if (bits)
		swap->has_bits = 1;

	return 0;
}

static long prefetch_bitmap_block_cb(u32 block_idx, void *ctx)
{
	u32 zero = 0;
	struct bitmap_prefetch_ctx *prefetch = ctx;
	struct token_stats *stats;
	u32 word;
	u64 *bitmap_word;
	u64 bit = 1ULL << (block_idx % 64);
	u64 offset;
	u64 addr;
	u64 length;

	if (!prefetch->va_space || !prefetch->base_addr || !prefetch->size)
		return 0;

	if (block_idx >= BITMAP_BITS)
		return 0;

	word = block_idx / 64;
	if (word >= BITMAP_WORDS)
		return 0;

	bitmap_word = bpf_map_lookup_elem(&fault_bitmap_prev, &word);
	if (!bitmap_word)
		return 0;

	if (!(*bitmap_word & bit))
		return 0;

	offset = (u64)block_idx * VA_BLOCK_SIZE;
	if (offset >= prefetch->size)
		return 0;

	addr = prefetch->base_addr + offset;
	if (addr < prefetch->base_addr)
		return 0;

	length = prefetch->size - offset;
	if (length > VA_BLOCK_SIZE)
		length = VA_BLOCK_SIZE;

	if (bpf_gpu_migrate_range(prefetch->va_space, addr, length) != 0)
		return 0;

	stats = bpf_map_lookup_elem(&token_stats_map, &zero);
	if (stats)
		__sync_fetch_and_add(&stats->prefetch_blocks, 1);

	return 0;
}

static int do_bitmap_prefetch(void *map, int *key, void *value)
{
	u32 zero = 0;
	struct prefetch_work_item *req = value;
	struct model_buffer_info *model;
	struct bitmap_prefetch_ctx ctx = {};

	if (!req || !req->va_space)
		return 0;

	model = bpf_map_lookup_elem(&model_buffer_map, &zero);
	if (!model || !model->base_addr || !model->size)
		return 0;

	ctx.va_space = req->va_space;
	ctx.base_addr = model->base_addr;
	ctx.size = model->size;

	bpf_loop(BITMAP_BITS, prefetch_bitmap_block_cb, &ctx, 0);
	return 0;
}

static __always_inline void reset_tracking_state(void)
{
	u32 zero = 0;
	u64 cleared_va_space = 0;
	u64 *first_out_of_range;
	u64 *last_sync_ns;
	struct pending_prefetch_state *pending;
	struct token_stats *stats;

	bpf_loop(BITMAP_WORDS, reset_bitmaps_cb, &zero, 0);
	bpf_loop(DBG_MAX_STATS, reset_debug_stats_cb, &zero, 0);
	bpf_map_update_elem(&va_space_map, &zero, &cleared_va_space, BPF_ANY);

	pending = bpf_map_lookup_elem(&pending_prefetch_map, &zero);
	if (pending) {
		pending->generation = 0;
		pending->drained_generation = 0;
	}

	last_sync_ns = bpf_map_lookup_elem(&last_sync_ns_map, &zero);
	if (last_sync_ns)
		*last_sync_ns = 0;

	first_out_of_range = bpf_map_lookup_elem(&first_block_start, &zero);
	if (first_out_of_range)
		*first_out_of_range = 0;

	stats = bpf_map_lookup_elem(&token_stats_map, &zero);
	if (stats) {
		stats->sync_count = 0;
		stats->prefetch_count = 0;
		stats->faults_recorded = 0;
		stats->prefetch_blocks = 0;
	}
}

static __always_inline void try_schedule_prefetch(void)
{
	u32 zero = 0;
	u64 *va_space;
	struct pending_prefetch_state *pending;
	struct prefetch_work_item *req;
	struct token_stats *stats;

	pending = bpf_map_lookup_elem(&pending_prefetch_map, &zero);
	if (!pending || pending->generation == pending->drained_generation)
		return;

	va_space = bpf_map_lookup_elem(&va_space_map, &zero);
	if (!va_space || !*va_space)
		return;

	req = bpf_map_lookup_elem(&wq_map, &zero);
	if (!req)
		return;

	req->va_space = *va_space;
	req->generation = pending->generation;

	bpf_wq_init(&req->work, &wq_map, 0);
	bpf_wq_set_callback(&req->work, do_bitmap_prefetch, 0);
	bpf_wq_start(&req->work, 0);

	pending->drained_generation = pending->generation;

	stats = bpf_map_lookup_elem(&token_stats_map, &zero);
	if (stats)
		__sync_fetch_and_add(&stats->prefetch_count, 1);
}

static __always_inline void record_fault_bitmap(uvm_gpu_chunk_t *chunk)
{
	u32 zero = 0;
	struct model_buffer_info *model;
	struct token_stats *stats;
	uvm_va_block_t *va_block;
	u64 block_start;
	u64 model_end;
	u64 offset;
	u32 block_idx;
	u32 word;
	u64 bit;
	u64 *bitmap_word;
	u64 old_bits;

	bump_debug_stat(DBG_RECORD_CALLED);

	model = bpf_map_lookup_elem(&model_buffer_map, &zero);
	if (!model) {
		bump_debug_stat(DBG_NO_MODEL);
		return;
	}

	if (!model->base_addr || !model->size) {
		bump_debug_stat(DBG_NO_MODEL_ADDR);
		return;
	}

	model_end = model->base_addr + model->size;
	if (model_end <= model->base_addr) {
		bump_debug_stat(DBG_OVERFLOW);
		return;
	}

	va_block = BPF_CORE_READ(chunk, va_block);
	if (!va_block) {
		bump_debug_stat(DBG_NO_VA_BLOCK);
		return;
	}

	block_start = BPF_CORE_READ(va_block, start);
	if (!block_start) {
		bump_debug_stat(DBG_NO_BLOCK_START);
		return;
	}

	if (block_start < model->base_addr || block_start >= model_end) {
		bump_debug_stat(DBG_OUT_OF_RANGE);
		remember_first_block_start(block_start);
		return;
	}

	offset = block_start - model->base_addr;
	block_idx = offset / VA_BLOCK_SIZE;
	word = block_idx / 64;
	if (word >= BITMAP_WORDS) {
		bump_debug_stat(DBG_OUT_OF_RANGE);
		remember_first_block_start(block_start);
		return;
	}

	bit = 1ULL << (block_idx % 64);

	bitmap_word = bpf_map_lookup_elem(&fault_bitmap_current, &word);
	if (!bitmap_word)
		return;

	old_bits = __sync_fetch_and_or(bitmap_word, bit);
	if (old_bits & bit) {
		bump_debug_stat(DBG_BITMAP_DUP);
		return;
	}

	bump_debug_stat(DBG_BITMAP_SET);

	stats = bpf_map_lookup_elem(&token_stats_map, &zero);
	if (stats)
		__sync_fetch_and_add(&stats->faults_recorded, 1);
}

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_space, uvm_va_block_t *va_block)
{
	u32 zero = 0;
	u32 *target_tgid;
	u32 owner_tgid = 0;
	u64 va_space_val = 0;

	if (!va_block)
		return 0;

	target_tgid = bpf_map_lookup_elem(&target_tgid_map, &zero);
	if (!target_tgid || *target_tgid == 0)
		return 0;

	uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
	if (!managed)
		return 0;

	uvm_va_space_t *va_space = BPF_CORE_READ(managed, va_range.va_space);
	if (!va_space)
		return 0;

	struct mm_struct *mm = BPF_CORE_READ(va_space, va_space_mm.mm);
	if (mm) {
		struct task_struct *owner = BPF_CORE_READ(mm, owner);
		if (owner)
			owner_tgid = BPF_CORE_READ(owner, tgid);
	}

	if (owner_tgid != *target_tgid)
		return 0;

	bpf_probe_read_kernel(&va_space_val, sizeof(va_space_val), &va_space);
	if (va_space_val)
		bpf_map_update_elem(&va_space_map, &zero, &va_space_val, BPF_ANY);

	return 0;
}

SEC("uprobe")
int BPF_UPROBE(cuda_malloc_managed_enter, void **dev_ptr, u64 size, u32 flags)
{
	u32 zero = 0;
	u32 current_tgid;
	u32 *target_tgid;
	u64 pid_tgid;
	u64 dev_ptr_addr = 0;
	struct alloc_args args = {};

	if (!comm_matches_target())
		return 0;

	if (size < min_alloc_size)
		return 0;

	current_tgid = bpf_get_current_pid_tgid() >> 32;
	target_tgid = bpf_map_lookup_elem(&target_tgid_map, &zero);
	if (target_tgid && *target_tgid && *target_tgid != current_tgid)
		return 0;

	bpf_probe_read_kernel(&dev_ptr_addr, sizeof(dev_ptr_addr), &dev_ptr);
	if (!dev_ptr_addr)
		return 0;

	pid_tgid = bpf_get_current_pid_tgid();
	args.size = size;
	args.dev_ptr_addr = dev_ptr_addr;
	bpf_map_update_elem(&alloc_args_map, &pid_tgid, &args, BPF_ANY);

	return 0;
}

SEC("uretprobe")
int BPF_URETPROBE(cuda_malloc_managed_ret, int ret)
{
	u32 zero = 0;
	u32 current_tgid = bpf_get_current_pid_tgid() >> 32;
	u32 *target_tgid;
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u64 model_base = 0;
	struct alloc_args *args;
	struct model_buffer_info *model;

	args = bpf_map_lookup_elem(&alloc_args_map, &pid_tgid);
	if (!args)
		return 0;

	if (ret != 0 || args->size < min_alloc_size || !args->dev_ptr_addr)
		goto out;

	target_tgid = bpf_map_lookup_elem(&target_tgid_map, &zero);
	if (target_tgid && *target_tgid && *target_tgid != current_tgid)
		goto out;

	if (bpf_probe_read_user(&model_base, sizeof(model_base),
				(void *)(unsigned long)args->dev_ptr_addr) != 0)
		goto out;
	if (!model_base)
		goto out;

	model = bpf_map_lookup_elem(&model_buffer_map, &zero);
	if (!model)
		goto out;

	if (model->tgid && model->tgid != current_tgid && model->base_addr)
		goto out;

	if (!model->base_addr || args->size >= model->size) {
		model->base_addr = model_base;
		model->size = args->size;
		model->tgid = current_tgid;
		model->reserved = 0;

		if (target_tgid)
			*target_tgid = current_tgid;

		reset_tracking_state();
	}

out:
	bpf_map_delete_elem(&alloc_args_map, &pid_tgid);
	return 0;
}

SEC("uprobe")
int BPF_UPROBE(cuda_sync_token_boundary, void *stream)
{
	u32 zero = 0;
	u32 current_tgid;
	u32 *target_tgid;
	u64 now_ns;
	u64 *last_sync_ns;
	struct token_stats *stats;
	struct pending_prefetch_state *pending;
	struct bitmap_swap_ctx swap = {};

	(void)stream;

	if (!comm_matches_target())
		return 0;

	current_tgid = bpf_get_current_pid_tgid() >> 32;
	target_tgid = bpf_map_lookup_elem(&target_tgid_map, &zero);
	if (!target_tgid || *target_tgid == 0 || *target_tgid != current_tgid)
		return 0;

	last_sync_ns = bpf_map_lookup_elem(&last_sync_ns_map, &zero);
	if (!last_sync_ns)
		return 0;

	now_ns = bpf_ktime_get_ns();
	if (*last_sync_ns && now_ns - *last_sync_ns < min_sync_gap_ns)
		return 0;

	*last_sync_ns = now_ns;

	stats = bpf_map_lookup_elem(&token_stats_map, &zero);
	if (stats)
		__sync_fetch_and_add(&stats->sync_count, 1);

	bpf_loop(BITMAP_WORDS, swap_bitmap_cb, &swap, 0);
	if (!swap.has_bits)
		return 0;

	pending = bpf_map_lookup_elem(&pending_prefetch_map, &zero);
	if (!pending)
		return 0;

	__sync_fetch_and_add(&pending->generation, 1);
	return 0;
}

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
	try_schedule_prefetch();
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
	u32 idx = chunk_hash(chunk);
	u8 *count;

	record_fault_bitmap(chunk);

	count = bpf_map_lookup_elem(&access_counts, &idx);
	if (!count)
		return 0;

	if (*count < 255)
		(*count)++;

	if (*count >= T1_FREQ_THRESHOLD) {
		bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
		return 1;
	}

	return 0;
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
struct gpu_mem_ops uvm_ops_moe_expert = {
	.gpu_test_trigger = (void *)gpu_test_trigger,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
