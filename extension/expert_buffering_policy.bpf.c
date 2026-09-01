/* SPDX-License-Identifier: GPL-2.0 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

#define EXPERT_BLOCK_BYTES (2ULL * 1024ULL * 1024ULL)
#define EXPERT_MAX_LAYOUT_BLOCKS 65536

enum expert_block_class {
	EXPERT_BLOCK_DEFAULT = 0,
	EXPERT_BLOCK_COLD = 1,
	EXPERT_BLOCK_HOT = 2,
	EXPERT_BLOCK_SHARED = 3,
};

enum expert_policy_mode {
	EXPERT_POLICY_PAGE_LIFO = 1,
	EXPERT_POLICY_HOT_LIFO = 2,
};

enum expert_policy_stat {
	EXPERT_STAT_ACTIVATE = 0,
	EXPERT_STAT_MAPPED,
	EXPERT_STAT_HOT_TAIL,
	EXPERT_STAT_COLD_HEAD,
	EXPERT_STAT_SHARED_TAIL,
	EXPERT_STAT_DEFAULT,
	EXPERT_STAT_SETTER_FAILURE,
	EXPERT_STAT_ACCESS,
	EXPERT_STAT_MAX,
};

struct expert_layout_control {
	u64 base;
	u32 blocks;
	u32 mode;
	u32 ready;
	u32 reserved;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, EXPERT_MAX_LAYOUT_BLOCKS);
	__type(key, u32);
	__type(value, u8);
} block_classes SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct expert_layout_control);
} layout_control SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, EXPERT_STAT_MAX);
	__type(key, u32);
	__type(value, u64);
} policy_stats SEC(".maps");

static __always_inline void count_stat(u32 key)
{
	u64 *value = bpf_map_lookup_elem(&policy_stats, &key);

	if (value)
		(*value)++;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
	     uvm_page_index_t page_index,
	     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	     uvm_va_block_region_t *max_prefetch_region,
	     uvm_bpf_prefetch_decision_t *decision_ctx)
{
	return 0;
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
	u32 zero = 0;
	struct expert_layout_control *control;
	uvm_va_block_t *va_block;
	u64 block_start;
	u64 offset;
	u32 index;
	u8 *class;
	u64 position;
	u32 stat;
	int ret;

	(void)pmm;
	count_stat(EXPERT_STAT_ACTIVATE);
	control = bpf_map_lookup_elem(&layout_control, &zero);
	if (!control || !control->ready || !control->blocks) {
		count_stat(EXPERT_STAT_DEFAULT);
		return 0;
	}

	va_block = BPF_CORE_READ(chunk, va_block);
	if (!va_block) {
		count_stat(EXPERT_STAT_DEFAULT);
		return 0;
	}
	block_start = BPF_CORE_READ(va_block, start);
	if (block_start < control->base) {
		count_stat(EXPERT_STAT_DEFAULT);
		return 0;
	}
	offset = block_start - control->base;
	index = offset / EXPERT_BLOCK_BYTES;
	if (index >= control->blocks) {
		count_stat(EXPERT_STAT_DEFAULT);
		return 0;
	}

	class = bpf_map_lookup_elem(&block_classes, &index);
	if (!class || *class == EXPERT_BLOCK_DEFAULT) {
		count_stat(EXPERT_STAT_DEFAULT);
		return 0;
	}
	count_stat(EXPERT_STAT_MAPPED);

	if (*class == EXPERT_BLOCK_SHARED) {
		position = NV_GPU_PMM_POSITION_TAIL;
		stat = EXPERT_STAT_SHARED_TAIL;
	} else if (*class == EXPERT_BLOCK_HOT &&
		   control->mode == EXPERT_POLICY_HOT_LIFO) {
		position = NV_GPU_PMM_POSITION_TAIL;
		stat = EXPERT_STAT_HOT_TAIL;
	} else {
		position = NV_GPU_PMM_POSITION_HEAD;
		stat = EXPERT_STAT_COLD_HEAD;
	}

	ret = bpf_gpu_request_reorder(decision_ctx,
				      NV_GPU_PMM_DESTINATION_USED,
				      position);
	if (ret)
		count_stat(EXPERT_STAT_SETTER_FAILURE);
	else
		count_stat(stat);
	return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
	     uvm_pmm_gpu_t *pmm,
	     uvm_gpu_chunk_t *chunk,
	     uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
	(void)pmm;
	(void)chunk;
	(void)decision_ctx;
	count_stat(EXPERT_STAT_ACCESS);
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
struct gpu_mem_ops uvm_ops_expert_buffering = {
	.gpu_test_trigger = (void *)gpu_test_trigger,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
