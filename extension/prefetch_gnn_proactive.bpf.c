/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Transparent GNN proactive prefetch
 *
 * Tracks the large UVM feature tensor via uvm_malloc() without modifying the
 * workload, then uses cudaDeviceSynchronize() as an epoch-boundary marker to
 * proactively migrate the first few 2MB VA blocks.
 *
 * Policy:
 *   - uprobe/uretprobe on uvm_malloc(): capture the large feature tensor
 *   - sleepable uprobe on cudaDeviceSynchronize(): direct proactive migrate
 *   - kprobe on uvm_perf_prefetch_get_hint_va_block(): capture va_space
 *   - struct_ops gpu_page_prefetch(): always_max + direction-aware XB
 *   - struct_ops gpu_block_access(): cycle_moe eviction
 *
 * If cudaDeviceSynchronize() fires before va_space is known, a pending request
 * is recorded and drained asynchronously from struct_ops on the next fault.
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
#define MAX_PREFETCH_BLOCKS 8
#define XB_WQ_SLOTS 64

const volatile __u32 prefetch_blocks = MAX_PREFETCH_BLOCKS;
const volatile __u64 min_alloc_size = 4000000000ULL;

struct feature_tensor_info {
	u64 addr;
	u64 size;
	u32 tgid;
	u32 reserved;
};

struct alloc_args {
	u64 size;
};

struct va_block_ctx {
	u64 va_start;
	u64 va_end;
	u64 va_space;
};

struct direction_ctx {
	u64 block_hist[3];
};

struct proactive_pending {
	u64 generation;
	u64 drained_generation;
	u32 requested_blocks;
	u32 reserved;
};

struct sync_state {
	u64 sync_calls;
	u64 direct_prefetches;
};

struct prefetch_work_item {
	u64 va_space;
	u64 addr;
	u64 length;
	struct bpf_wq work;
};

enum {
	STAT_KPROBE = 0,
	STAT_ALLOC_ENTER,
	STAT_ALLOC_TRACK,
	STAT_SYNC_HIT,
	STAT_SYNC_DIRECT,
	STAT_SYNC_DIRECT_FAIL,
	STAT_SYNC_NO_FEATURE,
	STAT_SYNC_NO_VA_SPACE,
	STAT_PENDING_SET,
	STAT_PENDING_DRAIN,
	STAT_PENDING_WQ_SCHED,
	STAT_WQ_CALLBACK,
	STAT_WQ_MIGRATE_OK,
	STAT_WQ_MIGRATE_FAIL,
	STAT_PREFETCH_HOOK,
	STAT_XB_WQ_SCHED,
	STAT_XB_DIR_SKIP,
	STAT_XB_DEDUP_SKIP,
	STAT_XB_RATELIMIT,
	STAT_EVICT_T1,
	NUM_STATS,
};

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
	__type(value, struct feature_tensor_info);
} feature_map SEC(".maps");

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
	__type(value, struct proactive_pending);
} pending_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct sync_state);
} sync_state_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct va_block_ctx);
} va_block_cache SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct direction_ctx);
} direction_cache SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} last_prefetch_target SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAX_PREFETCH_BLOCKS);
	__type(key, int);
	__type(value, struct prefetch_work_item);
} proactive_wq_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, XB_WQ_SLOTS);
	__type(key, int);
	__type(value, struct prefetch_work_item);
} xb_wq_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, COUNTER_SLOTS);
	__type(key, u32);
	__type(value, u8);
} access_counts SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, NUM_STATS);
	__type(key, u32);
	__type(value, u64);
} gnn_stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
	u64 *val = bpf_map_lookup_elem(&gnn_stats, &key);

	if (val)
		__sync_fetch_and_add(val, 1);
}

static __always_inline u32 clamp_prefetch_blocks(void)
{
	u32 blocks = prefetch_blocks;

	if (blocks == 0)
		return 1;
	if (blocks > MAX_PREFETCH_BLOCKS)
		return MAX_PREFETCH_BLOCKS;
	return blocks;
}

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
	u64 ptr = 0;

	bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
	return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

static __always_inline int is_target_process(struct feature_tensor_info *feature)
{
	u32 current_tgid = bpf_get_current_pid_tgid() >> 32;

	return feature && feature->tgid && feature->tgid == current_tgid;
}

static __always_inline u64 proactive_prefix_len(u64 size, u32 blocks)
{
	u64 wanted = (u64)blocks * VA_BLOCK_SIZE;

	return size < wanted ? size : wanted;
}

static int do_prefetch_cb(void *map, int *key, void *value)
{
	struct prefetch_work_item *req = value;

	stat_inc(STAT_WQ_CALLBACK);

	if (!req || !req->va_space || !req->length)
		return 0;

	if (bpf_gpu_migrate_range(req->va_space, req->addr, req->length) == 0)
		stat_inc(STAT_WQ_MIGRATE_OK);
	else
		stat_inc(STAT_WQ_MIGRATE_FAIL);

	return 0;
}

static __always_inline void schedule_pending_prefetch(u64 va_space,
						      const struct feature_tensor_info *feature,
						      u32 blocks)
{
	int i;

#pragma unroll
	for (i = 0; i < MAX_PREFETCH_BLOCKS; i++) {
		struct prefetch_work_item *req;
		u64 offset;
		u64 length;
		int key = i;

		if (i >= blocks)
			break;

		offset = (u64)i * VA_BLOCK_SIZE;
		if (offset >= feature->size)
			break;

		req = bpf_map_lookup_elem(&proactive_wq_map, &key);
		if (!req)
			break;

		length = feature->size - offset;
		if (length > VA_BLOCK_SIZE)
			length = VA_BLOCK_SIZE;

		req->va_space = va_space;
		req->addr = feature->addr + offset;
		req->length = length;

		stat_inc(STAT_PENDING_WQ_SCHED);
		bpf_wq_init(&req->work, &proactive_wq_map, 0);
		bpf_wq_set_callback(&req->work, do_prefetch_cb, 0);
		bpf_wq_start(&req->work, 0);
	}
}

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block, uvm_va_block_t *va_block)
{
	u32 zero = 0;
	struct va_block_ctx *info;

	stat_inc(STAT_KPROBE);

	info = bpf_map_lookup_elem(&va_block_cache, &zero);
	if (!info)
		return 0;

	if (!va_block) {
		info->va_start = 0;
		info->va_end = 0;
		info->va_space = 0;
		return 0;
	}

	info->va_start = BPF_CORE_READ(va_block, start);
	info->va_end = BPF_CORE_READ(va_block, end);
	info->va_space = 0;

	uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
	if (managed) {
		uvm_va_space_t *vs = BPF_CORE_READ(managed, va_range.va_space);
		u64 vs_val = 0;

		bpf_probe_read_kernel(&vs_val, sizeof(vs_val), &vs);
		info->va_space = vs_val;
		if (vs_val)
			bpf_map_update_elem(&va_space_map, &zero, &vs_val, BPF_ANY);
	}

	return 0;
}

SEC("uprobe")
int BPF_UPROBE(uvm_malloc_enter, u64 size, int device, void *stream)
{
	u64 pid_tgid;
	struct alloc_args args = {};

	if (size < min_alloc_size)
		return 0;

	stat_inc(STAT_ALLOC_ENTER);

	pid_tgid = bpf_get_current_pid_tgid();
	args.size = size;
	bpf_map_update_elem(&alloc_args_map, &pid_tgid, &args, BPF_ANY);
	return 0;
}

SEC("uretprobe")
int BPF_URETPROBE(uvm_malloc_ret, void *ret)
{
	u32 zero = 0;
	u32 current_tgid = bpf_get_current_pid_tgid() >> 32;
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u64 cleared = 0;
	struct alloc_args *args;
	struct feature_tensor_info *feature;
	struct proactive_pending *pending;
	struct sync_state *sync_state;

	args = bpf_map_lookup_elem(&alloc_args_map, &pid_tgid);
	if (!args)
		return 0;

	if (!ret || args->size < min_alloc_size) {
		bpf_map_delete_elem(&alloc_args_map, &pid_tgid);
		return 0;
	}

	feature = bpf_map_lookup_elem(&feature_map, &zero);
	if (!feature) {
		bpf_map_delete_elem(&alloc_args_map, &pid_tgid);
		return 0;
	}

	if (feature->tgid && feature->tgid != current_tgid) {
		bpf_map_delete_elem(&alloc_args_map, &pid_tgid);
		return 0;
	}

	if (!feature->addr || args->size >= feature->size) {
		feature->addr = (u64)ret;
		feature->size = args->size;
		feature->tgid = current_tgid;
		feature->reserved = 0;
		stat_inc(STAT_ALLOC_TRACK);

		bpf_map_update_elem(&va_space_map, &zero, &cleared, BPF_ANY);

		pending = bpf_map_lookup_elem(&pending_map, &zero);
		if (pending) {
			pending->generation = 0;
			pending->drained_generation = 0;
			pending->requested_blocks = clamp_prefetch_blocks();
		}

		sync_state = bpf_map_lookup_elem(&sync_state_map, &zero);
		if (sync_state) {
			sync_state->sync_calls = 0;
			sync_state->direct_prefetches = 0;
		}
	}

	bpf_map_delete_elem(&alloc_args_map, &pid_tgid);
	return 0;
}

SEC("uprobe.s")
int BPF_UPROBE(cuda_sync_epoch_boundary)
{
	u32 zero = 0;
	u32 blocks;
	u64 *va_space;
	u64 length;
	struct feature_tensor_info *feature;
	struct proactive_pending *pending;
	struct sync_state *sync_state;

	stat_inc(STAT_SYNC_HIT);

	feature = bpf_map_lookup_elem(&feature_map, &zero);
	if (!feature || !feature->addr || !is_target_process(feature)) {
		stat_inc(STAT_SYNC_NO_FEATURE);
		return 0;
	}

	sync_state = bpf_map_lookup_elem(&sync_state_map, &zero);
	if (sync_state)
		__sync_fetch_and_add(&sync_state->sync_calls, 1);

	blocks = clamp_prefetch_blocks();
	length = proactive_prefix_len(feature->size, blocks);
	if (!length)
		return 0;

	va_space = bpf_map_lookup_elem(&va_space_map, &zero);
	if (va_space && *va_space) {
		if (bpf_gpu_migrate_range(*va_space, feature->addr, length) == 0) {
			stat_inc(STAT_SYNC_DIRECT);
			if (sync_state)
				__sync_fetch_and_add(&sync_state->direct_prefetches, 1);
		} else {
			stat_inc(STAT_SYNC_DIRECT_FAIL);
		}
		return 0;
	}

	stat_inc(STAT_SYNC_NO_VA_SPACE);

	pending = bpf_map_lookup_elem(&pending_map, &zero);
	if (!pending)
		return 0;

	pending->requested_blocks = blocks;
	__sync_fetch_and_add(&pending->generation, 1);
	stat_inc(STAT_PENDING_SET);
	return 0;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
	     uvm_page_index_t page_index,
	     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	     uvm_va_block_region_t *max_prefetch_region,
	     uvm_bpf_prefetch_decision_t *decision_ctx)
{
	u32 zero = 0;
	u64 prefetch_target = 0;
	struct feature_tensor_info *feature;
	struct proactive_pending *pending;
	struct va_block_ctx *blk;
	struct direction_ctx *dir;
	uvm_page_index_t max_first;
	uvm_page_index_t max_outer;

	stat_inc(STAT_PREFETCH_HOOK);

	/* always_max: always use full max_prefetch_region */
	max_first = BPF_CORE_READ(max_prefetch_region, first);
	max_outer = BPF_CORE_READ(max_prefetch_region, outer);
	bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);

	/* Drain pending proactive prefetch if any */
	feature = bpf_map_lookup_elem(&feature_map, &zero);
	pending = bpf_map_lookup_elem(&pending_map, &zero);
	if (feature && pending &&
	    pending->generation != pending->drained_generation &&
	    feature->addr && feature->size) {
		u64 *va_space = bpf_map_lookup_elem(&va_space_map, &zero);
		u32 blocks = pending->requested_blocks;

		if (va_space && *va_space) {
			if (blocks == 0 || blocks > clamp_prefetch_blocks())
				blocks = clamp_prefetch_blocks();
			schedule_pending_prefetch(*va_space, feature, blocks);
			pending->drained_generation = pending->generation;
			stat_inc(STAT_PENDING_DRAIN);
		}
	}

	blk = bpf_map_lookup_elem(&va_block_cache, &zero);
	if (!blk || !blk->va_space || !blk->va_end)
		return 1;

	dir = bpf_map_lookup_elem(&direction_cache, &zero);
	if (!dir)
		return 1;

	u64 block_end = blk->va_end;
	u64 h1 = dir->block_hist[1];
	u64 h2 = dir->block_hist[2];

	if (block_end != h2) {
		dir->block_hist[0] = h1;
		dir->block_hist[1] = h2;
		dir->block_hist[2] = block_end;
	} else {
		stat_inc(STAT_XB_RATELIMIT);
		return 1;
	}

	if (h1 == 0 || h2 == 0) {
		stat_inc(STAT_XB_DIR_SKIP);
		return 1;
	}

	s64 d1 = (s64)(h2 - h1);
	s64 d2 = (s64)(block_end - h2);

	if ((d1 > 0 && d2 > 0) || (d1 < 0 && d2 < 0))
		prefetch_target = block_end + (d2 > 0 ? 1 : -(s64)VA_BLOCK_SIZE);
	else {
		stat_inc(STAT_XB_DIR_SKIP);
		return 1;
	}

	u64 *last_target = bpf_map_lookup_elem(&last_prefetch_target, &zero);
	if (last_target) {
		if (*last_target == prefetch_target) {
			stat_inc(STAT_XB_DEDUP_SKIP);
			return 1;
		}
		*last_target = prefetch_target;
	}

	int wq_key = bpf_get_smp_processor_id() % XB_WQ_SLOTS;
	struct prefetch_work_item *req = bpf_map_lookup_elem(&xb_wq_map, &wq_key);
	if (req) {
		req->va_space = blk->va_space;
		req->addr = prefetch_target;
		req->length = VA_BLOCK_SIZE;

		stat_inc(STAT_XB_WQ_SCHED);
		bpf_wq_init(&req->work, &xb_wq_map, 0);
		bpf_wq_set_callback(&req->work, do_prefetch_cb, 0);
		bpf_wq_start(&req->work, 0);
	}

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
	     struct list_head *list)
{
	u32 idx = chunk_hash(chunk);
	u8 *count = bpf_map_lookup_elem(&access_counts, &idx);

	if (!count)
		return 0;

	u8 c = *count;
	if (c < 255)
		*count = c + 1;

	if (c + 1 >= T1_FREQ_THRESHOLD) {
		bpf_gpu_block_move_tail(chunk, list);
		stat_inc(STAT_EVICT_T1);
		return 1;
	}

	return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
	     uvm_pmm_gpu_t *pmm,
	     uvm_gpu_chunk_t *chunk,
	     struct list_head *list)
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
struct gpu_mem_ops uvm_ops_gnn_proactive = {
	.gpu_test_trigger = (void *)gpu_test_trigger,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
