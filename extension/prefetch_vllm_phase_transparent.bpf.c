/* SPDX-License-Identifier: GPL-2.0 */
/*
 * vLLM Transparent Uprobe Phase Detection: Prefill vs Decode
 *
 * Hooks paged_attention_v1()/paged_attention_v2() in vLLM's _C.abi3.so
 * without requiring any vLLM source changes.
 *
 * PREFILL: always_max + direction-aware cross-block prefetch
 * DECODE:  always_max only (no cross-block)
 * Eviction: cycle_moe (T1 protect + DEFAULT non-T1)
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
#define DEFAULT_PREFETCH_LEN VA_BLOCK_SIZE
#define PREFILL_TIMEOUT_FAULTS 500

#define PHASE_PREFILL 1
#define PHASE_DECODE  2

/* ===== Shared phase state ===== */

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u32);
} phase_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} faults_since_decode SEC(".maps");

/* ===== Per-CPU va_block context ===== */

struct va_block_ctx {
	u64 va_start;
	u64 va_end;
	u64 va_space;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct va_block_ctx);
} va_block_cache SEC(".maps");

/* ===== Direction tracking ===== */

struct direction_ctx {
	u64 block_hist[3];
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct direction_ctx);
} direction_cache SEC(".maps");

/* ===== T1 eviction counters ===== */

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, COUNTER_SLOTS);
	__type(key, u32);
	__type(value, u8);
} access_counts SEC(".maps");

/* ===== bpf_wq cross-block prefetch ===== */

struct prefetch_data {
	u64 va_space;
	u64 addr;
	u64 length;
	struct bpf_wq work;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 64);
	__type(key, int);
	__type(value, struct prefetch_data);
} wq_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} last_prefetch_target SEC(".maps");

/* ===== Stats ===== */

#define STAT_KPROBE                 0
#define STAT_PREFETCH               1
#define STAT_WQ_SCHED               2
#define STAT_WQ_CALLBACK            3
#define STAT_MIGRATE_OK             4
#define STAT_MIGRATE_FAIL           5
#define STAT_RATELIMIT              6
#define STAT_WQ_SKIP                7
#define STAT_DIR_SKIP               8
#define STAT_DEDUP_SKIP             9
#define STAT_PHASE_DECODE_V1        10
#define STAT_PHASE_DECODE_V2        11
#define STAT_PHASE_TIMEOUT_PREFILL  12
#define STAT_PREFILL_XB             13
#define STAT_DECODE_SKIP            14
#define STAT_DECODE_FAULT           15
#define NUM_STATS                   16

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, NUM_STATS);
	__type(key, u32);
	__type(value, u64);
} vtp_stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
	u64 *val = bpf_map_lookup_elem(&vtp_stats, &key);
	if (val)
		__sync_fetch_and_add(val, 1);
}

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
	u64 ptr = 0;
	bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
	return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

static __always_inline void set_decode_phase(u32 stat_key)
{
	u32 zero = 0;
	u32 *phase = bpf_map_lookup_elem(&phase_map, &zero);
	u64 *faults = bpf_map_lookup_elem(&faults_since_decode, &zero);

	if (phase)
		*phase = PHASE_DECODE;
	if (faults)
		*faults = 0;

	stat_inc(stat_key);
}

/* ===== UPROBES: decode marker ===== */

SEC("uprobe")
int BPF_UPROBE(paged_attention_v1_decode)
{
	set_decode_phase(STAT_PHASE_DECODE_V1);
	return 0;
}

SEC("uprobe")
int BPF_UPROBE(paged_attention_v2_decode)
{
	set_decode_phase(STAT_PHASE_DECODE_V2);
	return 0;
}

/* ===== kprobe: capture va_block context + timeout back to PREFILL ===== */

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block, uvm_va_block_t *va_block)
{
	stat_inc(STAT_KPROBE);

	u32 zero = 0;
	struct va_block_ctx *info = bpf_map_lookup_elem(&va_block_cache, &zero);
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

	u32 *phase = bpf_map_lookup_elem(&phase_map, &zero);
	u64 *faults = bpf_map_lookup_elem(&faults_since_decode, &zero);
	u32 cur_phase = PHASE_PREFILL;

	if (phase && *phase == PHASE_DECODE)
		cur_phase = PHASE_DECODE;

	if (cur_phase == PHASE_DECODE && faults) {
		u64 new_faults = __sync_fetch_and_add(faults, 1) + 1;

		stat_inc(STAT_DECODE_FAULT);

		if (new_faults > PREFILL_TIMEOUT_FAULTS) {
			if (phase)
				*phase = PHASE_PREFILL;
			*faults = 0;
			cur_phase = PHASE_PREFILL;
			stat_inc(STAT_PHASE_TIMEOUT_PREFILL);
		}
	}

	if (cur_phase == PHASE_DECODE)
		return 0;

	uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
	if (managed) {
		uvm_va_space_t *vs = BPF_CORE_READ(managed, va_range.va_space);
		u64 vs_val = 0;
		bpf_probe_read_kernel(&vs_val, sizeof(vs_val), &vs);
		info->va_space = vs_val;
	}

	return 0;
}

/* ===== bpf_wq callback ===== */

static int do_prefetch(void *map, int *key, void *value)
{
	stat_inc(STAT_WQ_CALLBACK);

	struct prefetch_data *data = value;
	if (!data || !data->va_space || !data->length) {
		stat_inc(STAT_WQ_SKIP);
		return 0;
	}

	int ret = bpf_gpu_migrate_range(data->va_space, data->addr, data->length);
	if (ret == 0)
		stat_inc(STAT_MIGRATE_OK);
	else
		stat_inc(STAT_MIGRATE_FAIL);

	return 0;
}

/* ===== PREFETCH: always_max + phase-gated cross-block ===== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
	     uvm_page_index_t page_index,
	     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	     uvm_va_block_region_t *max_prefetch_region,
	     uvm_bpf_prefetch_decision_t *decision_ctx)
{
	stat_inc(STAT_PREFETCH);

	uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
	uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
	bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);

	u32 zero = 0;
	u32 *phase = bpf_map_lookup_elem(&phase_map, &zero);
	u32 cur_phase = PHASE_PREFILL;

	if (phase && *phase == PHASE_DECODE)
		cur_phase = PHASE_DECODE;

	if (cur_phase == PHASE_DECODE) {
		stat_inc(STAT_DECODE_SKIP);
		return 1;
	}

	struct va_block_ctx *blk = bpf_map_lookup_elem(&va_block_cache, &zero);
	if (!blk || !blk->va_space || !blk->va_end)
		return 1;

	u64 va_space = blk->va_space;
	u64 block_end = blk->va_end;

	struct direction_ctx *dir = bpf_map_lookup_elem(&direction_cache, &zero);
	if (!dir)
		return 1;

	u64 h1 = dir->block_hist[1];
	u64 h2 = dir->block_hist[2];

	if (block_end != h2) {
		dir->block_hist[0] = h1;
		dir->block_hist[1] = h2;
		dir->block_hist[2] = block_end;
	} else {
		stat_inc(STAT_RATELIMIT);
		return 1;
	}

	if (h1 == 0 || h2 == 0) {
		stat_inc(STAT_DIR_SKIP);
		return 1;
	}

	s64 d1 = (s64)(h2 - h1);
	s64 d2 = (s64)(block_end - h2);
	u64 prefetch_target = 0;

	if ((d1 > 0 && d2 > 0) || (d1 < 0 && d2 < 0))
		prefetch_target = block_end + (d2 > 0 ? 1 : -(s64)VA_BLOCK_SIZE);
	else {
		stat_inc(STAT_DIR_SKIP);
		return 1;
	}

	if (!prefetch_target)
		return 1;

	u64 *last_target = bpf_map_lookup_elem(&last_prefetch_target, &zero);
	if (last_target) {
		if (*last_target == prefetch_target) {
			stat_inc(STAT_DEDUP_SKIP);
			return 1;
		}
		*last_target = prefetch_target;
	}

	stat_inc(STAT_PREFILL_XB);

	int wq_key = bpf_get_smp_processor_id() % 64;
	struct prefetch_data *data = bpf_map_lookup_elem(&wq_map, &wq_key);
	if (data) {
		data->va_space = va_space;
		data->addr = prefetch_target;
		data->length = DEFAULT_PREFETCH_LEN;

		stat_inc(STAT_WQ_SCHED);
		bpf_wq_init(&data->work, &wq_map, 0);
		bpf_wq_set_callback(&data->work, do_prefetch, 0);
		bpf_wq_start(&data->work, 0);
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

/* ===== EVICTION: cycle_moe ===== */

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
struct gpu_mem_ops uvm_ops_vllm_phase_transparent = {
	.gpu_test_trigger = (void *)gpu_test_trigger,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
