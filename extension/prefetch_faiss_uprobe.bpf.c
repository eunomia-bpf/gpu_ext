/* SPDX-License-Identifier: GPL-2.0 */
/*
 * FAISS Uprobe-Based Phase Detection + Cross-Block Prefetch
 *
 * Uses uprobes on FAISS's GpuIndex::add_with_ids() and GpuIndex::search()
 * to detect BUILD vs SEARCH phases with zero per-fault overhead.
 * This eliminates the 25% nprobe=1 overhead of the sliding-window
 * phase detection in prefetch_faiss_phase.bpf.c.
 *
 * BUILD phase:  always_max intra-block + direction-aware cross-block (bpf_wq)
 * SEARCH phase: always_max intra-block only (no cross-block)
 * Eviction: default_lru (return 0 for all -- kernel LRU handles it)
 *           cycle_moe HURTS FAISS search, so we use pure LRU.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* VA block size = 2MB */
#define VA_BLOCK_SIZE (2ULL * 1024 * 1024)

/* Default cross-block prefetch length = 2MB */
#define DEFAULT_PREFETCH_LEN VA_BLOCK_SIZE

/* Phase constants */
#define PHASE_UNKNOWN 0
#define PHASE_BUILD   1
#define PHASE_SEARCH  2

/* ===== Phase map (set by uprobe, read by struct_ops) ===== */

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u32);
} phase_map SEC(".maps");

/* ===== Per-CPU context from kprobe ===== */

struct va_block_ctx {
	u64 va_start;
	u64 va_end;
	u64 va_space;   /* opaque handle for bpf_gpu_migrate_range() */
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct va_block_ctx);
} va_block_cache SEC(".maps");

/* ===== Direction tracking for cross-block ===== */

struct direction_ctx {
	u64 block_hist[3];    /* [0]=oldest, [1]=prev, [2]=current */
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct direction_ctx);
} direction_cache SEC(".maps");

/* ===== Cross-block prefetch request data + embedded bpf_wq ===== */

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

/* Global dedup: last prefetched target address */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} last_prefetch_target SEC(".maps");

/* ===== Stats counters ===== */
/*
 * key 0: kprobe fires
 * key 1: prefetch hook fires
 * key 2: wq scheduled
 * key 3: wq callback ran
 * key 4: migrate success (ret == 0)
 * key 5: migrate failed (ret != 0)
 * key 6: rate-limit skip (same block as last on this CPU)
 * key 7: wq callback skipped (invalid data)
 * key 8: direction skip (no consistent momentum)
 * key 9: dedup skip
 * key 10: uprobe -> BUILD
 * key 11: uprobe -> SEARCH
 * key 12: search_skip (XB skipped during SEARCH)
 * key 13: build_prefetch (XB scheduled during BUILD)
 */
#define STAT_KPROBE          0
#define STAT_PREFETCH        1
#define STAT_WQ_SCHED        2
#define STAT_WQ_CALLBACK     3
#define STAT_MIGRATE_OK      4
#define STAT_MIGRATE_FAIL    5
#define STAT_RATELIMIT       6
#define STAT_WQ_SKIP         7
#define STAT_DIR_SKIP        8
#define STAT_DEDUP_SKIP      9
#define STAT_PHASE_BUILD     10
#define STAT_PHASE_SEARCH    11
#define STAT_SEARCH_SKIP     12
#define STAT_BUILD_PREFETCH  13
#define NUM_STATS            14

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, NUM_STATS);
	__type(key, u32);
	__type(value, u64);
} fu_stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
	u64 *val = bpf_map_lookup_elem(&fu_stats, &key);
	if (val)
		__sync_fetch_and_add(val, 1);
}

/* ===== UPROBE: FAISS phase detection ===== */

SEC("uprobe")
int BPF_UPROBE(faiss_add_start)
{
	u32 key = 0;
	u32 phase = PHASE_BUILD;
	bpf_map_update_elem(&phase_map, &key, &phase, BPF_ANY);
	stat_inc(STAT_PHASE_BUILD);
	return 0;
}

SEC("uprobe")
int BPF_UPROBE(faiss_search_start)
{
	u32 key = 0;
	u32 phase = PHASE_SEARCH;
	bpf_map_update_elem(&phase_map, &key, &phase, BPF_ANY);
	stat_inc(STAT_PHASE_SEARCH);
	return 0;
}

/* ===== kprobe: capture va_block context ===== */

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block,
               uvm_va_block_t *va_block)
{
	stat_inc(STAT_KPROBE);

	u32 key = 0;
	struct va_block_ctx *info = bpf_map_lookup_elem(&va_block_cache, &key);
	if (!info)
		return 0;

	if (!va_block) {
		info->va_start = 0;
		info->va_end = 0;
		info->va_space = 0;
		return 0;
	}

	/* Always capture VA range */
	info->va_start = BPF_CORE_READ(va_block, start);
	info->va_end = BPF_CORE_READ(va_block, end);

	/* Only capture va_space during BUILD phase (expensive pointer chain).
	 * During SEARCH, cross-block is disabled so va_space is never used. */
	u32 zero = 0;
	u32 *phase = bpf_map_lookup_elem(&phase_map, &zero);
	if (phase && *phase == PHASE_SEARCH) {
		/* Skip expensive pointer chase -- va_space not needed */
		return 0;
	}

	/* Navigate: va_block -> managed_range -> va_range.va_space */
	uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
	if (managed) {
		uvm_va_space_t *vs = BPF_CORE_READ(managed, va_range.va_space);
		u64 vs_val = 0;
		bpf_probe_read_kernel(&vs_val, sizeof(vs_val), &vs);
		info->va_space = vs_val;
	} else {
		info->va_space = 0;
	}

	return 0;
}

/* ===== bpf_wq callback: process context, sleepable ===== */

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

	/* 1) Intra-block: always_max -- prefetch entire VA block */
	uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
	uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
	bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);

	/* 2) Phase-gated cross-block prefetch.
	 * During SEARCH: skip cross-block entirely (just always_max).
	 * During BUILD: direction-aware cross-block via bpf_wq. */
	u32 zero = 0;
	u32 *phase = bpf_map_lookup_elem(&phase_map, &zero);
	u32 cur_phase = phase ? *phase : PHASE_UNKNOWN;

	if (cur_phase != PHASE_BUILD) {
		stat_inc(STAT_SEARCH_SKIP);
		return 1; /* BYPASS (always_max only) */
	}

	/* BUILD phase: attempt cross-block prefetch */
	struct va_block_ctx *blk = bpf_map_lookup_elem(&va_block_cache, &zero);
	if (!blk || !blk->va_space || !blk->va_end)
		return 1;

	u64 va_space = blk->va_space;
	u64 block_end = blk->va_end;

	/* Per-CPU direction tracking */
	struct direction_ctx *dir = bpf_map_lookup_elem(&direction_cache, &zero);
	if (!dir)
		return 1;

	u64 h1 = dir->block_hist[1]; /* prev */
	u64 h2 = dir->block_hist[2]; /* current */

	/* Update history: shift window */
	if (block_end != h2) {
		dir->block_hist[0] = h1;
		dir->block_hist[1] = h2;
		dir->block_hist[2] = block_end;
	} else {
		/* Same block as last on this CPU -- rate limit */
		stat_inc(STAT_RATELIMIT);
		return 1;
	}

	/* Direction-aware: require 2 consecutive same-direction transitions */
	if (h1 == 0 || h2 == 0) {
		stat_inc(STAT_DIR_SKIP);
		return 1;
	}
	s64 d1 = (s64)(h2 - h1);
	s64 d2 = (s64)(block_end - h2);

	u64 prefetch_target = 0;
	if ((d1 > 0 && d2 > 0) || (d1 < 0 && d2 < 0)) {
		prefetch_target = block_end + (d2 > 0 ? 1 : -(s64)VA_BLOCK_SIZE);
	} else {
		stat_inc(STAT_DIR_SKIP);
		return 1;
	}

	if (!prefetch_target)
		return 1;

	/* Global dedup: prevent duplicate cross-CPU prefetch */
	u64 *last_target = bpf_map_lookup_elem(&last_prefetch_target, &zero);
	if (last_target) {
		if (*last_target == prefetch_target) {
			stat_inc(STAT_DEDUP_SKIP);
			return 1;
		}
		*last_target = prefetch_target;
	}

	stat_inc(STAT_BUILD_PREFETCH);

	/* Schedule cross-block prefetch via bpf_wq */
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

	return 1; /* BYPASS */
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

/* ===== EVICTION: default LRU (let kernel handle everything) ===== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
	return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
	/* Default LRU: return 0 (DEFAULT) for all, let kernel handle */
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
struct gpu_mem_ops uvm_ops_faiss_uprobe = {
	.gpu_test_trigger = (void *)gpu_test_trigger,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
