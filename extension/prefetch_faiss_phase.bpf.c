/* SPDX-License-Identifier: GPL-2.0 */
/*
 * FAISS Phase-Adaptive Cross-Block Prefetch v2
 *
 * Auto-detects BUILD (sequential K-means) vs SEARCH (random query) phases
 * using a sliding window of SEQUENTIAL STRIDE detection. A block transition
 * counts as "sequential" only if it is exactly +1 VA block (2MB forward).
 * Cross-block prefetch is only enabled during BUILD phase.
 *
 * v1 used direction consistency (forward vs backward) which failed because
 * FAISS search still has 62.5% forward consistency. v2 uses strict
 * adjacent-stride detection which clearly separates sequential scan from
 * random access.
 *
 * Eviction: cycle_moe (T1 protect + DEFAULT non-T1) hardcoded.
 * Intra-block: always_max (prefetch entire VA block).
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

/* VA block size = 2MB */
#define VA_BLOCK_SIZE (2ULL * 1024 * 1024)

/* Default cross-block prefetch length = 2MB */
#define DEFAULT_PREFETCH_LEN VA_BLOCK_SIZE

/* Sliding window size for phase detection */
#define WINDOW_SIZE 32
#define WINDOW_MASK (WINDOW_SIZE - 1)

/* Phase detection thresholds (out of WINDOW_SIZE=32)
 * v2: count of sequential (+1 VA block) transitions in window.
 * BUILD: >50% sequential -> sequential scan detected
 * SEARCH: <25% sequential -> random access detected */
#define BUILD_THRESHOLD  16  /* >50% sequential -> BUILD */
#define SEARCH_THRESHOLD  8  /* <25% sequential -> SEARCH */

/* Phase constants */
#define PHASE_SEARCH 0
#define PHASE_BUILD  1

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

/* ===== Phase detection state (global, shared across CPUs) ===== */

struct phase_state {
	u8  stride_window[WINDOW_SIZE]; /* circular buffer: 1=sequential(+1 block), 0=other */
	u32 window_idx;                 /* current write position */
	u32 seq_count;                  /* count of sequential (1) entries in window */
	u32 phase;                      /* PHASE_BUILD or PHASE_SEARCH */
	u64 last_block_end;             /* last seen block end VA */
	u64 phase_transitions;          /* count of phase changes */
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct phase_state);
} phase_map SEC(".maps");

/* ===== Maps ===== */

/* Per-CPU access frequency counters for eviction */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, COUNTER_SLOTS);
	__type(key, u32);
	__type(value, u8);
} access_counts SEC(".maps");

/* Cross-block prefetch request data + embedded bpf_wq.
 * Use max_entries=64 indexed by cpu_id%64 to avoid data corruption
 * when multiple CPUs schedule work simultaneously. */
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

/* ===== Stats counters ===== */
/*
 * key 0: kprobe fires
 * key 1: prefetch hook fires
 * key 2: wq scheduled
 * key 3: wq callback ran
 * key 4: migrate success (ret == 0)
 * key 5: migrate failed (ret != 0)
 * key 6: rate-limit skip (same block as last)
 * key 7: wq callback skipped (invalid data)
 * key 8: build prefetch (cross-block scheduled during BUILD)
 * key 9: search skip (cross-block skipped during SEARCH)
 * key 10: phase set to BUILD
 * key 11: phase set to SEARCH
 */
#define STAT_KPROBE          0
#define STAT_PREFETCH        1
#define STAT_WQ_SCHED        2
#define STAT_WQ_CALLBACK     3
#define STAT_MIGRATE_OK      4
#define STAT_MIGRATE_FAIL    5
#define STAT_RATELIMIT       6
#define STAT_WQ_SKIP         7
#define STAT_BUILD_PREFETCH  8
#define STAT_SEARCH_SKIP     9
#define STAT_PHASE_BUILD     10
#define STAT_PHASE_SEARCH    11
#define NUM_STATS            12

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, NUM_STATS);
	__type(key, u32);
	__type(value, u64);
} fp_stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
	u64 *val = bpf_map_lookup_elem(&fp_stats, &key);
	if (val)
		__sync_fetch_and_add(val, 1);
}

/* ===== Helpers ===== */

static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
	u64 ptr = 0;
	bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
	return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
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

	/* Always capture VA range (cheap, needed for phase detection) */
	info->va_start = BPF_CORE_READ(va_block, start);
	info->va_end = BPF_CORE_READ(va_block, end);

	/* Only capture va_space during BUILD phase (expensive pointer chain).
	 * During SEARCH, cross-block is disabled so va_space is never used. */
	u32 zero = 0;
	struct phase_state *ps = bpf_map_lookup_elem(&phase_map, &zero);
	if (ps && ps->phase == PHASE_SEARCH) {
		/* Skip expensive pointer chase — va_space not needed */
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

/* ===== PREFETCH: always_max + phase-adaptive cross-block ===== */

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

	/* 2) Phase-adaptive cross-block prefetch
	 *
	 * Phase detection uses only va_end (always captured by kprobe).
	 * va_space is only needed when scheduling cross-block prefetch,
	 * and is only captured during BUILD phase (kprobe optimization).
	 *
	 * BUG FIX: Previously checked va_space before phase detection,
	 * creating a deadlock: SEARCH phase -> kprobe skips va_space ->
	 * prefetch hook bails on !va_space -> phase detection never runs
	 * -> stuck in SEARCH forever. Now phase detection runs first,
	 * using only va_end.
	 */
	u32 zero = 0;
	struct va_block_ctx *blk = bpf_map_lookup_elem(&va_block_cache, &zero);
	if (!blk || !blk->va_end)
		return 1; /* BYPASS (always_max only) */

	u64 block_end = blk->va_end;

	/* Get phase state */
	struct phase_state *ps = bpf_map_lookup_elem(&phase_map, &zero);
	if (!ps)
		return 1;

	/* Check if this is a new block transition */
	if (block_end == ps->last_block_end) {
		/* Same block as last -- rate limit */
		stat_inc(STAT_RATELIMIT);
		return 1;
	}

	/* Compute stride: is this exactly +1 VA block from last? */
	u8 is_sequential;
	if (ps->last_block_end == 0) {
		/* First block seen, initialize and skip cross-block */
		ps->last_block_end = block_end;
		return 1;
	}

	/* Sequential = exactly +1 VA block forward (end - last_end == VA_BLOCK_SIZE) */
	u64 stride = block_end - ps->last_block_end;
	is_sequential = (stride == VA_BLOCK_SIZE) ? 1 : 0;

	/* Update sliding window */
	u32 idx = ps->window_idx & WINDOW_MASK;
	u8 old_val = ps->stride_window[idx];
	ps->stride_window[idx] = is_sequential;
	ps->window_idx = (ps->window_idx + 1) & WINDOW_MASK;

	/* Update seq_count incrementally */
	u32 new_seq = ps->seq_count;
	if (old_val)
		new_seq--;
	if (is_sequential)
		new_seq++;
	/* Clamp to [0, WINDOW_SIZE] for safety */
	if (new_seq > WINDOW_SIZE)
		new_seq = 0;
	ps->seq_count = new_seq;

	/* Phase detection with hysteresis */
	u32 old_phase = ps->phase;
	if (new_seq >= BUILD_THRESHOLD) {
		ps->phase = PHASE_BUILD;
		if (old_phase != PHASE_BUILD) {
			stat_inc(STAT_PHASE_BUILD);
			ps->phase_transitions++;
		}
	} else if (new_seq <= SEARCH_THRESHOLD) {
		ps->phase = PHASE_SEARCH;
		if (old_phase != PHASE_SEARCH) {
			stat_inc(STAT_PHASE_SEARCH);
			ps->phase_transitions++;
		}
	}
	/* else: keep current phase (hysteresis band) */

	ps->last_block_end = block_end;

	/* Cross-block only during BUILD phase */
	if (ps->phase != PHASE_BUILD) {
		stat_inc(STAT_SEARCH_SKIP);
		return 1; /* BYPASS */
	}

	stat_inc(STAT_BUILD_PREFETCH);

	/* va_space is only captured by kprobe during BUILD phase.
	 * If we just transitioned to BUILD, va_space may not be available
	 * for the first few calls until the kprobe fires again. */
	u64 va_space = blk->va_space;
	if (!va_space)
		return 1; /* va_space not yet captured, skip cross-block this time */

	/* Sequential stride detected — always prefetch next block forward */
	u64 prefetch_target = block_end + 1;

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
struct gpu_mem_ops uvm_ops_faiss_phase = {
	.gpu_test_trigger = (void *)gpu_test_trigger,
	.gpu_page_prefetch = (void *)gpu_page_prefetch,
	.gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
	.gpu_block_activate = (void *)gpu_block_activate,
	.gpu_block_access = (void *)gpu_block_access,
	.gpu_evict_prepare = (void *)gpu_evict_prepare,
};
