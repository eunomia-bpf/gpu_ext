/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Cross-VA-Block Prefetch v2: BPF Workqueue + Always-Max + Configurable Eviction
 *
 * Uses kprobe on uvm_perf_prefetch_get_hint_va_block to capture va_block
 * and va_space into a per-CPU map. The struct_ops hook reads this context
 * and schedules cross-block prefetch via bpf_wq.
 *
 * Config (via xb_config map):
 *   key 0 = eviction mode (0=passive_mru, 1=cycle_moe, 2=default_lru, 3=fifo)
 *   key 1 = prefetch length in bytes (0 = default 2MB)
 *
 * Pattern reference: extension/prefetch_trace.bpf.c
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

/*
 * Eviction mode (set via config map key 0):
 *   0 = passive MRU (T1 protect + BYPASS for non-T1, default)
 *   1 = cycle_moe (T1 protect + DEFAULT for non-T1, kernel LRU refresh)
 *   2 = no eviction BPF (all DEFAULT, pure kernel LRU)
 *   3 = FIFO (all BYPASS, freeze all positions)
 */
#define EVICT_PASSIVE_MRU  0
#define EVICT_CYCLE_MOE    1
#define EVICT_DEFAULT_LRU  2
#define EVICT_FIFO         3

/* ===== Maps ===== */

/* Configuration: key 0 = eviction mode, key 1 = prefetch length (bytes) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, u32);
    __type(value, u64);
} xb_config SEC(".maps");

/* Direction tracking: per-CPU history of last 3 block VAs for momentum detection.
 * Config key 2:
 *   0 = direction-aware 2-step (default)
 *   1 = blind adjacent
 *   2 = direction-aware 3-step (stricter) */
struct direction_ctx {
    u64 block_hist[3];    /* [0]=oldest, [1]=prev, [2]=current */
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct direction_ctx);
} direction_cache SEC(".maps");

/* Global dedup: last prefetched target address (avoids duplicate cross-CPU prefetch) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} last_prefetch_target SEC(".maps");

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
 * key 6: rate-limit skip (same block as last on this CPU)
 * key 7: wq callback skipped (invalid data)
 * key 8: direction skip (no consistent forward/backward momentum)
 * key 9: dedup skip (another CPU already prefetched this target)
 */
#define STAT_KPROBE       0
#define STAT_PREFETCH     1
#define STAT_WQ_SCHED     2
#define STAT_WQ_CALLBACK  3
#define STAT_MIGRATE_OK   4
#define STAT_MIGRATE_FAIL 5
#define STAT_RATELIMIT    6
#define STAT_WQ_SKIP      7
#define STAT_DIR_SKIP     8
#define STAT_DEDUP_SKIP   9
#define NUM_STATS         10

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 10);
    __type(key, u32);
    __type(value, u64);
} xb_stats SEC(".maps");

static __always_inline void stat_inc(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&xb_stats, &key);
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

    if (va_block) {
        info->va_start = BPF_CORE_READ(va_block, start);
        info->va_end = BPF_CORE_READ(va_block, end);

        /* Navigate: va_block → managed_range → va_range.va_space */
        uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
        if (managed) {
            uvm_va_space_t *vs = BPF_CORE_READ(managed, va_range.va_space);
            u64 vs_val = 0;
            bpf_probe_read_kernel(&vs_val, sizeof(vs_val), &vs);
            info->va_space = vs_val;
        } else {
            info->va_space = 0;
        }
    } else {
        info->va_start = 0;
        info->va_end = 0;
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

/* ===== PREFETCH: always_max + bpf_wq cross-block ===== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    stat_inc(STAT_PREFETCH);

    /* 1) Intra-block: always_max — prefetch entire VA block */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(decision_ctx, max_first, max_outer);

    /* 2) Cross-block: direction-aware prefetch via bpf_wq.
     *
     * Direction detection: track last 2 block VAs per-CPU.
     * Only prefetch when 2+ consecutive same-direction transitions detected.
     * Global dedup: prevent multiple CPUs from prefetching the same target.
     *
     * Config key 1 = prefetch length (0 = 2MB default)
     * Config key 2 = mode: 0 = direction-aware (default), 1 = blind adjacent */
    u32 zero = 0;
    struct va_block_ctx *blk = bpf_map_lookup_elem(&va_block_cache, &zero);
    if (!blk || !blk->va_space || !blk->va_end)
        return 1; /* BYPASS (always_max only) */

    u64 va_space = blk->va_space;
    u64 block_end = blk->va_end;

    /* Per-CPU direction tracking */
    struct direction_ctx *dir = bpf_map_lookup_elem(&direction_cache, &zero);
    if (!dir)
        return 1;

    u64 h0 = dir->block_hist[0]; /* oldest */
    u64 h1 = dir->block_hist[1]; /* prev */
    u64 h2 = dir->block_hist[2]; /* current */

    /* Update history: shift window */
    if (block_end != h2) {
        dir->block_hist[0] = h1;
        dir->block_hist[1] = h2;
        dir->block_hist[2] = block_end;
    } else {
        /* Same block as last on this CPU — rate limit */
        stat_inc(STAT_RATELIMIT);
        return 1;
    }

    /* Read prefetch mode from config key 2 */
    u32 cfg_mode_key = 2;
    u64 *cfg_mode = bpf_map_lookup_elem(&xb_config, &cfg_mode_key);
    u64 pf_mode = cfg_mode ? *cfg_mode : 0;

    u64 prefetch_target = 0;

    if (pf_mode == 1) {
        /* Blind adjacent: always prefetch next block */
        prefetch_target = block_end + 1;
    } else if (pf_mode == 2) {
        /* 3-step direction-aware: require 3 consecutive same-direction */
        if (h0 == 0 || h1 == 0 || h2 == 0) {
            stat_inc(STAT_DIR_SKIP);
            return 1;
        }
        s64 d0 = (s64)(h1 - h0);
        s64 d1 = (s64)(h2 - h1);
        s64 d2 = (s64)(block_end - h2);

        if ((d0 > 0 && d1 > 0 && d2 > 0) || (d0 < 0 && d1 < 0 && d2 < 0)) {
            prefetch_target = block_end + (d2 > 0 ? 1 : -(s64)VA_BLOCK_SIZE);
        } else {
            stat_inc(STAT_DIR_SKIP);
            return 1;
        }
    } else if (pf_mode == 3) {
        /* Adjacent-stride: require all transitions are exactly ±1 VA block */
        if (h0 == 0 || h1 == 0 || h2 == 0) {
            stat_inc(STAT_DIR_SKIP);
            return 1;
        }
        s64 d0 = (s64)(h1 - h0);
        s64 d1 = (s64)(h2 - h1);
        s64 d2 = (s64)(block_end - h2);
        s64 blk = (s64)VA_BLOCK_SIZE;

        /* All transitions must be exactly +1 or -1 block (adjacent) */
        if ((d0 == blk && d1 == blk && d2 == blk) ||
            (d0 == -blk && d1 == -blk && d2 == -blk)) {
            prefetch_target = block_end + (d2 > 0 ? 1 : -(s64)VA_BLOCK_SIZE);
        } else {
            stat_inc(STAT_DIR_SKIP);
            return 1;
        }
    } else {
        /* 2-step direction-aware (default): require 2 consecutive same-direction */
        if (h1 == 0 || h2 == 0) {
            stat_inc(STAT_DIR_SKIP);
            return 1;
        }
        s64 d1 = (s64)(h2 - h1);
        s64 d2 = (s64)(block_end - h2);

        if ((d1 > 0 && d2 > 0) || (d1 < 0 && d2 < 0)) {
            prefetch_target = block_end + (d2 > 0 ? 1 : -(s64)VA_BLOCK_SIZE);
        } else {
            stat_inc(STAT_DIR_SKIP);
            return 1;
        }
    }

    if (!prefetch_target)
        return 1;

    /* Global dedup: check if another CPU already prefetched this target */
    u64 *last_target = bpf_map_lookup_elem(&last_prefetch_target, &zero);
    if (last_target) {
        if (*last_target == prefetch_target) {
            stat_inc(STAT_DEDUP_SKIP);
            return 1;
        }
        *last_target = prefetch_target;
    }

    /* Read configurable prefetch length (key 1), default = 2MB */
    u32 cfg_len_key = 1;
    u64 *cfg_len = bpf_map_lookup_elem(&xb_config, &cfg_len_key);
    u64 prefetch_len = (cfg_len && *cfg_len > 0) ? *cfg_len : VA_BLOCK_SIZE;

    /* Use CPU-indexed wq_map entry to avoid data corruption across CPUs */
    int wq_key = bpf_get_smp_processor_id() % 64;
    struct prefetch_data *data = bpf_map_lookup_elem(&wq_map, &wq_key);
    if (data) {
        data->va_space = va_space;
        data->addr = prefetch_target;
        data->length = prefetch_len;

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

/* ===== EVICTION: configurable ===== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* Read eviction mode from config */
    u32 cfg_key = 0;
    u64 *mode = bpf_map_lookup_elem(&xb_config, &cfg_key);
    u64 evict_mode = mode ? *mode : EVICT_PASSIVE_MRU;

    /* Mode 2 (default LRU): return DEFAULT, let kernel handle everything */
    if (evict_mode == EVICT_DEFAULT_LRU)
        return 0;

    /* Mode 3 (FIFO): return BYPASS for all, freeze positions */
    if (evict_mode == EVICT_FIFO)
        return 1;

    /* Modes 0,1: use T1 frequency tracking */
    u32 idx = chunk_hash(chunk);
    u8 *count;

    count = bpf_map_lookup_elem(&access_counts, &idx);
    if (!count)
        return 0;

    u8 c = *count;
    if (c < 255)
        *count = c + 1;

    if (c + 1 >= T1_FREQ_THRESHOLD) {
        /* T1: protect */
        bpf_gpu_block_move_tail(chunk, list);
        return 1;
    }

    /* Mode 0 (passive MRU): BYPASS without move */
    if (evict_mode == EVICT_PASSIVE_MRU)
        return 1;

    /* Mode 1 (cycle_moe): DEFAULT, let kernel LRU refresh non-T1 */
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
struct gpu_mem_ops uvm_ops_cross_block_v2 = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
