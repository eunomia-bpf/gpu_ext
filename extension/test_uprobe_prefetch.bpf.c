/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Uprobe-driven GPU Prefetch POC
 *
 * Architecture:
 *   kprobe on uvm_perf_prefetch_get_hint_va_block → captures va_space → stores in map
 *   uprobe on user's request_prefetch(addr, len) → writes request to pending_req map
 *   struct_ops gpu_page_prefetch → on next fault, checks pending_req, schedules bpf_wq
 *
 * Note: bpf_wq is NOT allowed in tracing (uprobe/kprobe) programs, only in
 * struct_ops. So the uprobe writes to a shared map, and the struct_ops callback
 * drains it on the next fault.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* ===== va_space capture (from kprobe, same as cross_block_v2) ===== */

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} va_space_map SEC(".maps");

SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_space, uvm_va_block_t *va_block)
{
    if (!va_block)
        return 0;

    uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
    if (!managed)
        return 0;

    uvm_va_space_t *vs = BPF_CORE_READ(managed, va_range.va_space);
    u64 vs_val = 0;
    bpf_probe_read_kernel(&vs_val, sizeof(vs_val), &vs);

    if (vs_val) {
        u32 key = 0;
        bpf_map_update_elem(&va_space_map, &key, &vs_val, BPF_ANY);
    }

    return 0;
}

/* ===== Pending uprobe request (written by uprobe, read by struct_ops) ===== */

struct pending_req {
    u64 addr;
    u64 length;
    u64 pending;  /* 1 = request waiting, 0 = idle */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct pending_req);
} pending_map SEC(".maps");

/* ===== bpf_wq for async prefetch (scheduled from struct_ops) ===== */

struct prefetch_req {
    u64 va_space;
    u64 addr;
    u64 length;
    struct bpf_wq work;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, int);
    __type(value, struct prefetch_req);
} wq_map SEC(".maps");

/* Stats */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

#define STAT_UPROBE_FIRE  0
#define STAT_WQ_SCHED     1
#define STAT_MIGRATE_OK   2
#define STAT_MIGRATE_FAIL 3

static __always_inline void stat_inc(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static int do_prefetch_cb(void *map, int *key, void *value)
{
    struct prefetch_req *req = value;
    if (!req || !req->va_space || !req->length)
        return 0;

    int ret = bpf_gpu_migrate_range(req->va_space, req->addr, req->length);
    if (ret == 0)
        stat_inc(STAT_MIGRATE_OK);
    else
        stat_inc(STAT_MIGRATE_FAIL);

    bpf_printk("uprobe prefetch: addr=0x%llx len=%llu ret=%d\n",
               req->addr, req->length, ret);

    return 0;
}

/* ===== uprobe: write request to pending_map (fallback if direct kfunc unavailable) ===== */

SEC("uprobe")
int BPF_UPROBE(uprobe_request_prefetch, void *addr, size_t length)
{
    stat_inc(STAT_UPROBE_FIRE);

    u32 key = 0;
    struct pending_req *pr = bpf_map_lookup_elem(&pending_map, &key);
    if (!pr)
        return 0;

    pr->addr = (u64)addr;
    pr->length = (u64)length;
    __sync_lock_test_and_set(&pr->pending, 1);

    bpf_printk("uprobe: queued prefetch addr=0x%llx len=%llu\n",
               (u64)addr, (u64)length);

    return 0;
}

/* ===== sleepable uprobe: directly call bpf_gpu_migrate_range ===== */

SEC("uprobe.s")
int BPF_UPROBE(uprobe_direct_prefetch, void *addr, size_t length)
{
    stat_inc(STAT_UPROBE_FIRE);

    u32 key = 0;
    u64 *vs = bpf_map_lookup_elem(&va_space_map, &key);
    if (!vs || !*vs) {
        bpf_printk("uprobe_direct: no va_space yet\n");
        return 0;
    }

    u64 target_addr = (u64)addr;
    u64 target_len = (u64)length;

    bpf_printk("uprobe_direct: migrate addr=0x%llx len=%llu\n",
               target_addr, target_len);

    int ret = bpf_gpu_migrate_range(*vs, target_addr, target_len);
    if (ret == 0)
        stat_inc(STAT_MIGRATE_OK);
    else
        stat_inc(STAT_MIGRATE_FAIL);

    stat_inc(STAT_WQ_SCHED); /* reuse counter for direct calls */
    return 0;
}

/* ===== struct_ops: always_max + drain pending uprobe requests ===== */

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_bpf_prefetch_decision_t *decision_ctx)
{
    /* Always-max: prefetch entire block */
    uvm_va_block_region_t max_region;
    bpf_probe_read_kernel(&max_region, sizeof(max_region), max_prefetch_region);
    bpf_gpu_set_prefetch_region(decision_ctx, max_region.first, max_region.outer);

    /* Check for pending uprobe request and schedule bpf_wq */
    u32 key = 0;
    struct pending_req *pr = bpf_map_lookup_elem(&pending_map, &key);
    if (pr && __sync_lock_test_and_set(&pr->pending, 0)) {
        u64 *vs = bpf_map_lookup_elem(&va_space_map, &key);
        if (vs && *vs) {
            int wq_key = 0;
            struct prefetch_req *req = bpf_map_lookup_elem(&wq_map, &wq_key);
            if (req) {
                req->va_space = *vs;
                req->addr = pr->addr;
                req->length = pr->length;

                stat_inc(STAT_WQ_SCHED);
                bpf_wq_init(&req->work, &wq_map, 0);
                bpf_wq_set_callback(&req->work, do_prefetch_cb, 0);
                bpf_wq_start(&req->work, 0);
            }
        }
    }

    return 1; /* UVM_BPF_ACTION_BYPASS */
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

SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_uprobe_prefetch = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
};
