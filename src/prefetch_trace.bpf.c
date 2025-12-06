// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * Prefetch Trace Tool - Trace prefetch calls using kprobes
 *
 * Traces uvm_bpf_call_before_compute_prefetch with VA block info
 * VA block info is captured from uvm_perf_prefetch_get_hint_va_block
 * and stored in per-CPU map for correlation
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "prefetch_trace_event.h"
#include "uvm_types.h"

char LICENSE[] SEC("license") = "GPL";

// Per-CPU storage for VA block info from get_hint_va_block
struct va_block_info {
    u64 va_block;
    u64 va_start;
    u64 va_end;
    u32 faulted_first;
    u32 faulted_outer;
    u32 fault_pid;      // PID from fault_authorized.first_pid
    u32 owner_tgid;     // Owner TGID from mm->owner->tgid
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct va_block_info);
} va_block_cache SEC(".maps");

// Ring buffer for outputting events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Statistics counters
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

#define STAT_GET_HINT 0
#define STAT_BEFORE_COMPUTE 1
#define STAT_ON_TREE_ITER 2
#define STAT_DROPPED 3

static __always_inline void inc_stat(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

// Count set bits in a 64-bit word
static __always_inline u32 popcount64(u64 x)
{
    u32 count = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        count += (x >> i) & 1;
    }
    return count;
}

/*
 * Hook: uvm_perf_prefetch_get_hint_va_block
 *
 * Captures VA block info and stores it in per-CPU map.
 * This is called BEFORE uvm_bpf_call_before_compute_prefetch.
 */
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(trace_get_hint_va_block,
               uvm_va_block_t *va_block,
               void *va_block_context,
               u32 new_residency,
               void *faulted_pages,
               u32 faulted_region_packed,  /* first:16 | outer:16 */
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
    u32 key = 0;
    struct va_block_info *info;

    inc_stat(STAT_GET_HINT);

    info = bpf_map_lookup_elem(&va_block_cache, &key);
    if (!info)
        return 0;

    // Store VA block info for later use by before_compute
    if (va_block) {
        info->va_block = (u64)va_block;
        info->va_start = BPF_CORE_READ(va_block, start);
        info->va_end = BPF_CORE_READ(va_block, end);

        // Get fault_authorized.first_pid - this is the PID that caused the fault
        info->fault_pid = BPF_CORE_READ(va_block, cpu.fault_authorized.first_pid);

        // Try to get owner TGID via va_block->managed_range->va_range.va_space->va_space_mm.mm->owner->tgid
        uvm_va_range_managed_t *managed_range = BPF_CORE_READ(va_block, managed_range);
        if (managed_range) {
            uvm_va_space_t *va_space = BPF_CORE_READ(managed_range, va_range.va_space);
            if (va_space) {
                struct mm_struct *mm = BPF_CORE_READ(va_space, va_space_mm.mm);
                if (mm) {
                    struct task_struct *owner = BPF_CORE_READ(mm, owner);
                    if (owner) {
                        info->owner_tgid = BPF_CORE_READ(owner, tgid);
                    } else {
                        info->owner_tgid = 0;
                    }
                } else {
                    info->owner_tgid = 0;
                }
            } else {
                info->owner_tgid = 0;
            }
        } else {
            info->owner_tgid = 0;
        }
    } else {
        info->va_block = 0;
        info->va_start = 0;
        info->va_end = 0;
        info->fault_pid = 0;
        info->owner_tgid = 0;
    }

    // Extract faulted_region (packed as first:16, outer:16)
    info->faulted_first = faulted_region_packed & 0xFFFF;
    info->faulted_outer = (faulted_region_packed >> 16) & 0xFFFF;

    return 0;
}

/*
 * Hook: uvm_bpf_call_before_compute_prefetch
 *
 * Main event output - combines VA block info from per-CPU cache
 * with prefetch computation parameters.
 */
SEC("kprobe/uvm_bpf_call_before_compute_prefetch")
int BPF_KPROBE(trace_before_compute,
               u32 page_index,
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
               uvm_va_block_region_t *max_prefetch_region,
               uvm_va_block_region_t *result_region)
{
    struct prefetch_event *e;
    u32 key = 0;
    struct va_block_info *cached_info;

    inc_stat(STAT_BEFORE_COMPUTE);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = HOOK_PREFETCH_BEFORE_COMPUTE;
    e->page_index = page_index;

    // Get cached VA block info from per-CPU map
    cached_info = bpf_map_lookup_elem(&va_block_cache, &key);
    if (cached_info) {
        e->va_block = cached_info->va_block;
        e->va_start = cached_info->va_start;
        e->va_end = cached_info->va_end;
        e->faulted_first = cached_info->faulted_first;
        e->faulted_outer = cached_info->faulted_outer;
        e->fault_pid = cached_info->fault_pid;
        e->owner_tgid = cached_info->owner_tgid;
    } else {
        e->va_block = 0;
        e->va_start = 0;
        e->va_end = 0;
        e->faulted_first = 0;
        e->faulted_outer = 0;
        e->fault_pid = 0;
        e->owner_tgid = 0;
    }

    // Read max_prefetch_region using CO-RE
    if (max_prefetch_region) {
        e->max_region_first = BPF_CORE_READ(max_prefetch_region, first);
        e->max_region_outer = BPF_CORE_READ(max_prefetch_region, outer);
    } else {
        e->max_region_first = 0;
        e->max_region_outer = 0;
    }

    // Read bitmap_tree info using CO-RE
    if (bitmap_tree) {
        e->tree_offset = BPF_CORE_READ(bitmap_tree, offset);
        e->tree_leaf_count = BPF_CORE_READ(bitmap_tree, leaf_count);
        e->tree_level_count = BPF_CORE_READ(bitmap_tree, level_count);

        // Count total pages accessed (popcount of bitmap)
        u32 total = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u64 bits = BPF_CORE_READ(bitmap_tree, pages.bitmap[i]);
            total += popcount64(bits);
        }
        e->pages_accessed = total;
    } else {
        e->tree_offset = 0;
        e->tree_leaf_count = 0;
        e->tree_level_count = 0;
        e->pages_accessed = 0;
    }

    bpf_ringbuf_submit(e, 0);
    return 0;
}
