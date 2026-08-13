// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __PREFETCH_TRACE_EVENT_H
#define __PREFETCH_TRACE_EVENT_H

// Hook types for prefetch tracing
#define HOOK_PREFETCH_BEFORE_COMPUTE 1
#define HOOK_PREFETCH_ON_TREE_ITER 2
#define HOOK_PREFETCH_GET_HINT 3
#define HOOK_PREFETCH_DECISION 4

// Event structure shared between BPF and userspace
struct prefetch_event {
    __u64 timestamp_ns;
    __u64 call_id;
    __u32 cpu;
    __u32 hook_type;

    __u32 current_pid;
    __u32 action;

    // Page fault info
    __u32 page_index;           // Triggering page index

    // max_prefetch_region
    __u32 max_region_first;     // max_prefetch_region.first
    __u32 max_region_outer;     // max_prefetch_region.outer

    // Policy callback output before UVM traversal and clamping
    __u32 policy_region_first;
    __u32 policy_region_outer;

    // Region actually returned by compute_prefetch_region()
    __u32 final_region_first;
    __u32 final_region_outer;
    __u32 final_pages;
    __u32 reserved;

    // bitmap_tree info
    __u32 tree_offset;          // bitmap_tree->offset
    __u32 tree_leaf_count;      // bitmap_tree->leaf_count
    __u32 tree_level_count;     // bitmap_tree->level_count
    __u32 pages_accessed;       // popcount of bitmap_tree->pages (how many pages already accessed)

    // VA block info (from uvm_perf_prefetch_get_hint_va_block)
    __u64 va_block;             // va_block pointer
    __u64 va_start;             // va_block->start
    __u64 va_end;               // va_block->end

    // faulted_region info
    __u32 faulted_first;        // faulted_region.first
    __u32 faulted_outer;        // faulted_region.outer

    // PID info (from va_block->cpu.fault_authorized or mm_struct)
    __u32 fault_pid;            // PID that caused the fault (from fault_authorized.first_pid)
    __u32 owner_tgid;           // Owner process TGID (from mm->owner->tgid)
};

#endif /* __PREFETCH_TRACE_EVENT_H */
