/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Adaptive Sequential Prefetch Policy with Direction Support
 *
 * Instead of iterating the bitmap tree, this policy directly takes a portion
 * of max_prefetch_region based on a configurable percentage (0-100).
 *
 * Userspace can dynamically adjust the prefetch percentage based on:
 * - PCIe bandwidth utilization
 * - GPU memory pressure
 * - Workload characteristics
 *
 * percentage = 100: equivalent to always_max (prefetch entire region)
 * percentage = 0:   equivalent to prefetch_none (no prefetch)
 * percentage = 50:  prefetch half of the max region
 *
 * Direction support (from prefetch_direction):
 * direction = 0 (FORWARD):       Prefetch pages AFTER the faulting page (higher addresses)
 *                                Use for sequential access patterns (low -> high)
 * direction = 1 (BACKWARD):      Prefetch pages BEFORE the faulting page (lower addresses)
 *                                Use for reverse access patterns (high -> low)
 * direction = 2 (FORWARD_START): Prefetch from region start [max_first, max_first+n)
 *                                Original sequential prefetch behavior
 *
 * num_pages = 0: Use percentage-based prefetch
 * num_pages > 0: Prefetch exactly num_pages (overrides percentage)
 */

#define PREFETCH_FORWARD       0
#define PREFETCH_BACKWARD      1
#define PREFETCH_FORWARD_START 2

/* BPF map: Stores prefetch percentage (0-100) set by userspace */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);  // Prefetch percentage (0-100)
} prefetch_pct_map SEC(".maps");

/* BPF map: Stores prefetch direction (0=forward, 1=backward) set by userspace */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} prefetch_direction_map SEC(".maps");

/* BPF map: Stores number of pages to prefetch (0=use percentage) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} prefetch_num_pages_map SEC(".maps");

/* Helper: Get prefetch percentage from userspace */
static __always_inline unsigned int get_prefetch_percentage(void)
{
    u32 key = 0;
    u32 *pct = bpf_map_lookup_elem(&prefetch_pct_map, &key);

    if (!pct)
        return 100;  // Default to 100% (always_max behavior) if not set

    return *pct;
}

/* Helper: Get prefetch direction from userspace */
static __always_inline unsigned int get_prefetch_direction(void)
{
    u32 key = 0;
    u32 *dir = bpf_map_lookup_elem(&prefetch_direction_map, &key);

    if (!dir)
        return PREFETCH_FORWARD;  /* Default to forward */

    return *dir;
}

/* Helper: Get number of pages to prefetch from userspace */
static __always_inline unsigned int get_prefetch_num_pages(void)
{
    u32 key = 0;
    u32 *num = bpf_map_lookup_elem(&prefetch_num_pages_map, &key);

    if (!num)
        return 0;  /* Default to 0 (use percentage) */

    return *num;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    u32 pct = get_prefetch_percentage();
    u32 direction = get_prefetch_direction();
    u32 num_pages = get_prefetch_num_pages();

    /* Read max_prefetch_region bounds */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    unsigned int total_pages = max_outer - max_first;

    /* Handle edge cases */
    if (pct == 0 || total_pages == 0) {
        /* No prefetch */
        bpf_uvm_set_va_block_region(result_region, 0, 0);
        return 1; /* UVM_BPF_ACTION_BYPASS */
    }

    /* Calculate how many pages to prefetch:
     * - If num_pages > 0, use that directly (overrides percentage)
     * - Otherwise, calculate from percentage
     */
    unsigned int prefetch_pages;
    if (num_pages > 0) {
        prefetch_pages = num_pages;
    } else if (pct >= 100) {
        prefetch_pages = total_pages;
    } else {
        prefetch_pages = (total_pages * pct) / 100;
        if (prefetch_pages == 0)
            prefetch_pages = 1;  // At least 1 page if pct > 0
    }

    uvm_page_index_t new_first, new_outer;

    if (direction == PREFETCH_BACKWARD) {
        /* Prefetch pages BEFORE page_index (lower addresses) */
        /* If page_index <= max_first, there's nothing to prefetch backward */
        if (page_index <= max_first) {
            bpf_uvm_set_va_block_region(result_region, 0, 0);
            return 1; /* UVM_BPF_ACTION_BYPASS */
        }

        new_outer = page_index;

        /* Calculate new_first */
        if (page_index >= prefetch_pages) {
            new_first = page_index - prefetch_pages;
        } else {
            new_first = 0;
        }
        /* Clamp to max_first */
        if (new_first < max_first)
            new_first = max_first;

        if (new_outer > max_outer)
            new_outer = max_outer;

        // bpf_printk("adaptive_seq: BACKWARD page=%u, n=%u, region=[%u,%u)\n",
        //            page_index, prefetch_pages, new_first, new_outer);
    } else if (direction == PREFETCH_FORWARD_START) {
        /* FORWARD_START: Prefetch from region start [max_first, max_first+n)
         * This is the original sequential prefetch behavior */
        new_first = max_first;
        new_outer = max_first + prefetch_pages;

        /* Clamp to max_outer */
        if (new_outer > max_outer)
            new_outer = max_outer;

        // bpf_printk("adaptive_seq: FORWARD_START page=%u, n=%u, region=[%u,%u)\n",
        //            page_index, prefetch_pages, new_first, new_outer);
    } else {
        /* FORWARD: Prefetch pages AFTER page_index (higher addresses) */
        new_first = page_index + 1;

        /* If page_index+1 >= max_outer, there's nothing to prefetch forward */
        if (new_first >= max_outer) {
            bpf_uvm_set_va_block_region(result_region, 0, 0);
            return 1; /* UVM_BPF_ACTION_BYPASS */
        }

        /* Clamp new_first to max_first */
        if (new_first < max_first)
            new_first = max_first;

        /* Calculate new_outer */
        new_outer = new_first + prefetch_pages;
        /* Clamp to max_outer */
        if (new_outer > max_outer)
            new_outer = max_outer;

        // bpf_printk("adaptive_seq: FORWARD page=%u, n=%u, region=[%u,%u)\n",
        //            page_index, prefetch_pages, new_first, new_outer);
    }

    bpf_uvm_set_va_block_region(result_region, new_first, new_outer);

    return 1; /* UVM_BPF_ACTION_BYPASS */
}

/* This hook is not used - we bypass tree iteration */
SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    /* Not used - we return BYPASS in before_compute */
    return 0;
}

/* Dummy implementation for test trigger */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_adaptive_sequential = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
