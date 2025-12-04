/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Directional Prefetch Policy
 *
 * Prefetch pages in a configurable direction relative to the faulting page,
 * with configurable number of pages to prefetch.
 *
 * direction = 0 (FORWARD):  Prefetch pages AFTER the faulting page (higher addresses)
 *                           [page_index+1, page_index+1+num_pages)
 *                           Use for sequential access patterns (low -> high)
 *
 * direction = 1 (BACKWARD): Prefetch pages BEFORE the faulting page (lower addresses)
 *                           [page_index-num_pages, page_index)
 *                           Use for reverse access patterns (high -> low)
 *
 * num_pages = 0: Prefetch all available pages in that direction (up to max_prefetch_region)
 * num_pages > 0: Prefetch exactly num_pages (or fewer if not available)
 *
 * Default: FORWARD, num_pages=0 (all available)
 */

#define PREFETCH_FORWARD  0
#define PREFETCH_BACKWARD 1

/* BPF map: Stores prefetch direction (0=forward, 1=backward) set by userspace */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} prefetch_direction_map SEC(".maps");

/* BPF map: Stores number of pages to prefetch (0=all available) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} prefetch_num_pages_map SEC(".maps");

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
        return 0;  /* Default to 0 (all available) */

    return *num;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    /* Read max_prefetch_region bounds */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);

    u32 direction = get_prefetch_direction();
    u32 num_pages = get_prefetch_num_pages();

    if (direction == PREFETCH_BACKWARD) {
        /* Prefetch pages BEFORE page_index (lower addresses) */
        /* If page_index <= max_first, there's nothing to prefetch backward */
        if (page_index <= max_first) {
            bpf_uvm_set_va_block_region(result_region, 0, 0);
            return 1; /* UVM_BPF_ACTION_BYPASS */
        }

        /* Calculate new_first based on num_pages */
        uvm_page_index_t new_first;
        uvm_page_index_t new_outer = page_index;

        if (num_pages == 0) {
            /* Prefetch all available: [max_first, page_index) */
            new_first = max_first;
        } else {
            /* Prefetch num_pages: [page_index - num_pages, page_index) */
            if (page_index >= num_pages) {
                new_first = page_index - num_pages;
            } else {
                new_first = 0;
            }
            /* Clamp to max_first */
            if (new_first < max_first)
                new_first = max_first;
        }

        if (new_outer > max_outer)
            new_outer = max_outer;

        // bpf_printk("prefetch_dir: BACKWARD fault=%u, n=%u, prefetch=[%u,%u)\n",
        //            page_index, num_pages, new_first, new_outer);

        bpf_uvm_set_va_block_region(result_region, new_first, new_outer);
    } else {
        /* FORWARD: Prefetch pages AFTER page_index (higher addresses) */
        uvm_page_index_t new_first = page_index + 1;

        /* If page_index+1 >= max_outer, there's nothing to prefetch forward */
        if (new_first >= max_outer) {
            bpf_uvm_set_va_block_region(result_region, 0, 0);
            return 1; /* UVM_BPF_ACTION_BYPASS */
        }

        /* Clamp new_first to max_first */
        if (new_first < max_first)
            new_first = max_first;

        /* Calculate new_outer based on num_pages */
        uvm_page_index_t new_outer;
        if (num_pages == 0) {
            /* Prefetch all available: [page_index+1, max_outer) */
            new_outer = max_outer;
        } else {
            /* Prefetch num_pages: [page_index+1, page_index+1+num_pages) */
            new_outer = new_first + num_pages;
            /* Clamp to max_outer */
            if (new_outer > max_outer)
                new_outer = max_outer;
        }

        // bpf_printk("prefetch_dir: FORWARD fault=%u, n=%u, prefetch=[%u,%u)\n",
        //            page_index, num_pages, new_first, new_outer);

        bpf_uvm_set_va_block_region(result_region, new_first, new_outer);
    }

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
struct uvm_gpu_ext uvm_ops_direction = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
