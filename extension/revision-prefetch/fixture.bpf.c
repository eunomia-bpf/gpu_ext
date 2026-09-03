/* SPDX-License-Identifier: GPL-2.0 */
/* Functional observer only: never use these instrumented cells as timings. */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "../uvm_types.h"
#include "../bpf_testmod.h"
#include "fixture.h"

char LICENSE[] SEC("license") = "GPL";
const volatile __u32 action = 99;

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, PREFETCH_MAX_FRAMES);
    __type(key, __u64);
    __type(value, struct prefetch_frame);
} frames SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct prefetch_metrics);
} metrics SEC(".maps");

static __always_inline struct prefetch_metrics *stats(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&metrics, &key);
}

static __always_inline __u64 pointer_id(const void *pointer)
{
    __u64 value = 0;
    /* Same opaque-scalar conversion as the existing delta/Markov fixture. */
    bpf_probe_read_kernel(&value, sizeof(value), &pointer);
    return value;
}

static __always_inline struct prefetch_frame *frame_for(
    const void *tree, struct prefetch_metrics *m)
{
    __u64 task = bpf_get_current_pid_tgid();
    struct prefetch_frame *f = bpf_map_lookup_elem(&frames, &task);
    if (!f) {
        m->missing_frame++;
        return 0;
    }
    if (f->tree != pointer_id(tree)) {
        m->identity_errors++;
        return 0;
    }
    return f;
}

static __always_inline void finish_decision(struct prefetch_frame *f,
                                           struct prefetch_metrics *m)
{
    if (!f->pending)
        return;
    m->decisions_complete++;
    if (f->action == 1) {
        m->bypass_decisions++;
        if (f->range_calls)
            m->traversal_errors++;
    } else if (f->action == 0 || f->action == 99) {
        m->native_decisions++;
        if (!f->range_calls)
            m->traversal_errors++;
    } else {
        m->action_errors++;
    }
    f->pending = 0;
}

/* PROG2 preserves the actual by-value region ABI; no instruction offsets. */
SEC("fentry/compute_prefetch_mask")
int BPF_PROG2(mask_enter, uvm_va_block_region_t, faulted,
              uvm_va_block_region_t, maximum,
              uvm_perf_prefetch_bitmap_tree_t *, tree,
              const uvm_page_mask_t *, faults, uvm_page_mask_t *, output)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame f = {};
    __u64 task = bpf_get_current_pid_tgid();
    if (!m)
        return 0;
    m->mask_enter++;
    if (bpf_map_lookup_elem(&frames, &task)) {
        m->nesting_errors++;
        return 0;
    }
    f.tree = pointer_id(tree);
    f.mask = pointer_id(output);
    f.first = maximum.first;
    f.outer = maximum.outer;
    if (!f.tree || !f.mask || f.first >= f.outer || f.outer > 512) {
        m->identity_errors++;
        return 0;
    }
    if (bpf_map_update_elem(&frames, &task, &f, BPF_NOEXIST))
        m->map_errors++;
    return 0;
}

SEC("fentry/uvm_bpf_call_gpu_page_prefetch")
int BPF_PROG(wrapper_enter, uvm_page_index_t page,
             uvm_perf_prefetch_bitmap_tree_t *tree,
             uvm_va_block_region_t *maximum, void *decision)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame *f;
    if (!m || !(f = frame_for(tree, m)))
        return 0;
    m->wrapper_enter++;
    if (f->in_wrapper)
        m->order_errors++;
    finish_decision(f, m);
    f->in_wrapper = 1;
    f->policy_seen = 0;
    f->range_calls = 0;
    return 0;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, uvm_page_index_t page,
             uvm_perf_prefetch_bitmap_tree_t *tree,
             uvm_va_block_region_t *maximum,
             uvm_bpf_prefetch_decision_t *decision)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame *f;
    if (!m)
        return 0;
    m->policy_calls++;
    if (!(f = frame_for(tree, m)) || !f->in_wrapper || f->policy_seen) {
        m->order_errors++;
        return 0; /* Missing observation never becomes an unscoped injection. */
    }
    f->policy_seen = 1;
    if (action != 1 && action != 99) {
        m->action_errors++;
        return 0;
    }
    if (bpf_gpu_set_prefetch_region(decision, 0, 0)) {
        m->request_errors++;
        return 0;
    }
    m->setter_ok++;
    return action;
}

SEC("fexit/uvm_bpf_call_gpu_page_prefetch")
int BPF_PROG(wrapper_exit, uvm_page_index_t page,
             uvm_perf_prefetch_bitmap_tree_t *tree,
             uvm_va_block_region_t *maximum, const void *decision,
             long long returned_action)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame *f;
    uvm_bpf_prefetch_decision_t request = {};
    if (!m || !(f = frame_for(tree, m)))
        return 0;
    m->wrapper_exit++;
    if (!f->in_wrapper || f->pending || f->policy_seen != (action != 0))
        m->order_errors++;
    if (bpf_probe_read_kernel(&request, sizeof(request), decision))
        m->read_errors++;
    else if (request.attempted != (action != 0) || request.conflict ||
             request.first || request.outer)
        m->request_errors++;
    if (returned_action != action)
        m->action_errors++;
    if (returned_action == 0)
        m->returned_default++;
    else if (returned_action == 1)
        m->returned_bypass++;
    else if (returned_action == 99)
        m->returned_invalid99++;
    f->action = returned_action; /* Actual return, not the fixture's input. */
    f->in_wrapper = 0;
    f->pending = 1;
    return 0;
}

SEC("fentry/uvm_perf_prefetch_bitmap_tree_iter_get_range")
int BPF_PROG(range_enter, const uvm_perf_prefetch_bitmap_tree_t *tree,
             const uvm_perf_prefetch_bitmap_tree_iter_t *iter)
{
    struct prefetch_metrics *m = stats();
    __u64 task = bpf_get_current_pid_tgid();
    struct prefetch_frame *f = bpf_map_lookup_elem(&frames, &task);
    if (!m || !f)
        return 0; /* Other callers outside a mask computation are not evidence. */
    if (f->tree != pointer_id(tree))
        m->identity_errors++;
    else if (f->in_wrapper || !f->pending)
        m->order_errors++;
    else {
        f->range_calls++;
        m->range_calls++;
    }
    return 0;
}

SEC("fentry/uvm_bpf_call_gpu_page_prefetch_iter")
int BPF_PROG(iterator_enter)
{
    struct prefetch_metrics *m = stats();
    if (m)
        m->iterator_calls++;
    return 0;
}

SEC("fexit/compute_prefetch_mask")
int BPF_PROG2(mask_exit, uvm_va_block_region_t, faulted,
              uvm_va_block_region_t, maximum,
              uvm_perf_prefetch_bitmap_tree_t *, tree,
              const uvm_page_mask_t *, faults, uvm_page_mask_t *, output)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame *f;
    uvm_page_mask_t actual = {};
    __u64 task = bpf_get_current_pid_tgid();
    __u64 nonempty = 0;
    if (!m || !(f = frame_for(tree, m)))
        return 0;
    if (f->in_wrapper)
        m->order_errors++;
    finish_decision(f, m);
    if (f->mask != pointer_id(output) || f->first != maximum.first ||
        f->outer != maximum.outer)
        m->identity_errors++;
    else if (bpf_probe_read_kernel(&actual, sizeof(actual), output))
        m->read_errors++;
    else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            __u64 allowed = ~0ULL;
            __u32 lo = i * 64, hi = lo + 64;
            if (f->outer <= lo || f->first >= hi)
                allowed = 0;
            else {
                if (f->first > lo)
                    allowed &= ~0ULL << ((f->first - lo) & 63);
                if (f->outer < hi)
                    allowed &= (1ULL << ((f->outer - lo) & 63)) - 1;
            }
            if (actual.bitmap[i] & ~allowed)
                m->mask_bounds_errors++;
            nonempty |= actual.bitmap[i];
            if (!m->mask_exit)
                m->sample_bitmap[i] = actual.bitmap[i];
        }
        if (!m->mask_exit) {
            m->sample_first = f->first;
            m->sample_outer = f->outer;
        }
        if (nonempty)
            m->nonempty_masks++;
        else
            m->empty_masks++;
    }
    m->mask_exit++;
    if (bpf_map_delete_elem(&frames, &task))
        m->map_errors++;
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops invalid_prefetch_ops = {
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
};
