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

static __always_inline struct prefetch_frame *current_frame(struct prefetch_metrics *m)
{
    __u64 task = bpf_get_current_pid_tgid();
    struct prefetch_frame *f = bpf_map_lookup_elem(&frames, &task);

    if (!f)
        m->missing_frame++;
    return f;
}

SEC("fentry/uvm_bpf_call_gpu_page_prefetch")
int BPF_PROG(wrapper_enter, uvm_page_index_t page,
             uvm_perf_prefetch_bitmap_tree_t *tree,
             uvm_va_block_region_t *maximum, void *decision)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame f = { .in_wrapper = 1 };
    __u64 task = bpf_get_current_pid_tgid();

    if (!m)
        return 0;
    m->wrapper_enter++;
    if (bpf_map_lookup_elem(&frames, &task)) {
        m->nesting_errors++;
        return 0;
    }
    if (bpf_map_update_elem(&frames, &task, &f, BPF_NOEXIST))
        m->map_errors++;
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
    f = current_frame(m);
    if (!f)
        return 0;
    if (!f->in_wrapper || f->pending || f->selected || f->policy_seen) {
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
    uvm_va_block_region_t bounds = {};

    if (!m)
        return 0;
    f = current_frame(m);
    if (!f)
        return 0;
    m->wrapper_exit++;
    if (!f->in_wrapper || f->pending || f->selected ||
        f->policy_seen != (action != 0))
        m->order_errors++;
    if (bpf_probe_read_kernel(&request, sizeof(request), decision) ||
        bpf_probe_read_kernel(&bounds, sizeof(bounds), maximum)) {
        m->read_errors++;
        return 0;
    }
    if (request.attempted != (action != 0) || request.conflict ||
        request.first || request.outer)
        m->request_errors++;
    if (returned_action != action)
        m->action_errors++;
    if (bounds.first >= bounds.outer || bounds.outer > 512)
        m->output_errors++;
    if (returned_action == 0)
        m->returned_default++;
    else if (returned_action == 1)
        m->returned_bypass++;
    else if (returned_action == 99)
        m->returned_invalid99++;
    else
        m->action_errors++;

    f->action = returned_action;
    f->request_attempted = request.attempted;
    f->request_conflict = request.conflict;
    f->requested_first = request.first;
    f->requested_outer = request.outer;
    f->max_first = bounds.first;
    f->max_outer = bounds.outer;
    f->in_wrapper = 0;
    f->pending = 1;
    return 0;
}

static __always_inline int selected_state_matches(
    const struct prefetch_frame *f,
    const struct uvm_bpf_prefetch_diagnostic_ctx *ctx)
{
    if (ctx->raw_action != f->action ||
        ctx->request_attempted != f->request_attempted ||
        ctx->request_conflict != f->request_conflict ||
        ctx->requested_first != f->requested_first ||
        ctx->requested_outer != f->requested_outer ||
        ctx->max_first != f->max_first || ctx->max_outer != f->max_outer)
        return 0;

    if (f->action == 0)
        return !ctx->request_attempted && !ctx->request_conflict &&
               !ctx->requested_first && !ctx->requested_outer &&
               ctx->initial_region_result == PREFETCH_RESULT_NOOP_DEFAULT &&
               ctx->initial_effect == PREFETCH_EFFECT_NATIVE;
    if (f->action == 1)
        return ctx->request_attempted && !ctx->request_conflict &&
               !ctx->requested_first && !ctx->requested_outer &&
               ctx->initial_region_result == PREFETCH_RESULT_APPLY &&
               ctx->initial_effect == PREFETCH_EFFECT_BYPASS;
    if (f->action == 99)
        return ctx->request_attempted && !ctx->request_conflict &&
               !ctx->requested_first && !ctx->requested_outer &&
               ctx->initial_region_result == PREFETCH_RESULT_APPLY &&
               ctx->initial_effect == PREFETCH_EFFECT_NATIVE;
    return 0;
}

static __always_inline void record_output(
    const struct prefetch_frame *f,
    const struct uvm_bpf_prefetch_diagnostic_ctx *ctx,
    struct prefetch_metrics *m)
{
    if (!ctx->output_first && !ctx->output_outer) {
        m->empty_outputs++;
        return;
    }
    if (ctx->output_first < ctx->output_outer &&
        ctx->output_first >= f->max_first &&
        ctx->output_outer <= f->max_outer && ctx->output_outer <= 512) {
        m->nonempty_outputs++;
        return;
    }
    m->output_errors++;
}

SEC("fentry/uvm_bpf_prefetch_diagnostic")
int BPF_PROG(diagnostic_enter,
             const struct uvm_bpf_prefetch_diagnostic_ctx *driver_ctx)
{
    struct prefetch_metrics *m = stats();
    struct prefetch_frame *f;
    struct uvm_bpf_prefetch_diagnostic_ctx diagnostic = {};
    __u64 task = bpf_get_current_pid_tgid();

    if (!m)
        return 0;
    m->diagnostic_calls++;
    if (bpf_probe_read_kernel(&diagnostic, sizeof(diagnostic), driver_ctx)) {
        m->read_errors++;
        return 0;
    }
    f = current_frame(m);
    if (!f)
        return 0;

    if (diagnostic.phase == PREFETCH_DIAG_SELECTED) {
        m->selected_events++;
        if (!f->pending || f->in_wrapper) {
            m->order_errors++;
            return 0;
        }
        if (f->selected) {
            m->nesting_errors++;
            return 0;
        }
        if (!selected_state_matches(f, &diagnostic))
            m->state_errors++;
        if (diagnostic.initial_region_result == PREFETCH_RESULT_NOOP_DEFAULT)
            m->region_noop_default++;
        else if (diagnostic.initial_region_result == PREFETCH_RESULT_APPLY)
            m->region_apply++;
        else
            m->state_errors++;
        if (diagnostic.initial_effect == PREFETCH_EFFECT_NATIVE)
            m->native_effects++;
        else if (diagnostic.initial_effect == PREFETCH_EFFECT_BYPASS)
            m->bypass_effects++;
        else
            m->state_errors++;
        f->initial_region_result = diagnostic.initial_region_result;
        f->initial_effect = diagnostic.initial_effect;
        f->selected = 1;
        return 0;
    }

    if (diagnostic.phase == PREFETCH_DIAG_FINISHED) {
        m->finished_events++;
        if (!f->pending || f->in_wrapper || !f->selected) {
            m->order_errors++;
            if (bpf_map_delete_elem(&frames, &task))
                m->map_errors++;
            return 0;
        }
        if (!selected_state_matches(f, &diagnostic) ||
            diagnostic.initial_region_result != f->initial_region_result ||
            diagnostic.initial_effect != f->initial_effect)
            m->state_errors++;

        record_output(f, &diagnostic, m);
        if (diagnostic.initial_effect == PREFETCH_EFFECT_BYPASS) {
            if (diagnostic.native_completed || diagnostic.native_iterations ||
                diagnostic.output_first || diagnostic.output_outer)
                m->traversal_errors++;
        }
        else if (diagnostic.initial_effect == PREFETCH_EFFECT_NATIVE) {
            if (diagnostic.native_completed != 1 || !diagnostic.native_iterations)
                m->traversal_errors++;
        }
        else {
            m->state_errors++;
        }
        m->native_completions += diagnostic.native_completed;
        m->native_iterations += diagnostic.native_iterations;
        m->decisions_complete++;
        if (bpf_map_delete_elem(&frames, &task))
            m->map_errors++;
        return 0;
    }

    m->phase_errors++;
    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops invalid_prefetch_ops = {
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
};
