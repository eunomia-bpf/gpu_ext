/* SPDX-License-Identifier: GPL-2.0 */
#ifndef REVISION_PREFETCH_FIXTURE_H
#define REVISION_PREFETCH_FIXTURE_H

/* All counters are per-CPU and summed only after the owned target has exited. */
#define PREFETCH_COUNTERS(X) \
    X(wrapper_enter) X(wrapper_exit) X(policy_calls) X(setter_ok) \
    X(diagnostic_calls) X(selected_events) X(finished_events) \
    X(decisions_complete) \
    X(returned_default) X(returned_bypass) X(returned_invalid99) \
    X(region_noop_default) X(region_apply) \
    X(native_effects) X(bypass_effects) \
    X(native_completions) X(native_iterations) \
    X(empty_outputs) X(nonempty_outputs) \
    X(map_errors) X(nesting_errors) X(missing_frame) \
    X(order_errors) X(read_errors) X(request_errors) X(action_errors) \
    X(state_errors) X(phase_errors) X(traversal_errors) X(output_errors)

struct prefetch_metrics {
#define COUNTER_FIELD(name) unsigned long long name;
    PREFETCH_COUNTERS(COUNTER_FIELD)
#undef COUNTER_FIELD
};

struct prefetch_frame {
    unsigned long long requested_first, requested_outer;
    unsigned long long max_first, max_outer;
    long long action;
    unsigned int request_attempted, request_conflict;
    unsigned int initial_region_result, initial_effect;
    unsigned int in_wrapper, pending, policy_seen, selected;
};

struct uvm_bpf_prefetch_diagnostic_ctx {
    long long raw_action;
    unsigned long long requested_first;
    unsigned long long requested_outer;
    unsigned long long max_first;
    unsigned long long max_outer;
    unsigned long long output_first;
    unsigned long long output_outer;
    unsigned int phase;
    unsigned int request_attempted;
    unsigned int request_conflict;
    unsigned int initial_region_result;
    unsigned int initial_effect;
    unsigned int native_iterations;
    unsigned int native_completed;
};

_Static_assert(sizeof(struct uvm_bpf_prefetch_diagnostic_ctx) == 88,
               "prefetch diagnostic ABI size");
_Static_assert(__builtin_offsetof(struct uvm_bpf_prefetch_diagnostic_ctx, phase) == 56,
               "prefetch diagnostic phase offset");
_Static_assert(__builtin_offsetof(struct uvm_bpf_prefetch_diagnostic_ctx, native_completed) == 80,
               "prefetch diagnostic completion offset");

#define PREFETCH_DIAG_SELECTED 1U
#define PREFETCH_DIAG_FINISHED 2U
#define PREFETCH_RESULT_APPLY 0U
#define PREFETCH_RESULT_NOOP_DEFAULT 1U
#define PREFETCH_EFFECT_NATIVE 0U
#define PREFETCH_EFFECT_BYPASS 1U
#define PREFETCH_MAX_FRAMES 256
#define PREFETCH_OBSERVER_COUNT 3
#endif
