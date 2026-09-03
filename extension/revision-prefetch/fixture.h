/* SPDX-License-Identifier: GPL-2.0 */
#ifndef REVISION_PREFETCH_FIXTURE_H
#define REVISION_PREFETCH_FIXTURE_H

/* All counters are per-CPU and summed only after the owned target has exited. */
#define PREFETCH_COUNTERS(X) \
    X(mask_enter) X(mask_exit) X(wrapper_enter) X(wrapper_exit) \
    X(policy_calls) X(setter_ok) X(decisions_complete) \
    X(returned_default) X(returned_bypass) X(returned_invalid99) \
    X(native_decisions) X(bypass_decisions) X(range_calls) \
    X(empty_masks) X(nonempty_masks) \
    X(map_errors) X(nesting_errors) X(missing_frame) X(identity_errors) \
    X(order_errors) X(read_errors) X(request_errors) X(action_errors) \
    X(traversal_errors) X(iterator_calls) X(mask_bounds_errors)

struct prefetch_metrics {
#define COUNTER_FIELD(name) unsigned long long name;
    PREFETCH_COUNTERS(COUNTER_FIELD)
#undef COUNTER_FIELD
    unsigned long long sample_first, sample_outer, sample_bitmap[8];
};

struct prefetch_frame {
    unsigned long long tree, mask;
    unsigned long long range_calls;
    unsigned int first, outer, in_wrapper, pending, policy_seen;
    long long action;
};

#define PREFETCH_MAX_FRAMES 256
#define PREFETCH_OBSERVER_COUNT 6
#endif
