/* SPDX-License-Identifier: GPL-2.0 */
#ifndef STALE_STATE_575_DRIVER_BRIDGE_V1_OBSERVER_H
#define STALE_STATE_575_DRIVER_BRIDGE_V1_OBSERVER_H

#include "abi.h"

#define STALE_STATE_V1_MODE_NATIVE 1U
#define STALE_STATE_V1_MODE_BPF 2U
#define STALE_STATE_V1_PHASE_DENSE 1U
#define STALE_STATE_V1_PHASE_SPARSE 2U
#define STALE_STATE_V1_DIAG_SELECTED 1U
#define STALE_STATE_V1_DIAG_FINISHED 2U
#define STALE_STATE_V1_STATUS_EFFECT_APPLIED 5U
#define STALE_STATE_V1_ACTION_PREFETCH_MAX 1U
#define STALE_STATE_V1_ACTION_DISCARD_PREFETCH 2U
#define STALE_STATE_V1_TRANSITION_APPLY 0U
#define STALE_STATE_V1_INITIAL_BYPASS 1U

struct uvm_stale_state_v1_diagnostic {
    struct uvm_stale_state_v1_input input;
    long long callback_return;
    unsigned long long decision_age_ns;
    unsigned long long requested_first;
    unsigned long long requested_outer;
    unsigned long long output_first;
    unsigned long long output_outer;
    unsigned int diagnostic_phase;
    unsigned int mode;
    unsigned int status;
    unsigned int action;
    unsigned int action_attempted;
    unsigned int action_conflict;
    unsigned int action_request_calls;
    unsigned int region_result;
    unsigned int initial_effect;
    unsigned int reserved;
};

struct stale_state_v1_observer_config {
    unsigned int target_tgid;
    unsigned int expected_mode;
};

struct stale_state_v1_observer_event {
    unsigned long long observed_mono_ns;
    unsigned long long pid_tgid;
    struct uvm_stale_state_v1_diagnostic diagnostic;
};

#define STALE_STATE_V1_OBSERVER_COUNTERS(X) \
    X(diagnostic_calls) X(selected_seen) X(finished_seen) X(records_emitted) \
    X(foreign_tgid) X(read_errors) X(ringbuf_drops) X(phase_errors)

struct stale_state_v1_observer_metrics {
#define STALE_STATE_V1_COUNTER_FIELD(name) unsigned long long name;
    STALE_STATE_V1_OBSERVER_COUNTERS(STALE_STATE_V1_COUNTER_FIELD)
#undef STALE_STATE_V1_COUNTER_FIELD
};

_Static_assert(sizeof(struct uvm_stale_state_v1_diagnostic) == 176,
               "diagnostic ABI size");
_Static_assert(__builtin_offsetof(struct uvm_stale_state_v1_diagnostic,
                                  diagnostic_phase) == 136,
               "diagnostic phase offset");
_Static_assert(sizeof(struct stale_state_v1_observer_event) == 192,
               "observer event ABI size");

#endif
