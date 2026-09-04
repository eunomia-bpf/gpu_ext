/* SPDX-License-Identifier: MIT */
#ifndef STALE_STATE_575_POLICY_MODEL_H
#define STALE_STATE_575_POLICY_MODEL_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Pure decision model intended to be shared by the future native and BPF
 * implementations.  It deliberately contains no clock read and no mutable
 * state: the driver must pass one atomically captured snapshot and the current
 * decision time.
 */
enum stale_state_575_phase {
    STALE_STATE_575_PHASE_INVALID = 0,
    STALE_STATE_575_PHASE_DENSE = 1,
    STALE_STATE_575_PHASE_SPARSE = 2,
};

enum stale_state_575_action {
    STALE_STATE_575_ACTION_REJECT = 0,
    STALE_STATE_575_ACTION_PREFETCH_MAX = 1,
    STALE_STATE_575_ACTION_DISCARD_PREFETCH = 2,
};

struct stale_state_575_snapshot {
    uint64_t sequence;
    uint64_t source_mono_ns;
    uint64_t published_mono_ns;
    uint32_t phase;
    uint32_t reserved;
};

struct stale_state_575_decision {
    uint64_t snapshot_sequence;
    uint64_t decision_age_ns;
    uint32_t snapshot_phase;
    uint32_t action;
};

static inline bool stale_state_575_snapshot_valid(
    const struct stale_state_575_snapshot *snapshot,
    uint64_t decision_mono_ns)
{
    if (snapshot == 0 || snapshot->sequence == 0 || snapshot->reserved != 0)
        return false;
    if (snapshot->phase != STALE_STATE_575_PHASE_DENSE &&
        snapshot->phase != STALE_STATE_575_PHASE_SPARSE)
        return false;
    if (snapshot->source_mono_ns == 0 || snapshot->published_mono_ns == 0)
        return false;
    if (snapshot->published_mono_ns < snapshot->source_mono_ns ||
        decision_mono_ns < snapshot->published_mono_ns)
        return false;
    return true;
}

static inline enum stale_state_575_action stale_state_575_choose(
    const struct stale_state_575_snapshot *snapshot,
    uint64_t decision_mono_ns,
    struct stale_state_575_decision *decision)
{
    enum stale_state_575_action action;

    if (!stale_state_575_snapshot_valid(snapshot, decision_mono_ns) ||
        decision == 0)
        return STALE_STATE_575_ACTION_REJECT;

    action = snapshot->phase == STALE_STATE_575_PHASE_DENSE
                 ? STALE_STATE_575_ACTION_PREFETCH_MAX
                 : STALE_STATE_575_ACTION_DISCARD_PREFETCH;
    decision->snapshot_sequence = snapshot->sequence;
    decision->decision_age_ns = decision_mono_ns - snapshot->source_mono_ns;
    decision->snapshot_phase = snapshot->phase;
    decision->action = action;
    return action;
}

static inline bool stale_state_575_wrong_phase(
    const struct stale_state_575_decision *decision,
    enum stale_state_575_phase host_phase)
{
    if (decision == 0 ||
        (host_phase != STALE_STATE_575_PHASE_DENSE &&
         host_phase != STALE_STATE_575_PHASE_SPARSE))
        return false;
    return decision->snapshot_phase != (uint32_t)host_phase;
}

#endif

