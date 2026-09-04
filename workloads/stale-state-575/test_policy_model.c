/* SPDX-License-Identifier: MIT */
#include "stale_state_policy_model.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static struct stale_state_575_snapshot snapshot(uint64_t sequence,
                                                 uint32_t phase,
                                                 uint64_t source,
                                                 uint64_t published)
{
    struct stale_state_575_snapshot value = {
        .sequence = sequence,
        .source_mono_ns = source,
        .published_mono_ns = published,
        .phase = phase,
        .reserved = 0,
    };
    return value;
}

static void test_dense_and_sparse_actions(void)
{
    struct stale_state_575_decision decision = {0};
    struct stale_state_575_snapshot dense =
        snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 120);
    struct stale_state_575_snapshot sparse =
        snapshot(2, STALE_STATE_575_PHASE_SPARSE, 200, 220);

    assert(stale_state_575_choose(&dense, 170, &decision) ==
           STALE_STATE_575_ACTION_PREFETCH_MAX);
    assert(decision.snapshot_sequence == 1);
    assert(decision.decision_age_ns == 70);
    assert(!stale_state_575_wrong_phase(&decision,
                                        STALE_STATE_575_PHASE_DENSE));
    assert(stale_state_575_wrong_phase(&decision,
                                       STALE_STATE_575_PHASE_SPARSE));

    memset(&decision, 0, sizeof(decision));
    assert(stale_state_575_choose(&sparse, 260, &decision) ==
           STALE_STATE_575_ACTION_DISCARD_PREFETCH);
    assert(decision.snapshot_sequence == 2);
    assert(decision.decision_age_ns == 60);
}

static void test_invalid_snapshots_reject(void)
{
    struct stale_state_575_decision decision = {0};
    struct stale_state_575_snapshot value =
        snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 120);

    assert(stale_state_575_choose(NULL, 130, &decision) ==
           STALE_STATE_575_ACTION_REJECT);
    assert(stale_state_575_choose(&value, 130, NULL) ==
           STALE_STATE_575_ACTION_REJECT);
    value.sequence = 0;
    assert(stale_state_575_choose(&value, 130, &decision) ==
           STALE_STATE_575_ACTION_REJECT);
    value = snapshot(1, STALE_STATE_575_PHASE_INVALID, 100, 120);
    assert(stale_state_575_choose(&value, 130, &decision) ==
           STALE_STATE_575_ACTION_REJECT);
    value = snapshot(1, STALE_STATE_575_PHASE_DENSE, 120, 100);
    assert(stale_state_575_choose(&value, 130, &decision) ==
           STALE_STATE_575_ACTION_REJECT);
    value = snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 140);
    assert(stale_state_575_choose(&value, 130, &decision) ==
           STALE_STATE_575_ACTION_REJECT);
    value = snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 120);
    value.reserved = 1;
    assert(stale_state_575_choose(&value, 130, &decision) ==
           STALE_STATE_575_ACTION_REJECT);
}

int main(void)
{
    _Static_assert(sizeof(struct stale_state_575_snapshot) == 32,
                   "snapshot ABI size");
    _Static_assert(sizeof(struct stale_state_575_decision) == 24,
                   "decision ABI size");
    test_dense_and_sparse_actions();
    test_invalid_snapshots_reject();
    puts("stale_state_policy_model: all CPU assertions passed");
    return 0;
}

