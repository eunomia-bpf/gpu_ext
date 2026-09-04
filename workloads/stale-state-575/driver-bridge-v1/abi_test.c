/* SPDX-License-Identifier: MIT */

#include <stdio.h>
#include <stdlib.h>

#include "abi.h"
#include "../stale_state_policy_model.h"

static unsigned int checks;

#define CHECK(condition)                                                        \
    do {                                                                        \
        ++checks;                                                               \
        if (!(condition)) {                                                     \
            fprintf(stderr, "%s:%d: check failed: %s\n",                    \
                    __FILE__, __LINE__, #condition);                            \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

static int compare(unsigned int phase,
                   unsigned long long source_mono_ns,
                   unsigned long long published_mono_ns,
                   unsigned long long decision_mono_ns,
                   unsigned int expected)
{
    struct uvm_stale_state_v1_input input = {
        .snapshot = {
            .sequence = 1,
            .source_mono_ns = source_mono_ns,
            .published_mono_ns = published_mono_ns,
            .phase = phase,
        },
        .decision_mono_ns = decision_mono_ns,
        .abi_version = STALE_STATE_DRIVER_V1_ABI_VERSION,
    };
    struct stale_state_575_snapshot canonical = {
        .sequence = input.snapshot.sequence,
        .source_mono_ns = input.snapshot.source_mono_ns,
        .published_mono_ns = input.snapshot.published_mono_ns,
        .phase = input.snapshot.phase,
        .reserved = input.snapshot.reserved,
    };
    struct stale_state_575_decision decision = {0};
    enum stale_state_575_action action =
        stale_state_575_choose(&canonical, decision_mono_ns, &decision);

    CHECK((unsigned int)action == expected);
    if (action != STALE_STATE_575_ACTION_REJECT) {
        CHECK(decision.snapshot_sequence == input.snapshot.sequence);
        CHECK(decision.snapshot_phase == input.snapshot.phase);
        CHECK(decision.decision_age_ns ==
              decision_mono_ns - input.snapshot.source_mono_ns);
    }
    return EXIT_SUCCESS;
}

int main(void)
{
    CHECK(STALE_STATE_DRIVER_V1_ABI_VERSION == 1);
    CHECK(compare(STALE_STATE_575_PHASE_DENSE, 100, 120, 170,
                  STALE_STATE_575_ACTION_PREFETCH_MAX) == EXIT_SUCCESS);
    CHECK(compare(STALE_STATE_575_PHASE_SPARSE, 100, 120, 180,
                  STALE_STATE_575_ACTION_DISCARD_PREFETCH) == EXIT_SUCCESS);
    CHECK(compare(STALE_STATE_575_PHASE_INVALID, 100, 120, 180,
                  STALE_STATE_575_ACTION_REJECT) == EXIT_SUCCESS);
    CHECK(compare(STALE_STATE_575_PHASE_DENSE, 100, 120, 119,
                  STALE_STATE_575_ACTION_REJECT) == EXIT_SUCCESS);
    printf("stale-state driver bridge ABI: %u checks passed\n", checks);
    return EXIT_SUCCESS;
}
