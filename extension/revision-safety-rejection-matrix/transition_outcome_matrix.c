/* SPDX-License-Identifier: MIT */
/* CPU-only paired outcomes through the production-shared transition header. */

#include <stdio.h>
#include <stdlib.h>

#include "nv-gpu-transition-validator.h"

static unsigned int pairs;
static unsigned int assertions;

#define EXPECT(condition)                                                       \
    do                                                                          \
    {                                                                           \
        ++assertions;                                                           \
        if (!(condition))                                                       \
        {                                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define PASS_PAIR(name, unsafe_name, control_name)                              \
    do                                                                          \
    {                                                                           \
        ++pairs;                                                                \
        printf("PASS layer=transition case=%s unsafe=%s control=%s\n", name,   \
               unsafe_name, control_name);                                      \
    } while (0)

static void test_range_pair(void)
{
    const nv_gpu_scheduler_snapshot_t snapshot = { 17, 3, 1 };
    nv_gpu_transition_u64_request_t invalid = { 0 };
    nv_gpu_transition_u64_request_t legal = { 0 };
    const nv_gpu_transition_u32_request_t no_interleave = { 0 };
    nv_gpu_scheduler_validation_t result;

    nv_gpu_transition_record_u64(&invalid, 9);
    result = nv_gpu_transition_validate_scheduler(
        &snapshot, &snapshot, 100, 1, 10, &invalid, &no_interleave);
    EXPECT(result.timeslice_result == NV_GPU_TRANSITION_REJECT_RANGE);
    EXPECT(result.timeslice == 100);

    nv_gpu_transition_record_u64(&legal, 10);
    result = nv_gpu_transition_validate_scheduler(
        &snapshot, &snapshot, 100, 1, 10, &legal, &no_interleave);
    EXPECT(result.timeslice_result == NV_GPU_TRANSITION_APPLY);
    EXPECT(result.timeslice == 10);
    PASS_PAIR("scheduler-range", "rejected-preserved-native",
              "accepted-applied-minimum");
}

static void test_stale_pair(void)
{
    const nv_gpu_scheduler_snapshot_t expected = { 17, 3, 1 };
    nv_gpu_scheduler_snapshot_t stale = expected;
    nv_gpu_transition_u64_request_t request = { 0 };
    const nv_gpu_transition_u32_request_t no_interleave = { 0 };
    nv_gpu_scheduler_validation_t result;

    nv_gpu_transition_record_u64(&request, 20);
    stale.phase++;
    result = nv_gpu_transition_validate_scheduler(
        &expected, &stale, 100, 1, 10, &request, &no_interleave);
    EXPECT(result.timeslice_result == NV_GPU_TRANSITION_NOOP_STALE);
    EXPECT(result.timeslice == 100);

    result = nv_gpu_transition_validate_scheduler(
        &expected, &expected, 100, 1, 10, &request, &no_interleave);
    EXPECT(result.timeslice_result == NV_GPU_TRANSITION_APPLY);
    EXPECT(result.timeslice == 20);
    PASS_PAIR("scheduler-snapshot", "stale-noop-preserved-native",
              "current-applied");
}

static void test_conflict_pair(void)
{
    const nv_gpu_scheduler_snapshot_t snapshot = { 17, 3, 1 };
    nv_gpu_transition_u64_request_t conflict = { 0 };
    nv_gpu_transition_u64_request_t repeat = { 0 };
    const nv_gpu_transition_u32_request_t no_interleave = { 0 };
    nv_gpu_scheduler_validation_t result;

    EXPECT(nv_gpu_transition_record_u64(&conflict, 20) ==
           NV_GPU_TRANSITION_APPLY);
    EXPECT(nv_gpu_transition_record_u64(&conflict, 30) ==
           NV_GPU_TRANSITION_NOOP_CONFLICT);
    result = nv_gpu_transition_validate_scheduler(
        &snapshot, &snapshot, 100, 1, 10, &conflict, &no_interleave);
    EXPECT(result.timeslice_result == NV_GPU_TRANSITION_NOOP_CONFLICT);
    EXPECT(result.timeslice == 100);

    EXPECT(nv_gpu_transition_record_u64(&repeat, 20) ==
           NV_GPU_TRANSITION_APPLY);
    EXPECT(nv_gpu_transition_record_u64(&repeat, 20) ==
           NV_GPU_TRANSITION_NOOP_REPEAT);
    result = nv_gpu_transition_validate_scheduler(
        &snapshot, &snapshot, 100, 1, 10, &repeat, &no_interleave);
    EXPECT(result.timeslice_result == NV_GPU_TRANSITION_APPLY);
    EXPECT(result.timeslice == 20);
    PASS_PAIR("scheduler-conflict", "conflict-noop-preserved-native",
              "idempotent-repeat-applied-once");
}

static void test_prefetch_action_pair(void)
{
    NvU32 action;

    EXPECT(nv_gpu_transition_validate_initial_action(99, &action) ==
           NV_GPU_TRANSITION_REJECT_ACTION);
    EXPECT(action == NV_GPU_TRANSITION_ACTION_DEFAULT);
    EXPECT(nv_gpu_transition_prefetch_initial_effect(
               99, NV_GPU_TRANSITION_APPLY) ==
           NV_GPU_PREFETCH_INITIAL_NATIVE);

    EXPECT(nv_gpu_transition_validate_initial_action(
               NV_GPU_TRANSITION_ACTION_BYPASS, &action) ==
           NV_GPU_TRANSITION_APPLY);
    EXPECT(action == NV_GPU_TRANSITION_ACTION_BYPASS);
    EXPECT(nv_gpu_transition_prefetch_initial_effect(
               NV_GPU_TRANSITION_ACTION_BYPASS,
               NV_GPU_TRANSITION_APPLY) ==
           NV_GPU_PREFETCH_INITIAL_BYPASS);
    PASS_PAIR("prefetch-action", "invalid99-rejected-native-route",
              "bypass-accepted-bypass-route");
}

static void test_pmm_identity_pair(void)
{
    const nv_gpu_pmm_snapshot_t expected = {
        1, 2, 3, NV_GPU_PMM_DESTINATION_USED
    };
    nv_gpu_pmm_snapshot_t wrong_owner = expected;
    nv_gpu_pmm_request_t request = { 0 };
    enum nv_gpu_transition_result result;

    nv_gpu_transition_record_pmm(&request, NV_GPU_PMM_DESTINATION_UNUSED,
                                 NV_GPU_PMM_POSITION_TAIL);
    wrong_owner.owner_id++;
    result = nv_gpu_transition_validate_pmm(&expected, &wrong_owner, &request);
    EXPECT(result == NV_GPU_TRANSITION_REJECT_IDENTITY);
    EXPECT(nv_gpu_transition_pmm_access_effect(
               NV_GPU_TRANSITION_ACTION_BYPASS, result) ==
           NV_GPU_PMM_ACCESS_PRESERVE);

    result = nv_gpu_transition_validate_pmm(&expected, &expected, &request);
    EXPECT(result == NV_GPU_TRANSITION_APPLY);
    EXPECT(nv_gpu_transition_pmm_access_effect(
               NV_GPU_TRANSITION_ACTION_BYPASS, result) ==
           NV_GPU_PMM_ACCESS_COMMIT);
    PASS_PAIR("pmm-identity", "wrong-owner-rejected-preserved",
              "matching-owner-committed");
}

int main(void)
{
    test_range_pair();
    test_stale_pair();
    test_conflict_pair();
    test_prefetch_action_pair();
    test_pmm_identity_pair();
    printf("PASS all: %u transition pairs, %u assertions\n", pairs, assertions);
    return EXIT_SUCCESS;
}
