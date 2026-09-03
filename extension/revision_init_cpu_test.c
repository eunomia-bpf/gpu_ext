/* SPDX-License-Identifier: GPL-2.0 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Explicitly selected 575 production header, not a second validator model. */
#include "nv-gpu-transition-validator.h"

static unsigned assertions;
#define EXPECT(condition) do {                                        \
    ++assertions;                                                     \
    if (!(condition)) {                                               \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1);                                                      \
    }                                                                 \
} while (0)

static struct nv_gpu_task_init_decision_ctx *current;
static int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *ctx, NvU64 value)
{
    EXPECT(ctx == &current->input);
    return nv_gpu_transition_record_u64(&current->timeslice_request, value);
}
static int bpf_nv_gpu_set_interleave(struct nv_gpu_task_init_ctx *ctx, NvU32 value)
{
    EXPECT(ctx == &current->input);
    return nv_gpu_transition_record_u32(&current->interleave_request, value);
}
#include "revision_init_requests.h"

_Static_assert(NV_GPU_TRANSITION_APPLY == 0, "recording/apply status");
_Static_assert(NV_GPU_TRANSITION_NOOP_DEFAULT == 1, "default status");
_Static_assert(NV_GPU_TRANSITION_NOOP_REPEAT == 2, "repeat status");
_Static_assert(NV_GPU_TRANSITION_NOOP_CONFLICT == 4, "conflict status");
_Static_assert(NV_GPU_TRANSITION_REJECT_RANGE == 6, "range status");
_Static_assert(sizeof(struct nv_gpu_task_init_ctx) == 32, "input ABI");
_Static_assert(offsetof(struct nv_gpu_task_init_ctx, default_timeslice) == 16,
               "timeslice input ABI");

struct expected_case {
    enum revision_init_case which;
    const char *name;
    struct revision_init_returns requests;
    enum nv_gpu_transition_result timeslice_result;
    enum nv_gpu_transition_result interleave_result;
    NvU32 final_interleave;
};
static const struct expected_case matrix[] = {
    {REVISION_INIT_NO_REQUEST, "no_request", {0, 0, {0}, {0}}, 1, 1, 1},
    {REVISION_INIT_LEGAL, "legal", {1, 1, {0}, {0}}, 0, 0, 0},
    {REVISION_INIT_INVALID_INTERLEAVE, "invalid_interleave",
        {0, 1, {0}, {0}}, 1, 6, 1},
    {REVISION_INIT_DUPLICATE, "duplicate", {2, 2, {0, 2}, {0, 2}}, 0, 0, 0},
    {REVISION_INIT_CONFLICT, "conflict", {3, 3, {0, 4, 4}, {0, 4, 4}}, 4, 4, 1},
    {REVISION_INIT_INDEPENDENT_INTERLEAVE, "independent_interleave",
        {1, 1, {0}, {0}}, 0, 6, 1},
    {REVISION_INIT_INDEPENDENT_TIMESLICE, "independent_timeslice",
        {2, 1, {0, 4}, {0}}, 4, 0, 0},
};

int main(void)
{
    /* The current 575 generated minimum HAL is zero. These synthetic defaults
     * additionally check zero and 64-bit endpoints; none is a GPU measurement. */
    const NvU64 defaults[] = {0, 1, 1024, ~(NvU64)0};
    const nv_gpu_scheduler_snapshot_t snapshot = {17, 3, 1};
    unsigned cases = 0;

    EXPECT(sizeof(matrix) / sizeof(matrix[0]) == REVISION_INIT_CASE_COUNT);
    for (unsigned d = 0; d < sizeof(defaults) / sizeof(defaults[0]); ++d) {
        for (unsigned i = 0; i < sizeof(matrix) / sizeof(matrix[0]); ++i) {
            const struct expected_case *expected = &matrix[i];
            struct nv_gpu_task_init_decision_ctx decision = {0};
            struct revision_init_returns requests = {0};
            nv_gpu_scheduler_validation_t validation;

            decision.input.tsg_id = snapshot.tsg_id;
            decision.input.engine_type = 1;
            decision.input.default_timeslice = defaults[d];
            decision.input.default_interleave = 1;
            decision.input.runlist_id = snapshot.runlist_id;
            struct nv_gpu_task_init_ctx before = decision.input;
            current = &decision;
            revision_init_issue_requests(expected->which, &decision.input, &requests);
            current = NULL;
            EXPECT(before.tsg_id == decision.input.tsg_id);
            EXPECT(before.engine_type == decision.input.engine_type);
            EXPECT(before.default_timeslice == decision.input.default_timeslice);
            EXPECT(before.default_interleave == decision.input.default_interleave);
            EXPECT(before.runlist_id == decision.input.runlist_id);
            EXPECT(requests.timeslice_count == expected->requests.timeslice_count);
            EXPECT(requests.interleave_count == expected->requests.interleave_count);
            for (unsigned slot = 0; slot < 3; ++slot) {
                EXPECT(requests.timeslice[slot] == expected->requests.timeslice[slot]);
                EXPECT(requests.interleave[slot] == expected->requests.interleave[slot]);
            }
            EXPECT(decision.timeslice_request.attempted == (requests.timeslice_count != 0));
            EXPECT(decision.interleave_request.attempted == (requests.interleave_count != 0));
            EXPECT(decision.timeslice_request.conflict ==
                   (expected->timeslice_result == NV_GPU_TRANSITION_NOOP_CONFLICT));
            EXPECT(decision.interleave_request.conflict ==
                   (expected->interleave_result == NV_GPU_TRANSITION_NOOP_CONFLICT));
            if (requests.timeslice_count)
                EXPECT(decision.timeslice_request.value == defaults[d]);
            if (requests.interleave_count)
                EXPECT(decision.interleave_request.value ==
                       (expected->interleave_result == NV_GPU_TRANSITION_REJECT_RANGE ? 3 : 0));

            validation = nv_gpu_transition_validate_scheduler(
                &snapshot, &snapshot, defaults[d], 1, 0,
                &decision.timeslice_request, &decision.interleave_request);
            EXPECT(validation.timeslice_result == expected->timeslice_result);
            EXPECT(validation.interleave_result == expected->interleave_result);
            EXPECT(validation.timeslice == defaults[d]);
            EXPECT(validation.interleave == expected->final_interleave);
            ++cases;
        }
    }
    printf("revision_init_cpu: fixtures=%u cases=%u assertions=%u "
           "scope=production_shared_recorder_validator native_execution=0\n",
           (unsigned)REVISION_INIT_CASE_COUNT, cases, assertions);
    return 0;
}
