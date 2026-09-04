/* SPDX-License-Identifier: GPL-2.0 */
#ifndef REVISION_INIT_REQUESTS_H
#define REVISION_INIT_REQUESTS_H

#include "revision_init_records.h"

/* One request sequence is shared by BPF fixtures and the CPU test of the
 * production 575 recorder/validator. No native actuator is mocked here. */
enum revision_init_case {
    REVISION_INIT_NO_REQUEST = 0,
    REVISION_INIT_LEGAL,
    REVISION_INIT_INVALID_INTERLEAVE,
    REVISION_INIT_DUPLICATE,
    REVISION_INIT_CONFLICT,
    REVISION_INIT_INDEPENDENT_INTERLEAVE,
    REVISION_INIT_INDEPENDENT_TIMESLICE,
    REVISION_INIT_CASE_COUNT,
};

/* Callers provide a zeroed result and the two bpf_nv_gpu_set_* functions.
 * D is a known native default, not an invented live minimum. D^1 differs
 * from D even at UINT64_MAX; conflict validation takes precedence over range. */
static inline void revision_init_issue_requests(
    enum revision_init_case which, struct nv_gpu_task_init_ctx *ctx,
    struct revision_init_returns *result)
{
    const unsigned long long d = ctx->default_timeslice;

#define REQUEST_TS(slot, value) do {                                  \
    result->timeslice[slot] = bpf_nv_gpu_set_timeslice(ctx, value);     \
    ++result->timeslice_count;                                        \
} while (0)
#define REQUEST_IL(slot, value) do {                                  \
    result->interleave[slot] = bpf_nv_gpu_set_interleave(ctx, value);   \
    ++result->interleave_count;                                       \
} while (0)

    switch (which) {
    case REVISION_INIT_NO_REQUEST:
        break;
    case REVISION_INIT_LEGAL:
        REQUEST_TS(0, d);
        REQUEST_IL(0, 0);
        break;
    case REVISION_INIT_INVALID_INTERLEAVE:
        REQUEST_IL(0, 3);
        break;
    case REVISION_INIT_DUPLICATE:
        REQUEST_TS(0, d);
        REQUEST_TS(1, d);
        REQUEST_IL(0, 0);
        REQUEST_IL(1, 0);
        break;
    case REVISION_INIT_CONFLICT:
        REQUEST_TS(0, d);
        REQUEST_TS(1, d ^ 1ULL);
        REQUEST_TS(2, d);
        REQUEST_IL(0, 0);
        REQUEST_IL(1, 2);
        REQUEST_IL(2, 0);
        break;
    case REVISION_INIT_INDEPENDENT_INTERLEAVE:
        REQUEST_TS(0, d);
        REQUEST_IL(0, 3);
        break;
    case REVISION_INIT_INDEPENDENT_TIMESLICE:
        REQUEST_TS(0, d);
        REQUEST_TS(1, d ^ 1ULL);
        REQUEST_IL(0, 0);
        break;
    case REVISION_INIT_CASE_COUNT:
        break;
    }

#undef REQUEST_TS
#undef REQUEST_IL
}
#endif
