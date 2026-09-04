/* SPDX-License-Identifier: GPL-2.0 */
#include "stale_state_policy_jit.h"

/*
 * Host-uBPF entry point for the future stale-state consumer. This program is
 * deliberately clock-free and state-free: the driver must supply one atomic
 * snapshot and the decision timestamp. An ABI mismatch fails closed.
 */
uint64_t stale_state_575_bpf(void *memory, uint64_t length)
{
    struct stale_state_575_jit_context *context = memory;
    enum stale_state_575_action action;

    if (context == 0 || length != sizeof(*context))
        return STALE_STATE_575_ACTION_REJECT;

    context->decision = (struct stale_state_575_decision){0};
    context->status = STALE_STATE_575_ACTION_REJECT;
    if (context->reserved != 0)
        return STALE_STATE_575_ACTION_REJECT;

    action = stale_state_575_choose(&context->snapshot,
                                    context->decision_mono_ns,
                                    &context->decision);
    context->status = (uint32_t)action;
    return (uint64_t)action;
}
