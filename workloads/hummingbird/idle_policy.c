/* SPDX-License-Identifier: GPL-2.0 */
#include "idle_policy.h"

/* Paper-described Hummingbird Algorithm 1 component, with the explicit
 * nonblocking event-completion guard documented in plan.md. This function
 * issues no CUDA operation; the common asynchronous executor applies actions.
 */
hb_u64 hb_decide(struct hb_call *call, unsigned long length)
{
    if (!call || length != sizeof(*call)) return HB_ERROR;
    const struct hb_input *in = &call->input;
    struct hb_output *out = &call->output;
    out->next_tick_ns = 0;
    out->action = HB_ERROR;
    out->bubble = HB_NO_BUBBLE;
    out->wait_reason = HB_NOT_WAITING;
    out->reserved = 0;
    if (in->hp_gpu_done > 1 || in->small_active > 1 || in->small_start_done > 1 ||
        in->lp_pending > 1 || in->lp_gpu_done > 1 || in->kernel_unstarted > 1 ||
        in->consolidate > 1 || in->now_ns < in->last_hp_activity_ns ||
        !in->large_after_ns || !in->split_ns || !in->whole_ns)
        return HB_ERROR;

    /* HP arrival closes LP admission before considering timers or LP progress. */
    if (in->hp_pending) {
        out->action = HB_STOP_LP;
        return out->action;
    }
    out->action = HB_WAIT;
    if (!in->lp_pending) {
        out->wait_reason = HB_WAIT_EMPTY;
        return out->action;
    }
    if (!in->hp_gpu_done) {
        out->wait_reason = HB_WAIT_HP;
        return out->action;
    }
    if (in->now_ns - in->last_hp_activity_ns >= in->large_after_ns)
        out->bubble = HB_LARGE_BUBBLE;
    else if (in->small_active && in->small_start_done)
        out->bubble = HB_SMALL_BUBBLE;
    if (!out->bubble) {
        out->wait_reason = HB_WAIT_BUBBLE;
        return out->action;
    }
    if (in->now_ns < in->tick_due_ns) {
        out->next_tick_ns = in->tick_due_ns;
        out->wait_reason = HB_WAIT_TICK;
        return out->action;
    }
    /* A profiled tick does not establish completion. Underprediction must not
     * accumulate launches. Count this wait separately in the real executor. */
    if (!in->lp_gpu_done) {
        out->wait_reason = HB_WAIT_LP_EVENT;
        return out->action;
    }
    out->action = out->bubble == HB_LARGE_BUBBLE && in->consolidate && in->kernel_unstarted
                ? HB_WHOLE : HB_SPLIT;
    const hb_u64 duration = out->action == HB_WHOLE ? in->whole_ns : in->split_ns;
    const hb_u64 interval = duration > in->launch_overhead_ns
                         ? duration - in->launch_overhead_ns : 0;
    if (in->now_ns > ~(hb_u64)0 - interval) {
        out->action = HB_ERROR;
        return HB_ERROR;
    }
    out->next_tick_ns = in->now_ns + interval;
    return out->action;
}
