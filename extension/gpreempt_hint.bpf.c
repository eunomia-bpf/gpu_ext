/* SPDX-License-Identifier: GPL-2.0 */
#define GP_BPF_ONLY
#include "gpreempt_bridge.h"

/* Host-JIT policy only: CUDA launches and GDRCopy writes stay upstream.
 * The daemon deliberately passes the SAME system_clock epoch for both times.
 * Do not replace this strict comparison with bpf_ktime_get_ns(). */
gp_u64 gpreempt_hint(const struct gp_hint_input *in)
{
    if (in->role > GP_BE || in->initialized > 1 || in->reserve > 1 ||
        in->event < GP_PREPROCESS || in->event > GP_INFER)
        return ~(gp_u64)0;
    if (in->role != GP_LC || !in->initialized)
        return 0;
    if (in->event == GP_PREPROCESS)
        return GP_RESET | (in->reserve ? GP_HINT : GP_BLOCK);
    if (in->event == GP_DUE)
        return in->now_ns > in->deadline_ns ? GP_BLOCK : 0;
    return GP_RELEASE;
}
