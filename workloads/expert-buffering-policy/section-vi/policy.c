/* The identical translation unit is compiled as native C and actual eBPF. */
#include "policy.h"

eb_u64 eb_select(struct eb_context *ctx)
{
    if (!ctx) return EB_INVALID;
    const struct eb_input *in = &ctx->input;
    struct eb_output *out = &ctx->output;
    out->batch_epoch = in->batch_epoch;
    out->status = EB_INVALID;
    out->victim = EB_NO_VICTIM;
    if (in->abi_version != EB_ABI_VERSION || !in->count ||
        in->count > EB_MAX_EXPERTS || !in->capacity ||
        in->capacity > in->count || in->incoming >= in->count ||
        !in->batch_epoch)
        return out->status;

    eb_u32 resident = 0;
    eb_u32 victim = EB_NO_VICTIM;
    eb_u32 victim_active = 1;
    eb_u64 latest = 0;
    for (eb_u32 i = 0; i < in->count; ++i) {
        const struct eb_entry *entry = &in->experts[i];
        if (entry->flags & ~(EB_RESIDENT | EB_ELIGIBLE)) return out->status;
        if (!(entry->flags & EB_RESIDENT)) {
            if (entry->flags || entry->admission) return out->status;
            continue;
        }
        if (!entry->admission) return out->status;
        ++resident;
        if (!(entry->flags & EB_ELIGIBLE)) continue;
        eb_u32 active = entry->token_count != 0;
        if (victim == EB_NO_VICTIM || active < victim_active ||
            (active == victim_active && entry->admission > latest)) {
            victim = i;
            victim_active = active;
            latest = entry->admission;
        }
    }
    if (resident > in->capacity || !in->experts[in->incoming].token_count)
        return out->status;
    if (in->experts[in->incoming].flags & EB_RESIDENT) {
        out->status = EB_HIT;
    } else if (resident < in->capacity) {
        out->status = EB_ADMIT;
    } else if (victim != EB_NO_VICTIM) {
        out->status = EB_EVICT;
        out->victim = victim;
    } else {
        out->status = EB_BLOCKED;
    }
    return out->status;
}
