/* SPDX-License-Identifier: Apache-2.0 */
/* Probability-scored eviction for the explicit paper-v3 algorithm port.
 * IEEE-754 ordering uses integers; no unsupported eBPF FP instructions. */
#define MEP_BPF_ONLY
#include "moe_expert_policy.h"

static __inline __attribute__((always_inline)) mep_u64 order_key(mep_u64 bits)
{
    /* Numeric +/-0 equality is essential for retaining the original first tie. */
    if ((bits << 1) == 0) bits = 0;
    return bits >> 63 ? ~bits : bits ^ 0x8000000000000000ULL;
}

mep_u64 moe_expert_scored(struct moe_expert_scored_snapshot *snapshot)
{
    if (snapshot->abi_version != MOE_EXPERT_POLICY_ABI || snapshot->reserved ||
        snapshot->count > MOE_EXPERT_MAX_CANDIDATES)
        return MOE_EXPERT_INVALID;
    mep_u64 minimum = order_key(0x7ff0000000000000ULL); /* +Inf sentinel */
    mep_u64 selected = MOE_EXPERT_NONE;
#pragma clang loop unroll(disable)
    for (mep_u32 index = 0; index < snapshot->count; ++index) {
        const struct moe_expert_scored_candidate *entry = &snapshot->entries[index];
        if (entry->reserved || (entry->flags & ~MOE_EXPERT_ELIGIBLE))
            return MOE_EXPERT_INVALID;
        const mep_u64 bits = entry->score_bits;
        if ((bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) continue; /* NaN */
        const mep_u64 key = order_key(bits);
        if (entry->flags == MOE_EXPERT_ELIGIBLE && key < minimum) {
            minimum = key;
            selected = index;
        }
    }
    return selected;
}
