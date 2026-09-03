/* SPDX-License-Identifier: Apache-2.0 */
/* Select all maximum cosine similarities in original order, entirely in BPF.
 * Predictor uses every tie; full EAMC replacement uses the first returned index.
 * Cosine tensor arithmetic is shared frontend work, not claimed as BPF. */
#define MEP_BPF_ONLY
#include "moe_expert_policy.h"

static __inline __attribute__((always_inline)) mep_u64 order_key(mep_u64 bits)
{
    if ((bits << 1) == 0) bits = 0; /* +/-0 equality */
    return bits >> 63 ? ~bits : bits ^ 0x8000000000000000ULL;
}

mep_u64 moe_expert_match(struct moe_expert_rank_snapshot *snapshot)
{
    const mep_u32 count = snapshot->count;
    if (snapshot->abi_version != MOE_EXPERT_POLICY_ABI || snapshot->reserved ||
        count > MOE_EXPERT_MAX_CANDIDATES)
        return MOE_EXPERT_INVALID;
    mep_u32 *indices = (mep_u32 *)(snapshot->entries + count);
    mep_u64 maximum = order_key(0xfff0000000000000ULL); /* -Inf sentinel */
    mep_u32 matched = 0;
#pragma clang loop unroll(disable)
    for (mep_u32 i = 0; i < count; ++i) {
        const struct moe_expert_rank_candidate *entry = &snapshot->entries[i];
        if (entry->reserved || entry->ordinal != i) return MOE_EXPERT_INVALID;
        const mep_u64 bits = entry->score_bits;
        if ((bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) continue;
        const mep_u64 key = order_key(bits);
        if (key > maximum) {
            maximum = key;
            matched = 0;
        }
        if (key == maximum) indices[matched++] = i;
    }
    return matched;
}
