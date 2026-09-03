/* SPDX-License-Identifier: Apache-2.0 */
/* Positive-score filtering and stable descending merge sort entirely in BPF.
 * Input order is the unsorted, unfiltered layer-major frontend order. */
#define MEP_BPF_ONLY
#include "moe_expert_policy.h"

mep_u64 moe_expert_rank(struct moe_expert_rank_snapshot *snapshot)
{
    const mep_u32 count = snapshot->count;
    if (snapshot->abi_version != MOE_EXPERT_POLICY_ABI || snapshot->reserved ||
        count > MOE_EXPERT_MAX_CANDIDATES)
        return MOE_EXPERT_INVALID;
    mep_u32 *indices = (mep_u32 *)(snapshot->entries + count);
    mep_u32 *scratch = indices + count;
    mep_u32 positive = 0;
#pragma clang loop unroll(disable)
    for (mep_u32 i = 0; i < count; ++i) {
        const struct moe_expert_rank_candidate *entry = &snapshot->entries[i];
        if (entry->reserved || entry->ordinal != i) return MOE_EXPERT_INVALID;
        // Positive IEEE-754 encodings are numerically ordered; +Inf qualifies,
        // whereas positive NaNs sort above +Inf and must explicitly be excluded.
        if (entry->score_bits > 0 && entry->score_bits <= 0x7ff0000000000000ULL)
            indices[positive++] = i;
    }
    // Bottom-up stable merge sort: O(n log n), caller-owned bounded scratch.
    // Equal scores take the left element, retaining original input order.
    mep_u32 *from = indices;
    mep_u32 *to = scratch;
#pragma clang loop unroll(disable)
    for (mep_u32 width = 1; width < positive; width *= 2) {
#pragma clang loop unroll(disable)
        for (mep_u32 start = 0; start < positive; start += 2 * width) {
            mep_u32 middle = start + width;
            if (middle > positive) middle = positive;
            mep_u32 end = start + 2 * width;
            if (end > positive) end = positive;
            mep_u32 left = start, right = middle;
#pragma clang loop unroll(disable)
            for (mep_u32 dest = start; dest < end; ++dest) {
                if (right >= end || (left < middle &&
                    snapshot->entries[from[left]].score_bits >=
                    snapshot->entries[from[right]].score_bits))
                    to[dest] = from[left++];
                else
                    to[dest] = from[right++];
            }
        }
        mep_u32 *swap = from;
        from = to;
        to = swap;
    }
    if (from != indices) {
#pragma clang loop unroll(disable)
        for (mep_u32 i = 0; i < positive; ++i) indices[i] = from[i];
    }
    return positive;
}
