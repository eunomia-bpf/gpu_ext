/* SPDX-License-Identifier: Apache-2.0 */
/* Selection rule from MoE-Infinity b766f8f, ExpertDispatcher::FindExpertEvict.
 * Host userspace eBPF/JIT only; no page policy or CUDA/migration actuator. */
#define MEP_BPF_ONLY
#include "moe_expert_policy.h"

mep_u64 moe_expert_select(struct moe_expert_snapshot *snapshot)
{
    if (snapshot->abi_version != MOE_EXPERT_POLICY_ABI || snapshot->reserved ||
        snapshot->count > MOE_EXPERT_MAX_CANDIDATES)
        return MOE_EXPERT_INVALID;
    mep_u64 minimum = 2147483647ULL; /* Original uint64_t min = INT_MAX. */
    mep_u64 selected = MOE_EXPERT_NONE;
#pragma clang loop unroll(disable)
    for (mep_u32 index = 0; index < snapshot->count; ++index) {
        const struct moe_expert_candidate *entry = &snapshot->entries[index];
        if (entry->reserved || (entry->flags & ~MOE_EXPERT_ELIGIBLE))
            return MOE_EXPERT_INVALID;
        if (entry->flags == MOE_EXPERT_ELIGIBLE && entry->incache_visit_count < minimum) {
            selected = index;
            minimum = entry->incache_visit_count;
        }
    }
    return selected;
}
