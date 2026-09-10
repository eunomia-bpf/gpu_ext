/* SPDX-License-Identifier: Apache-2.0 */
/* Governor admission rule: admit the largest whole-expert ranked prefix that
 * fits under budget minus exact outstanding speculative bytes. Integer-only. */
#define MOE_SPEC_BPF_ONLY
#include "moe_spec_admission.h"

mep_u64 moe_spec_admission(struct moe_spec_snapshot *snapshot)
{
    if (snapshot->abi_version != MOE_SPEC_ADMISSION_ABI || snapshot->reserved ||
        snapshot->count > MOE_SPEC_MAX_CANDIDATES)
        return MOE_SPEC_INVALID;
#pragma clang loop unroll(disable)
    for (mep_u32 index = 0; index < snapshot->count; ++index) {
        if (snapshot->entries[index].reserved)
            return MOE_SPEC_INVALID;
    }
    /* Saturating headroom: budget - outstanding, floored at zero. */
    const mep_u64 budget = snapshot->budget_bytes;
    const mep_u64 outstanding = snapshot->outstanding_bytes;
    const mep_u64 allowed = outstanding < budget ? budget - outstanding : 0;
    mep_u64 used = 0;
    mep_u64 admitted = 0;
#pragma clang loop unroll(disable)
    for (mep_u32 index = 0; index < snapshot->count; ++index) {
        const mep_u64 bytes = snapshot->entries[index].payload_bytes;
        const mep_u64 next = used + bytes; /* candidates are < 2 GiB, no overflow */
        if (bytes == 0 || next > allowed) break;
        used = next;
        admitted = index + 1;
    }
    return admitted;
}
