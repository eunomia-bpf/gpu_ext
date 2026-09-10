/* SPDX-License-Identifier: Apache-2.0 */
/* Governor admission snapshot ABI for the speculation byte-budget rule. */
#ifndef MOE_SPEC_ADMISSION_H
#define MOE_SPEC_ADMISSION_H

#include "moe_expert_policy.h"

#define MOE_SPEC_ADMISSION_ABI 1U
#define MOE_SPEC_MAX_CANDIDATES 65536U
#define MOE_SPEC_NONE (~(mep_u64)0)
#define MOE_SPEC_INVALID (MOE_SPEC_NONE - 1)

/* Ranked whole-expert candidate prefix, already sorted by the shared frontend.
 * The bridge never re-ranks: BPF only decides the admitted prefix length k. */
struct moe_spec_candidate {
    mep_u64 payload_bytes;
    mep_u64 reserved;
};

struct moe_spec_snapshot {
    mep_u64 budget_bytes;    /* current adaptive budget */
    mep_u64 outstanding_bytes; /* exact queued+inflight speculative bytes */
    mep_u32 abi_version;
    mep_u32 count;
    mep_u64 reserved;
    struct moe_spec_candidate entries[];
};

struct moe_spec_admission_stats {
    mep_u64 calls, candidates, admitted, empty, errors;
};

#ifndef MOE_SPEC_BPF_ONLY
#ifdef __cplusplus
extern "C" {
#endif
/* init/select execute only real ubpf JIT, never a C fallback. Negative returns
 * are fatal to the caller. A successful NONE means "admit nothing". */
int moe_spec_admission_init_v1(const char *absolute_bytecode_path);
int moe_spec_admission_prefix_v1(const struct moe_spec_candidate *entries,
                                 mep_u32 count, mep_u64 budget_bytes,
                                 mep_u64 outstanding_bytes, mep_u64 *admitted);
void moe_spec_admission_stats_v1(struct moe_spec_admission_stats *output);
#ifdef __cplusplus
}
#endif
#endif
#endif
