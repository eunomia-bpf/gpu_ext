/* SPDX-License-Identifier: Apache-2.0 */
#ifndef MOE_EXPERT_POLICY_H
#define MOE_EXPERT_POLICY_H

typedef unsigned long long mep_u64;
typedef unsigned int mep_u32;

#define MOE_EXPERT_POLICY_ABI 1U
#define MOE_EXPERT_MAX_CANDIDATES 65536U
#define MOE_EXPERT_NONE (~(mep_u64)0)
#define MOE_EXPERT_INVALID (MOE_EXPERT_NONE - 1)
#define MOE_EXPERT_NODE_PRESENT 1U
#define MOE_EXPERT_DEVICE_CUDA 2U
#define MOE_EXPERT_PENDING_ZERO 4U
#define MOE_EXPERT_EXEC_IDLE 8U
#define MOE_EXPERT_ELIGIBLE 15U

/* Exactly the original cached_experts_[gpu] iteration order. Identity is the
 * existing (layer << 32) | expert key; never sort or renumber candidates. */
struct moe_expert_candidate {
    mep_u64 identity;
    mep_u64 incache_visit_count;
    mep_u32 flags;
    mep_u32 reserved;
};

struct moe_expert_snapshot {
    mep_u32 abi_version;
    mep_u32 count;
    mep_u64 reserved;
    struct moe_expert_candidate entries[];
};

struct moe_expert_policy_stats {
    mep_u64 calls, candidates, selected, no_victim, errors;
};

#ifndef MEP_BPF_ONLY
#ifdef __cplusplus
extern "C" {
#endif
/* init/select execute only real ubpf JIT, never a C fallback. Negative returns
 * are fatal to the caller. A successful NONE is the original nullptr outcome.
 * A null init path uses absolute MOE_EXPERT_POLICY_CODE; one program per process. */
int moe_expert_policy_init_v1(const char *absolute_bytecode_path);
int moe_expert_policy_select_v1(const struct moe_expert_candidate *entries,
                               mep_u32 count, mep_u64 *selected_index);
void moe_expert_policy_stats_v1(struct moe_expert_policy_stats *output);
#ifdef __cplusplus
}
#endif
#endif
#endif
