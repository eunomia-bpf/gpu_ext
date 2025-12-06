/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Common definitions for GPU memory eviction policies
 *
 * This header is shared between:
 * - BPF programs (eviction_*.bpf.c)
 * - Userspace loaders (eviction_*.c)
 */

#ifndef __EVICTION_COMMON_H__
#define __EVICTION_COMMON_H__

/* Configuration map keys - common across all eviction policies */
#define CONFIG_PRIORITY_PID 0       /* High priority PID */
#define CONFIG_PRIORITY_PARAM 1     /* High priority parameter (quota %, decay N, etc.) */
#define CONFIG_LOW_PRIORITY_PID 2   /* Low priority PID */
#define CONFIG_LOW_PRIORITY_PARAM 3 /* Low priority parameter */
#define CONFIG_DEFAULT_PARAM 4      /* Default parameter for other PIDs */

/*
 * Per-PID statistics structure - unified across all eviction policies
 *
 * This structure tracks chunk allocation and policy decisions.
 * The policy_* fields have different meanings depending on the policy:
 *
 * For quota-based policy:
 *   - policy_allow = in_quota (moved to tail, protected)
 *   - policy_deny  = over_quota (not moved, easier to evict)
 *
 * For frequency decay policy:
 *   - policy_allow = moved (moved to tail when decay threshold reached)
 *   - policy_deny  = skipped (not moved, access count not at threshold)
 */
struct pid_chunk_stats {
    __u64 current_count;    /* Current active chunk count */
    __u64 total_activate;   /* Total chunks activated */
    __u64 total_used;       /* Total chunk_used calls */
    __u64 policy_allow;     /* Times policy allowed move (protected) */
    __u64 policy_deny;      /* Times policy denied move (easier to evict) */
};

#endif /* __EVICTION_COMMON_H__ */
