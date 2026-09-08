/* SPDX-License-Identifier: GPL-2.0 */
#include "xg_bpf_abi.h"

/* XSched Level-2 guardian rule as a device eBPF program (real compiled
 * eBPF, exported by ptxpass to a sm_120 device .func). This program computes
 * the bounded leader decision of the original guardian:
 *
 *   - check_preempt leader logic (inject.cu): read the per-hwQueue global
 *     deactivation flag and the recorded preempt_idx; decide whether this
 *     CTA aborts, and whether it records itself as the first preempted
 *     command (preempt_idx, block restore flag).
 *   - restore_exec decision (inject.cu): whether a resumed block ran to
 *     completion before deactivation (skip) or must run now (clear flag).
 *
 * It only reads device state and returns/records a decision word; the
 * trusted actuator performs the record writes, the fence/barrier and the
 * thread exit, exactly like the original trusted __threadfence_block /
 * __syncthreads / exit glue.
 */

#ifndef SEC
#define SEC(name) __attribute__((section(name), used))
#endif

SEC(XG_BPF_SECTION)
unsigned long long xsched_guardian(unsigned long long *ctx)
{
	unsigned long long buf = ctx[0];
	unsigned long long kernel_idx = ctx[1];
	unsigned long long block_and_type = ctx[2];
	unsigned long long block_idx = block_and_type & 0xffffffffULL;
	unsigned long long launch_type = block_and_type >> 32;
	unsigned long long decision;

	if (launch_type == XG_LAUNCH_RESUME) {
		/* restore_exec: all threads observe the flag (read-only, no
		 * concurrent writer for this block during a resume launch). */
		unsigned int *restore = (unsigned int *)(buf +
			4ULL * (2ULL * block_idx + 5ULL));
		decision = (*restore == 0) ? XG_D_RESUME_SKIP : XG_D_RESUME_CLEAR;
		ctx[3] = decision;
		return decision;
	}

	/* check_preempt leader decision: a single reader of the mutable
	 * global deactivation flag, so every thread of the CTA later acts on
	 * one consistent decision via the trusted barrier. */
	unsigned int *global_exit_flag = (unsigned int *)buf;
	if (*global_exit_flag == 0) {
		decision = XG_D_CLEAR_EXIT;
		ctx[3] = decision;
		return decision;
	}

	decision = XG_D_ABORT;
	unsigned long long *preempt_idx = (unsigned long long *)(buf + 8ULL);
	unsigned long long recorded = *preempt_idx;
	if (recorded == 0) {
		decision |= XG_D_RECORD_IDX | XG_D_RECORD_RESTORE;
	} else if (recorded == kernel_idx) {
		decision |= XG_D_RECORD_RESTORE;
	}
	ctx[3] = decision;
	return decision;
}
