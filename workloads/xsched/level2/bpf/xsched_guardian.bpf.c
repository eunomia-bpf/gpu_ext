/* SPDX-License-Identifier: GPL-2.0 */
#include "xg_bpf_abi.h"

/* XSched Level-2 guardian rule as a device eBPF program (real compiled
 * eBPF, exported to a sm_120 device .func through the context-adapted
 * sass-kretprobe bpf_to_ptx exporter). The program consumes ONLY the
 * bounded scalar snapshot that the trusted trampoline materialized from
 * the device preempt buffer:
 *
 *   ctx[0] kernel_idx             ctx[1] block_and_type (low u32 block,
 *                                 high u32 launch type)
 *   ctx[2] global_exit_flag       ctx[3] recorded preempt_idx
 *   ctx[4] block_restore_flag     ctx[5] decision (write-back)
 *
 * The 6 x u64 = 48-byte context is exactly the verify-time PREVAIL context
 * size of the exported ABI. It contains no device pointers: this program
 * never dereferences device memory. Decision/actuation split unchanged:
 * the program records only the decision word (slot 5); the trusted
 * actuator performs the record writes, the fence/barrier and the thread
 * exit, exactly like the original trusted __threadfence_block /
 * __syncthreads / exit glue.
 */

#ifndef SEC
#define SEC(name) __attribute__((section(name), used))
#endif

SEC(XG_BPF_SECTION)
unsigned long long xsched_guardian(unsigned long long *ctx)
{
	unsigned long long kernel_idx = ctx[0];
	unsigned long long block_and_type = ctx[1];
	unsigned long long global_exit_flag = ctx[2];
	unsigned long long recorded = ctx[3];
	unsigned long long restore_flag = ctx[4];
	unsigned long long decision;

	if ((block_and_type >> 32) == XG_LAUNCH_RESUME) {
		/* restore_exec decision from the same per-block snapshot the
		 * native-C decision consumes */
		decision = (restore_flag == 0) ? XG_D_RESUME_SKIP : XG_D_RESUME_CLEAR;
		ctx[5] = decision;
		return decision;
	}

	/* check_preempt leader decision: a single consistent reader of the
	 * mutable global deactivation flag (the trampoline guarantees one
	 * leader per CTA), so every thread of the CTA later acts on one
	 * decision via the trusted barrier. */
	if (global_exit_flag == 0) {
		decision = XG_D_CLEAR_EXIT;
		ctx[5] = decision;
		return decision;
	}

	decision = XG_D_ABORT;
	if (recorded == 0) {
		decision |= XG_D_RECORD_IDX | XG_D_RECORD_RESTORE;
	} else if (recorded == kernel_idx) {
		decision |= XG_D_RECORD_RESTORE;
	}
	ctx[5] = decision;
	return decision;
}
