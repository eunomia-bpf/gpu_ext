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
 *
 * The rule is expressed branchless (straight-line u64 arithmetic, no
 * conditional jumps): the strict GPU warp-uniform branch verifier rejects
 * branch predicates that vary per lane, and no snapshot value may be
 * marked uniform to paper over that. The identical policy of the native-C
 * decision is therefore computed with: a nonzero mask via (x | -x) >> 63,
 * full-u64 equality via XOR of the compared values, and decision
 * selection via a (0 - c) all-bits/zero mask. Each computed mask value
 * passes an empty volatile asm compiler barrier, which keeps the idiom
 * from being re-folded into a comparison (and hence a conditional jump)
 * by the BPF backend while the emitted instructions remain plain
 * straight-line ALU ops. Semantics are unchanged, including the
 * recorded==0 precedence over recorded==kernel_idx in the else-if chain.
 */

#ifndef SEC
#define SEC(name) __attribute__((section(name), used))
#endif

/* Branchless u64 idioms as straight-line BPF ALU statements; each forms
 * no conditional jump. The 0/1 equality mask comes from the operand XOR
 * and the (x | -x) >> 63 nonzero mask; the select mask is the all-bits/
 * zero value 0 - c. The empty volatile asm barrier keeps each computed
 * piece in a BPF register so the compiler cannot fold the idiom back
 * into a comparison and hence into a conditional branch:
 *   XG_EQ_ASSIGN(m, a, b)    : m = 1 if a == b else 0 (full u64 equality)
 *   XG_SEL_ASSIGN(o, c, t,f) : o = t if c is 1 else f (c in {0,1})
 */
#define XG_EQ_ASSIGN(m, a, b)                                            \
	do {                                                               \
		unsigned long long xg_d = (a) ^ (b);                           \
		unsigned long long xg_neg = 0ull - xg_d;                       \
		asm volatile("" : "+r"(xg_neg));                                \
		(m) = 1ull - ((xg_d | xg_neg) >> 63);                           \
	} while (0)

#define XG_SEL_ASSIGN(o, c, t, f)                                         \
	do {                                                                \
		unsigned long long xg_mask = 0ull - (c);                       \
		asm volatile("" : "+r"(xg_mask));                               \
		(o) = (xg_mask & ((t) ^ (f))) ^ (f);                             \
	} while (0)

SEC(XG_BPF_SECTION)
unsigned long long xsched_guardian(unsigned long long *ctx)
{
	unsigned long long kernel_idx = ctx[0];
	unsigned long long block_and_type = ctx[1];
	unsigned long long global_exit_flag = ctx[2];
	unsigned long long recorded = ctx[3];
	unsigned long long restore_flag = ctx[4];
	unsigned long long decision;
	unsigned long long m_resume, m_restore0, m_exit0, m_rec0, m_receq;
	unsigned long long d_resume, d_preempt, rec_restore, rec_bits;

	/* 0/1 masks from the full u64 snapshot values */
	XG_EQ_ASSIGN(m_resume, block_and_type >> 32, XG_LAUNCH_RESUME);
	XG_EQ_ASSIGN(m_restore0, restore_flag, 0ull);
	XG_EQ_ASSIGN(m_exit0, global_exit_flag, 0ull);
	XG_EQ_ASSIGN(m_rec0, recorded, 0ull);
	XG_EQ_ASSIGN(m_receq, recorded, kernel_idx);

	/* restore_exec decision from the same per-block snapshot the
	 * native-C decision consumes */
	XG_SEL_ASSIGN(d_resume, m_restore0, XG_D_RESUME_SKIP,
	              XG_D_RESUME_CLEAR);

	/* check_preempt record bits with the native else-if precedence:
	 * recorded == 0 takes the full record set even when kernel_idx is
	 * zero */
	XG_SEL_ASSIGN(rec_restore, m_receq, XG_D_RECORD_RESTORE, 0ull);
	XG_SEL_ASSIGN(rec_bits, m_rec0, XG_D_RECORD_IDX | XG_D_RECORD_RESTORE,
	              rec_restore);
	d_preempt = XG_D_ABORT | rec_bits;

	XG_SEL_ASSIGN(decision, m_exit0, XG_D_CLEAR_EXIT, d_preempt);
	XG_SEL_ASSIGN(decision, m_resume, d_resume, decision);

	/* Single write-back to slot 5, exactly as the original */
	ctx[5] = decision;
	return decision;
}
