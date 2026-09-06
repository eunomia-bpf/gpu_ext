// SPDX-License-Identifier: GPL-2.0
/*
 * GDS storage scheduling policy for the custom nvidia_uvm module (575).
 *
 * Implements the gpu_storage_ops struct_ops callback gpu_storage_decide()
 * with the same deterministic precedence and bounds as policy.py's
 * matched_native_decide() contract:
 *
 *  1. demand read                    -> SUBMIT_NOW
 *  2. cheaper recomputable read that fits in the deadline slack -> RECOMPUTE
 *  3. speculative + safe-to-defer read at HBM pressure >= 800 permille
 *                                    -> DEFER (unbatched)
 *  4. safe-to-defer write at HBM pressure >= 600 permille
 *                                    -> DEFER (batched to the queue depth)
 *  5. everything else                -> SUBMIT_NOW
 *
 * Every decision is recorded with bpf_gpu_storage_record(), which clamps
 * defer_ns to the 10 ms maximum, priority to <= 7, and batch_target to
 * 1..64 exactly like the policy contract (MIN/MAX_DEFER_NS,
 * MIN/MAX_BATCH_SIZE). Flags/ops/actions mirror the live ABI:
 * DEMAND=1 SPECULATIVE=2 RECOMPUTABLE=4 SAFE_TO_DEFER=8,
 * READ=0 WRITE=1, SUBMIT_NOW=0 DEFER=1 RECOMPUTE=2.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

/* Storage request ABI mirror of the live nvidia_uvm BTF
 * (/sys/kernel/btf/nvidia_uvm: uvm_bpf_storage_request, decision, ctx,
 * gpu_storage_ops, bpf_gpu_storage_record). Field names, types, and
 * layout are fixed by uvm_bpf_struct_ops.h; preserve_access_index keeps
 * the CO-RE relocations against the module BTF. */
#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
#endif

#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif

typedef unsigned int NvU32;
typedef unsigned long long NvU64;

typedef struct uvm_bpf_storage_request {
	NvU32 abi_version;
	NvU32 op;
	NvU32 request_flags;
	NvU32 input_priority;
	NvU64 request_id;
	NvU64 object_id;
	NvU64 bytes;
	NvU64 tenant_id;
	NvU64 caller_hint;
	NvU64 deadline_ns;
	NvU64 slack_ns;
	NvU64 estimated_transfer_ns;
	NvU64 recompute_ns;
	NvU32 queue_depth;
	NvU32 hbm_pressure_permille;
} uvm_bpf_storage_request_t;

typedef struct uvm_bpf_storage_decision {
	NvU32 action;
	NvU64 defer_ns;
	NvU32 priority;
	NvU32 batch_target;
} uvm_bpf_storage_decision_t;

typedef struct uvm_bpf_storage_decision_ctx {
	uvm_bpf_storage_request_t request;
	uvm_bpf_storage_decision_t decision;
	NvU32 recorded;
} uvm_bpf_storage_decision_ctx_t;

struct gpu_storage_ops {
	int (*gpu_storage_decide)(uvm_bpf_storage_decision_ctx_t *decision_ctx);
};

/* Kfunc from the live nvidia_uvm module (BTF_KFUNCS uvm_bpf_kfunc_ids_set). */
extern int bpf_gpu_storage_record(uvm_bpf_storage_decision_ctx_t *decision_ctx,
				  u32 action, u64 defer_ns,
				  u32 priority, u32 batch_target) __weak __ksym;

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif

/* Fixed-width ABI and policy constants (policy.py / uvm_gpu_storage). */
#define GDS_ABI_VERSION			1

#define OP_READ				0
#define OP_WRITE			1

#define FLAG_DEMAND			1
#define FLAG_SPECULATIVE		2
#define FLAG_RECOMPUTABLE		4
#define FLAG_SAFE_TO_DEFER		8

#define ACTION_SUBMIT_NOW		0
#define ACTION_DEFER			1
#define ACTION_RECOMPUTE		2

#define MIN_DEFER_NS			0
#define MAX_DEFER_NS			10000000ULL
#define MAX_INPUT_PRIORITY		7
#define MIN_BATCH_SIZE			1
#define MAX_BATCH_SIZE			64

#define READ_DEFER_PRESSURE_PERMILLE	800
#define WRITE_DEFER_PRESSURE_PERMILLE	600

/* priority: the request's input priority, clamped by the kfunc to <= 7. */
static void gds_submit(uvm_bpf_storage_decision_ctx_t *ctx, u32 priority)
{
	bpf_gpu_storage_record(ctx, ACTION_SUBMIT_NOW, MIN_DEFER_NS,
			       priority, MIN_BATCH_SIZE);
}

static void gds_recompute(uvm_bpf_storage_decision_ctx_t *ctx, u32 priority)
{
	bpf_gpu_storage_record(ctx, ACTION_RECOMPUTE, MIN_DEFER_NS,
			       priority, MIN_BATCH_SIZE);
}

/* batched == 0: read defer keeps batch_size at the minimum;
 * batched == 1: batch_size follows the queue depth (1..64). */
static void gds_defer(uvm_bpf_storage_decision_ctx_t *ctx, u32 priority,
		      u64 slack_ns, u32 queue_depth, int batched)
{
	u64 defer_ns = slack_ns;
	u32 batch_size = MIN_BATCH_SIZE;

	if (defer_ns > MAX_DEFER_NS)
		defer_ns = MAX_DEFER_NS;
	if (batched) {
		batch_size = queue_depth;
		if (batch_size < MIN_BATCH_SIZE)
			batch_size = MIN_BATCH_SIZE;
		else if (batch_size > MAX_BATCH_SIZE)
			batch_size = MAX_BATCH_SIZE;
	}
	bpf_gpu_storage_record(ctx, ACTION_DEFER, defer_ns,
			       priority, batch_size);
}

SEC("struct_ops/gpu_storage_decide")
int BPF_PROG(gds_gpu_storage_decide,
	     uvm_bpf_storage_decision_ctx_t *decision_ctx)
{
	u32 flags, op, priority, pressure;
	u64 recompute_ns, transfer_ns, slack_ns;

	if (!decision_ctx)
		return 0;

	flags = decision_ctx->request.request_flags;
	op = decision_ctx->request.op;
	priority = decision_ctx->request.input_priority;
	pressure = decision_ctx->request.hbm_pressure_permille;
	recompute_ns = decision_ctx->request.recompute_ns;
	transfer_ns = decision_ctx->request.estimated_transfer_ns;
	slack_ns = decision_ctx->request.slack_ns;

	/* Demand reads are always submitted immediately. */
	if (op == OP_WRITE)
		goto write;

	if (flags & FLAG_DEMAND) {
		gds_submit(decision_ctx, priority);
		return 0;
	}

	/* Cheaper recomputable read that still finishes within the
	 * deadline slack is recomputed instead of fetched. */
	if (flags & FLAG_RECOMPUTABLE) {
		if (recompute_ns < transfer_ns && recompute_ns <= slack_ns) {
			gds_recompute(decision_ctx, priority);
			return 0;
		}
	}

	/* Speculative reads under high HBM pressure: defer unbatched. */
	if ((flags & FLAG_SPECULATIVE) && (flags & FLAG_SAFE_TO_DEFER) &&
	    pressure >= READ_DEFER_PRESSURE_PERMILLE) {
		gds_defer(decision_ctx, priority, slack_ns, 0, 0);
		return 0;
	}

	gds_submit(decision_ctx, priority);
	return 0;

write:
	if (!(op == OP_WRITE))
		/* Unknown op falls through to the SUBMIT_NOW default. */
		goto submit;

	/* Safe-to-defer writes under high HBM pressure: defer batched to
	 * the current queue depth. */
	if ((flags & FLAG_SAFE_TO_DEFER) &&
	    pressure >= WRITE_DEFER_PRESSURE_PERMILLE) {
		gds_defer(decision_ctx, priority, slack_ns,
			  decision_ctx->request.queue_depth, 1);
		return 0;
	}

submit:
	gds_submit(decision_ctx, priority);
	return 0;
}

/* The struct_ops map registered against the live gpu_storage_ops type. */
SEC(".struct_ops")
struct gpu_storage_ops gds_ops = {
	.gpu_storage_decide = (void *)gds_gpu_storage_decide,
};
