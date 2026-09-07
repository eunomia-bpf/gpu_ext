// SPDX-License-Identifier: GPL-2.0
/*
 * KV reclaim candidate-selection policy for the custom nvidia_uvm module
 * (575).
 *
 * Implements the gpu_kv_reclaim_ops struct_ops callback
 * gpu_kv_reclaim_choose() on a bounded candidate vector (max 8,
 * fixed-width scalars only). The policy runs the shared bounded-integer
 * algorithm from kv_reclaim_abi.h - the same source the native entrypoint
 * and the mirrored kernel definitions use - to minimize estimated recovery
 * cost per actually freeable KV byte, restricted to the worst priority
 * class, with the stock victim kept on ties or unusable telemetry.
 *
 * The struct_ops ctx is BTF-typed and check_ptr_to_btf_access() in v6.15
 * rejects any non-constant variable offset on it, so the candidate vector
 * is first copied into a per-CPU snapshot map at unrolled fixed slots and
 * the shared algorithm runs against that snapshot.
 *
 * Only a decision that differs from the stock default is recorded with
 * bpf_kv_reclaim_record(). The kfunc is mechanism-only: it validates the
 * index against the kernel precomputed eligible mask (worst priority class,
 * positive freeable bytes, consistent telemetry), the echoed cookie, and
 * the route range, and clamps the saturating estimate - it does not enforce
 * any particular cost algorithm, so different policies are permitted.
 *
 * Priority keeps vLLM integer semantics: lower value is more important, the
 * worst class is the maximum value. The disk read is priced on the
 * provider's actual backed transfer bytes. No fd, file offset, GPU pointer,
 * arbitrary user pointer, stream, or completion crosses this hook.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

/* preserve_access_index keeps the CO-RE relocations against the live
 * nvidia_uvm module BTF for the shared context/decision types; the shared
 * header is included inside the region so its record types are annotated. */
#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
#endif

#include "kv_reclaim_abi.h"

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif

#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif

/* Kfunc from the live nvidia_uvm module (BTF_KFUNCS sets). */
extern int bpf_kv_reclaim_record(uvm_bpf_kv_reclaim_decision_ctx_t *decision_ctx,
 				 u32 index, u32 route, u64 cookie,
 				 u64 estimated_ns) __weak __ksym;

/* Per-CPU candidate snapshot. Plain key_size/value_size with no BTF types
 * on either side (a BTF key next to a BTF-less value is rejected at map
 * creation), so the value pointer carries no BTF type and bounded
 * variable offsets into it are verifier-allowed, unlike the BTF-typed
 * ctx. Per-CPU keeps concurrent ioctl callers from sharing a slot;
 * same-CPU callers run serialized. */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__uint(key_size, 4);
	__uint(value_size,
	       UVM_KV_RECLAIM_MAX_CANDIDATES * sizeof(uvm_bpf_kv_reclaim_candidate_t));
} kv_reclaim_snap SEC(".maps");

/* The shared struct_ops type registered by the live module; the map in the
 * .struct_ops section below is matched against this type by name. */
struct gpu_kv_reclaim_ops {
	int (*gpu_kv_reclaim_choose)(uvm_bpf_kv_reclaim_decision_ctx_t *decision_ctx);
};

/* Fixed-width field copy for one fixed candidate slot. The volatile-typed
 * source keeps every ctx load at its field width: the compiler cannot
 * merge adjacent u32 members (stock_index+pad0, priority+flags) into a
 * 64-bit load that would cross a BTF member boundary on the BTF-typed
 * ctx. Stores into the plain map value may merge freely. */
#define KV_RECLAIM_COPY_SLOT(k) do {					\
	snap[k].cookie = vctx->candidates[k].cookie;		\
	snap[k].freeable_bytes = vctx->candidates[k].freeable_bytes;	\
	snap[k].computed_tokens = vctx->candidates[k].computed_tokens;	\
	snap[k].disk_backed_tokens = vctx->candidates[k].disk_backed_tokens; \
	snap[k].disk_backed_bytes = vctx->candidates[k].disk_backed_bytes; \
	snap[k].priority = vctx->candidates[k].priority;	\
	snap[k].flags = vctx->candidates[k].flags;		\
} while (0)

SEC("struct_ops/gpu_kv_reclaim_choose")
int BPF_PROG(kv_reclaim_gpu_kv_reclaim_choose,
	     uvm_bpf_kv_reclaim_decision_ctx_t *decision_ctx)
{
	uvm_bpf_kv_reclaim_request_t req;
	uvm_bpf_kv_reclaim_decision_t d;
	uvm_bpf_kv_reclaim_candidate_t *snap;
	const volatile uvm_bpf_kv_reclaim_decision_ctx_t *vctx;
	u32 n, stock;
	u32 zero = 0;
	u8 *v;

	if (!decision_ctx)
		return 0;

	/* Fixed-width, constant-offset member reads through a
	 * volatile-typed pointer: each request field is its own load, so
	 * no 64-bit load can ever span two u32 members. */
	vctx = (const volatile uvm_bpf_kv_reclaim_decision_ctx_t *)decision_ctx;
	req.abi_version = vctx->request.abi_version;
	req.n_candidates = vctx->request.n_candidates;
	req.stock_index = vctx->request.stock_index;
	req.pad0 = vctx->request.pad0;
	req.disk_read_ns_per_kib = vctx->request.disk_read_ns_per_kib;
	req.recompute_ns_per_token = vctx->request.recompute_ns_per_token;

	n = req.n_candidates;
	if (n > UVM_KV_RECLAIM_MAX_CANDIDATES)
		n = UVM_KV_RECLAIM_MAX_CANDIDATES;
	stock = req.stock_index;
	if (n == 0 || stock >= n)
		return 0;

	/* Fixed-slot copy into the per-CPU snapshot, one field per load,
	 * guarded by the locally bounded n. Stale slots beyond n are
	 * never read because the shared algorithm only touches indexes
	 * < n. */
	v = bpf_map_lookup_elem(&kv_reclaim_snap, &zero);
	if (!v)
		return 0;
	snap = (uvm_bpf_kv_reclaim_candidate_t *)v;
	if (n > 0)
		KV_RECLAIM_COPY_SLOT(0);
	if (n > 1)
		KV_RECLAIM_COPY_SLOT(1);
	if (n > 2)
		KV_RECLAIM_COPY_SLOT(2);
	if (n > 3)
		KV_RECLAIM_COPY_SLOT(3);
	if (n > 4)
		KV_RECLAIM_COPY_SLOT(4);
	if (n > 5)
		KV_RECLAIM_COPY_SLOT(5);
	if (n > 6)
		KV_RECLAIM_COPY_SLOT(6);
	if (n > 7)
		KV_RECLAIM_COPY_SLOT(7);

	uvm_kv_reclaim_choose(&req, snap, n, stock, &d);

	/* Stock default: record nothing, the kernel keeps the stock victim. */
	if (d.route == UVM_KV_RECLAIM_ROUTE_STOCK)
		return 0;

	bpf_kv_reclaim_record(decision_ctx, d.index, d.route, d.cookie,
			      d.estimated_ns);
	return 0;
}

/* The struct_ops map registered against the live gpu_kv_reclaim_ops type. */
SEC(".struct_ops")
struct gpu_kv_reclaim_ops kv_reclaim_ops = {
	.gpu_kv_reclaim_choose = (void *)kv_reclaim_gpu_kv_reclaim_choose,
};
