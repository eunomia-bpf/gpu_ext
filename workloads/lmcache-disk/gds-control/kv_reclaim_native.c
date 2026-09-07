// SPDX-License-Identifier: GPL-2.0
/*
 * Native decision entrypoint for real KV reclaim candidate selection.
 *
 * Runs the shared bounded-integer algorithm from kv_reclaim_abi.h - the
 * same source the BPF policy and the mirrored kernel definitions use - on a
 * caller-supplied candidate vector (max 8, fixed-width scalars only).
 * Exposed as a flat, ctypes-friendly C function so the serving integration
 * can select a victim from real candidates without the kernel.
 *
 * The kernel ioctl path returns the stock victim when no BPF policy is
 * attached; this entrypoint is the stock/native arm of that comparison.
 *
 * Return status: 0 = decision produced (route STOCK means the stock victim
 * was kept, possibly with its real route when it is the unique minimum),
 * 1 = hard-invalid input (ABI version, candidate count, stock index,
 * priority range, flag bits, or null outputs); outputs still carry the
 * stock defaults where they are defined.
 */

#include <stddef.h>

#include "kv_reclaim_abi.h"

int kv_reclaim_native_choose(
    unsigned int abi_version,
    unsigned int n_candidates,
    unsigned int stock_index,
    unsigned long long disk_read_ns_per_kib,
    unsigned long long recompute_ns_per_token,
    const unsigned long long *cookie,
    const unsigned long long *freeable_bytes,
    const unsigned long long *computed_tokens,
    const unsigned long long *disk_backed_tokens,
    const unsigned long long *disk_backed_bytes,
    const unsigned int *priority,
    const unsigned int *flags,
    unsigned int *out_index,
    unsigned int *out_route,
    unsigned long long *out_cookie,
    unsigned long long *out_estimated_ns)
{
    uvm_bpf_kv_reclaim_decision_ctx_t ctx;
    uvm_bpf_kv_reclaim_decision_t d;
    unsigned int i, bad = 0;

    if (!out_index || !out_route || !out_cookie || !out_estimated_ns)
        return 1;

    /* Hard range/default validation; any failure keeps the stock defaults. */
    if (abi_version != UVM_KV_RECLAIM_ABI_VERSION ||
        n_candidates == 0 ||
        n_candidates > UVM_KV_RECLAIM_MAX_CANDIDATES ||
        stock_index >= n_candidates) {
        *out_index = (stock_index < n_candidates) ? stock_index : 0;
        *out_route = UVM_KV_RECLAIM_ROUTE_STOCK;
        *out_cookie = (cookie && n_candidates <= UVM_KV_RECLAIM_MAX_CANDIDATES &&
                       stock_index < n_candidates) ? cookie[stock_index] : 0;
        *out_estimated_ns = 0;
        return 1;
    }
    for (i = 0; i < n_candidates; i++) {
        if (!cookie || !freeable_bytes || !computed_tokens ||
            !disk_backed_tokens || !disk_backed_bytes ||
            !priority || !flags ||
            priority[i] > UVM_KV_RECLAIM_MAX_PRIORITY ||
            (flags[i] & ~UVM_KV_RECLAIM_CANDIDATE_FLAGS_ALL)) {
            bad = 1;
            break;
        }
    }
    if (bad) {
        *out_index = stock_index;
        *out_route = UVM_KV_RECLAIM_ROUTE_STOCK;
        *out_cookie = cookie ? cookie[stock_index] : 0;
        *out_estimated_ns = 0;
        return 1;
    }

    ctx.request.abi_version = abi_version;
    ctx.request.n_candidates = n_candidates;
    ctx.request.stock_index = stock_index;
    ctx.request.pad0 = 0;
    ctx.request.disk_read_ns_per_kib = disk_read_ns_per_kib;
    ctx.request.recompute_ns_per_token = recompute_ns_per_token;
    ctx.eligible_mask = 0;
    ctx.recorded = 0;
    for (i = 0; i < UVM_KV_RECLAIM_MAX_CANDIDATES; i++) {
        ctx.candidates[i].cookie = (i < n_candidates) ? cookie[i] : 0;
        ctx.candidates[i].freeable_bytes = (i < n_candidates) ? freeable_bytes[i] : 0;
        ctx.candidates[i].computed_tokens = (i < n_candidates) ? computed_tokens[i] : 0;
        ctx.candidates[i].disk_backed_tokens = (i < n_candidates) ? disk_backed_tokens[i] : 0;
        ctx.candidates[i].disk_backed_bytes = (i < n_candidates) ? disk_backed_bytes[i] : 0;
        ctx.candidates[i].priority = (i < n_candidates) ? priority[i] : 0;
        ctx.candidates[i].flags = (i < n_candidates) ? flags[i] : 0;
    }

    uvm_kv_reclaim_choose(&ctx.request, ctx.candidates, n_candidates,
                          stock_index, &d);

    *out_index = d.index;
    *out_route = d.route;
    *out_cookie = d.cookie;
    *out_estimated_ns = d.estimated_ns;
    return 0;
}
