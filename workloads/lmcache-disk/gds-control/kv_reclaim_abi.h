// SPDX-License-Identifier: GPL-2.0
/*
 * Shared fixed-width ABI and bounded-integer candidate-selection policy for
 * real KV reclaim victim selection (UVM ioctl 83 / gpu_kv_reclaim_ops).
 *
 * This header is the single source of the ABI and the shared selection
 * algorithm, used by:
 *   - the driver patch (kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h mirrors
 *     these definitions; kv-reclaim-driver-575.patch),
 *   - the BPF policy (kv_reclaim_policy.bpf.c),
 *   - the native decision entrypoint (kv_reclaim_native.c),
 *   - the ctypes binding (kv_reclaim_binding.py), which calls the native
 *     entrypoint and the ioctl transport.
 *
 * No fd, file offset, GPU pointer, arbitrary user pointer, stream, or
 * completion object crosses this interface. Every call carries its own
 * bounded candidate vector (max 8, fixed-width scalars only) plus shared
 * observed rates; the kernel keeps no caller state and the answer is a
 * selected index/cookie/route.
 *
 * Selection goal: minimize estimated recovery cost per actually freeable KV
 * byte. For each candidate the cheaper of
 *   - full recompute:            computed_tokens * recompute_ns_per_token, and
 *   - disk-backed-prefix read +
 *     recomputing the unsaved tail: ceil(disk_backed_bytes/1024) *
 *     disk_read_ns_per_kib + (computed_tokens - disk_backed_tokens) *
 *     recompute_ns_per_token
 * is used when coverage is known and both rates are known. The disk read is
 * priced on the provider's actual backed transfer bytes
 * (disk_backed_bytes), NOT clamped to the freeable count: freeable KV
 * blocks exclude shared/referenced blocks while the disk transfer may still
 * restore the full backing objects. Unknown coverage is NOT a fully backed
 * request (its disk-backed prefix is priced as zero); an unknown disk rate
 * prices the disk route as unbounded. All products and sums are saturating
 * against UVM_KV_RECLAIM_COST_SAT; the cost-per-freeable-byte comparison
 * cross-multiplies with an explicit overflow flag.
 *
 * Ablation selector: defining UVM_KV_RECLAIM_POLICY_TOTAL_COST at compile
 * time ranks eligible victims by ABSOLUTE estimated recovery ns instead of
 * recovery ns per actually freeable byte. The per-candidate estimator,
 * priority semantics, eligibility, unique-min/tie-to-stock rule, routes,
 * and decision metadata are unchanged; the default (macro undefined) keeps
 * the original cost-per-freeable-byte policy.
 *
 * Priority keeps vLLM integer semantics: LOWER numeric value is MORE
 * important, and the scheduler would pick the max(priority, arrival)
 * victim. The worst priority class is therefore the MAXIMUM priority value
 * among candidates with positive freeable bytes; only that class is
 * eligible, and no policy may silently invert this. On ties, unknown or
 * unusable telemetry, or zero freeable candidates the decision stays the
 * caller's stock victim.
 */

#ifndef _KV_RECLAIM_ABI_H
#define _KV_RECLAIM_ABI_H

#ifndef NvU32
typedef unsigned int NvU32;
#endif
#ifndef NvU64
typedef unsigned long long NvU64;
#endif

#define UVM_KV_RECLAIM_ABI_VERSION                  1
#define UVM_KV_RECLAIM_MAX_CANDIDATES               8
#define UVM_KV_RECLAIM_MAX_PRIORITY                 7U

/* Candidate metadata flags (fixed-width scalars only). */
#define UVM_KV_RECLAIM_CANDIDATE_FLAG_COVERAGE_KNOWN  0x00000001
#define UVM_KV_RECLAIM_CANDIDATE_FLAGS_ALL            \
    UVM_KV_RECLAIM_CANDIDATE_FLAG_COVERAGE_KNOWN

/* Recovery-route codes returned with the selected candidate. */
#define UVM_KV_RECLAIM_ROUTE_STOCK            0 /* keep the caller's default victim */
#define UVM_KV_RECLAIM_ROUTE_FULL_RECOMPUTE   1 /* recompute every token */
#define UVM_KV_RECLAIM_ROUTE_DISK_PREFIX      2 /* read the disk-backed prefix, recompute the unsaved tail */

/* Saturating cap for estimated costs: keeps every product and sum inside
 * the explicit bounded 64-bit representation below. Real values (ns for
 * GB-scale KV) stay far below it. */
#define UVM_KV_RECLAIM_COST_SAT               0x7FFFFFFFFFFFFFFFULL

typedef struct uvm_bpf_kv_reclaim_candidate
{
    NvU64 cookie;             /* opaque caller token, echoed back only */
    NvU64 freeable_bytes;     /* KV bytes actually freeable by reclaim */
    NvU64 computed_tokens;    /* tokens computed so far */
    NvU64 disk_backed_tokens; /* contiguous disk-backed prefix tokens */
    NvU64 disk_backed_bytes;  /* provider's actual backed transfer bytes */
    NvU32 priority;           /* 0..UVM_KV_RECLAIM_MAX_PRIORITY, vLLM semantics: lower is more important */
    NvU32 flags;             /* UVM_KV_RECLAIM_CANDIDATE_FLAG_* */
} uvm_bpf_kv_reclaim_candidate_t;

typedef struct uvm_bpf_kv_reclaim_request
{
    NvU32 abi_version;
    NvU32 n_candidates;       /* 1..UVM_KV_RECLAIM_MAX_CANDIDATES */
    NvU32 stock_index;        /* caller's default victim index */
    NvU32 pad0;               /* reserved, must be 0 */
    NvU64 disk_read_ns_per_kib;   /* shared observed rate, 0 = unknown */
    NvU64 recompute_ns_per_token; /* shared observed rate, 0 = unknown */
} uvm_bpf_kv_reclaim_request_t;

typedef struct uvm_bpf_kv_reclaim_decision
{
    NvU32 index;              /* selected candidate index */
    NvU32 route;              /* UVM_KV_RECLAIM_ROUTE_* */
    NvU64 cookie;
    NvU64 estimated_ns;       /* saturating estimated recovery cost */
} uvm_bpf_kv_reclaim_decision_t;

/* Callback-local context: all inputs plus the recorded decision.
 * eligible_mask is precomputed by the kernel before the policy is invoked
 * and is the only kernel-trusted view of eligibility. */
typedef struct uvm_bpf_kv_reclaim_decision_ctx
{
    uvm_bpf_kv_reclaim_request_t request;
    uvm_bpf_kv_reclaim_candidate_t candidates[UVM_KV_RECLAIM_MAX_CANDIDATES];
    uvm_bpf_kv_reclaim_decision_t decision;
    NvU32 eligible_mask;
    NvU32 recorded;
} uvm_bpf_kv_reclaim_decision_ctx_t;

/* ---- bounded integer helpers (shared kernel/BPF/native) ----
 *
 * BPF-safe by construction: no 64-bit division by a runtime value and no
 * 128-bit type. 64x64->128 products are built from four 32-bit products
 * plus 64-bit adds/shifts/masks, which the BPF backend supports directly.
 */

/* Exact 64x64 -> 128 product (*hi : *lo) from four 32-bit products.
 * a = ah*2^32 + al, b = bh*2^32 + bl;
 * a*b = ah*bh*2^64 + (ah*bl + al*bh)*2^32 + al*bl. */
static inline void uvm_kv_reclaim_mul128(NvU64 a, NvU64 b,
                                         NvU64 *hi, NvU64 *lo)
{
    NvU32 al = (NvU32)a, ah = (NvU32)(a >> 32);
    NvU32 bl = (NvU32)b, bh = (NvU32)(b >> 32);
    NvU64 p0 = (NvU64)al * bl;   /* < 2^64 */
    NvU64 p1 = (NvU64)al * bh;   /* < 2^64 */
    NvU64 p2 = (NvU64)ah * bl;   /* < 2^64 */
    NvU64 p3 = (NvU64)ah * bh;   /* < 2^64 */
    NvU64 s1 = (p0 >> 32) + (NvU32)p1 + (NvU32)p2;    /* < 3*2^32 */
    NvU64 s2 = (p1 >> 32) + (p2 >> 32) + (NvU32)p3 + (s1 >> 32); /* < 3*2^32 */

    *lo = ((s1 & 0xffffffffULL) << 32) | (p0 & 0xffffffffULL);
    *hi = (((p3 >> 32) + (s2 >> 32)) << 32) | (s2 & 0xffffffffULL);
}

/* Saturating product: the exact 128-bit result is formed first; any product
 * above cap saturates to cap. No unchecked product, no division. */
static inline NvU64 uvm_kv_reclaim_sat_mul(NvU64 a, NvU64 b, NvU64 cap)
{
    NvU64 hi, lo;

    uvm_kv_reclaim_mul128(a, b, &hi, &lo);
    if (hi != 0 || lo > cap)
        return cap;
    return lo;
}

static inline NvU64 uvm_kv_reclaim_sat_add(NvU64 a, NvU64 b, NvU64 cap)
{
    if (a >= cap || b > cap - a)
        return cap;
    return a + b;
}

/* ceil(bytes / 1024) as a shift, no overflow on bytes == U64_MAX. */
static inline NvU64 uvm_kv_reclaim_ceil_div_kib(NvU64 bytes)
{
    return bytes ? ((bytes - 1) >> 10) + 1 : 0;
}

/* ---- per-candidate policy (shared kernel/BPF/native) ----
 *
 * Every policy function takes the request and the candidate vector as
 * separate pointers, never a whole ctx. The BPF arm passes a stack-local
 * request copy and a per-CPU snapshot map value; native passes the fields
 * of its kernel ctx. This keeps every variable-offset BPF read on a stack
 * or map-value pointer; the BTF-typed struct_ops ctx is read only at
 * compile-time-constant offsets (the request copy and the unrolled
 * fixed-slot candidate copy), because check_ptr_to_btf_access() in v6.15
 * rejects any non-constant variable offset on a BTF-typed ctx.
 *
 * Verifier state containment: under clang -target bpf the branchy policy
 * functions below are emitted as out-of-line subprograms; native keeps
 * them inlined. That linkage change alone is not a general proof that
 * each subprogram is verified only once. What it provides here is a
 * smaller inlined body plus the factored selection - the worst priority
 * class is computed once per decision, each candidate's cost is computed
 * once, and there is one 128-bit cross-product comparison per candidate -
 * and that source verified and attached on the live 575 module without
 * raising the verifier limit. Decisions, tie handling, and telemetry
 * gating are exactly the same as a naive per-candidate rescan. */

#ifdef __BPF__
#define UVM_KV_RECLAIM_FN static __attribute__((noinline))
#else
#define UVM_KV_RECLAIM_FN static inline
#endif

/* Worst (least important) priority class among candidates with positive
 * freeable bytes, in vLLM integer semantics: lower value is more important,
 * so the worst class is the MAXIMUM priority value. Returns 0 when none;
 * eligibility fails either way. */
UVM_KV_RECLAIM_FN NvU32 uvm_kv_reclaim_worst_class(
    const uvm_bpf_kv_reclaim_candidate_t *cands, NvU32 n)
{
    NvU32 i, worst = 0, any = 0;

    for (i = 0; i < n; i++) {
        if (cands[i].freeable_bytes == 0)
            continue;
        if (!any || cands[i].priority > worst) {
            worst = cands[i].priority;
            any = 1;
        }
    }
    return any ? worst : 0;
}

/* Eligible: positive freeable bytes, usable recompute rate, consistent
 * telemetry, and the precomputed worst priority class. */
UVM_KV_RECLAIM_FN NvU32 uvm_kv_reclaim_eligible(
    const uvm_bpf_kv_reclaim_request_t *req,
    const uvm_bpf_kv_reclaim_candidate_t *cands, NvU32 n, NvU32 i,
    NvU32 worst)
{
    const uvm_bpf_kv_reclaim_candidate_t *c = &cands[i];

    if (i >= n || c->freeable_bytes == 0)
        return 0;
    if (req->recompute_ns_per_token == 0)
        return 0;
    if (c->disk_backed_tokens > c->computed_tokens)
        return 0;
    if (c->priority != worst)
        return 0;
    return 1;
}

/* Estimated recovery cost in ns for candidate i, with the cheaper recovery
 * route in *route. The disk read is priced on the provider's actual backed
 * transfer bytes (disk_backed_bytes), not clamped to freeable_bytes:
 * freeable KV blocks exclude shared/referenced blocks while the disk
 * transfer may still restore the full backing objects. Unknown coverage
 * prices the disk-backed prefix as zero; an unknown disk rate prices the
 * disk route as unbounded. Callers must pass a candidate whose
 * disk_backed_tokens <= computed_tokens. */
UVM_KV_RECLAIM_FN NvU64 uvm_kv_reclaim_cost_route(
    const uvm_bpf_kv_reclaim_request_t *req,
    const uvm_bpf_kv_reclaim_candidate_t *c, NvU32 *route)
{
    NvU64 rate_disk = req->disk_read_ns_per_kib;
    NvU64 rate_comp = req->recompute_ns_per_token;
    NvU64 covered = c->flags & UVM_KV_RECLAIM_CANDIDATE_FLAG_COVERAGE_KNOWN;
    NvU64 backed_tokens = covered ? c->disk_backed_tokens : 0;
    NvU64 backed_bytes = covered ? c->disk_backed_bytes : 0;
    NvU64 tail_tokens, full, disk_cost, tail_cost, mixed;

    if (backed_tokens > c->computed_tokens)
        backed_tokens = c->computed_tokens;
    tail_tokens = c->computed_tokens - backed_tokens;

    full = uvm_kv_reclaim_sat_mul(c->computed_tokens, rate_comp,
                                  UVM_KV_RECLAIM_COST_SAT);
    disk_cost = (rate_disk == 0) ?
        UVM_KV_RECLAIM_COST_SAT :
        uvm_kv_reclaim_sat_mul(uvm_kv_reclaim_ceil_div_kib(backed_bytes),
                               rate_disk, UVM_KV_RECLAIM_COST_SAT);
    tail_cost = uvm_kv_reclaim_sat_mul(tail_tokens, rate_comp,
                                       UVM_KV_RECLAIM_COST_SAT);
    mixed = uvm_kv_reclaim_sat_add(disk_cost, tail_cost,
                                   UVM_KV_RECLAIM_COST_SAT);

    if (mixed < full) {
        *route = UVM_KV_RECLAIM_ROUTE_DISK_PREFIX;
        return mixed;
    }
    *route = UVM_KV_RECLAIM_ROUTE_FULL_RECOMPUTE;
    return full;
}

/* -1/0/1 for two (cost, freeable-bytes) pairs under the ACTIVE ranking
 * policy. Default: a is cheaper per actually freeable byte than b iff
 * cost_a*free_b < cost_b*free_a, compared by exact 128-bit cross-products
 * (no float, no division, no overflow ambiguity; the products are exact in
 * 128 bits); 0 means equal cost-per-freeable-byte. With
 * UVM_KV_RECLAIM_POLICY_TOTAL_COST defined: plain absolute order of the two
 * estimated recovery ns (the freeable-byte arguments are then unused); 0
 * means equal absolute cost. This is the only candidate comparison shared
 * by uvm_kv_reclaim_compare() and uvm_kv_reclaim_choose(), so both keep
 * consistent semantics under either policy. */
UVM_KV_RECLAIM_FN int uvm_kv_reclaim_cost_cmp(
    NvU64 cost_a, NvU64 free_a, NvU64 cost_b, NvU64 free_b)
{
#ifdef UVM_KV_RECLAIM_POLICY_TOTAL_COST
    (void)free_a;
    (void)free_b;
    if (cost_a < cost_b)
        return -1;
    if (cost_a > cost_b)
        return 1;
    return 0;
#else
    NvU64 hi_a, lo_a, hi_b, lo_b;

    uvm_kv_reclaim_mul128(cost_a, free_b, &hi_a, &lo_a);
    uvm_kv_reclaim_mul128(cost_b, free_a, &hi_b, &lo_b);

    if (hi_a < hi_b)
        return -1;
    if (hi_a > hi_b)
        return 1;
    if (lo_a < lo_b)
        return -1;
    if (lo_a > lo_b)
        return 1;
    return 0;
#endif
}

/* -1/0/1: candidate a vs b under the active ranking policy (default:
 * estimated cost per actually freeable byte); each candidate's cost/route
 * is computed exactly once and the two costs are compared with
 * uvm_kv_reclaim_cost_cmp(). 0 means equal under the active policy. */
UVM_KV_RECLAIM_FN int uvm_kv_reclaim_compare(
    const uvm_bpf_kv_reclaim_request_t *req,
    const uvm_bpf_kv_reclaim_candidate_t *cands, NvU32 a, NvU32 b)
{
    NvU32 route_a, route_b;
    NvU64 cost_a, cost_b;

    cost_a = uvm_kv_reclaim_cost_route(req, &cands[a], &route_a);
    cost_b = uvm_kv_reclaim_cost_route(req, &cands[b], &route_b);
    return uvm_kv_reclaim_cost_cmp(cost_a, cands[a].freeable_bytes,
                                   cost_b, cands[b].freeable_bytes);
}

/* The full shared selection. *out defaults to the caller's stock victim;
 * a candidate replaces it only as the unique strict minimum under the
 * active ranking policy (default: estimated cost per actually freeable
 * byte; UVM_KV_RECLAIM_POLICY_TOTAL_COST: absolute estimated recovery ns)
 * among eligible candidates. Ties, no
 * eligible candidate, and unusable telemetry all keep the stock victim
 * (route STOCK, estimated_ns 0). If the stock victim itself is the unique
 * minimum it is returned with its real route and estimate.
 *
 * n and stock are exact locally bounded values (stock < n <=
 * UVM_KV_RECLAIM_MAX_CANDIDATES); the BPF arm keeps them in
 * verifier-bounded registers so every vector read is a fixed-size strided
 * access at a bounded offset on the snapshot, never an index reloaded from
 * the BTF-typed ctx. The worst priority class is computed once and each
 * candidate's cost is computed once; decisions, ties, and telemetry
 * gating are identical to a naive per-candidate rescan. */
UVM_KV_RECLAIM_FN void uvm_kv_reclaim_choose(
    const uvm_bpf_kv_reclaim_request_t *req,
    const uvm_bpf_kv_reclaim_candidate_t *cands, NvU32 n, NvU32 stock,
    uvm_bpf_kv_reclaim_decision_t *out)
{
    NvU32 i, best = 0, any = 0, tie = 0, route_i, worst;
    NvU32 best_route = UVM_KV_RECLAIM_ROUTE_STOCK;
    NvU64 cost_i, best_cost = 0;
    int cmp;

    out->index = stock;
    out->route = UVM_KV_RECLAIM_ROUTE_STOCK;
    out->cookie = cands[stock].cookie;
    out->estimated_ns = 0;

    worst = uvm_kv_reclaim_worst_class(cands, n);

    for (i = 0; i < n; i++) {
        if (!uvm_kv_reclaim_eligible(req, cands, n, i, worst))
            continue;

        cost_i = uvm_kv_reclaim_cost_route(req, &cands[i], &route_i);
        if (!any) {
            best = i;
            best_cost = cost_i;
            best_route = route_i;
            any = 1;
            continue;
        }
        cmp = uvm_kv_reclaim_cost_cmp(cost_i, cands[i].freeable_bytes,
                                      best_cost, cands[best].freeable_bytes);
        if (cmp < 0) {
            best = i;
            best_cost = cost_i;
            best_route = route_i;
            tie = 0;
        } else if (cmp == 0) {
            tie = 1;
        }
    }

    if (!any || tie)
        return;

    out->index = best;
    out->route = best_route;
    out->cookie = cands[best].cookie;
    out->estimated_ns = best_cost;
}

#endif /* _KV_RECLAIM_ABI_H */
