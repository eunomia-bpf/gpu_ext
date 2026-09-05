/* SPDX-License-Identifier: MIT */
/*
 * Workload-Aware cache Replacement (WARC) decision model.
 *
 * Port of the cache-replacement policy only from DALI (arXiv:2602.03495v1),
 * Section 4.3 and Algorithm 2: a per-MoE-layer expert cache of fixed size
 * cache_size, a sliding workload-score window of w_size recent steps, and a
 * symmetric swap of u_size experts at each window boundary (highest-scoring
 * non-resident experts loaded, lowest-scoring resident experts evicted).
 *
 * This header is the single shared specification for both selector arms in
 * this workload. It contains no clock read and no allocation at steady
 * state. The native arm and the gpubpf-selector (host uBPF JIT) arm must
 * reach identical decisions on identical inputs; differential tests in
 * test_dali_policy.c enforce that. Nothing here claims the complete DALI
 * system (no greedy CPU/GPU assignment, no residual-based prefetching).
 *
 * Workload convention (documented port deviation, see plan.md):
 *   - one trace step = one compute-graph evaluation (one token window);
 *   - one route event per (step, layer, expert) means the expert was active
 *     for that token window, so per-step workload per expert is a count;
 *   - one sequence = one request; per-sequence state is reset at request
 *     start; the window itself is the recency mechanism (w_size steps).
 */
#ifndef DALI_POLICY_MODEL_H
#define DALI_POLICY_MODEL_H

#include <stdint.h>

#define DALI_WARC_MAX_LAYERS 32U
#define DALI_WARC_MAX_EXPERTS 128U
#define DALI_WARC_MAX_WINDOW 64U
#define DALI_WARC_MAX_U 16U
#define DALI_WARC_ERROR (-1)

typedef struct dali_warc_layer {
    uint8_t resident[DALI_WARC_MAX_EXPERTS]; /* 1 = expert on GPU */
    uint32_t score[DALI_WARC_MAX_EXPERTS];  /* window workload accumulators */
    uint32_t residents;                     /* invariant: == cache_size */
    uint32_t step_in_window;                /* steps since last swap */
} dali_warc_layer_t;

typedef struct dali_warc_state {
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t cache_size;
    uint32_t w_size;
    uint32_t u_size;
    uint32_t wide_threshold; /* n_selected > wide_threshold marks a prefill step */
    dali_warc_layer_t layer[DALI_WARC_MAX_LAYERS];

    /* Cumulative performance metrics (performance only; no correctness gate). */
    uint64_t routed_events;
    uint64_t hits;
    uint64_t misses;
    uint64_t routed_events_prefill;
    uint64_t hits_prefill;
    uint64_t misses_prefill;
    uint64_t routed_events_decode;
    uint64_t hits_decode;
    uint64_t misses_decode;
    uint64_t loads;
    uint64_t evictions;
    uint64_t windows;
    uint64_t empty_windows; /* window boundary produced no positive load set */
    uint64_t requests;

    /* Selector accounting. */
    uint64_t selector_calls; /* logical window decisions made */
    uint64_t selector_errors;
    uint64_t bpf_rank_calls; /* BPF arm only: raw JIT selection calls */
    uint64_t decision_ns;    /* wall time spent inside selector choose calls */
} dali_warc_state_t;

/*
 * One logical selection: return the symmetric swap for a layer.
 * load[0..k-1] = non-resident experts to bring to the GPU, highest window
 * score first; evict[0..k-1] = resident experts to return to the CPU,
 * lowest window score first; 0 <= k <= u_size. When fewer than u_size
 * non-resident experts have positive window score, k is that smaller count
 * and exactly k residents are evicted, so the cache size is invariant.
 * Deterministic tie-break: equal scores keep the lower expert ID in both
 * directions. Return 0 on success, DALI_WARC_ERROR on contract failure.
 * ctx is selector-owned; score/resident are read-only layer views.
 */
typedef struct dali_warc_selector {
    void *ctx;
    int (*choose)(void *ctx,
                  const uint32_t *score,
                  const uint8_t *resident,
                  uint32_t n_experts,
                  uint32_t u_size,
                  uint32_t *load,
                  uint32_t *evict,
                  uint32_t *k_out);
} dali_warc_selector_t;

static inline void dali_warc_layer_init(dali_warc_layer_t *layer,
                                        uint32_t n_experts,
                                        uint32_t cache_size)
{
    uint32_t e;
    for (e = 0; e < n_experts; ++e) {
        layer->resident[e] = e < cache_size ? 1 : 0;
        layer->score[e] = 0;
    }
    layer->residents = cache_size;
    layer->step_in_window = 0;
}

/* Deterministic native selection; also the oracle for differential tests.
 * Inputs are read-only. Ascending expert-ID scans with strict improvement
 * rules make equal scores keep the lowest expert ID. */
static inline int dali_warc_native_choose(void *ctx,
                                          const uint32_t *score,
                                          const uint8_t *resident,
                                          uint32_t n_experts,
                                          uint32_t u_size,
                                          uint32_t *load,
                                          uint32_t *evict,
                                          uint32_t *k_out)
{
    uint8_t taken_load[DALI_WARC_MAX_EXPERTS];
    uint8_t taken_evict[DALI_WARC_MAX_EXPERTS];
    uint32_t k, e, i;

    (void)ctx;
    if (u_size == 0 || u_size > DALI_WARC_MAX_U || n_experts > DALI_WARC_MAX_EXPERTS)
        return DALI_WARC_ERROR;
    for (e = 0; e < n_experts; ++e) {
        taken_load[e] = 0;
        taken_evict[e] = 0;
    }

    /* Top-k non-resident experts: highest positive score, lower ID on ties. */
    for (k = 0; k < u_size; ++k) {
        uint32_t best = n_experts, best_score = 0;
        for (e = 0; e < n_experts; ++e) {
            if (resident[e] || taken_load[e] || score[e] == 0)
                continue;
            if (best == n_experts || score[e] > best_score) {
                best = e;
                best_score = score[e];
            }
        }
        if (best == n_experts)
            break; /* fewer than u_size positive non-resident experts */
        load[k] = best;
        taken_load[best] = 1;
    }

    /* Bottom-k resident experts: lowest score, lower ID on ties. */
    for (i = 0; i < k; ++i) {
        uint32_t best = n_experts;
        for (e = 0; e < n_experts; ++e) {
            if (!resident[e] || taken_evict[e])
                continue;
            if (best == n_experts || score[e] < score[best] ||
                (score[e] == score[best] && e < best)) {
                best = e;
            }
        }
        evict[i] = best;
        taken_evict[best] = 1;
    }

    *k_out = k;
    return 0;
}

/*
 * Advance one layer by one step of routed experts.
 * selected[0..n_selected-1] are the distinct expert IDs active in this step;
 * n_selected is 0..n_experts, no duplicates within a step.
 * Metrics are updated before the window-boundary swap, matching Algorithm 2
 * (workload is observed, then at i mod w_size == 0 the swap applies).
 */
static inline int dali_warc_step(dali_warc_state_t *st,
                                 const dali_warc_selector_t *sel,
                                 uint32_t layer,
                                 const uint32_t *selected,
                                 uint32_t n_selected,
                                 int prefill)
{
    dali_warc_layer_t *L;
    uint32_t e;

    if (st == 0 || sel == 0 || layer >= st->n_layers)
        return DALI_WARC_ERROR;
    if (n_selected > st->n_experts)
        return DALI_WARC_ERROR;
    L = &st->layer[layer];

    for (e = 0; e < n_selected; ++e) {
        uint32_t expert = selected[e];
        int is_hit;
        if (expert >= st->n_experts)
            return DALI_WARC_ERROR;
        is_hit = L->resident[expert] != 0;
        ++st->routed_events;
        if (is_hit)
            ++st->hits;
        else
            ++st->misses;
        if (prefill) {
            ++st->routed_events_prefill;
            if (is_hit)
                ++st->hits_prefill;
            else
                ++st->misses_prefill;
        } else {
            ++st->routed_events_decode;
            if (is_hit)
                ++st->hits_decode;
            else
                ++st->misses_decode;
        }
        ++L->score[expert];
    }

    if (st->w_size == 0 || st->w_size > DALI_WARC_MAX_WINDOW ||
        st->u_size == 0 || st->u_size > DALI_WARC_MAX_U)
        return DALI_WARC_ERROR;
    ++L->step_in_window;
    if (L->step_in_window < st->w_size)
        return 0;

    {
        uint32_t load[DALI_WARC_MAX_U];
        uint32_t evict[DALI_WARC_MAX_U];
        uint32_t k = 0;
        int rc;

        rc = sel->choose(sel->ctx, L->score, L->resident, st->n_experts,
                         st->u_size, load, evict, &k);
        ++st->selector_calls;
        if (rc != 0) {
            ++st->selector_errors;
            return DALI_WARC_ERROR;
        }
        if (k > st->u_size || k > L->residents || k > st->n_experts - L->residents)
            return DALI_WARC_ERROR;
        for (e = 0; e < k; ++e) {
            if (load[e] >= st->n_experts || evict[e] >= st->n_experts ||
                L->resident[load[e]] || !L->resident[evict[e]])
                return DALI_WARC_ERROR;
        }
        for (e = 0; e < k; ++e) {
            L->resident[evict[e]] = 0;
            L->resident[load[e]] = 1;
            ++st->loads;
            ++st->evictions;
        }
        for (e = 0; e < st->n_experts; ++e)
            L->score[e] = 0;
        L->step_in_window = 0;
        ++st->windows;
        if (k == 0)
            ++st->empty_windows;
    }
    return 0;
}

/* Per-sequence reset (DALI EOS break): state back to the initial resident
 * set with zero scores; cumulative metrics keep running. */
static inline int dali_warc_request_start(dali_warc_state_t *st)
{
    uint32_t i;
    if (st == 0)
        return DALI_WARC_ERROR;
    for (i = 0; i < st->n_layers; ++i)
        dali_warc_layer_init(&st->layer[i], st->n_experts, st->cache_size);
    ++st->requests;
    return 0;
}

#endif /* DALI_POLICY_MODEL_H */
