/* SPDX-License-Identifier: MIT */
/*
 * Two selector arms over the shared WARC decision model:
 *   arm "native": selection executed in C (the reference implementation).
 *   arm "bpf":    the same policy decisions, with every top-k/bottom-k
 *                 selection executed by the existing gpubpf rank selector
 *                 (extension/.output/libmoe_expert_policy.so, moe_expert_rank
 *                 bytecode under the host uBPF JIT). No new BPF program and
 *                 no native fallback: any selector failure is fatal to the
 *                 step, matching the fail-closed contract of the component.
 *
 * Load transform fed to the rank selector: score_bits(e) = window score of
 * e when e is non-resident, else +0.0. The rank program keeps score > 0 in
 * stable descending order, so its output is exactly the top-u non-resident
 * experts, lower expert ID first on ties.
 * Evict transform: score_bits(e) = Smax - score(e) + 1.0 when e is resident,
 * else +0.0, with Smax the maximum resident window score. Positive entries
 * therefore order resident experts by ascending window score, stable by
 * expert ID, which is exactly the bottom-u eviction set.
 */
#define _POSIX_C_SOURCE 200809L

#include "dali_policy_abi.h"
#include "dali_policy_model.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "moe_expert_policy.h"

struct bpf_arm {
    void *lib;
    int (*rank_init)(const char *path);
    int (*rank_run)(const struct moe_expert_rank_candidate *entries,
                    mep_u32 count, mep_u32 *indices, mep_u32 capacity,
                    mep_u32 *selected_count);
    void (*rank_stats)(struct moe_expert_rank_stats *out);
};

struct dali_policy_state {
    dali_warc_state_t model;
    char arm[8];
    struct bpf_arm bpf;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void store_bits(mep_u64 *bits_out, double value)
{
    memcpy(bits_out, &value, sizeof(value));
}

/* gpubpf-selector selection for one WARC window decision. */
static int bpf_choose(void *ctx,
                      const uint32_t *score,
                      const uint8_t *resident,
                      uint32_t n_experts,
                      uint32_t u_size,
                      uint32_t *load,
                      uint32_t *evict,
                      uint32_t *k_out)
{
    struct dali_policy_state *state = ctx;
    struct bpf_arm *bpf = &state->bpf;
    mep_u32 indices[DALI_WARC_MAX_EXPERTS];
    mep_u32 selected;
    mep_u32 e, i, k;
    int rc;
    uint64_t started, finished;

    if (state->model.n_experts != n_experts)
        return DALI_WARC_ERROR;

    /* Load candidates: non-resident experts with positive window score. */
    {
        struct moe_expert_rank_candidate entries[DALI_WARC_MAX_EXPERTS];
        for (e = 0; e < n_experts; ++e) {
            double v = resident[e] ? 0.0 : (double)score[e];
            store_bits(&entries[e].score_bits, v);
            entries[e].identity = e;
            entries[e].ordinal = e;
            entries[e].reserved = 0;
        }
        started = now_ns();
        rc = bpf->rank_run(entries, n_experts, indices, n_experts, &selected);
        finished = now_ns();
        state->model.bpf_rank_calls += 1;
        state->model.decision_ns += finished - started;
        if (rc != 0)
            return DALI_WARC_ERROR;
    }
    k = selected < u_size ? selected : u_size;
    for (i = 0; i < k; ++i) {
        load[i] = indices[i];
        if (load[i] >= n_experts || resident[load[i]] || score[load[i]] == 0)
            return DALI_WARC_ERROR;
    }
    if (k == 0) {
        *k_out = 0;
        return 0;
    }

    /* Evict candidates: resident experts, transformed to ascending score. */
    {
        struct moe_expert_rank_candidate entries[DALI_WARC_MAX_EXPERTS];
        uint32_t smax = 0;
        for (e = 0; e < n_experts; ++e) {
            if (resident[e] && score[e] > smax)
                smax = score[e];
        }
        for (e = 0; e < n_experts; ++e) {
            double v = resident[e] ? (double)(smax - score[e] + 1u) : 0.0;
            store_bits(&entries[e].score_bits, v);
            entries[e].identity = e;
            entries[e].ordinal = e;
            entries[e].reserved = 0;
        }
        started = now_ns();
        rc = bpf->rank_run(entries, n_experts, indices, n_experts, &selected);
        finished = now_ns();
        state->model.bpf_rank_calls += 1;
        state->model.decision_ns += finished - started;
        if (rc != 0 || selected < k)
            return DALI_WARC_ERROR;
    }
    for (i = 0; i < k; ++i) {
        evict[i] = indices[i];
        if (evict[i] >= n_experts || !resident[evict[i]])
            return DALI_WARC_ERROR;
        for (mep_u32 j = 0; j < i; ++j)
            if (evict[j] == evict[i])
                return DALI_WARC_ERROR;
    }
    *k_out = k;
    return 0;
}

static int native_choose_timed(void *ctx,
                               const uint32_t *score,
                               const uint8_t *resident,
                               uint32_t n_experts,
                               uint32_t u_size,
                               uint32_t *load,
                               uint32_t *evict,
                               uint32_t *k_out)
{
    struct dali_policy_state *state = ctx;
    uint64_t started, finished;
    int rc;

    started = now_ns();
    rc = dali_warc_native_choose(ctx, score, resident, n_experts, u_size,
                                 load, evict, k_out);
    finished = now_ns();
    if (rc == 0)
        state->model.decision_ns += finished - started;
    return rc;
}

int dali_policy_create(const char *arm,
                       const char *bpf_library,
                       const char *bpf_bytecode,
                       uint32_t n_layers,
                       uint32_t n_experts,
                       uint32_t cache_size,
                       uint32_t w_size,
                       uint32_t u_size,
                       uint32_t wide_threshold,
                       void **state_out)
{
    struct dali_policy_state *state;
    uint32_t i;

    if (arm == 0 || state_out == 0)
        return DALI_WARC_ERROR;
    if (strcmp(arm, "native") != 0 && strcmp(arm, "bpf") != 0)
        return DALI_WARC_ERROR;
    if (n_layers == 0 || n_layers > DALI_WARC_MAX_LAYERS ||
        n_experts == 0 || n_experts > DALI_WARC_MAX_EXPERTS ||
        cache_size == 0 || cache_size > n_experts ||
        w_size == 0 || w_size > DALI_WARC_MAX_WINDOW ||
        u_size == 0 || u_size > DALI_WARC_MAX_U ||
        wide_threshold >= n_experts)
        return DALI_WARC_ERROR;

    state = calloc(1, sizeof(*state));
    if (state == 0)
        return DALI_WARC_ERROR;
    state->model.n_layers = n_layers;
    state->model.n_experts = n_experts;
    state->model.cache_size = cache_size;
    state->model.w_size = w_size;
    state->model.u_size = u_size;
    state->model.wide_threshold = wide_threshold;
    for (i = 0; i < n_layers; ++i)
        dali_warc_layer_init(&state->model.layer[i], n_experts, cache_size);
    if (strcmp(arm, "native") == 0) {
        snprintf(state->arm, sizeof(state->arm), "native");
    } else {
        int rc;
        if (bpf_library == 0 || bpf_bytecode == 0) {
            free(state);
            return DALI_WARC_ERROR;
        }
        state->bpf.lib = dlopen(bpf_library, RTLD_NOW | RTLD_LOCAL);
        if (state->bpf.lib == 0) {
            free(state);
            return DALI_WARC_ERROR;
        }
        state->bpf.rank_init = (int (*)(const char *))dlsym(
            state->bpf.lib, "moe_expert_rank_init_v1");
        state->bpf.rank_run = (int (*)(const struct moe_expert_rank_candidate *,
                                       mep_u32, mep_u32 *, mep_u32, mep_u32 *))
            dlsym(state->bpf.lib, "moe_expert_rank_v1");
        state->bpf.rank_stats = (void (*)(struct moe_expert_rank_stats *))
            dlsym(state->bpf.lib, "moe_expert_rank_stats_v1");
        if (state->bpf.rank_init == 0 || state->bpf.rank_run == 0 ||
            state->bpf.rank_stats == 0) {
            dlclose(state->bpf.lib);
            free(state);
            return DALI_WARC_ERROR;
        }
        rc = state->bpf.rank_init(bpf_bytecode);
        if (rc != 0) {
            dlclose(state->bpf.lib);
            free(state);
            return DALI_WARC_ERROR;
        }
        snprintf(state->arm, sizeof(state->arm), "bpf");
    }
    *state_out = state;
    return 0;
}

static dali_warc_selector_t selector_for(void *state_in)
{
    struct dali_policy_state *state = state_in;
    dali_warc_selector_t sel;
    sel.ctx = state;
    sel.choose = strcmp(state->arm, "bpf") == 0 ? bpf_choose
                                               : native_choose_timed;
    return sel;
}

int dali_policy_step(void *state_in,
                     uint32_t layer,
                     const uint32_t *selected,
                     uint32_t n_selected,
                     int prefill)
{
    struct dali_policy_state *state = state_in;
    dali_warc_selector_t sel;
    if (state == 0)
        return DALI_WARC_ERROR;
    sel = selector_for(state);
    return dali_warc_step(&state->model, &sel, layer, selected, n_selected,
                          prefill);
}

int dali_policy_request_start(void *state_in)
{
    struct dali_policy_state *state = state_in;
    if (state == 0)
        return DALI_WARC_ERROR;
    return dali_warc_request_start(&state->model);
}

int dali_policy_stats(void *state_in, struct dali_policy_stats *out)
{
    struct dali_policy_state *state = state_in;
    if (state == 0 || out == 0)
        return DALI_WARC_ERROR;
    out->routed_events = state->model.routed_events;
    out->hits = state->model.hits;
    out->misses = state->model.misses;
    out->routed_events_prefill = state->model.routed_events_prefill;
    out->hits_prefill = state->model.hits_prefill;
    out->misses_prefill = state->model.misses_prefill;
    out->routed_events_decode = state->model.routed_events_decode;
    out->hits_decode = state->model.hits_decode;
    out->misses_decode = state->model.misses_decode;
    out->loads = state->model.loads;
    out->evictions = state->model.evictions;
    out->windows = state->model.windows;
    out->empty_windows = state->model.empty_windows;
    out->requests = state->model.requests;
    out->selector_calls = state->model.selector_calls;
    out->selector_errors = state->model.selector_errors;
    out->bpf_rank_calls = state->model.bpf_rank_calls;
    out->decision_ns = state->model.decision_ns;
    out->bpf_calls = 0;
    out->bpf_candidates = 0;
    out->bpf_ranked = 0;
    out->bpf_empty = 0;
    out->bpf_errors = 0;
    out->warmup_events = 0;
    if (strcmp(state->arm, "bpf") == 0) {
        struct moe_expert_rank_stats rs;
        state->bpf.rank_stats(&rs);
        out->bpf_calls = rs.calls;
        out->bpf_candidates = rs.candidates;
        out->bpf_ranked = rs.ranked;
        out->bpf_empty = rs.empty;
        out->bpf_errors = rs.errors;
    }
    return 0;
}

/* Resident-set probe used by tests to diff arm states step by step. */
int dali_policy_resident_layer(void *state_in,
                               uint32_t layer,
                               uint8_t *resident_out)
{
    struct dali_policy_state *state = state_in;
    if (state == 0 || layer >= state->model.n_layers || resident_out == 0)
        return DALI_WARC_ERROR;
    memcpy(resident_out, state->model.layer[layer].resident,
           state->model.n_experts);
    return 0;
}

void dali_policy_destroy(void *state_in)
{
    struct dali_policy_state *state = state_in;
    if (state == 0)
        return;
    if (strcmp(state->arm, "bpf") == 0 && state->bpf.lib != 0)
        dlclose(state->bpf.lib);
    free(state);
}
