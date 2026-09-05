/* SPDX-License-Identifier: MIT */
/*
 * Public C ABI of the WARC experiment arms. One shared library
 * (libdali_policy.so) exports both arms; the arm is selected at creation.
 * Fail-closed: any negative return is a contract failure. There is no
 * selector fallback and no decision is produced on error.
 */
#ifndef DALI_POLICY_ABI_H
#define DALI_POLICY_ABI_H

#include <stdint.h>

#define DALI_WARC_ERROR (-1)

struct dali_policy_stats {
    uint64_t routed_events, hits, misses;
    uint64_t routed_events_prefill, hits_prefill, misses_prefill;
    uint64_t routed_events_decode, hits_decode, misses_decode;
    uint64_t loads, evictions, windows, empty_windows, requests;
    uint64_t selector_calls, selector_errors, bpf_rank_calls, decision_ns;
    uint64_t bpf_calls, bpf_candidates, bpf_ranked, bpf_empty, bpf_errors;
    uint64_t warmup_events;
};

/* arm is "native" or "bpf". For "bpf", bpf_library is the absolute path of
 * the gpubpf selector shared object and bpf_bytecode the absolute path of
 * the rank bytecode; both are ignored (and may be NULL) for "native".
 * wide_threshold: routed-expert counts strictly above it mark prefill steps.
 */
int dali_policy_create(const char *arm,
                       const char *bpf_library,
                       const char *bpf_bytecode,
                       uint32_t n_layers,
                       uint32_t n_experts,
                       uint32_t cache_size,
                       uint32_t w_size,
                       uint32_t u_size,
                       uint32_t wide_threshold,
                       void **state_out);

/* selected[0..n_selected-1]: distinct expert IDs routed this step, layer.
 * prefill classifies the step for prefill/decode metric splits. */
int dali_policy_step(void *state,
                     uint32_t layer,
                     const uint32_t *selected,
                     uint32_t n_selected,
                     int prefill);

/* Per-sequence (request) reset of all per-layer policy state. */
int dali_policy_request_start(void *state);

int dali_policy_stats(void *state, struct dali_policy_stats *out);

/* resident_out receives n_experts 0/1 marks for one layer (test diffing). */
int dali_policy_resident_layer(void *state, uint32_t layer, uint8_t *out);

void dali_policy_destroy(void *state);

#endif /* DALI_POLICY_ABI_H */
