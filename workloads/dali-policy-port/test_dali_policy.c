/* SPDX-License-Identifier: MIT */
/*
 * CPU-only validation for the WARC policy port. No GPU, no driver, no
 * correctness/engagement/verifier/precision/hash gates: the checks are
 * decision determinism, native-vs-gpubpf-selector agreement (zero
 * mismatches on identical inputs), contract/fail-closed behavior, and
 * state invariants. Performance metrics (hit rates, transfer bytes,
 * decision overhead) are computed on the frozen 35,360-step real routing
 * trace. Selector engagement counts (bpf_rank_calls, bpf_errors,
 * selector_errors) are retained and printed as metadata only; they are
 * never a gate, retry, filter, or result rejection. bpf_errors is the
 * gpubpf library's process-global counter, so it also reflects the
 * intentional fail-closed negative test.
 *
 * Usage:
 *   test_dali_policy [trace.jsonl] [bpf_lib.so] [bpf_bytecode.bin]
 * Defaults resolve to this repository's frozen expert-buffering trace block
 * 01 and the existing extension/.output selector artifacts.
 */
#define _POSIX_C_SOURCE 200809L

#include "dali_policy_abi.h"
#include "dali_policy_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                    msg);                                            \
            ++failures;                                               \
        }                                                             \
    } while (0)

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static uint64_t rng_next(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static uint32_t rng_below(uint32_t n)
{
    return (uint32_t)(rng_next() % (uint64_t)n);
}

/* ----------------------- unit: native oracle ----------------------- */

static void test_native_oracle(void)
{
    uint32_t score[DALI_WARC_MAX_EXPERTS];
    uint8_t resident[DALI_WARC_MAX_EXPERTS];
    uint32_t load[DALI_WARC_MAX_U], evict[DALI_WARC_MAX_U], k;
    uint32_t e;
    memset(score, 0, sizeof(score));
    for (e = 0; e < DALI_WARC_MAX_EXPERTS; ++e)
        resident[e] = e < 8 ? 1 : 0; /* cache 8 of 128: experts 0..7 */

    /* Non-resident candidates 9:5 and 41:5 tie; lower ID (9) loads first,
     * then 77:1. Resident 4:3 and 5:1 stay; bottom-2 residents are the
     * score-0 tie {0, 1} in ID order. */
    score[9] = 5;
    score[41] = 5;
    score[77] = 1;
    score[4] = 3;
    score[5] = 1;

    CHECK(dali_warc_native_choose(0, score, resident, 128, 2, load, evict,
                                  &k) == 0,
          "oracle rc");
    CHECK(k == 2, "oracle k");
    CHECK(load[0] == 9 && load[1] == 41, "oracle load tie order");
    CHECK(evict[0] == 0 && evict[1] == 1, "oracle evict tie order");

    /* Fewer positives than u: only one non-resident positive -> k == 1. */
    memset(score, 0, sizeof(score));
    score[10] = 4;
    CHECK(dali_warc_native_choose(0, score, resident, 128, 3, load, evict,
                                  &k) == 0,
          "oracle rc 2");
    CHECK(k == 1 && load[0] == 10 && evict[0] == 0, "oracle short load set");
    /* No positive non-resident score -> k == 0, nothing evicted. */
    memset(score, 0, sizeof(score));
    score[0] = 7; /* resident only */
    CHECK(dali_warc_native_choose(0, score, resident, 128, 1, load, evict,
                                  &k) == 0, "oracle rc 3");
    CHECK(k == 0, "oracle empty window");

    CHECK(dali_warc_native_choose(0, score, resident, 128, 0, load, evict,
                                  &k) == DALI_WARC_ERROR,
          "oracle u=0 rejected");
    CHECK(dali_warc_native_choose(0, score, resident, 129, 1, load, evict,
                                  &k) == DALI_WARC_ERROR,
          "oracle n_experts overflow rejected");
}

/* ----------------------- unit: state invariants --------------------- */

static void test_invariants(void)
{
    void *native = 0;
    uint8_t resident[DALI_WARC_MAX_EXPERTS];
    uint32_t selected[16];
    struct dali_policy_stats s;
    uint32_t i, step;

    CHECK(dali_policy_create("native", 0, 0, 4, 32, 8, 4, 2, 4, &native) == 0,
          "native create");
    CHECK(native != 0, "native state");
    CHECK(dali_policy_create("bogus", 0, 0, 4, 32, 8, 4, 2, 4, &native)
              == DALI_WARC_ERROR,
          "unknown arm rejected");
    CHECK(dali_policy_create("native", 0, 0, 4, 32, 33, 4, 2, 4, &native)
              == DALI_WARC_ERROR,
          "cache > experts rejected");
    CHECK(dali_policy_create("native", 0, 0, 4, 32, 8, 0, 2, 4, &native)
              == DALI_WARC_ERROR,
          "w=0 rejected");
    CHECK(dali_policy_create("native", 0, 0, 4, 32, 8, 4, 17, 4, &native)
              == DALI_WARC_ERROR,
          "u overflow rejected");
    CHECK(dali_policy_create("native", 0, 0, 33, 32, 8, 4, 2, 4, &native)
              == DALI_WARC_ERROR,
          "layer overflow rejected");
    CHECK(dali_policy_create("native", 0, 0, 4, 32, 8, 4, 2, 32, &native)
              == DALI_WARC_ERROR,
          "wide threshold >= experts rejected");

    for (step = 0; step < 500; ++step) {
        uint32_t layer = rng_below(4);
        uint32_t n = 4 + rng_below(12);
        uint32_t e;
        for (e = 0; e < n; ++e)
            selected[e] = rng_below(32);
        CHECK(dali_policy_step(native, layer, selected, n, n > 4) == 0,
              "step rc");
        for (layer = 0; layer < 4; ++layer) {
            uint32_t residents = 0;
            CHECK(dali_policy_resident_layer(native, layer, resident) == 0,
                  "resident probe");
            for (e = 0; e < 32; ++e)
                residents += resident[e];
            CHECK(residents == 8, "cache size invariant");
        }
        if (step % 67 == 0)
            CHECK(dali_policy_request_start(native) == 0, "request start");
    }
    CHECK(dali_policy_step(native, 4, selected, 1, 0) == DALI_WARC_ERROR,
          "out-of-range layer rejected");
    CHECK(dali_policy_stats(native, &s) == 0, "stats rc");
    CHECK(s.requests >= 7, "request counter advanced");
    CHECK(s.routed_events == s.hits + s.misses, "hit/miss partition");
    CHECK(s.selector_errors == 0, "no selector errors (native)");
    dali_policy_destroy(native);
    (void)i;
}

/* ----------------- differential: native vs bpf arm ------------------ */

static void test_differential_random(const char *lib, const char *code)
{
    uint32_t caches[] = {8, 16, 32, 64};
    uint32_t windows[] = {1, 2, 4, 8};
    uint32_t usizes[] = {1, 2, 4};
    uint32_t ci, wi, ui;
    int bpf_ok = 0;

    for (ci = 0; ci < 4; ++ci)
        for (wi = 0; wi < 4; ++wi)
            for (ui = 0; ui < 3; ++ui) {
            void *native = 0, *bpf = 0;
            uint8_t rn[DALI_WARC_MAX_EXPERTS], rb[DALI_WARC_MAX_EXPERTS];
            struct dali_policy_stats sn, sb;
            uint32_t steps, step, layer, e;

            rng_state = 0x1234567ULL ^
                        ((uint64_t)ci << 40) ^ ((uint64_t)wi << 32) ^
                        ((uint64_t)ui << 24);
            CHECK(dali_policy_create("native", 0, 0, 4, 128, caches[ci],
                                     windows[wi], usizes[ui], 4, &native)
                      == 0,
                  "diff native create");
            CHECK(dali_policy_create("bpf", lib, code, 4, 128, caches[ci],
                                     windows[wi], usizes[ui], 4, &bpf)
                      == 0,
                  "diff bpf create");
            if (bpf == 0)
                continue; /* fail-closed already counted */
            bpf_ok = 1;
            steps = 1200;
            for (step = 0; step < steps; ++step) {
                for (layer = 0; layer < 4; ++layer) {
                    uint32_t n = 4 + rng_below(24);
                    uint32_t selected[28];
                    for (e = 0; e < n; ++e)
                        selected[e] = rng_below(128);
                    CHECK(dali_policy_step(native, layer, selected, n, n > 4)
                              == 0,
                          "diff native step");
                    CHECK(dali_policy_step(bpf, layer, selected, n, n > 4)
                              == 0,
                          "diff bpf step");
                }
                for (layer = 0; layer < 4; ++layer) {
                    CHECK(dali_policy_resident_layer(native, layer, rn) == 0
                              &&
                          dali_policy_resident_layer(bpf, layer, rb) == 0,
                          "diff resident probe");
                    CHECK(memcmp(rn, rb, 128) == 0,
                          "diff resident mismatch");
                }
                if (step % 251 == 0)
                    CHECK(dali_policy_request_start(native) == 0 &&
                              dali_policy_request_start(bpf) == 0,
                          "diff request start");
            }
            CHECK(dali_policy_stats(native, &sn) == 0 &&
                      dali_policy_stats(bpf, &sb) == 0,
                  "diff stats rc");
            CHECK(sn.hits == sb.hits && sn.misses == sb.misses &&
                      sn.loads == sb.loads && sn.evictions == sb.evictions &&
                      sn.windows == sb.windows &&
                      sn.empty_windows == sb.empty_windows &&
                      sn.selector_calls == sb.selector_calls &&
                      sn.routed_events == sb.routed_events,
                  "diff cumulative metric mismatch");
            /* Engagement and overhead counts are metadata only: retained
             * and printed, never a gate, retry, filter, or rejection. */
            printf("cell c=%u w=%u u=%u: native hit %.4f, bpf hit %.4f, "
                   "bpf_rank_calls %llu, bpf_errors %llu, "
                   "selector_errors native %llu bpf %llu, "
                   "decision_ns native %llu bpf %llu\n",
                   caches[ci], windows[wi], usizes[ui],
                   (double)sn.hits / (double)sn.routed_events,
                   (double)sb.hits / (double)sb.routed_events,
                   (unsigned long long)sb.bpf_rank_calls,
                   (unsigned long long)sb.bpf_errors,
                   (unsigned long long)sn.selector_errors,
                   (unsigned long long)sb.selector_errors,
                   (unsigned long long)sn.decision_ns,
                   (unsigned long long)sb.decision_ns);
            dali_policy_destroy(native);
            dali_policy_destroy(bpf);
        }
    CHECK(bpf_ok, "bpf arm never created; fail-closed path taken");
}

/* ------------------------- trace replay ----------------------------- */

#define MAX_GRAPHS 8192U
#define MAX_LAYERS_TRACE 32U

static void test_trace_replay(const char *trace, const char *lib,
                              const char *code)
{
    FILE *fp = fopen(trace, "r");
    static uint8_t picks[MAX_GRAPHS][MAX_LAYERS_TRACE][DALI_WARC_MAX_EXPERTS];
    static uint8_t counts[MAX_GRAPHS][MAX_LAYERS_TRACE];
    char line[8192];
    unsigned long long max_graph = 0;
    uint64_t route_events = 0;
    uint32_t g, l, e;
    void *native = 0, *bpf = 0;
    struct dali_policy_stats sn, sb;
    uint8_t rn[DALI_WARC_MAX_EXPERTS], rb[DALI_WARC_MAX_EXPERTS];
    uint64_t requests = 0, steps = 0;
    int last_wide = 0;

    CHECK(fp != 0, "trace open");
    if (fp == 0)
        return;
    while (fgets(line, sizeof(line), fp)) {
        char *p;
        int graph;
        if (strstr(line, "\"event\":\"route\"") == 0)
            continue;
        p = strstr(line, "\"graph\":");
        if (p == 0)
            continue;
        graph = (int)strtoll(p + 8, 0, 10);
        ++route_events;
        if ((unsigned long long)graph > max_graph)
            max_graph = (unsigned long long)graph;
    }
    fclose(fp);
    CHECK(route_events > 0, "trace has route events");
    CHECK(max_graph < (unsigned long long)MAX_GRAPHS, "graph cap");
    if (max_graph >= (unsigned long long)MAX_GRAPHS)
        return;

    /* Second pass: map tensor_base -> layer via layout names. */
    {
        static int base_layer[MAX_GRAPHS];
        static unsigned long long base_seen[MAX_GRAPHS];
        unsigned n_base = 0;
        unsigned i;
        fp = fopen(trace, "r");
        CHECK(fp != 0, "trace reopen");
        if (fp == 0)
            return;
        memset(picks, 0, sizeof(picks));
        memset(counts, 0, sizeof(counts));
        while (fgets(line, sizeof(line), fp)) {
            char *p;
            if (strstr(line, "\"event\":\"layout\"")) {
                unsigned long long base;
                int layer = -1;
                p = strstr(line, "\"base\":");
                base = p ? strtoull(p + 7, 0, 10) : 0;
                p = strstr(line, "ffn_gate_exps.weight");
                if (p != 0) {
                    /* name looks like "blk.N.ffn_gate_exps.weight" */
                    p = strstr(line, "\"name\":\"blk.");
                    if (p != 0)
                        layer = (int)strtol(p + 12, 0, 10);
                }
                if (base && layer >= 0 &&
                    n_base < sizeof(base_seen) / sizeof(base_seen[0])) {
                    base_seen[n_base] = base;
                    base_layer[n_base] = layer;
                    ++n_base;
                }
            } else if (strstr(line, "\"event\":\"route\"")) {
                unsigned long long base;
                int graph, expert, layer = -1;
                p = strstr(line, "\"graph\":");
                graph = p ? (int)strtol(p + 8, 0, 10) : -1;
                p = strstr(line, "\"tensor_base\":");
                base = p ? strtoull(p + 14, 0, 10) : 0;
                p = strstr(line, "\"expert_id\":");
                expert = p ? (int)strtol(p + 12, 0, 10) : -1;
                for (i = 0; i < n_base; ++i)
                    if (base_seen[i] == base) {
                        layer = base_layer[i];
                        break;
                    }
                if (graph >= 0 && expert >= 0 && layer >= 0 &&
                    (unsigned long long)graph < MAX_GRAPHS) {
                    ++picks[graph][layer][expert];
                    counts[graph][layer] += 1;
                }
            }
        }
        fclose(fp);
        (void)i;
    }

    /* Replay both arms in lockstep over (graph, layer). */
    CHECK(dali_policy_create("native", 0, 0, MAX_LAYERS_TRACE,
                             DALI_WARC_MAX_EXPERTS, 64, 4, 1, 4, &native)
              == 0,
          "replay native create");
    CHECK(dali_policy_create("bpf", lib, code, MAX_LAYERS_TRACE,
                             DALI_WARC_MAX_EXPERTS, 64, 4, 1, 4, &bpf)
              == 0,
          "replay bpf create");
    if (native == 0 || bpf == 0) {
        dali_policy_destroy(native);
        dali_policy_destroy(bpf);
        return;
    }
    for (g = 1; g <= max_graph; ++g) {
        uint32_t active_layers = 0, any_wide = 0, any_narrow = 0;
        for (l = 0; l < MAX_LAYERS_TRACE; ++l) {
            uint32_t n = counts[g][l];
            if (n == 0)
                continue;
            active_layers += 1;
            if (n > 4)
                any_wide += 1;
            else
                any_narrow += 1;
        }
        if (active_layers == 0)
            continue;
        if (any_wide > 0 && !last_wide) {
            /* Documented request-boundary heuristic on the frozen campaign
             * structure: a wide (prefill-chunk) graph that does not directly
             * follow another wide graph starts a new sequence, so per-layer
             * policy state is reset at request starts. */
            CHECK(dali_policy_request_start(native) == 0 &&
                      dali_policy_request_start(bpf) == 0,
                  "replay request start");
            ++requests;
        }
        last_wide = any_wide > 0;
        for (l = 0; l < MAX_LAYERS_TRACE; ++l) {
            uint32_t selected[DALI_WARC_MAX_EXPERTS];
            uint32_t n = 0;
            for (e = 0; e < DALI_WARC_MAX_EXPERTS; ++e)
                if (picks[g][l][e])
                    selected[n++] = e;
            CHECK(dali_policy_step(native, l, selected, n, n > 4) == 0,
                  "replay native step");
            CHECK(dali_policy_step(bpf, l, selected, n, n > 4) == 0,
                  "replay bpf step");
            CHECK(dali_policy_resident_layer(native, l, rn) == 0 &&
                      dali_policy_resident_layer(bpf, l, rb) == 0,
                  "replay probe");
            CHECK(memcmp(rn, rb, DALI_WARC_MAX_EXPERTS) == 0,
                  "replay resident mismatch");
            ++steps;
        }
    }
    CHECK(dali_policy_stats(native, &sn) == 0 &&
              dali_policy_stats(bpf, &sb) == 0,
          "replay stats rc");
    CHECK(sn.hits == sb.hits && sn.misses == sb.misses &&
              sn.loads == sb.loads && sn.evictions == sb.evictions &&
              sn.windows == sb.windows &&
              sn.empty_windows == sb.empty_windows &&
              sn.requests == sb.requests,
          "replay cumulative metric mismatch");
    /* Engagement and overhead counts are metadata only: retained and
     * printed, never a gate, retry, filter, or rejection. */
    {
        double hrn = sn.routed_events ? (double)sn.hits /
                                          (double)sn.routed_events
                                      : 0.0;
        double hrs = sb.routed_events ? (double)sb.hits /
                                          (double)sb.routed_events
                                      : 0.0;
        double hrd = sn.routed_events_decode ? (double)sn.hits_decode /
                                                  (double)sn.routed_events_decode
                                              : 0.0;
        double hrsd = sb.routed_events_decode ? (double)sb.hits_decode /
                                                    (double)sb.routed_events_decode
                                                : 0.0;
        printf("trace: steps %llu, requests %llu, routed %llu | native "
               "hit %.4f (decode %.4f), bpf hit %.4f (decode %.4f), "
               "misses native %llu bpf %llu, windows %llu, "
               "bpf_rank_calls %llu, bpf_errors %llu, "
               "selector_errors native %llu bpf %llu, "
               "decision_ns native %llu bpf %llu, bytes_missed %llu\n",
               (unsigned long long)steps, (unsigned long long)requests,
               (unsigned long long)sn.routed_events, hrn, hrd, hrs, hrsd,
               (unsigned long long)sn.misses, (unsigned long long)sb.misses,
               (unsigned long long)sn.windows,
               (unsigned long long)sb.bpf_rank_calls,
               (unsigned long long)sb.bpf_errors,
               (unsigned long long)sn.selector_errors,
               (unsigned long long)sb.selector_errors,
               (unsigned long long)sn.decision_ns,
               (unsigned long long)sb.decision_ns,
               (unsigned long long)(sn.misses * 13253760ULL));
    }
    dali_policy_destroy(native);
    dali_policy_destroy(bpf);
}

/* --------------------------- fail-closed ---------------------------- */

static void test_fail_closed(const char *lib, const char *code)
{
    void *state = 0;
    struct dali_policy_stats s;
    uint32_t selected = 7;

    CHECK(dali_policy_create("bpf", "/nonexistent/lib.so", code, 1, 8, 4, 2,
                             1, 4, &state)
              == DALI_WARC_ERROR,
          "bad bpf library rejected");
    CHECK(dali_policy_create("bpf", lib, "/nonexistent/code.bin", 1, 8, 4, 2,
                             1, 4, &state)
              == DALI_WARC_ERROR,
          "bad bpf bytecode rejected");
    CHECK(dali_policy_create("native", 0, 0, 1, 8, 4, 2, 1, 4, &state) == 0,
          "native create ok");
    CHECK(state != 0, "native state");
    CHECK(dali_policy_step(state, 0, &selected, 1, 0) == 0, "step ok");
    CHECK(dali_policy_step(0, 0, &selected, 1, 0) == DALI_WARC_ERROR,
          "null state rejected");
    CHECK(dali_policy_stats(state, &s) == 0, "stats rc");
    CHECK(s.bpf_calls == 0 && s.bpf_ranked == 0,
          "native arm has no bpf counters");
    dali_policy_destroy(state);
    dali_policy_destroy(0); /* must not crash */
}

int main(int argc, char **argv)
{
    const char *trace =
        "/home/yunwei37/workspace/gpu/gpu_ext/workloads/"
        "expert-buffering-policy/raw/timing/block-01/llama_ncmoe32/"
        "trace.jsonl";
    const char *lib = "/home/yunwei37/workspace/gpu/gpu_ext/extension/"
                      ".output/libmoe_expert_policy.so";
    const char *code = "/home/yunwei37/workspace/gpu/gpu_ext/extension/"
                       ".output/moe_expert_policy_rank.bin";

    if (argc > 1)
        trace = argv[1];
    if (argc > 2)
        lib = argv[2];
    if (argc > 3)
        code = argv[3];

    test_native_oracle();
    test_invariants();
    test_fail_closed(lib, code);
    test_differential_random(lib, code);
    test_trace_replay(trace, lib, code);

    if (failures == 0) {
        printf("=== TEST SUMMARY === {\"pass\": true, \"failures\": 0}\n");
        return 0;
    }
    printf("=== TEST SUMMARY === {\"pass\": false, \"failures\": %d}\n",
           failures);
    return 1;
}
