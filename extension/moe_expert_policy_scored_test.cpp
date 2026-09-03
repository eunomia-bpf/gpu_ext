// SPDX-License-Identifier: Apache-2.0
#include "moe_expert_policy.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

static std::atomic<mep_u64> decisions{0}, ranked{0}, scored_selected{0}, ranked_empty{0};
static std::atomic<mep_u64> matched{0}, match_empty{0};
static void require(bool condition, const char *message)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
static double number(mep_u64 bits)
{
    static_assert(sizeof(double) == sizeof(bits) && std::numeric_limits<double>::is_iec559,
                  "IEEE-754 float64 is required");
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Raw model facts and IEEE double arithmetic form the independent native oracle.
// No BPF bit-order transform or BPF sort implementation is reused here.
struct Raw { mep_u64 identity, bits; bool present, cuda; int pending, state; };

static void check(const std::vector<Raw> &raw)
{
    std::vector<moe_expert_scored_candidate> scored;
    std::vector<moe_expert_rank_candidate> rank;
    mep_u64 expected_victim = MOE_EXPERT_NONE;
    double minimum = std::numeric_limits<double>::infinity();
    std::vector<mep_u32> expected_rank;
    std::vector<mep_u32> expected_match;
    double maximum = -std::numeric_limits<double>::infinity();
    for (mep_u32 i = 0; i < raw.size(); ++i) {
        const Raw &r = raw[i];
        const double score = number(r.bits);
        if (r.present && r.cuda && r.pending == 0 && r.state == 0 && score < minimum) {
            minimum = score;
            expected_victim = i;
        }
        if (score > 0) expected_rank.push_back(i);
        if (score > maximum) { maximum = score; expected_match.clear(); }
        if (score == maximum) expected_match.push_back(i);
        const mep_u32 flags = (r.present ? MOE_EXPERT_NODE_PRESENT : 0) |
            (r.cuda ? MOE_EXPERT_DEVICE_CUDA : 0) | (r.pending == 0 ? MOE_EXPERT_PENDING_ZERO : 0) |
            (r.state == 0 ? MOE_EXPERT_EXEC_IDLE : 0);
        scored.push_back({r.identity, r.bits, flags, 0});
        rank.push_back({r.identity, r.bits, i, 0});
    }
    std::stable_sort(expected_rank.begin(), expected_rank.end(), [&](mep_u32 left, mep_u32 right) {
        return number(raw[left].bits) > number(raw[right].bits);
    });
    mep_u64 victim = 0;
    require(moe_expert_scored_select_v1(scored.data(), scored.size(), &victim) == 0,
            "scored JIT failed");
    require(victim == expected_victim, "numeric min / JIT victim mismatch");
    const mep_u32 canary = 0xfe12abcdU;
    std::vector<mep_u32> indices(raw.size() + 2, canary);
    mep_u32 count = 0;
    require(moe_expert_rank_v1(rank.data(), rank.size(), indices.data() + 1, rank.size(), &count) == 0,
            "rank JIT failed");
    require(count == expected_rank.size(), "positive filtering count mismatch");
    require(indices.front() == canary && indices.back() == canary, "rank output overrun");
    require(std::equal(expected_rank.begin(), expected_rank.end(), indices.begin() + 1),
            "native stable_sort / BPF merge-sort mismatch");
    const mep_u32 rank_count = count;
    std::fill(indices.begin(), indices.end(), canary);
    require(moe_expert_match_v1(rank.data(), rank.size(), indices.data() + 1, rank.size(), &count) == 0,
            "match JIT failed");
    require(count == expected_match.size(), "maximum tie count mismatch");
    require(indices.front() == canary && indices.back() == canary, "match output overrun");
    require(std::equal(expected_match.begin(), expected_match.end(), indices.begin() + 1),
            "native numeric maximum / BPF match mismatch");
    ++decisions;
    ranked += rank_count;
    matched += count;
    if (victim != MOE_EXPERT_NONE) ++scored_selected;
    if (!rank_count) ++ranked_empty;
    if (!count) ++match_empty;
}

static const mep_u64 special[]{
    0, 0x8000000000000000ULL, // +/- zero
    1, 0x8000000000000001ULL, // +/- smallest subnormal
    0x0010000000000000ULL, 0x8010000000000000ULL, // +/- smallest normal
    0x3ff0000000000000ULL, 0xbff0000000000000ULL, // +/- one
    0x3ff0000000000001ULL, // nextafter(1,+Inf): must not round to float32 tie
    0x7fefffffffffffffULL, 0xffefffffffffffffULL, // +/- largest finite
    0x7ff0000000000000ULL, 0xfff0000000000000ULL, // +/- infinity
    0x7ff8000000000000ULL, 0xfff8000000000000ULL, // +/- quiet NaN
    0x7ff0000000000001ULL, 0xfff0000000000001ULL, // +/- signaling NaN
};

static void random_cases(unsigned seed, unsigned count)
{
    std::mt19937_64 random(seed);
    for (unsigned trial = 0; trial < count; ++trial) {
        std::vector<Raw> raw;
        unsigned size = random() % 257;
        for (unsigned i = 0; i < size; ++i) {
            mep_u64 bits = random() % 2 ? random() : special[random() % (sizeof(special) / sizeof(*special))];
            raw.push_back({random(), bits, bool(random() % 2), bool(random() % 2),
                           static_cast<int>(random() % 3) - 1, static_cast<int>(random() % 3)});
        }
        check(raw);
    }
}

int main(int argc, char **argv)
{
    require(argc == 4, "scored, rank, and match bytecode arguments required");
    unsetenv("MOE_EXPERT_SCORED_CODE");
    unsetenv("MOE_EXPERT_RANK_CODE");
    unsetenv("MOE_EXPERT_MATCH_CODE");
    require(moe_expert_scored_init_v1(nullptr) < 0, "missing scored program accepted");
    require(moe_expert_rank_init_v1(nullptr) < 0, "missing rank program accepted");
    require(moe_expert_match_init_v1(nullptr) < 0, "missing match program accepted");
    require(moe_expert_scored_init_v1(argv[1]) == 0, "scored JIT initialization failed");
    require(moe_expert_rank_init_v1(argv[2]) == 0, "rank JIT initialization failed");
    require(moe_expert_match_init_v1(argv[3]) == 0, "match JIT initialization failed");
    check({});
    for (mep_u64 a : special) {
        check({{99, a, true, true, 0, 0}});
        for (mep_u64 b : special) {
            check({{99, a, true, true, 0, 0}, {0, b, true, true, 0, 0}});
            check({{0, b, true, true, 0, 0}, {99, a, true, true, 0, 0}});
        }
        for (unsigned facts = 0; facts < 16; ++facts)
            check({{0, a, bool(facts & 1), bool(facts & 2), facts & 4 ? 0 : 1, facts & 8 ? 0 : 2}});
    }
    std::vector<Raw> maximum;
    for (unsigned i = 0; i < MOE_EXPERT_MAX_CANDIDATES; ++i)
        maximum.push_back({MOE_EXPERT_MAX_CANDIDATES - i,
                           0x3ff0000000000000ULL + (i % 137), true, true, 0, 0});
    check(maximum);
    std::reverse(maximum.begin(), maximum.end());
    check(maximum);
    for (Raw &r : maximum) r.bits = 0x3ff0000000000000ULL;
    check(maximum); // maximum all-tie workload must preserve every index
    random_cases(581, 20000);
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < 4; ++i) threads.emplace_back(random_cases, 781 + i, 5000);
    for (auto &thread : threads) thread.join();

    moe_expert_policy_stats score_stats{};
    moe_expert_scored_stats_v1(&score_stats);
    moe_expert_rank_stats rank_stats{};
    moe_expert_rank_stats_v1(&rank_stats);
    require(score_stats.calls == decisions && score_stats.selected == scored_selected &&
            score_stats.selected + score_stats.no_victim == decisions, "scored counters wrong");
    require(rank_stats.calls == decisions && rank_stats.ranked == ranked && rank_stats.empty == ranked_empty,
            "rank counters wrong");
    moe_expert_match_stats match_stats{};
    moe_expert_match_stats_v1(&match_stats);
    require(match_stats.calls == decisions && match_stats.matched == matched && match_stats.empty == match_empty,
            "match counters wrong");
    mep_u64 victim;
    mep_u32 count, index;
    moe_expert_scored_candidate s{0, 0, MOE_EXPERT_ELIGIBLE, 0};
    moe_expert_rank_candidate r{0, 0x3ff0000000000000ULL, 0, 0};
    require(moe_expert_scored_select_v1(nullptr, 1, &victim) < 0, "scored null accepted");
    require(moe_expert_scored_select_v1(&s, 1, nullptr) < 0, "scored null output accepted");
    require(moe_expert_scored_select_v1(&s, MOE_EXPERT_MAX_CANDIDATES + 1, &victim) < 0, "scored oversized accepted");
    s.flags = 16;
    require(moe_expert_scored_select_v1(&s, 1, &victim) < 0, "scored unknown flags accepted");
    s.flags = 15; s.reserved = 1;
    require(moe_expert_scored_select_v1(&s, 1, &victim) < 0, "scored reserved accepted");
    require(moe_expert_rank_v1(nullptr, 1, &index, 1, &count) < 0, "rank null accepted");
    require(moe_expert_rank_v1(&r, 1, nullptr, 1, &count) < 0, "rank null output accepted");
    require(moe_expert_rank_v1(&r, 1, &index, 1, nullptr) < 0, "rank null count accepted");
    require(moe_expert_rank_v1(&r, 1, &index, 0, &count) < 0, "rank insufficient output accepted");
    require(moe_expert_rank_v1(&r, MOE_EXPERT_MAX_CANDIDATES + 1, &index,
                              MOE_EXPERT_MAX_CANDIDATES + 1, &count) < 0, "rank oversized accepted");
    r.ordinal = 1;
    require(moe_expert_rank_v1(&r, 1, &index, 1, &count) < 0, "rank reordered ordinal accepted");
    r.ordinal = 0; r.reserved = 1;
    require(moe_expert_rank_v1(&r, 1, &index, 1, &count) < 0, "rank reserved accepted");
    r.reserved = 0;
    require(moe_expert_match_v1(nullptr, 1, &index, 1, &count) < 0, "match null accepted");
    require(moe_expert_match_v1(&r, 1, nullptr, 1, &count) < 0, "match null output accepted");
    require(moe_expert_match_v1(&r, 1, &index, 1, nullptr) < 0, "match null count accepted");
    require(moe_expert_match_v1(&r, 1, &index, 0, &count) < 0, "match insufficient output accepted");
    require(moe_expert_match_v1(&r, MOE_EXPERT_MAX_CANDIDATES + 1, &index,
                               MOE_EXPERT_MAX_CANDIDATES + 1, &count) < 0, "match oversized accepted");
    r.ordinal = 1;
    require(moe_expert_match_v1(&r, 1, &index, 1, &count) < 0, "match reordered ordinal accepted");
    r.ordinal = 0; r.reserved = 1;
    require(moe_expert_match_v1(&r, 1, &index, 1, &count) < 0, "match reserved accepted");
    moe_expert_scored_stats_v1(&score_stats);
    moe_expert_rank_stats_v1(&rank_stats);
    moe_expert_match_stats_v1(&match_stats);
    require(score_stats.errors == 6 && rank_stats.errors == 8 && match_stats.errors == 8,
            "negative cases not counted");
    std::printf("moe_expert_policy_scored_test: backend=ubpf-jit scored_decisions=%llu "
                "rank_decisions=%llu match_decisions=%llu ranked_indices=%llu matched_indices=%llu "
                "mismatch=0 negative_cases=22 threads=4 maximum_candidates=%u\n",
                decisions.load(), decisions.load(), decisions.load(), ranked.load(), matched.load(),
                MOE_EXPERT_MAX_CANDIDATES);
}
