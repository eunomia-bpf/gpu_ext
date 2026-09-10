// SPDX-License-Identifier: Apache-2.0
// Native oracle for the governor admission rule: reads raw facts, applies the
// published rule in C++, and demands exact parity with the BPF arm. No BPF
// implementation detail is reused here.
#include "moe_spec_admission.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static std::atomic<mep_u64> cases{0}, admitted_cases{0}, empty_cases{0};
static void require(bool condition, const char *message)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}

struct Raw { mep_u64 budget, outstanding; };

static size_t native_prefix(const std::vector<mep_u64> &payloads,
                            mep_u64 budget, mep_u64 outstanding)
{
    const mep_u64 allowed = outstanding < budget ? budget - outstanding : 0;
    mep_u64 used = 0;
    size_t admitted = 0;
    for (size_t i = 0; i < payloads.size(); ++i) {
        const mep_u64 bytes = payloads[i];
        if (bytes == 0 || used + bytes > allowed) break;
        used += bytes;
        admitted = i + 1;
    }
    return admitted;
}

static void check(const std::vector<mep_u64> &payloads, const Raw &facts)
{
    std::vector<moe_spec_candidate> candidates;
    for (const mep_u64 bytes : payloads)
        candidates.push_back({bytes, 0});
    mep_u64 admitted = MOE_SPEC_NONE;
    const int rc = moe_spec_admission_prefix_v1(
        candidates.data(), static_cast<mep_u32>(candidates.size()),
        facts.budget, facts.outstanding, &admitted);
    require(rc == 0, "spec admission call failed");
    require(admitted != MOE_SPEC_INVALID, "spec admission returned INVALID on a valid snapshot");
    const mep_u64 expected = native_prefix(payloads, facts.budget, facts.outstanding);
    require(admitted == expected, "BPF admission disagrees with native rule oracle");
    ++cases;
    if (admitted == 0) ++empty_cases; else ++admitted_cases;
}

int main(int argc, char **argv)
{
    require(argc == 2, "usage: moe_spec_admission_test <absolute admission .bin>");
    require(moe_spec_admission_init_v1(argv[1]) == 0, "admission BPF init failed");
    std::mt19937_64 rng(20260904ULL);
    // Boundary cases: empty, zero budget, saturated outstanding, one candidate,
    // exact-fit prefix, first-candidate-too-large.
    const std::vector<std::vector<mep_u64>> boundary = {
        {},
        {13ULL << 20},
        {128ULL << 20},
        {64ULL << 20, 64ULL << 20},
        {13ULL << 20, 13ULL << 20, 13ULL << 20},
    };
    const std::vector<Raw> boundary_facts = {
        {256ULL << 20, 0},
        {0, 0},
        {128ULL << 20, 128ULL << 20},
        {256ULL << 20, 64ULL << 20},
        {512ULL << 20, 448ULL << 20},
        {13ULL << 20, 0},
    };
    for (const auto &payloads : boundary)
        for (const Raw &facts : boundary_facts)
            check(payloads, facts);
    // Randomized sweep over realistic expert payload sizes mixed with noise.
    std::uniform_int_distribution<mep_u64> size_distribution(1ULL << 20, 32ULL << 20);
    std::uniform_int_distribution<mep_u64> budget_distribution(64ULL << 20, 2048ULL << 20);
    std::uniform_int_distribution<mep_u64> outstanding_distribution(0, 1024ULL << 20);
    for (int iteration = 0; iteration < 4096; ++iteration) {
        std::vector<mep_u64> payloads;
        const size_t count = 1 + rng() % 200;
        payloads.reserve(count);
        for (size_t i = 0; i < count; ++i) payloads.push_back(size_distribution(rng));
        const Raw facts{budget_distribution(rng), outstanding_distribution(rng)};
        check(payloads, facts);
    }
    struct moe_spec_admission_stats stats{};
    moe_spec_admission_stats_v1(&stats);
    require(stats.errors == 0, "bridge recorded errors");
    require(stats.calls == cases.load(), "bridge call count mismatch");
    std::printf("spec admission parity OK: cases=%llu admitted=%llu empty=%llu errors=%llu\n",
                cases.load(), admitted_cases.load(), empty_cases.load(), stats.errors);
    return 0;
}
