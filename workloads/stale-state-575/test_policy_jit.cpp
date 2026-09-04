/* SPDX-License-Identifier: MIT */
#include "stale_state_policy_jit.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct Publication {
    stale_state_575_snapshot snapshot;
    uint64_t eligible_mono_ns;
};

struct DelayCounters {
    uint64_t decisions = 0;
    uint64_t dense = 0;
    uint64_t sparse = 0;
    uint64_t wrong_phase = 0;
};

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

stale_state_575_snapshot snapshot(uint64_t sequence,
                                  uint32_t phase,
                                  uint64_t source,
                                  uint64_t published)
{
    return stale_state_575_snapshot{sequence, source, published, phase, 0};
}

int native_choose(stale_state_575_jit_context *context)
{
    context->decision = stale_state_575_decision{};
    context->status = STALE_STATE_575_ACTION_REJECT;
    if (context->reserved != 0)
        return STALE_STATE_575_ACTION_REJECT;
    const auto action = stale_state_575_choose(&context->snapshot,
                                                context->decision_mono_ns,
                                                &context->decision);
    context->status = static_cast<uint32_t>(action);
    return action;
}

void compare_one(void *jit,
                 const stale_state_575_snapshot &input,
                 uint64_t decision_mono_ns,
                 uint32_t context_reserved = 0)
{
    stale_state_575_jit_context expected{
        input,
        decision_mono_ns,
        {},
        STALE_STATE_575_ACTION_REJECT,
        context_reserved,
    };
    stale_state_575_jit_context actual = expected;
    std::memset(&actual.decision, 0xa5, sizeof(actual.decision));
    actual.status = 0xa5a5a5a5U;

    const int expected_action = native_choose(&expected);
    const int actual_action = stale_state_575_jit_choose(jit, &actual, sizeof(actual));
    require(actual_action == expected_action, "JIT/native action mismatch");
    require(std::memcmp(&actual, &expected, sizeof(actual)) == 0,
            "JIT/native context mismatch");
}

void run_boundaries(void *jit)
{
    const auto valid = snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 120);
    compare_one(jit, valid, 170);
    compare_one(jit, snapshot(2, STALE_STATE_575_PHASE_SPARSE, 200, 220), 260);
    compare_one(jit, snapshot(0, STALE_STATE_575_PHASE_DENSE, 100, 120), 170);
    compare_one(jit, snapshot(1, STALE_STATE_575_PHASE_INVALID, 100, 120), 170);
    compare_one(jit, snapshot(1, 99, 100, 120), 170);
    compare_one(jit, snapshot(1, STALE_STATE_575_PHASE_DENSE, 0, 120), 170);
    compare_one(jit, snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 0), 170);
    compare_one(jit, snapshot(1, STALE_STATE_575_PHASE_DENSE, 120, 100), 170);
    compare_one(jit, snapshot(1, STALE_STATE_575_PHASE_DENSE, 100, 180), 170);
    auto torn = valid;
    torn.reserved = 1;
    compare_one(jit, torn, 170);
    compare_one(jit, valid, 170, 1);

    stale_state_575_jit_context undersized{};
    const auto calls_before = stale_state_575_jit_calls(jit);
    require(stale_state_575_jit_choose(jit, &undersized, sizeof(undersized) - 1) ==
                STALE_STATE_575_ACTION_REJECT,
            "undersized context did not fail closed");
    require(stale_state_575_jit_calls(jit) == calls_before + 1,
            "undersized context did not execute the BPF ABI guard");
    require(stale_state_575_jit_choose(jit, nullptr, 0) == -1,
            "null context was not rejected by wrapper");
    require(stale_state_575_jit_calls(jit) == calls_before + 1,
            "null wrapper input unexpectedly reached JIT");
}

DelayCounters run_stream(void *jit, uint64_t delay_ns)
{
    constexpr uint64_t epoch_ns = 10'000'000'000ULL;
    constexpr uint64_t bootstrap_ns = 1'200'000'000ULL;
    constexpr uint64_t phase_ns = 2'000'000'000ULL;
    constexpr uint64_t cycles = 1000;
    constexpr uint64_t samples_per_phase = 17;
    std::minstd_rand random(static_cast<unsigned int>(20260903 + delay_ns / 1'000'000));
    std::uniform_int_distribution<uint64_t> sample_offset(0, phase_ns - 1);
    std::vector<Publication> publications;
    publications.reserve(1 + cycles * 6);

    publications.push_back({
        snapshot(1, STALE_STATE_575_PHASE_SPARSE, epoch_ns, epoch_ns + delay_ns),
        epoch_ns + delay_ns,
    });
    uint64_t sequence = 1;
    for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
        for (uint64_t phase = 0; phase < 6; ++phase) {
            ++sequence;
            const uint64_t source = epoch_ns + bootstrap_ns +
                                    (cycle * 6 + phase) * phase_ns;
            const uint32_t kind = phase % 2 == 0
                                      ? STALE_STATE_575_PHASE_DENSE
                                      : STALE_STATE_575_PHASE_SPARSE;
            publications.push_back({snapshot(sequence, kind, source, source + delay_ns),
                                    source + delay_ns});
        }
    }

    DelayCounters counters;
    size_t latest = 0;
    for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
        for (uint64_t phase = 0; phase < 6; ++phase) {
            const uint64_t source = epoch_ns + bootstrap_ns +
                                    (cycle * 6 + phase) * phase_ns;
            const uint32_t host_phase = phase % 2 == 0
                                            ? STALE_STATE_575_PHASE_DENSE
                                            : STALE_STATE_575_PHASE_SPARSE;
            std::array<uint64_t, samples_per_phase> offsets{};
            for (auto &offset : offsets)
                offset = sample_offset(random);
            std::sort(offsets.begin(), offsets.end());
            for (const uint64_t offset : offsets) {
                const uint64_t decision_time = source + offset;
                while (latest + 1 < publications.size() &&
                       publications[latest + 1].eligible_mono_ns <= decision_time)
                    ++latest;
                require(publications[latest].eligible_mono_ns <= decision_time,
                        "stream has no eligible snapshot");

                stale_state_575_jit_context expected{
                    publications[latest].snapshot,
                    decision_time,
                    {},
                    STALE_STATE_575_ACTION_REJECT,
                    0,
                };
                stale_state_575_jit_context actual = expected;
                const int expected_action = native_choose(&expected);
                const int actual_action = stale_state_575_jit_choose(jit, &actual, sizeof(actual));
                require(actual_action == expected_action,
                        "seeded stream JIT/native action mismatch");
                require(std::memcmp(&actual, &expected, sizeof(actual)) == 0,
                        "seeded stream JIT/native output mismatch");
                require(actual_action != STALE_STATE_575_ACTION_REJECT,
                        "eligible stream snapshot was rejected");

                ++counters.decisions;
                counters.dense += actual_action == STALE_STATE_575_ACTION_PREFETCH_MAX;
                counters.sparse += actual_action == STALE_STATE_575_ACTION_DISCARD_PREFETCH;
                counters.wrong_phase += stale_state_575_wrong_phase(
                    &actual.decision,
                    static_cast<stale_state_575_phase>(host_phase));
            }
        }
    }

    require(counters.dense != 0 && counters.sparse != 0,
            "seeded stream did not exercise both actions");
    if (delay_ns == 0)
        require(counters.wrong_phase == 0,
                "fresh stream observed a wrong-phase decision");
    else
        require(counters.wrong_phase != 0,
                "delayed stream did not exercise stale-phase decisions");
    return counters;
}

} // namespace

int main(int argc, char **argv)
{
    static_assert(sizeof(stale_state_575_jit_context) == 72, "JIT context ABI size");
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " BYTECODE\n";
        return 64;
    }

    char error[256]{};
    require(stale_state_575_jit_open(nullptr, error, sizeof(error)) == nullptr,
            "missing bytecode path was accepted");
    require(error[0] != '\0', "missing bytecode rejection had no diagnostic");

    void *jit = stale_state_575_jit_open(argv[1], error, sizeof(error));
    require(jit != nullptr, error[0] ? error : "JIT open failed");
    run_boundaries(jit);

    const auto fresh = run_stream(jit, 0);
    const auto delay_100 = run_stream(jit, 100'000'000ULL);
    const auto delay_1000 = run_stream(jit, 1'000'000'000ULL);
    const uint64_t expected_calls = 12 + fresh.decisions +
                                    delay_100.decisions + delay_1000.decisions;
    require(stale_state_575_jit_calls(jit) == expected_calls,
            "JIT call counter did not close");
    require(stale_state_575_jit_contract_errors(jit) == 0,
            "JIT contract error counter is nonzero");

    std::cout << "{\"test\":\"stale_state_native_vs_host_ubpf_jit\","
              << "\"seed\":20260903,\"boundary_jit_calls\":12,"
              << "\"calls\":" << expected_calls << ",\"contract_errors\":0,"
              << "\"delays\":["
              << "{\"delay_ms\":0,\"decisions\":" << fresh.decisions
              << ",\"dense\":" << fresh.dense << ",\"sparse\":" << fresh.sparse
              << ",\"wrong_phase\":" << fresh.wrong_phase << "},"
              << "{\"delay_ms\":100,\"decisions\":" << delay_100.decisions
              << ",\"dense\":" << delay_100.dense << ",\"sparse\":" << delay_100.sparse
              << ",\"wrong_phase\":" << delay_100.wrong_phase << "},"
              << "{\"delay_ms\":1000,\"decisions\":" << delay_1000.decisions
              << ",\"dense\":" << delay_1000.dense << ",\"sparse\":" << delay_1000.sparse
              << ",\"wrong_phase\":" << delay_1000.wrong_phase << "}]}\n";

    stale_state_575_jit_close(jit);
    return 0;
}
