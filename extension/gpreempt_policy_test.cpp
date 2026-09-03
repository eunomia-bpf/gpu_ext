// SPDX-License-Identifier: GPL-2.0
#include "gpreempt_bridge.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

// Independent direct transcription of upstream gpreemptclient.cpp branches;
// BPF decisions execute through the same linked bridge as the actual client.
static int original(unsigned event, unsigned role, gp_u64 now, gp_u64 deadline,
                    bool initialized, bool reserve)
{
    if (role != 0 || !initialized) return 0;
    switch (event) {
    case GP_PREPROCESS: return reserve ? 3 : 5;
    case GP_DUE: return now > deadline ? 4 : 0;
    case GP_INFER: return 8;
    default: throw std::runtime_error("invalid test input");
    }
}

int main()
{
    gp_u64 checked = 0;
    auto check = [&](unsigned event, unsigned role, gp_u64 now, gp_u64 deadline,
                     bool initialized, bool reserve) {
        int expected = original(event, role, now, deadline, initialized, reserve);
        int actual = gpreempt_hint_decide(event, role, now, deadline, initialized, reserve);
        if (actual != expected) throw std::runtime_error("hint differs from original GPReempt branch");
        ++checked;
    };
    const std::array<gp_u64, 8> boundaries{0, 1, 99, 100, 101, 100000,
                                          1788388800000000000ULL, std::numeric_limits<gp_u64>::max()};
    for (unsigned event = 1; event <= 3; ++event)
        for (unsigned role = 0; role <= 1; ++role)
            for (unsigned initialized = 0; initialized <= 1; ++initialized)
                for (unsigned reserve = 0; reserve <= 1; ++reserve)
                    for (gp_u64 now : boundaries)
                        for (gp_u64 deadline : boundaries)
                            check(event, role, now, deadline, initialized, reserve);
    std::mt19937_64 random(20260902);
    for (unsigned i = 0; i < 100000; ++i) {
        gp_u64 now = random(), deadline = random();
        if (i % 5 == 0) deadline = now; // Equal deadline must NOT block.
        check(1 + random() % 3, random() % 2, now, deadline, random() % 2, random() % 2);
    }
    if (gpreempt_hint_decide(0, 0, 0, 0, 1, 1) != -1 ||
        gpreempt_hint_decide(4, 0, 0, 0, 1, 1) != -1 ||
        gpreempt_hint_decide(1, 2, 0, 0, 1, 1) != -1 ||
        gpreempt_hint_decide(1, 0, 0, 0, 2, 1) != -1 ||
        gpreempt_hint_decide(1, 0, 0, 0, 1, 2) != -1)
        throw std::runtime_error("invalid hint input did not fail closed");
    if (gpreempt_bpf_enabled()) {
        unsetenv("GPREEMPT_BPF_MAPS");
        if (gpreempt_ctx_begin(GP_LC) != -1)
            throw std::runtime_error("missing kernel attachment was accepted");
    }
    std::cout << "{\"test\":\"gpreempt_original_branches_vs_bridge\",\"backend\":\""
              << (gpreempt_bpf_enabled() ? "ubpf-jit" : "original-c") << "\",\"decisions\":"
              << checked << ",\"mismatches\":0,\"invalid_inputs_rejected\":5,"
              << "\"strict_deadline\":true,\"gpu_executed\":false}" << std::endl;
}
