#include "selector_policy.h"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

static unsigned reference_op(unsigned p, unsigned d, unsigned ticket, bool prop) {
    if (!prop) return ticket % 2;
    if (p <= d) return ticket % (d / p + 1) == 0 ? 0 : 1;
    return ticket % (p / d + 1) < p / d ? 0 : 1;
}

static PodSelectorContext context(std::vector<unsigned> &counts, unsigned p,
                                 unsigned d, unsigned sm, bool prop) {
    PodSelectorContext c{};
    c.counters = reinterpret_cast<pod_u64>(counts.data());
    c.abi_version = POD_ABI_VERSION;
    c.nsmid = counts.size() - 2;
    c.smid = sm;
    c.prefill_slots = p;
    c.decode_slots = d;
    c.proportional = prop;
    c.grid_ctas = p + d;
    return c;
}

int main() {
    static_assert(offsetof(PodSelectorContext, out_op) == 36);
    static_assert(offsetof(PodSelectorContext, engine) == 48);
    std::mt19937 rng(20260903);
    for (bool prop : {false, true}) {
        for (unsigned p : {1u, 2u, 5u, 32u, 513u}) {
            for (unsigned d : {1u, 3u, 17u, 128u, 1025u}) {
                std::vector<unsigned> counts(130), tickets(128);
                std::vector<bool> seen_p(p), seen_d(d);
                unsigned valid = 0;
                for (unsigned i = 0; i < p + d + 9; ++i) {
                    // Deliberately sparse identifiers, beyond an active SM count.
                    unsigned sm = (rng() % 42) * 3;
                    auto c = context(counts, p, d, sm, prop);
                    auto expected_ticket = tickets[sm]++;
                    auto first_op = reference_op(p, d, expected_ticket, prop);
                    auto first_claim = counts[128 + first_op];
                    bool fallback = first_claim >= (first_op == 0 ? p : d);
                    auto final_op = fallback ? 1 - first_op : first_op;
                    auto final_claim = fallback ? counts[128 + final_op] : first_claim;
                    pod_select_policy(&c, sizeof(c), POD_ENGINE_CUDA);
                    assert(c.ticket == expected_ticket && c.first_op == first_op);
                    assert(c.first_claim == first_claim && c.out_op == final_op);
                    assert(c.out_cta == final_claim);
                    assert(c.fallback_claim == (fallback ? final_claim : POD_UNSET));
                    assert(c.engine == POD_ENGINE_CUDA);
                    if (c.status == POD_WORK) {
                        auto &seen = c.out_op == 0 ? seen_p : seen_d;
                        assert(c.out_cta < seen.size() && !seen[c.out_cta]);
                        seen[c.out_cta] = true;
                        ++valid;
                    } else assert(c.status == POD_EXHAUSTED);
                }
                assert(valid == p + d);
                for (bool seen : seen_p) assert(seen);
                for (bool seen : seen_d) assert(seen);
            }
        }
    }
    std::vector<unsigned> counts(130);
    for (unsigned field = 0; field < 7; ++field) {
        auto c = context(counts, 3, 7, 127, true);
        if (field == 0) c.smid = 128;
        if (field == 1) c.abi_version = 0;
        if (field == 2) c.prefill_slots = 0;
        if (field == 3) c.decode_slots = 0;
        if (field == 4) c.counters = 0;
        if (field == 5) c.proportional = 2;
        if (field == 6) c.grid_ctas = 0;
        pod_select_policy(&c, sizeof(c), POD_ENGINE_CUDA);
        assert(c.status == POD_BAD_INPUT && c.out_op == POD_UNSET);
        for (auto v : counts) assert(v == 0);
    }
    auto short_ctx = context(counts, 3, 7, 127, true);
    pod_select_policy(&short_ctx, sizeof(short_ctx) - 1, POD_ENGINE_CUDA);
    assert(short_ctx.engine == 0);
    std::cout << "PASS: ABI, original ratio rule, sparse SM IDs, atomic claims, fallback, "
                 "exactly-once slots, exhaustion and invalid input (CPU only)\n";
}
