// CUDA-free lifecycle tests. Completion calls below are CPU fixtures, not IO.
#include "adapter_state.h"
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace eb_adapter;
static unsigned checks = 0;
static void Check(bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("adapter state assertion failed");
}
template <class F> static void Reject(F action) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error &) { rejected = true; }
    Check(rejected);
}
static eb_context Step(State &state, eb_u32 layer, eb_u32 expert) {
    auto ctx = state.Decide(layer, expert, std::vector<bool>(state.Get(layer).nodes.size(), true));
    if (ctx.output.status == EB_EVICT) state.Evicted(ctx);
    if (ctx.output.status == EB_ADMIT || ctx.output.status == EB_EVICT) {
        state.CanAdmit(layer, ctx.input.batch_epoch, expert);
        state.Admitted(layer, ctx.input.batch_epoch, expert); // successful-copy fixture
    }
    return ctx;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    try {
        const std::string library = argv[1], bytecode = argv[2];
        for (auto arm : {Arm::Fifo, Arm::Native, Arm::Bpf}) {
            State state(arm, 2, library, bytecode);
            auto epoch = state.Begin(7, 0, {100, 101, 102, 103}, {1, 1, 1, 0});
            Check(state.Locate(101) == std::make_pair(7u, 1u));
            Reject([&] { state.Begin(8, 0, {200, 201}, {1, 1}); });
            Reject([&] { state.ActiveEpoch(7, 3); });
            state.CanAdmit(7, epoch, 0);
            Check(state.Get(7).serial == 0 && state.Stats().admissions == 0);
            auto empty = state.Decide(7, 0, {true, true, true, true});
            for (auto status : std::array<eb_u32, 6>{EB_HIT, EB_EVICT, EB_BLOCKED, EB_INVALID, 99, 0xffffffffu}) {
                auto corrupt = empty;
                corrupt.output.status = status;
                Reject([&] { state.Validate(corrupt); });
            }
            auto stale = Step(state, 7, 0);
            Reject([&] { state.Validate(stale); });
            Check(Step(state, 7, 1).output.status == EB_ADMIT);
            Check(Step(state, 7, 0).output.status == EB_HIT);
            Check(state.Get(7).input.experts[0].admission == 1);
            auto hit = state.Decide(7, 0, {true, true, true, true});
            hit.output.status = EB_ADMIT;
            Reject([&] { state.Validate(hit); });
            auto blocked = state.Decide(7, 2, {false, false, false, false});
            Check(blocked.output.status == EB_BLOCKED);
            Reject([&] { state.Evicted(blocked); });
            auto forged_evict = blocked;
            forged_evict.output.status = EB_EVICT;
            forged_evict.output.victim = 0;
            Reject([&] { state.Validate(forged_evict); });
            auto eligible_miss = state.Decide(7, 2, {true, true, true, true});
            for (auto status : {EB_BLOCKED, EB_ADMIT, EB_HIT}) {
                auto corrupt = eligible_miss;
                corrupt.output.status = status;
                corrupt.output.victim = EB_NO_VICTIM;
                Reject([&] { state.Validate(corrupt); });
            }
            auto evict = Step(state, 7, 2);
            Check(evict.output.victim == (arm == Arm::Fifo ? 0u : 1u));
            Check(state.Get(7).serial == 3 && state.Stats().evictions == 1);
            auto stats = state.Stats();
            Check(stats.jit_calls == (arm == Arm::Bpf ? stats.decisions : 0));
            state.End(7, epoch);
            Reject([&] { state.CanAdmit(7, epoch, 3); });
            Reject([&] { state.Begin(7, 0, {100, 101, 102, 104}, {1, 1, 1, 1}); });
            Reject([&] { state.Begin(8, 0, {100, 105}, {1, 1}); });
            Reject([&] { state.Begin(8, 0, {105, 105}, {1, 1}); });
            epoch = state.Begin(7, 0, {100, 101, 102, 103}, {0, 0, 1, 1});
            Reject([&] { state.Validate(evict); });
            auto active = state.Decide(7, 3, {true, true, true, true});
            Check(active.output.status == EB_EVICT);
            auto corrupt = active;
            corrupt.output.batch_epoch++;
            Reject([&] { state.Validate(corrupt); });
            corrupt = active;
            corrupt.input.experts[2].admission++;
            Reject([&] { state.Evicted(corrupt); });
            state.End(7, epoch);
        }

        State native(Arm::Native, 16, library, bytecode);
        State bpf(Arm::Bpf, 16, library, bytecode);
        std::vector<NodeId> nodes;
        for (unsigned i = 0; i < 60; ++i) nodes.push_back(500 + i);
        std::mt19937 random(60024);
        for (unsigned batch = 0; batch < 32; ++batch) {
            Counts counts;
            for (unsigned i = 0; i < 60; ++i) counts.push_back(random() % 3 ? 0 : 1 + random() % 4);
            auto ne = native.Begin(0, 0, nodes, counts);
            auto be = bpf.Begin(0, 0, nodes, counts);
            Check(ne == be);
            for (unsigned expert = 0; expert < 60; ++expert) {
                if (!counts[expert]) continue;
                auto a = Step(native, 0, expert), b = Step(bpf, 0, expert);
                Check(a.output.status == b.output.status && a.output.victim == b.output.victim);
                Check(native.Get(0).serial == bpf.Get(0).serial);
            }
            native.End(0, ne);
            bpf.End(0, be);
        }
        auto nstats = native.Stats(), bstats = bpf.Stats();
        Check(nstats.decisions == bstats.decisions && bstats.decisions == bstats.jit_calls);
        Check(nstats.admissions == bstats.admissions && nstats.evictions == bstats.evictions);
        Reject([] { State::NextSerial(std::numeric_limits<eb_u64>::max()); });
        Reject([&] { State bad(Arm::Bpf, 2, library, "/nonexistent-eb-bytecode"); });
        std::cout << "adapter_state_checks=" << checks
                  << " paired_native_decisions=" << nstats.decisions
                  << " actual_bpf_jit_decisions=" << bstats.jit_calls << "\n";
    } catch (const std::exception &error) {
        std::cerr << "EB_ADAPTER_CPU_FAILURE: " << error.what() << "\n";
        return 1;
    }
}
