// SPDX-License-Identifier: GPL-2.0
// Synthetic parser fixtures; no GPU initialization or experiment evidence.
#include "idle_executor.h"
#include <functional>
#include <iostream>
using namespace hummingbird;
Json::Value valid() {
    Json::Value p;
    p["schema_version"] = 1; p["model"] = "resnet152"; p["gpu_correctness_validated"] = true;
    p["launch_overhead_ns"] = 1000; p["large_after_ns"] = 40000;
    p["small_input_enabled"] = false; p["small_output_enabled"] = false;
    Json::Value k; k["index"] = 0; k["name"] = "synthetic_not_a_result";
    k["argument_count"] = 2; k["cap"] = 4; k["split_ns"] = 2000; k["whole_ns"] = 5000;
    for (auto value : {8, 2, 1}) k["grid"].append(value);
    for (auto value : {256, 1, 1}) k["block"].append(value);
    p["kernels"].append(k); return p;
}
int main(int argc, char **argv) {
    if (argc != 2) throw std::runtime_error("test_profile requires actual BPF bytecode path");
    size_t rejected = 0;
    auto check_bad = [&](std::function<void(Json::Value &)> mutate) {
        auto p = valid(); mutate(p); bool caught = false;
        try { Profile parsed(p); } catch (const std::exception &) { caught = true; }
        if (!caught) throw std::runtime_error("invalid synthetic profile was accepted");
        ++rejected;
    };
    Profile good(valid());
    if (good.kernels.size() != 1 || good.kernels[0].cap != 4 || good.small_input_enabled)
        throw std::runtime_error("valid synthetic profile parsed incorrectly");
    check_bad([](auto &p) { p["schema_version"] = 2; });
    check_bad([](auto &p) { p["model"] = "vgg"; });
    check_bad([](auto &p) { p["gpu_correctness_validated"] = false; });
    check_bad([](auto &p) { p["launch_overhead_ns"] = 0; });
    check_bad([](auto &p) { p["large_after_ns"] = -1; });
    check_bad([](auto &p) { p.removeMember("small_input_enabled"); });
    check_bad([](auto &p) { p["small_output_enabled"] = 1; });
    check_bad([](auto &p) { p["kernels"].clear(); });
    check_bad([](auto &p) { p["kernels"][0]["index"] = 1; });
    check_bad([](auto &p) { p["kernels"][0].removeMember("index"); });
    check_bad([](auto &p) { p["kernels"][0]["name"] = ""; });
    check_bad([](auto &p) { p["kernels"][0]["cap"] = 0; });
    check_bad([](auto &p) { p["kernels"][0]["cap"] = 17; });
    check_bad([](auto &p) { p["kernels"][0]["split_ns"] = 0; });
    check_bad([](auto &p) { p["kernels"][0]["argument_count"] = 65; });
    check_bad([](auto &p) { p["kernels"][0]["grid"][0] = 0; });
    check_bad([](auto &p) { p["kernels"][0]["grid"][0] = Json::UInt64(UINT32_MAX); });
    check_bad([](auto &p) { p["kernels"][0]["block"][1] = 5; });
    check_bad([](auto &p) { p["kernels"][0]["name"] = "nop"; });
    Policy c(false, ""), bpf(true, argv[1]);
    hb_input state{}; state.now_ns = 1000000; state.last_hp_activity_ns = 1;
    state.large_after_ns = 1000; state.launch_overhead_ns = 500; state.split_ns = 5000;
    state.whole_ns = 10000; state.hp_gpu_done = state.lp_gpu_done = state.lp_pending = 1;
    state.kernel_unstarted = state.consolidate = 1;
    for (unsigned int pending : {0, 1}) {
        state.hp_pending = pending;
        const auto left = c.decide(state), right = bpf.decide(state);
        if (left.action != (pending ? HB_STOP_LP : HB_WHOLE) || left.action != right.action ||
            left.next_tick_ns != right.next_tick_ns || left.bubble != right.bubble)
            throw std::runtime_error("actual runtime policy wrapper differs from expected semantics");
    }
    if (c.decisions != 2 || c.jit_decisions != 0 || bpf.decisions != 2 || bpf.jit_decisions != 2)
        throw std::runtime_error("actual runtime policy counters failed");
    std::cout << "profile_parser_cpu: synthetic_valid=1 synthetic_rejected=" << rejected
              << " runtime_policy_wrapper_actual_jit_cases=2 gpu_run=0\n";
}
