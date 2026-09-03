// SPDX-License-Identifier: GPL-2.0
// Pure CPU semantic cases and actual ubpf JIT execution; no CUDA dependency.
#include "idle_policy.h"
#include "ebpf-vm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

static hb_call valid()
{
    hb_call call{};
    call.input.now_ns = 1000000;
    call.input.last_hp_activity_ns = 999000;
    call.input.large_after_ns = 100000;
    call.input.launch_overhead_ns = 5000;
    call.input.split_ns = 20000;
    call.input.whole_ns = 90000;
    call.input.hp_gpu_done = 1;
    call.input.small_active = 1;
    call.input.small_start_done = 1;
    call.input.lp_pending = 1;
    call.input.lp_gpu_done = 1;
    call.input.kernel_unstarted = 1;
    call.input.consolidate = 1;
    return call;
}

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv)
{
    ebpf_vm *vm = nullptr;
    try {
        if (argc != 2) throw std::runtime_error("usage: test_idle_policy PATH_TO_BPF_BINARY");
        std::ifstream input(argv[1], std::ios::binary);
        std::vector<char> code{std::istreambuf_iterator<char>(input), {}};
        require(input.is_open() && !code.empty() && code.size() <= 65536, "invalid bytecode");
        vm = ebpf_create("ubpf");
        require(vm != nullptr, "cannot create ubpf VM");
        char *error = nullptr;
        if (ebpf_load(vm, code.data(), code.size(), &error)) {
            std::string message = error ? error : "bytecode load failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        ebpf_jit_fn execute = ebpf_compile(vm, &error);
        if (!execute) {
            std::string message = error ? error : "JIT failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        size_t checked = 0;
        auto check = [&](hb_call call, hb_action action, hb_wait reason, hb_u64 tick) {
            hb_call c = call, bpf = call;
            const auto c_result = hb_decide(&c, sizeof(c));
            const auto bpf_result = execute(&bpf, sizeof(bpf));
            require(c_result == action && c.output.action == action, "unexpected C action");
            require(c.output.wait_reason == reason, "unexpected C wait reason");
            require(c.output.next_tick_ns == tick, "unexpected next tick");
            require(c_result == bpf_result, "C/BPF return mismatch");
            require(std::memcmp(&c, &bpf, sizeof(c)) == 0, "C/BPF output mismatch");
            require(std::memcmp(&c.input, &call.input, sizeof(call.input)) == 0, "policy changed input");
            ++checked;
        };
        check(valid(), HB_SPLIT, HB_NOT_WAITING, 1015000);
        auto call = valid(); call.input.hp_pending = 307;
        check(call, HB_STOP_LP, HB_NOT_WAITING, 0);
        call = valid(); call.input.lp_pending = 0;
        check(call, HB_WAIT, HB_WAIT_EMPTY, 0);
        call = valid(); call.input.hp_gpu_done = 0;
        check(call, HB_WAIT, HB_WAIT_HP, 0);
        call = valid(); call.input.small_start_done = 0;
        check(call, HB_WAIT, HB_WAIT_BUBBLE, 0);
        call = valid(); call.input.small_active = 0;
        check(call, HB_WAIT, HB_WAIT_BUBBLE, 0);
        call = valid(); call.input.tick_due_ns = 1000001;
        check(call, HB_WAIT, HB_WAIT_TICK, 1000001);
        call = valid(); call.input.lp_gpu_done = 0;
        check(call, HB_WAIT, HB_WAIT_LP_EVENT, 0);
        call = valid(); call.input.last_hp_activity_ns = 900000;
        check(call, HB_WHOLE, HB_NOT_WAITING, 1085000);
        call.input.kernel_unstarted = 0;
        check(call, HB_SPLIT, HB_NOT_WAITING, 1015000);
        call = valid(); call.input.last_hp_activity_ns = 900000; call.input.consolidate = 0;
        check(call, HB_SPLIT, HB_NOT_WAITING, 1015000);
        call = valid(); call.input.launch_overhead_ns = 20001;
        check(call, HB_SPLIT, HB_NOT_WAITING, 1000000);
        call = valid(); call.input.hp_gpu_done = 2;
        check(call, HB_ERROR, HB_NOT_WAITING, 0);
        call = valid(); call.input.now_ns = 1;
        check(call, HB_ERROR, HB_NOT_WAITING, 0);
        call = valid(); call.input.large_after_ns = 0;
        check(call, HB_ERROR, HB_NOT_WAITING, 0);
        call = valid(); call.input.now_ns = ~0ULL;
        check(call, HB_ERROR, HB_NOT_WAITING, 0);
        // Exhaustive combinations of genuine state flags at both timer edges.
        for (unsigned int bits = 0; bits < 256; ++bits) {
            for (hb_u64 elapsed : {99999ULL, 100000ULL}) {
                hb_call c = valid(), bpf;
                c.input.hp_pending = (bits >> 0) & 1;
                c.input.hp_gpu_done = (bits >> 1) & 1;
                c.input.small_active = (bits >> 2) & 1;
                c.input.small_start_done = (bits >> 3) & 1;
                c.input.lp_pending = (bits >> 4) & 1;
                c.input.lp_gpu_done = (bits >> 5) & 1;
                c.input.kernel_unstarted = (bits >> 6) & 1;
                c.input.consolidate = (bits >> 7) & 1;
                c.input.last_hp_activity_ns = c.input.now_ns - elapsed;
                bpf = c;
                const auto c_result = hb_decide(&c, sizeof(c));
                require(c_result == execute(&bpf, sizeof(bpf)), "exhaustive return mismatch");
                require(std::memcmp(&c, &bpf, sizeof(c)) == 0, "exhaustive state mismatch");
                if (c_result == HB_SPLIT || c_result == HB_WHOLE)
                    require(!c.input.hp_pending && c.input.hp_gpu_done && c.input.lp_gpu_done,
                            "launch without genuine completion/priority permission");
                ++checked;
            }
        }
        call = valid();
        require(hb_decide(&call, sizeof(call) - 1) == HB_ERROR, "C accepted short input");
        require(execute(&call, sizeof(call) - 1) == HB_ERROR, "BPF accepted short input");
        std::printf("idle_policy_cpu: backend=ubpf-jit semantic_and_parity_cases=%zu passed=1 gpu_run=0\n", checked);
        ebpf_destroy(vm);
        return 0;
    } catch (const std::exception &error) {
        if (vm) ebpf_destroy(vm);
        std::fprintf(stderr, "idle_policy_cpu_error: %s\n", error.what());
        return 1;
    }
}
