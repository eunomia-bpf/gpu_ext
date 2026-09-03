// SPDX-License-Identifier: Apache-2.0
#include "moe_expert_policy.h"
#include "ebpf-vm.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(moe_expert_candidate) == 24 && sizeof(moe_expert_snapshot) == 16,
              "expert snapshot ABI");

namespace {
struct State {
    std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm{nullptr, ebpf_destroy};
    ebpf_jit_fn execute = nullptr;
    std::once_flag once;
    std::string path;
    std::atomic<mep_u64> calls{0}, candidates{0}, selected{0}, no_victim{0}, errors{0};
    ~State() {
        std::printf("moe_expert_policy_stats: backend=%s calls=%llu candidates=%llu "
                    "selected=%llu no_victim=%llu errors=%llu\n",
                    execute ? "ubpf-jit" : "uninitialized", calls.load(), candidates.load(),
                    selected.load(), no_victim.load(), errors.load());
        std::fflush(stdout);
    }
};
State &state() { static State s; return s; }

void initialize(const char *explicit_path)
{
    State &s = state();
    std::call_once(s.once, [&] {
        const char *path = explicit_path ? explicit_path : std::getenv("MOE_EXPERT_POLICY_CODE");
        if (!path || path[0] != '/')
            throw std::runtime_error("absolute MOE_EXPERT_POLICY_CODE or explicit bytecode path required");
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        const auto size = input.tellg();
        if (!input.is_open() || size <= 0 || size > 65536 || size % 8)
            throw std::runtime_error("invalid expert BPF bytecode file");
        std::vector<char> code(static_cast<size_t>(size));
        input.seekg(0);
        if (!input.read(code.data(), code.size())) throw std::runtime_error("incomplete bytecode read");
        std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm{ebpf_create("ubpf"), ebpf_destroy};
        if (!vm) throw std::runtime_error("cannot create ubpf VM");
        char *error = nullptr;
        if (ebpf_load(vm.get(), code.data(), code.size(), &error) != 0) {
            std::string message = error ? error : "expert BPF load failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        ebpf_jit_fn execute = ebpf_compile(vm.get(), &error);
        if (!execute) {
            std::string message = error ? error : "expert BPF JIT failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        s.path = path;
        s.vm = std::move(vm);
        s.execute = execute;
        std::printf("moe_expert_policy_ready: backend=ubpf-jit abi=%u instructions=%zu\n",
                    MOE_EXPERT_POLICY_ABI, code.size() / 8);
        std::fflush(stdout);
    });
    if (explicit_path && s.path != explicit_path)
        throw std::runtime_error("expert BPF program cannot change after initialization");
}

int fail(const std::exception &error)
{
    ++state().errors;
    std::fprintf(stderr, "moe_expert_policy_error: %s\n", error.what());
    return -1;
}
}

extern "C" int moe_expert_policy_init_v1(const char *path)
{
    try { initialize(path); return 0; }
    catch (const std::exception &error) { return fail(error); }
}

extern "C" int moe_expert_policy_select_v1(const moe_expert_candidate *entries,
                                           mep_u32 count, mep_u64 *selected_index)
{
    State &s = state();
    ++s.calls;
    if (selected_index) *selected_index = MOE_EXPERT_NONE;
    try {
        if (!selected_index || (!entries && count) || count > MOE_EXPERT_MAX_CANDIDATES)
            throw std::runtime_error("invalid expert snapshot arguments");
        initialize(nullptr);
        // Per-thread, contiguous, eight-byte-aligned memory; there are no VM helpers,
        // shared mutable BPF maps, pointer chasing, or external reads during selection.
        thread_local std::vector<mep_u64> storage;
        const size_t bytes = sizeof(moe_expert_snapshot) + sizeof(*entries) * count;
        storage.resize(bytes / sizeof(mep_u64));
        const moe_expert_snapshot header{MOE_EXPERT_POLICY_ABI, count, 0};
        std::memcpy(storage.data(), &header, sizeof(header));
        if (count) std::memcpy(reinterpret_cast<char *>(storage.data()) + sizeof(header),
                               entries, sizeof(*entries) * count);
        s.candidates.fetch_add(count, std::memory_order_relaxed);
        const mep_u64 result = s.execute(storage.data(), bytes);
        if (result != MOE_EXPERT_NONE && result >= count)
            throw std::runtime_error("expert BPF rejected snapshot or returned invalid index");
        if (result == MOE_EXPERT_NONE) ++s.no_victim;
        else ++s.selected;
        *selected_index = result;
        return 0;
    } catch (const std::exception &error) { return fail(error); }
}

extern "C" void moe_expert_policy_stats_v1(moe_expert_policy_stats *out)
{
    if (!out) return;
    State &s = state();
    *out = {s.calls.load(), s.candidates.load(), s.selected.load(), s.no_victim.load(), s.errors.load()};
}
