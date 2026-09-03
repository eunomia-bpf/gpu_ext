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
static_assert(sizeof(moe_expert_scored_candidate) == 24 && sizeof(moe_expert_rank_candidate) == 24 &&
              sizeof(moe_expert_scored_snapshot) == 16 && sizeof(moe_expert_rank_snapshot) == 16,
              "float64 policy ABI");

namespace {
struct State {
    const char *kind;
    const char *environment;
    std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm{nullptr, ebpf_destroy};
    ebpf_jit_fn execute = nullptr;
    std::once_flag once;
    std::string path;
    std::atomic<mep_u64> calls{0}, candidates{0}, selected{0}, no_victim{0}, errors{0};
    State(const char *kind_, const char *environment_) : kind(kind_), environment(environment_) {}
    ~State() {
        std::printf("moe_expert_policy_stats: backend=%s kind=%s calls=%llu candidates=%llu "
                    "selected=%llu no_victim=%llu errors=%llu\n",
                    execute ? "ubpf-jit" : "uninitialized", kind, calls.load(), candidates.load(),
                    selected.load(), no_victim.load(), errors.load());
        std::fflush(stdout);
    }
};
State &state() { static State s{"current_count", "MOE_EXPERT_POLICY_CODE"}; return s; }
State &scored_state() { static State s{"paper_scored", "MOE_EXPERT_SCORED_CODE"}; return s; }
State &rank_state() { static State s{"paper_rank", "MOE_EXPERT_RANK_CODE"}; return s; }
State &match_state() { static State s{"paper_match", "MOE_EXPERT_MATCH_CODE"}; return s; }

void initialize(State &s, const char *explicit_path)
{
    std::call_once(s.once, [&] {
        const char *path = explicit_path ? explicit_path : std::getenv(s.environment);
        if (!path || path[0] != '/')
            throw std::runtime_error(std::string("absolute ") + s.environment + " or explicit bytecode path required");
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
        std::printf("moe_expert_policy_ready: backend=ubpf-jit kind=%s abi=%u instructions=%zu\n",
                    s.kind, MOE_EXPERT_POLICY_ABI, code.size() / 8);
        std::fflush(stdout);
    });
    if (explicit_path && s.path != explicit_path)
        throw std::runtime_error("expert BPF program cannot change after initialization");
}

int fail(State &s, const std::exception &error)
{
    ++s.errors;
    std::fprintf(stderr, "moe_expert_policy_error: kind=%s %s\n", s.kind, error.what());
    return -1;
}

template<typename Entry> int select(State &s, const Entry *entries,
                                   mep_u32 count, mep_u64 *selected_index)
{
    ++s.calls;
    if (selected_index) *selected_index = MOE_EXPERT_NONE;
    try {
        if (!selected_index || (!entries && count) || count > MOE_EXPERT_MAX_CANDIDATES)
            throw std::runtime_error("invalid expert snapshot arguments");
        initialize(s, nullptr);
        // Per-thread, contiguous, eight-byte-aligned memory; no VM helpers,
        // shared mutable maps, pointer chasing, or external reads during selection.
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
    } catch (const std::exception &error) { return fail(s, error); }
}

void stats(State &s, moe_expert_policy_stats *out)
{
    if (out) *out = {s.calls.load(), s.candidates.load(), s.selected.load(), s.no_victim.load(), s.errors.load()};
}
}

extern "C" int moe_expert_policy_init_v1(const char *path)
{
    try { initialize(state(), path); return 0; }
    catch (const std::exception &error) { return fail(state(), error); }
}

extern "C" int moe_expert_policy_select_v1(const moe_expert_candidate *entries,
                                           mep_u32 count, mep_u64 *selected_index)
{
    return select(state(), entries, count, selected_index);
}

extern "C" void moe_expert_policy_stats_v1(moe_expert_policy_stats *out)
{
    stats(state(), out);
}

extern "C" int moe_expert_scored_init_v1(const char *path)
{
    try { initialize(scored_state(), path); return 0; }
    catch (const std::exception &error) { return fail(scored_state(), error); }
}

extern "C" int moe_expert_scored_select_v1(const moe_expert_scored_candidate *entries,
                                           mep_u32 count, mep_u64 *selected_index)
{
    return select(scored_state(), entries, count, selected_index);
}

extern "C" void moe_expert_scored_stats_v1(moe_expert_policy_stats *out)
{
    stats(scored_state(), out);
}

extern "C" int moe_expert_rank_init_v1(const char *path)
{
    try { initialize(rank_state(), path); return 0; }
    catch (const std::exception &error) { return fail(rank_state(), error); }
}

static int select_indices(State &s, const moe_expert_rank_candidate *entries, mep_u32 count,
                          mep_u32 *indices, mep_u32 capacity, mep_u32 *selected_count)
{
    ++s.calls;
    if (selected_count) *selected_count = 0;
    try {
        if (!selected_count || (!entries && count) || (!indices && count) || capacity < count ||
            count > MOE_EXPERT_MAX_CANDIDATES)
            throw std::runtime_error("invalid rank snapshot arguments");
        initialize(s, nullptr);
        thread_local std::vector<mep_u64> storage;
        const size_t offset = sizeof(moe_expert_rank_snapshot) + sizeof(*entries) * count;
        const size_t bytes = offset + 2 * sizeof(mep_u32) * count;
        storage.resize(bytes / sizeof(mep_u64));
        const moe_expert_rank_snapshot header{MOE_EXPERT_POLICY_ABI, count, 0};
        std::memcpy(storage.data(), &header, sizeof(header));
        if (count) std::memcpy(reinterpret_cast<char *>(storage.data()) + sizeof(header),
                               entries, sizeof(*entries) * count);
        s.candidates.fetch_add(count, std::memory_order_relaxed);
        const mep_u64 result = s.execute(storage.data(), bytes);
        if (result > count) throw std::runtime_error("rank BPF rejected snapshot or returned invalid count");
        const char *output = reinterpret_cast<char *>(storage.data()) + offset;
        if (result) std::memcpy(indices, output, sizeof(mep_u32) * result);
        // Boundary validation is not a native ranking fallback or oracle.
        for (mep_u32 i = 0; i < result; ++i)
            if (indices[i] >= count) throw std::runtime_error("rank BPF returned invalid index");
        s.selected.fetch_add(result, std::memory_order_relaxed);
        if (!result) ++s.no_victim;
        *selected_count = result;
        return 0;
    } catch (const std::exception &error) { return fail(s, error); }
}

extern "C" int moe_expert_rank_v1(const moe_expert_rank_candidate *entries, mep_u32 count,
                                  mep_u32 *indices, mep_u32 capacity, mep_u32 *selected_count)
{
    return select_indices(rank_state(), entries, count, indices, capacity, selected_count);
}

extern "C" void moe_expert_rank_stats_v1(moe_expert_rank_stats *out)
{
    if (!out) return;
    State &s = rank_state();
    *out = {s.calls.load(), s.candidates.load(), s.selected.load(), s.no_victim.load(), s.errors.load()};
}

extern "C" int moe_expert_match_init_v1(const char *path)
{
    try { initialize(match_state(), path); return 0; }
    catch (const std::exception &error) { return fail(match_state(), error); }
}

extern "C" int moe_expert_match_v1(const moe_expert_rank_candidate *entries, mep_u32 count,
                                   mep_u32 *indices, mep_u32 capacity, mep_u32 *selected_count)
{
    return select_indices(match_state(), entries, count, indices, capacity, selected_count);
}

extern "C" void moe_expert_match_stats_v1(moe_expert_match_stats *out)
{
    if (!out) return;
    State &s = match_state();
    *out = {s.calls.load(), s.candidates.load(), s.selected.load(), s.no_victim.load(), s.errors.load()};
}
