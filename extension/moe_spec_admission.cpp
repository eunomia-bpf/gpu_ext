// SPDX-License-Identifier: Apache-2.0
// Governor admission bridge: assemble the ABI snapshot, execute only the real
// uBPF JIT, validate the result boundary. No native fallback exists.
#include "moe_spec_admission.h"
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

static_assert(sizeof(moe_spec_candidate) == 16 && sizeof(moe_spec_snapshot) == 32,
              "spec admission snapshot ABI");

namespace {
struct State {
    std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm{nullptr, ebpf_destroy};
    ebpf_jit_fn execute = nullptr;
    std::once_flag once;
    std::string path;
    std::atomic<mep_u64> calls{0}, candidates{0}, admitted{0}, empty{0}, errors{0};
    ~State() {
        std::printf("moe_spec_admission_stats: backend=%s calls=%llu candidates=%llu "
                    "admitted=%llu empty=%llu errors=%llu\n",
                    execute ? "ubpf-jit" : "uninitialized", calls.load(),
                    candidates.load(), admitted.load(), empty.load(), errors.load());
        std::fflush(stdout);
    }
};
State &state() { static State s; return s; }

void initialize(const char *explicit_path)
{
    std::call_once(state().once, [&] {
        const char *path = explicit_path ? explicit_path : std::getenv("MOE_SPEC_ADMISSION_CODE");
        if (!path || path[0] != '/')
            throw std::runtime_error("absolute MOE_SPEC_ADMISSION_CODE or explicit bytecode path required");
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        const auto size = input.tellg();
        if (!input.is_open() || size <= 0 || size > 65536 || size % 8)
            throw std::runtime_error("invalid spec admission BPF bytecode file");
        std::vector<char> code(static_cast<size_t>(size));
        input.seekg(0);
        if (!input.read(code.data(), code.size())) throw std::runtime_error("incomplete bytecode read");
        std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm{ebpf_create("ubpf"), ebpf_destroy};
        if (!vm) throw std::runtime_error("cannot create ubpf VM");
        char *error = nullptr;
        if (ebpf_load(vm.get(), code.data(), code.size(), &error) != 0) {
            std::string message = error ? error : "spec admission BPF load failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        ebpf_jit_fn execute = ebpf_compile(vm.get(), &error);
        if (!execute) {
            std::string message = error ? error : "spec admission BPF JIT failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        state().path = path;
        state().vm = std::move(vm);
        state().execute = execute;
        std::printf("moe_spec_admission_ready: backend=ubpf-jit abi=%u instructions=%zu\n",
                    MOE_SPEC_ADMISSION_ABI, code.size() / 8);
        std::fflush(stdout);
    });
    if (explicit_path && state().path != explicit_path)
        throw std::runtime_error("spec admission BPF program cannot change after initialization");
}

int fail(const std::exception &error)
{
    ++state().errors;
    std::fprintf(stderr, "moe_spec_admission_error: %s\n", error.what());
    return -1;
}
}  // namespace

extern "C" int moe_spec_admission_init_v1(const char *path)
{
    try { initialize(path); return 0; }
    catch (const std::exception &error) { return fail(error); }
}

extern "C" int moe_spec_admission_prefix_v1(const struct moe_spec_candidate *entries,
                                            mep_u32 count, mep_u64 budget_bytes,
                                            mep_u64 outstanding_bytes, mep_u64 *admitted)
{
    State &s = state();
    ++s.calls;
    if (admitted) *admitted = MOE_SPEC_NONE;
    try {
        if (!admitted || (!entries && count) || count > MOE_SPEC_MAX_CANDIDATES)
            throw std::runtime_error("invalid spec admission arguments");
        initialize(nullptr);
        thread_local std::vector<mep_u64> storage;
        const size_t bytes = sizeof(moe_spec_snapshot) + sizeof(*entries) * count;
        storage.resize(bytes / sizeof(mep_u64));
        const moe_spec_snapshot header{budget_bytes, outstanding_bytes,
                                       MOE_SPEC_ADMISSION_ABI, count, 0};
        std::memcpy(storage.data(), &header, sizeof(header));
        if (count) std::memcpy(reinterpret_cast<char *>(storage.data()) + sizeof(header),
                               entries, sizeof(*entries) * count);
        s.candidates.fetch_add(count, std::memory_order_relaxed);
        const mep_u64 result = s.execute(storage.data(), bytes);
        if (result != MOE_SPEC_NONE && result > count)
            throw std::runtime_error("spec admission BPF rejected snapshot or returned invalid prefix");
        if (result == MOE_SPEC_NONE) ++s.empty;
        else if (count) s.admitted.fetch_add(result, std::memory_order_relaxed);
        *admitted = result;
        return 0;
    } catch (const std::exception &error) { return fail(error); }
}

extern "C" void moe_spec_admission_stats_v1(struct moe_spec_admission_stats *output)
{
    if (!output) return;
    State &s = state();
    *output = {s.calls.load(), s.candidates.load(), s.admitted.load(), s.empty.load(), s.errors.load()};
}
