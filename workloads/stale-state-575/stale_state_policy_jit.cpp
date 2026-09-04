/* SPDX-License-Identifier: MIT */
/* Real host-uBPF JIT wrapper. There is no native-policy fallback. */
#include "stale_state_policy_jit.h"

#include "ubpf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct StaleStateJit {
    std::unique_ptr<ubpf_vm, decltype(&ubpf_destroy)> vm{ubpf_create(), ubpf_destroy};
    ubpf_jit_fn execute = nullptr;
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> contract_errors{0};
};

extern "C" void *stale_state_575_jit_open(const char *path,
                                            char *error,
                                            size_t capacity)
{
    try {
        if (path == nullptr)
            throw std::runtime_error("missing BPF bytecode path");

        std::ifstream input(path, std::ios::binary | std::ios::ate);
        const auto end = input.tellg();
        if (!input || end <= 0 || end > 65536)
            throw std::runtime_error("invalid BPF bytecode size");
        input.seekg(0);

        std::vector<char> code(static_cast<size_t>(end));
        if (!input.read(code.data(), end))
            throw std::runtime_error("BPF bytecode read failed");

        auto handle = std::make_unique<StaleStateJit>();
        if (!handle->vm)
            throw std::runtime_error("uBPF allocation failed");

        char *message = nullptr;
        const int loaded = ubpf_load(handle->vm.get(), code.data(), code.size(), &message);
        if (loaded == 0)
            handle->execute = ubpf_compile(handle->vm.get(), &message);
        if (loaded != 0 || handle->execute == nullptr) {
            const std::string detail = message ? message : "uBPF load/JIT failed";
            std::free(message);
            throw std::runtime_error(detail);
        }
        std::free(message);

        if (error != nullptr && capacity != 0)
            error[0] = '\0';
        return handle.release();
    }
    catch (const std::exception &failure) {
        if (error != nullptr && capacity != 0)
            std::snprintf(error, capacity, "%s", failure.what());
        return nullptr;
    }
}

extern "C" int stale_state_575_jit_choose(
    void *opaque,
    struct stale_state_575_jit_context *context,
    size_t context_bytes)
{
    auto *handle = static_cast<StaleStateJit *>(opaque);
    if (handle == nullptr || context == nullptr)
        return -1;

    if (context_bytes != sizeof(*context)) {
        const uint64_t result = handle->execute(context, context_bytes);
        ++handle->calls;
        if (result != STALE_STATE_575_ACTION_REJECT)
            ++handle->contract_errors;
        return result == STALE_STATE_575_ACTION_REJECT
                   ? STALE_STATE_575_ACTION_REJECT
                   : -2;
    }

    const struct stale_state_575_snapshot before_snapshot = context->snapshot;
    const uint64_t before_time = context->decision_mono_ns;
    const uint32_t before_reserved = context->reserved;
    const uint64_t result = handle->execute(context, context_bytes);
    ++handle->calls;

    const bool input_changed =
        std::memcmp(&before_snapshot, &context->snapshot, sizeof(before_snapshot)) != 0 ||
        before_time != context->decision_mono_ns ||
        before_reserved != context->reserved;
    const bool invalid_result = result > STALE_STATE_575_ACTION_DISCARD_PREFETCH;
    const bool incoherent_output = context->status != result ||
                                   context->decision.action != result;
    if (input_changed || invalid_result || incoherent_output) {
        ++handle->contract_errors;
        return -2;
    }
    return static_cast<int>(result);
}

extern "C" uint64_t stale_state_575_jit_calls(void *opaque)
{
    auto *handle = static_cast<StaleStateJit *>(opaque);
    return handle ? handle->calls.load() : 0;
}

extern "C" uint64_t stale_state_575_jit_contract_errors(void *opaque)
{
    auto *handle = static_cast<StaleStateJit *>(opaque);
    return handle ? handle->contract_errors.load() : 0;
}

extern "C" void stale_state_575_jit_close(void *opaque)
{
    delete static_cast<StaleStateJit *>(opaque);
}
