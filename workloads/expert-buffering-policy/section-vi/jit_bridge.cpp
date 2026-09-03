/* CUDA-free host uBPF JIT. No native fallback on load or execution failure. */
#include "policy.h"
#include "ubpf.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

struct EbJit {
    std::unique_ptr<ubpf_vm, decltype(&ubpf_destroy)> vm{ubpf_create(), ubpf_destroy};
    ubpf_jit_fn execute = nullptr;
    std::atomic<eb_u64> calls{0};
};

extern "C" void *eb_jit_open(const char *path, char *error, size_t capacity)
{
    try {
        if (!path) throw std::runtime_error("missing BPF bytecode path");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        auto size = file.tellg();
        if (!file || size <= 0 || size > 65536)
            throw std::runtime_error("invalid BPF bytecode size");
        file.seekg(0);
        std::vector<char> code(static_cast<size_t>(size));
        if (!file.read(code.data(), size)) throw std::runtime_error("BPF read failed");
        auto handle = std::make_unique<EbJit>();
        if (!handle->vm) throw std::runtime_error("uBPF allocation failed");
        char *message = nullptr;
        int status = ubpf_load(handle->vm.get(), code.data(), code.size(), &message);
        if (!status) handle->execute = ubpf_compile(handle->vm.get(), &message);
        if (status || !handle->execute) {
            std::string detail = message ? message : "uBPF load/JIT failed";
            std::free(message);
            throw std::runtime_error(detail);
        }
        std::free(message);
        if (error && capacity) error[0] = '\0';
        return handle.release();
    } catch (const std::exception &failure) {
        if (error && capacity) std::snprintf(error, capacity, "%s", failure.what());
        return nullptr;
    }
}

extern "C" int eb_jit_select(void *opaque, eb_context *ctx)
{
    auto *handle = static_cast<EbJit *>(opaque);
    if (!handle || !ctx) return -1;
    const eb_input before = ctx->input;
    const auto result = handle->execute(ctx, sizeof(*ctx));
    ++handle->calls;
    const auto &out = ctx->output;
    if (result > EB_BLOCKED || out.status != result ||
        out.batch_epoch != before.batch_epoch ||
        std::memcmp(&before, &ctx->input, sizeof(before)))
        return -2;
    if (result == EB_EVICT) {
        if (out.victim >= before.count || out.victim >= EB_MAX_EXPERTS ||
            out.victim == before.incoming ||
            before.experts[out.victim].flags != (EB_RESIDENT | EB_ELIGIBLE))
            return -2;
    } else if (out.victim != EB_NO_VICTIM) {
        return -2;
    }
    return static_cast<int>(result);
}

extern "C" eb_u64 eb_jit_calls(void *opaque)
{
    auto *handle = static_cast<EbJit *>(opaque);
    return handle ? handle->calls.load() : 0;
}

extern "C" void eb_jit_close(void *opaque)
{
    delete static_cast<EbJit *>(opaque);
}
