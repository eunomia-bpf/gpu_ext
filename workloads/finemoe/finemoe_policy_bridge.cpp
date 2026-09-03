/* Actual host uBPF JIT, with explicit failure and no native fallback. */
#include "finemoe_policy.h"
#include "ubpf.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

struct FineMoeJit {
    std::unique_ptr<ubpf_vm, decltype(&ubpf_destroy)> vm{ubpf_create(), ubpf_destroy};
    ubpf_jit_fn execute = nullptr;
    std::atomic<fm_u64> calls{0};
};
extern "C" void *finemoe_jit_open(const char *path, char *error, size_t capacity)
{
    try {
        if (!path) throw std::runtime_error("missing BPF bytecode path");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        auto size = file.tellg();
        if (!file || size <= 0 || size > 65536)
            throw std::runtime_error("invalid BPF bytecode size");
        file.seekg(0);
        std::vector<char> code(static_cast<size_t>(size));
        if (!file.read(code.data(), size)) throw std::runtime_error("BPF bytecode read failed");
        auto handle = std::make_unique<FineMoeJit>();
        if (!handle->vm) throw std::runtime_error("uBPF allocation failed");
        char *message = nullptr;
        int status = ubpf_load(handle->vm.get(), code.data(), code.size(), &message);
        if (status == 0) handle->execute = ubpf_compile(handle->vm.get(), &message);
        if (status != 0 || !handle->execute) {
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
extern "C" int finemoe_select_bpf(void *opaque, fm_context *ctx)
{
    auto *handle = static_cast<FineMoeJit *>(opaque);
    if (!handle || !ctx) return -1;
    auto result = handle->execute(ctx, sizeof(*ctx));
    ++handle->calls;
    if (result > 1 || ctx->output.status != result) return -2;
    if (result) return ctx->output.mask || ctx->output.selected ? -2 : 1;
    if (ctx->input.count > FM_MAX_EXPERTS || !ctx->input.count ||
        ctx->output.mask >> ctx->input.count ||
        __builtin_popcountll(ctx->output.mask) != static_cast<int>(ctx->output.selected))
        return -2;
    return 0;
}
extern "C" fm_u64 finemoe_jit_calls(void *opaque)
{
    auto *handle = static_cast<FineMoeJit *>(opaque);
    return handle ? handle->calls.load() : 0;
}
extern "C" void finemoe_jit_close(void *opaque)
{
    delete static_cast<FineMoeJit *>(opaque);
}
