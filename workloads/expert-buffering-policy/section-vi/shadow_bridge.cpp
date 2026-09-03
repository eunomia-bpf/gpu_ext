// SPDX-License-Identifier: Apache-2.0
// Untimed correctness only: real JIT first, then native on the identical before
// snapshot. Never replace the JIT result. CPU tests are not live-run evidence.
#include "policy.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>

namespace {
struct Runtime {
    void *library = nullptr, *jit = nullptr;
    eb_u64 (*native)(eb_context *) = nullptr;
    void *(*open)(const char *, char *, size_t) = nullptr;
    int (*select)(void *, eb_context *) = nullptr;
    eb_u64 (*calls)(void *) = nullptr;
    void (*close)(void *) = nullptr;
    bool failed = false;
    ~Runtime() noexcept {
        try { if (jit && close) close(jit); } catch (...) {}
        if (library) dlclose(library);
    }
};
std::mutex mutex;
std::unique_ptr<Runtime> active;
// One active instance; successful reopen resets counters. Close preserves them.
eb_u64 checks = 0, mismatches = 0, jit_calls = 0;

bool Enabled() {
    const char *flag = std::getenv("EB_SECTION_VI_UNTIMED_SHADOW");
    return flag && std::strcmp(flag, "1") == 0;
}
void Error(char *buffer, size_t size, const char *message) {
    if (buffer && size) std::snprintf(buffer, size, "%s", message);
}
template <class T> T Symbol(void *library, const char *name) {
    return reinterpret_cast<T>(dlsym(library, name));
}
} // namespace

// State resolves this even for its BPF arm. Native/FIFO must never use shadow.
extern "C" eb_u64 eb_select(eb_context *) { return EB_INVALID; }

extern "C" void *eb_jit_open(const char *path, char *error, size_t capacity) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        if (!Enabled()) {
            Error(error, capacity, "shadow requires EB_SECTION_VI_UNTIMED_SHADOW=1");
            return nullptr;
        }
        if (active) {
            Error(error, capacity, "shadow permits only one active instance");
            return nullptr;
        }
        const char *library = std::getenv("EB_SECTION_VI_REAL_LIBRARY");
        if (!library || library[0] != '/') {
            Error(error, capacity, "shadow requires an absolute EB_SECTION_VI_REAL_LIBRARY");
            return nullptr;
        }
        auto next = std::make_unique<Runtime>();
        next->library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (!next->library) {
            Error(error, capacity, dlerror());
            return nullptr;
        }
        next->native = Symbol<decltype(next->native)>(next->library, "eb_select");
        next->open = Symbol<decltype(next->open)>(next->library, "eb_jit_open");
        next->select = Symbol<decltype(next->select)>(next->library, "eb_jit_select");
        next->calls = Symbol<decltype(next->calls)>(next->library, "eb_jit_calls");
        next->close = Symbol<decltype(next->close)>(next->library, "eb_jit_close");
        if (!next->native || !next->open || !next->select || !next->calls ||
            !next->close || next->open == &eb_jit_open) {
            Error(error, capacity, "shadow real library has missing or recursive symbols");
            return nullptr;
        }
        next->jit = next->open(path, error, capacity);
        if (!next->jit) return nullptr;
        if (next->calls(next->jit) != 0) {
            Error(error, capacity, "shadow real JIT did not start with zero calls");
            return nullptr;
        }
        active = std::move(next);
        checks = mismatches = jit_calls = 0;
        Error(error, capacity, "");
        return active.get();
    } catch (...) {
        Error(error, capacity, "shadow open exception");
        return nullptr;
    }
}

extern "C" int eb_jit_select(void *handle, eb_context *ctx) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        if (!Enabled() || !active || handle != active.get() || !ctx || active->failed)
            return -1;
        if (checks == std::numeric_limits<eb_u64>::max()) {
            active->failed = true;
            return -1;
        }
        const eb_context before = *ctx;
        active->failed = true; // Remains poisoned if any provider call throws.
        const eb_u64 previous_calls = active->calls(active->jit);
        const int result = active->select(active->jit, ctx); // Actual decision first.
        jit_calls = active->calls(active->jit);
        eb_context reference = before;
        const eb_u64 expected = active->native(&reference); // Check, never preselect.
        ++checks;
        // Includes all input/output bytes, return value, input immutability and
        // exactly one actual JIT call. The caller's context stays the JIT result.
        if (result < EB_HIT || result > EB_BLOCKED || expected != eb_u64(result) ||
            std::memcmp(ctx, &reference, sizeof(*ctx)) ||
            std::memcmp(&ctx->input, &before.input, sizeof(before.input)) ||
            previous_calls == std::numeric_limits<eb_u64>::max() ||
            jit_calls != previous_calls + 1 || jit_calls != checks) {
            ++mismatches;
            return -2;
        }
        active->failed = false;
        return result;
    } catch (...) {
        // The ABI is fail-closed even if a provider unexpectedly throws.
        return -1;
    }
}

extern "C" eb_u64 eb_jit_calls(void *handle) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        return active && handle == active.get() ? jit_calls : 0;
    } catch (...) { return 0; }
}

extern "C" void eb_jit_close(void *handle) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        if (active && handle == active.get()) active.reset();
    } catch (...) {}
}

// Python must retain its CDLL reference from before configuration through this
// snapshot: counters survive handle close, not library unload. All pointers are
// required; mismatches includes ABI/engagement failures above.
extern "C" int eb_shadow_snapshot(eb_u64 *out_checks, eb_u64 *out_mismatches,
                                   eb_u64 *out_jit_calls) noexcept {
    if (!out_checks || !out_mismatches || !out_jit_calls) return -1;
    try {
        std::lock_guard<std::mutex> lock(mutex);
        *out_checks = checks;
        *out_mismatches = mismatches;
        *out_jit_calls = jit_calls;
        return 0;
    } catch (...) { return -1; }
}
