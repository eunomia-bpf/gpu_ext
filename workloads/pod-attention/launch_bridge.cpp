// POD-only compatibility bridge for the final CUfunction launch. The existing
// bpftime agent creates a new CUfunction without copying the original runtime
// cudaFuncSetAttribute opt-in. This library does not change bpftime or policy.
#include "launch_bridge.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <tuple>

namespace {
[[noreturn]] void fail(const char *operation, int error = -1) {
    std::fprintf(stderr, "POD_BRIDGE_FATAL operation=%s code=%d\n", operation, error);
    std::fflush(stderr);
    // Returning a CUDA error to the pre-existing agent would allow its native
    // fallback. A failed compatibility/launch check is a failed process instead.
    std::_Exit(86);
}

#ifdef POD_BRIDGE_TEST
extern void *test_symbol(const char *name);
#endif
template<class T> T next_symbol(const char *name) {
#ifdef POD_BRIDGE_TEST
    void *p = test_symbol(name);
#else
    void *p = dlsym(RTLD_NEXT, name);
#endif
    if (!p) fail(name);
    return reinterpret_cast<T>(p);
}

bool enabled() {
    const char *mode = std::getenv("POD_LAUNCH_BRIDGE");
    if (!mode || !*mode || std::strcmp(mode, "off") == 0) return false;
    if (std::strcmp(mode, "cuda") && std::strcmp(mode, "bpf")) fail("invalid bridge mode");
    return true;
}

void check(CUresult code, const char *operation) {
    if (code != CUDA_SUCCESS) fail(operation, int(code));
}

bool is_pod(CUfunction function) {
    static auto get_name = next_symbol<decltype(&cuFuncGetName)>("cuFuncGetName");
    const char *name = nullptr;
    check(get_name(&name, function), "cuFuncGetName");
    if (!name) fail("null CUfunction name");
    return std::strstr(name, "true_fused_tb_fwd_kernel") != nullptr;
}

struct Attributes { int dynamic_bytes, static_bytes, device_bytes; };
struct State {
    std::mutex mutex;
    std::map<std::tuple<uintptr_t, uintptr_t>, Attributes> prepared;
    PodBridgeStats stats{};
};
State &state() { static State value; return value; }

void prepare(CUfunction function, unsigned bytes) {
    static auto get_context = next_symbol<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
    static auto get_device = next_symbol<decltype(&cuCtxGetDevice)>("cuCtxGetDevice");
    static auto device_attr = next_symbol<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    static auto function_attr = next_symbol<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute");
    static auto set_attr = next_symbol<decltype(&cuFuncSetAttribute)>("cuFuncSetAttribute");
    CUcontext context = nullptr;
    check(get_context(&context), "cuCtxGetCurrent");
    if (!context || bytes > INT_MAX) fail("invalid context/shared memory request");
    auto &s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    auto key = std::make_tuple(reinterpret_cast<uintptr_t>(context),
                              reinterpret_cast<uintptr_t>(function));
    auto it = s.prepared.find(key);
    if (it == s.prepared.end() || static_cast<unsigned>(it->second.dynamic_bytes) < bytes) {
        CUdevice device;
        Attributes a{};
        int prior_dynamic = 0;
        check(get_device(&device), "cuCtxGetDevice");
        check(device_attr(&a.device_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                          device), "device opt-in limit");
        check(function_attr(&a.static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
              "function static shared memory");
        check(function_attr(&prior_dynamic, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function),
              "function current dynamic limit");
        if (a.static_bytes < 0 || a.device_bytes <= 0 || prior_dynamic < 0 ||
            static_cast<uint64_t>(a.static_bytes) + bytes > static_cast<uint64_t>(a.device_bytes))
            fail("requested static+dynamic exceeds actual device opt-in limit");
        // Cover the actual launch request, without lowering an existing opt-in
        // when this function is also used with a smaller shared-memory request.
        // Do not substitute a datacenter Blackwell constant or another function.
        int requested_limit = prior_dynamic > int(bytes) ? prior_dynamic : int(bytes);
        if (static_cast<uint64_t>(a.static_bytes) + requested_limit > static_cast<uint64_t>(a.device_bytes))
            fail("existing function opt-in exceeds actual device limit");
        check(set_attr(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, requested_limit),
              "cuFuncSetAttribute actual launch function");
        check(function_attr(&a.dynamic_bytes, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function),
              "cuFuncGetAttribute opt-in readback");
        if (a.dynamic_bytes < 0 || static_cast<unsigned>(a.dynamic_bytes) < bytes ||
            static_cast<uint64_t>(a.static_bytes) + a.dynamic_bytes > static_cast<uint64_t>(a.device_bytes))
            fail("opt-in readback does not cover actual launch");
        if (it == s.prepared.end()) {
            it = s.prepared.emplace(key, a).first;
            ++s.stats.prepared_functions;
        } else {
            it->second = a;
        }
        std::fprintf(stderr, "POD_BRIDGE_PREPARED requested=%u verified=%d static=%d device_optin=%d\n",
                     bytes, a.dynamic_bytes, a.static_bytes, a.device_bytes);
    }
    const auto &a = it->second;
    s.stats.requested_dynamic_bytes = bytes;
    s.stats.verified_dynamic_bytes = a.dynamic_bytes;
    s.stats.static_shared_bytes = a.static_bytes;
    s.stats.device_optin_bytes = a.device_bytes;
}

using DriverLaunch = CUresult (CUDAAPI *)(CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
using RuntimeLaunch = cudaError_t (CUDARTAPI *)(const void *, dim3, dim3, void **, size_t, cudaStream_t);

CUresult driver_launch(const char *symbol, CUfunction function, unsigned gx, unsigned gy, unsigned gz,
                      unsigned bx, unsigned by, unsigned bz, unsigned bytes, CUstream stream,
                      void **args, void **extra) {
    auto original = next_symbol<DriverLaunch>(symbol);
    bool pod = enabled() && is_pod(function);
    if (pod) prepare(function, bytes);
    CUresult result = original(function, gx, gy, gz, bx, by, bz, bytes, stream, args, extra);
    if (pod) {
        check(result, "cuLaunchKernel actual POD function");
        auto &s = state();
        std::lock_guard<std::mutex> guard(s.mutex);
        ++s.stats.launches;
    }
    return result;
}
} // namespace

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned bytes, CUstream stream, void **args, void **extra) {
    return driver_launch("cuLaunchKernel", function, gx, gy, gz, bx, by, bz, bytes, stream, args, extra);
}

extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned bytes, CUstream stream, void **args, void **extra) {
    return driver_launch("cuLaunchKernel_ptsz", function, gx, gy, gz, bx, by, bz, bytes, stream, args, extra);
}

namespace {
cudaError_t runtime_launch(const char *symbol, const void *function, dim3 grid, dim3 block,
                          void **args, size_t bytes, cudaStream_t stream, bool per_thread) {
    auto original = next_symbol<RuntimeLaunch>(symbol);
    if (!enabled()) return original(function, grid, block, args, bytes, stream);
    // The adapter-control arm must use the same final driver opt-in bridge;
    // CUDA's internal runtime launch is not guaranteed to interpose a driver
    // symbol. cudaGetFuncBySymbol preserves the real registered official kernel.
    static auto get_function = next_symbol<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol");
    cudaFunction_t actual = nullptr;
    cudaError_t result = get_function(&actual, function);
    if (result != cudaSuccess) fail("cudaGetFuncBySymbol", int(result));
    if (!actual) fail("null registered CUDA function");
    if (!is_pod(actual)) return original(function, grid, block, args, bytes, stream);
    if (bytes > UINT_MAX) fail("dynamic shared memory size overflow");
    {
        auto &s = state();
        std::lock_guard<std::mutex> guard(s.mutex);
        ++s.stats.runtime_redirects;
    }
    CUresult launched = per_thread ?
        cuLaunchKernel_ptsz(actual, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                           unsigned(bytes), stream, args, nullptr) :
        cuLaunchKernel(actual, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                       unsigned(bytes), stream, args, nullptr);
    check(launched, "POD runtime-to-driver bridge");
    return cudaSuccess;
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void *function, dim3 grid, dim3 block,
    void **args, size_t bytes, cudaStream_t stream) {
    return runtime_launch("cudaLaunchKernel", function, grid, block, args, bytes, stream, false);
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void *function, dim3 grid, dim3 block,
    void **args, size_t bytes, cudaStream_t stream) {
    return runtime_launch("cudaLaunchKernel_ptsz", function, grid, block, args, bytes, stream, true);
}

extern "C" int pod_bridge_get_stats(PodBridgeStats *out, uint64_t size) {
    if (!out || size != sizeof(*out)) return -1;
    auto &s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    *out = s.stats;
    return 0;
}
