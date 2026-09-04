// CPU-only driver doubles: this binary neither loads nor calls CUDA libraries.
#define POD_BRIDGE_TEST
#include "launch_bridge.cpp"
#include <cassert>
#include <sys/wait.h>
#include <unistd.h>

namespace {
int dynamic_bytes = 48 * 1024, static_bytes = 1024, device_bytes = 99 * 1024;
int sets = 0, driver_calls = 0, runtime_calls = 0;
bool bad_readback = false, bad_set = false, bad_launch = false, other_kernel = false;

CUresult fake_name(const char **name, CUfunction) {
    *name = other_kernel ? "ordinary_kernel" : "_Z_true_fused_tb_fwd_kernel_h128";
    return CUDA_SUCCESS;
}
CUresult fake_context(CUcontext *context) { *context = reinterpret_cast<CUcontext>(1); return CUDA_SUCCESS; }
CUresult fake_device(CUdevice *device) { *device = 0; return CUDA_SUCCESS; }
CUresult fake_device_attr(int *value, CUdevice_attribute attr, CUdevice) {
    assert(attr == CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    *value = device_bytes;
    return CUDA_SUCCESS;
}
CUresult fake_attr(int *value, CUfunction_attribute attr, CUfunction) {
    *value = attr == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES ? static_bytes :
        (bad_readback && sets ? 1024 : dynamic_bytes);
    return CUDA_SUCCESS;
}
CUresult fake_set(CUfunction, CUfunction_attribute attr, int value) {
    assert(attr == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    if (bad_set) return CUDA_ERROR_INVALID_VALUE;
    dynamic_bytes = value;
    ++sets;
    return CUDA_SUCCESS;
}
CUresult fake_driver(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                     unsigned, unsigned bytes, CUstream, void **, void **) {
    ++driver_calls;
    assert(other_kernel || !enabled() || dynamic_bytes >= int(bytes));
    return bad_launch ? CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : CUDA_SUCCESS;
}
cudaError_t fake_runtime(const void *, dim3, dim3, void **, size_t, cudaStream_t) {
    ++runtime_calls;
    return cudaSuccess;
}
cudaError_t fake_get_function(cudaFunction_t *actual, const void *symbol) {
    *actual = reinterpret_cast<cudaFunction_t>(const_cast<void *>(symbol));
    return cudaSuccess;
}
void *test_symbol(const char *name) {
#define SYMBOL(n, f) if (!std::strcmp(name, n)) return reinterpret_cast<void *>(&f)
    SYMBOL("cuFuncGetName", fake_name);
    SYMBOL("cuCtxGetCurrent", fake_context);
    SYMBOL("cuCtxGetDevice", fake_device);
    SYMBOL("cuDeviceGetAttribute", fake_device_attr);
    SYMBOL("cuFuncGetAttribute", fake_attr);
    SYMBOL("cuFuncSetAttribute", fake_set);
    SYMBOL("cuLaunchKernel", fake_driver);
    SYMBOL("cuLaunchKernel_ptsz", fake_driver);
    SYMBOL("cudaLaunchKernel", fake_runtime);
    SYMBOL("cudaLaunchKernel_ptsz", fake_runtime);
    SYMBOL("cudaGetFuncBySymbol", fake_get_function);
#undef SYMBOL
    return nullptr;
}
void launch(uintptr_t id, unsigned bytes = 80 * 1024) {
    assert(cuLaunchKernel(reinterpret_cast<CUfunction>(id), 1, 1, 1, 256, 1, 1,
                          bytes, nullptr, nullptr, nullptr) == CUDA_SUCCESS);
}
template<class F> void must_fail(F fn) {
    pid_t child = fork();
    assert(child >= 0);
    if (!child) { fn(); std::_Exit(0); }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
} // namespace

int main() {
    assert(setenv("POD_LAUNCH_BRIDGE", "cuda", 1) == 0);
    launch(10);
    assert(sets == 1 && driver_calls == 1 && dynamic_bytes == 80 * 1024);
    PodBridgeFirstLaunch first_once{};
    assert(pod_bridge_get_first_launch(&first_once, sizeof(first_once), 0) == 0);
    launch(10);
    launch(10, 64 * 1024); // Never lower the opt-in for a smaller launch.
    assert(sets == 1 && dynamic_bytes == 80 * 1024);
    launch(10, 88 * 1024);
    assert(sets == 2 && dynamic_bytes == 88 * 1024);
    assert(cudaLaunchKernel(reinterpret_cast<const void *>(11), dim3(1), dim3(256),
                            nullptr, 80 * 1024, nullptr) == cudaSuccess);
    assert(cudaLaunchKernel_ptsz(reinterpret_cast<const void *>(11), dim3(1), dim3(256),
                                 nullptr, 80 * 1024, nullptr) == cudaSuccess);
    PodBridgeStats stats{};
    assert(pod_bridge_get_stats(&stats, sizeof(stats)) == 0);
    assert(stats.launches == 6 && stats.prepared_functions == 2 && stats.runtime_redirects == 2);
    assert(stats.first_launches == 1);
    assert(stats.requested_dynamic_bytes == 80 * 1024 && stats.verified_dynamic_bytes >= 80 * 1024);
    assert(pod_bridge_get_stats(&stats, sizeof(stats) - 1) == -1);
    PodBridgeFirstLaunch first{};
    assert(pod_bridge_get_first_launch(&first, sizeof(first), 0) == 0);
    assert(first.monotonic_ns > 0);
    assert(first.monotonic_ns == first_once.monotonic_ns);
    assert(std::strcmp(first.kernel, "_Z_true_fused_tb_fwd_kernel_h128") == 0);
    assert(pod_bridge_get_first_launch(&first, sizeof(first), 1) == 1);
    assert(pod_bridge_get_first_launch(&first, sizeof(first) - 1, 0) == -1);
    must_fail([] { launch(20, 99 * 1024); }); // Static + dynamic exceeds sm_120 limit.
    must_fail([] { bad_set = true; launch(21); });
    must_fail([] { bad_readback = true; launch(22); });
    must_fail([] { bad_launch = true; launch(23); });
    other_kernel = true;
    assert(cudaLaunchKernel(reinterpret_cast<const void *>(30), dim3(1), dim3(1),
                            nullptr, 0, nullptr) == cudaSuccess);
    assert(runtime_calls == 1);
    assert(setenv("POD_LAUNCH_BRIDGE", "off", 1) == 0);
    assert(cudaLaunchKernel(reinterpret_cast<const void *>(31), dim3(1), dim3(1),
                            nullptr, 0, nullptr) == cudaSuccess);
    assert(runtime_calls == 2);
    std::puts("PASS: CPU launch bridge, actual-function opt-in/readback and fail-closed paths");
}
