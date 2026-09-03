// Explicit alternative transport canary, NOT the original GDRCopy actuator.
// The polling kernel has a device-side deadline as well as the outer runner.
#include <cuda_runtime.h>
#include <cstdio>
#include <thread>
#include <chrono>

__global__ void bounded_poll(volatile unsigned *flag, unsigned *observed) {
    const auto begin = clock64();
    while (*flag != 0x13579bdfU && clock64() - begin < 1000000000ULL) {}
    *observed = *flag;
}

int main() {
    unsigned *host = nullptr, *device = nullptr, *observed = nullptr;
    unsigned value = 0;
    cudaStream_t stream = nullptr;
    int rdma = -1, mapped = -1, result = 1, completed = 0;
#define CUDA_CHECK(call) do { const auto err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err)); \
    goto cleanup; } } while (0)
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceGetAttribute(&rdma, cudaDevAttrGPUDirectRDMASupported, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&mapped, cudaDevAttrCanMapHostMemory, 0));
    std::printf("HOST_FLAG_CAPABILITIES rdma=%d host_mapping=%d\n", rdma, mapped);
    if (!mapped) goto cleanup;
    CUDA_CHECK(cudaHostAlloc(&host, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
    CUDA_CHECK(cudaMalloc(&observed, sizeof(*observed)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < 64; ++i) {
        __atomic_store_n(host, 0, __ATOMIC_RELEASE);
        bounded_poll<<<1, 1, 0, stream>>>(device, observed);
        CUDA_CHECK(cudaGetLastError());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        __atomic_store_n(host, 0x13579bdfU, __ATOMIC_RELEASE);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&value, observed, sizeof(value), cudaMemcpyDeviceToHost));
        if (value != 0x13579bdfU) {
            std::fprintf(stderr, "host flag roundtrip failed at %d: %x\n", i, value);
            goto cleanup;
        }
        ++completed;
    }
    result = 0;
cleanup:
    // Release any in-flight polling kernel before destroying its storage.
    if (host) __atomic_store_n(host, 0x13579bdfU, __ATOMIC_RELEASE);
    if (stream && cudaStreamSynchronize(stream) != cudaSuccess) result = 1;
    if (stream && cudaStreamDestroy(stream) != cudaSuccess) result = 1;
    if (observed && cudaFree(observed) != cudaSuccess) result = 1;
    if (host && cudaFreeHost(host) != cudaSuccess) result = 1;
    if (cudaDeviceReset() != cudaSuccess) result = 1;
    if (!result && completed == 64)
        std::puts("PASS host-mapped flag: 64 exact roundtrips; compatibility transport only, not GDRCopy");
    return result;
}
