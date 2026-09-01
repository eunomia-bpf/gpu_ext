/* SPDX-License-Identifier: MIT */

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CUDA_OK(call) do {                                                     \
    cudaError_t status_ = (call);                                               \
    if (status_ != cudaSuccess) {                                               \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_)); \
        return 1;                                                               \
    }                                                                           \
} while (0)

typedef void (*layout_marker_fn)(const char *, const void *, uint64_t, uint64_t,
                                 uint32_t, uint32_t);

__global__ static void touch_pages(uint8_t *allocation, size_t bytes)
{
    size_t page = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = page * 4096;

    if (offset < bytes)
        allocation[offset] = (uint8_t)page;
}

int main(int argc, char **argv)
{
    size_t allocation_mib = 8;
    size_t bytes;
    size_t per_expert;
    void *library;
    layout_marker_fn layout;
    uint8_t *allocation = NULL;
    int device = 0;
    int input;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s LIBGGML_BASE [ALLOCATION_MIB]\n", argv[0]);
        return 2;
    }
    if (argc == 3) {
        char *end = NULL;
        unsigned long long parsed = strtoull(argv[2], &end, 10);

        if (!end || *end != '\0' || parsed < 8 || parsed > 65536 || parsed % 4) {
            fprintf(stderr, "ALLOCATION_MIB must be a multiple of 4 in [8,65536]\n");
            return 2;
        }
        allocation_mib = (size_t)parsed;
    }
    bytes = allocation_mib * 1024ULL * 1024ULL;
    per_expert = bytes / 4;
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    layout = (layout_marker_fn)dlsym(library, "gpubpf_expert_tensor_layout");
    if (!layout) {
        fprintf(stderr, "layout marker symbol is missing\n");
        dlclose(library);
        return 1;
    }

    CUDA_OK(cudaGetDevice(&device));
    CUDA_OK(cudaMallocManaged(&allocation, bytes));
    CUDA_OK(cudaMemAdvise(allocation, bytes, cudaMemAdviseSetPreferredLocation,
                          cudaCpuDeviceId));
    CUDA_OK(cudaMemPrefetchAsync(allocation, bytes, cudaCpuDeviceId));
    CUDA_OK(cudaDeviceSynchronize());

    layout("blk.0.ffn_gate_exps.weight", allocation, bytes,
           per_expert, 4, 0);
    printf("policy_smoke_ready pid=%ld base=%llu bytes=%llu\n",
           (long)getpid(),
           (unsigned long long)(uintptr_t)allocation,
           (unsigned long long)bytes);
    printf("press Enter after the policy reports ready\n");
    fflush(stdout);
    input = getchar();
    if (input == EOF) {
        fprintf(stderr, "stdin closed before policy admission\n");
        cudaFree(allocation);
        dlclose(library);
        return 1;
    }

    CUDA_OK(cudaMemAdvise(allocation, bytes, cudaMemAdviseSetPreferredLocation,
                          device));
    CUDA_OK(cudaMemPrefetchAsync(allocation, bytes, device));
    CUDA_OK(cudaDeviceSynchronize());

    const size_t pages = bytes / 4096;
    touch_pages<<<(pages + 255) / 256, 256>>>(allocation, bytes);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    printf("policy_smoke_touch_complete pages=%llu\n",
           (unsigned long long)pages);

    CUDA_OK(cudaFree(allocation));
    dlclose(library);
    return 0;
}
