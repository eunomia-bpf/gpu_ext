// Compile-only regression for NVCC's external device-call ABI. Never launched;
// actual attention PTX must independently retain the same two-parameter call.
#include "selector_cuda.cuh"
extern "C" __global__ void pod_selector_abi_probe(PodSelectorContext *ctx) {
    if (threadIdx.x == 0) pod_device_selector(ctx, sizeof(*ctx));
}
