/*
 * diskuvm_fault.cu
 *
 * Fault-inducing GPU traversal for the LMCache disk-UVM GPU-promotion
 * integration (see lmcache_diskuvm_backing.py / lmcache_diskuvm_backing_cuda.py).
 *
 * This is the same full-range volatile source-load traversal used by the
 * completed primitive (gds-control/disk-uvm/disk_uvm_perf.cu,
 * kv_read_traverse).  It is exposed through a narrow C ABI so the Python
 * CUDA helper can drive it after a managed range has been sealed, offloaded
 * (on-disk), and GPU-promotion-enabled:
 *
 *   extern "C" int diskuvm_fault_read(size_t va, size_t size, int device);
 *
 * While the range is durably on disk, resident nowhere, and GPU promotion is
 * enabled, these first-touch GPU reads make the UVM driver perform the
 * CPU-staged disk -> GPU copy-engine restore (the characterized "gpu_restore"
 * step).  No P2P, no transparent/automatic offload.
 *
 * Build (root owns the GPU build):
 *   nvcc -shared -arch=sm_120 -Xcompiler -fPIC -o libdiskuvm_fault.so \
 *        diskuvm_fault.cu -lcudart
 *
 * The library is loaded lazily at restore time only; the put/offload path
 * (prepare) never needs it, so a missing library degrades a single demand
 * get to the stock GDS read instead of breaking preparation.
 */

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#define DISK_UVM_GPU_CTAS    1024
#define DISK_UVM_GPU_THREADS 256

__global__ void diskuvm_read_traverse(const volatile uint32_t *p, size_t n_words,
                                      uint32_t *sample_out)
{
    volatile uint32_t sink = 0u;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;
    size_t base = (size_t)blockIdx.x * (size_t)blockDim.x;
    size_t i;

    for (i = base + threadIdx.x; i < n_words; i += stride)
        sink = p[i];

    if (threadIdx.x == 0 && base < n_words)
        sample_out[blockIdx.x] = p[base];
}

extern "C" int diskuvm_fault_read(size_t va, size_t size, int device)
{
    cudaError_t st;
    cudaStream_t stream = NULL;
    uint32_t *dev_samples = NULL;

    if (size == 0)
        return 0;

    st = cudaSetDevice(device);
    if (st != cudaSuccess)
        return (int)st;

    st = cudaStreamCreate(&stream);
    if (st != cudaSuccess)
        return (int)st;

    st = cudaMalloc((void **)&dev_samples, sizeof(uint32_t) * DISK_UVM_GPU_CTAS);
    if (st != cudaSuccess) {
        cudaStreamDestroy(stream);
        return (int)st;
    }

    size_t n_words = size / sizeof(uint32_t);
    diskuvm_read_traverse<<<DISK_UVM_GPU_CTAS, DISK_UVM_GPU_THREADS, 0, stream>>>(
        (const volatile uint32_t *)va, n_words, dev_samples);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaFree(dev_samples);
        cudaStreamDestroy(stream);
        return (int)st;
    }
    st = cudaStreamSynchronize(stream);

    cudaFree(dev_samples);
    cudaStreamDestroy(stream);
    if (st != cudaSuccess)
        return (int)st;
    return 0;
}
