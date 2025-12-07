/**
 * conv2d.cuh - 2D Convolution kernel (3x3 stencil pattern)
 *
 * SOURCE: PolyBench/GPU Benchmark Suite
 * Original file: /memory/uvm_bench/polybenchGpu/CUDA/2DCONV/2DConvolution.cu
 *
 * Access pattern: Sequential (3x3 neighbor access)
 * Representative of: Image processing, CNN convolution layers
 *
 * This is a seq_stream equivalent with convolution computation.
 *
 * NOTE: Kernel code below is EXACT COPY from original source.
 */

#ifndef CONV2D_CUH
#define CONV2D_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ============================================================================
// PolyBench Compatibility Macros (from original polybench.h)
// ============================================================================

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef DIM_THREAD_BLOCK_X
#define DIM_THREAD_BLOCK_X 32
#endif

#ifndef DIM_THREAD_BLOCK_Y
#define DIM_THREAD_BLOCK_Y 8
#endif

// ============================================================================
// Original PolyBench 2DCONV Kernel - EXACT COPY from:
// /memory/uvm_bench/polybenchGpu/CUDA/2DCONV/2DConvolution.cu lines 102-119
// ============================================================================

__global__ void convolution2D_kernel(int ni, int nj, DATA_TYPE *A, DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < ni-1) && (j < nj-1) && (i > 0) && (j > 0))
	{
		B[i * nj + j] =  c11 * A[(i - 1) * nj + (j - 1)]  + c21 * A[(i - 1) * nj + (j + 0)] + c31 * A[(i - 1) * nj + (j + 1)]
			+ c12 * A[(i + 0) * nj + (j - 1)]  + c22 * A[(i + 0) * nj + (j + 0)] +  c32 * A[(i + 0) * nj + (j + 1)]
			+ c13 * A[(i + 1) * nj + (j - 1)]  + c23 * A[(i + 1) * nj + (j + 0)] +  c33 * A[(i + 1) * nj + (j + 1)];
	}
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

struct Conv2DResult {
    size_t bytes_accessed;
    float median_ms;
    float min_ms;
    float max_ms;
};

inline void run_conv2d(size_t total_working_set, const std::string& mode,
                       size_t stride_bytes, int iterations,
                       std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // Conv2D uses fixed 3x3 stencil pattern

    // Calculate grid size based on working set
    // Working set = 2 arrays (A, B) * NI * NJ * sizeof(DATA_TYPE)
    size_t array_bytes = total_working_set / 2;
    int grid_size = (int)sqrt((double)array_bytes / sizeof(DATA_TYPE));

    int NI = grid_size;
    int NJ = grid_size;

    DATA_TYPE *A_gpu, *B_gpu;

    // Allocate UVM memory
    cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NI * NJ);

    // Initialize data (same as original PolyBench init)
    for (int i = 0; i < NI; ++i) {
        for (int j = 0; j < NJ; ++j) {
            A_gpu[i * NJ + j] = (float)rand() / RAND_MAX;
        }
    }

    // Apply UVM hints if needed
    if (mode == "uvm_prefetch") {
        int dev;
        cudaGetDevice(&dev);
        cudaMemPrefetchAsync(A_gpu, sizeof(DATA_TYPE) * NI * NJ, dev, 0);
        cudaMemPrefetchAsync(B_gpu, sizeof(DATA_TYPE) * NI * NJ, dev, 0);
        cudaDeviceSynchronize();
    }

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil(((float)NI) / ((float)block.x)),
              (size_t)ceil(((float)NJ) / ((float)block.y)));

    // Warmup
    for (int w = 0; w < 2; w++) {
        convolution2D_kernel<<<grid, block>>>(NI, NJ, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        cudaEventRecord(start);
        convolution2D_kernel<<<grid, block>>>(NI, NJ, A_gpu, B_gpu);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        runtimes.push_back(ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute statistics
    std::sort(runtimes.begin(), runtimes.end());
    result.median_ms = runtimes[runtimes.size() / 2];
    result.min_ms = runtimes.front();
    result.max_ms = runtimes.back();

    // Bytes accessed: read 9 elements from A, write 1 to B per output pixel
    size_t output_pixels = (NI - 2) * (NJ - 2);
    result.bytes_accessed = output_pixels * (9 + 1) * sizeof(DATA_TYPE);

    // Cleanup
    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

#endif // CONV2D_CUH
