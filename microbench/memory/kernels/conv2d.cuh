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
		// Use size_t to avoid integer overflow for large grids (nj > 46340)
		size_t idx = (size_t)i * nj + j;
		size_t nj_s = (size_t)nj;
		B[idx] =  c11 * A[idx - nj_s - 1]  + c21 * A[idx - nj_s] + c31 * A[idx - nj_s + 1]
			+ c12 * A[idx - 1]  + c22 * A[idx] +  c32 * A[idx + 1]
			+ c13 * A[idx + nj_s - 1]  + c23 * A[idx + nj_s] +  c33 * A[idx + nj_s + 1];
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
    // Use size_t to avoid overflow for large grids
    size_t total_elements = (size_t)NI * NJ;

    DATA_TYPE *A_gpu, *B_gpu;

    // Allocate UVM memory
    cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * total_elements);
    cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * total_elements);

    // Use cudaMemset for large allocations to avoid CPU page faults
    cudaMemset(A_gpu, 0, sizeof(DATA_TYPE) * total_elements);
    cudaMemset(B_gpu, 0, sizeof(DATA_TYPE) * total_elements);

    // Apply UVM hints if needed
    if (mode == "uvm_prefetch") {
        int dev;
        cudaGetDevice(&dev);
        cudaMemPrefetchAsync(A_gpu, sizeof(DATA_TYPE) * total_elements, dev, 0);
        cudaMemPrefetchAsync(B_gpu, sizeof(DATA_TYPE) * total_elements, dev, 0);
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
    // Use size_t cast to avoid integer overflow
    size_t output_pixels = (size_t)(NI - 2) * (NJ - 2);
    result.bytes_accessed = output_pixels * (9 + 1) * sizeof(DATA_TYPE);

    // Cleanup
    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

#endif // CONV2D_CUH
