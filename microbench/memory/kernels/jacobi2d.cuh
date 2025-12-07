/**
 * jacobi2d.cuh - Jacobi 2D iterative solver kernel (5-point stencil)
 *
 * SOURCE: PolyBench/GPU Benchmark Suite
 * Original file: /memory/uvm_bench/polybenchGpu/CUDA/JACOBI2D/jacobi2D.cu
 *
 * Access pattern: Sequential (5-point stencil with iterative refinement)
 * Representative of: PDE solvers, iterative methods, scientific computing
 *
 * This is a seq_stream equivalent with iterative computation.
 *
 * NOTE: Kernel code below is EXACT COPY from original source.
 */

#ifndef JACOBI2D_CUH
#define JACOBI2D_CUH

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
// Original PolyBench JACOBI2D Kernels - EXACT COPY from:
// /memory/uvm_bench/polybenchGpu/CUDA/JACOBI2D/jacobi2D.cu lines 74-95
// ============================================================================

__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
	{
		B[i*n + j] = 0.2f * (A[i*n + j] + A[i*n + (j-1)] + A[i*n + (1 + j)] + A[(1 + i)*n + j] + A[(i-1)*n + j]);
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
	{
		A[i*n + j] = B[i*n + j];
	}
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

struct Jacobi2DResult {
    size_t bytes_accessed;
    float median_ms;
    float min_ms;
    float max_ms;
};

inline void run_jacobi2d(size_t total_working_set, const std::string& mode,
                         size_t stride_bytes, int iterations,
                         std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // Jacobi2D uses fixed 5-point stencil pattern

    // Calculate grid size based on working set
    // Working set = 2 arrays (A, B) * N * N * sizeof(DATA_TYPE)
    size_t array_bytes = total_working_set / 2;
    int N = (int)sqrt((double)array_bytes / sizeof(DATA_TYPE));

    // Align to block size
    N = (N / DIM_THREAD_BLOCK_X) * DIM_THREAD_BLOCK_X;
    if (N < DIM_THREAD_BLOCK_X) N = DIM_THREAD_BLOCK_X;

    int TSTEPS = 20;  // Number of Jacobi iterations

    DATA_TYPE *A_gpu, *B_gpu;

    // Allocate UVM memory
    cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * N * N);

    // Initialize data (same as original PolyBench init)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_gpu[i * N + j] = ((DATA_TYPE)(i * (j + 2) + 10)) / N;
            B_gpu[i * N + j] = ((DATA_TYPE)((i - 4) * (j - 1) + 11)) / N;
        }
    }

    // Apply UVM hints if needed
    if (mode == "uvm_prefetch") {
        int dev;
        cudaGetDevice(&dev);
        cudaMemPrefetchAsync(A_gpu, sizeof(DATA_TYPE) * N * N, dev, 0);
        cudaMemPrefetchAsync(B_gpu, sizeof(DATA_TYPE) * N * N, dev, 0);
        cudaDeviceSynchronize();
    }

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)N) / ((float)block.x)),
              (unsigned int)ceil(((float)N) / ((float)block.y)));

    // Warmup
    for (int w = 0; w < 2; w++) {
        for (int t = 0; t < TSTEPS; t++) {
            runJacobiCUDA_kernel1<<<grid, block>>>(N, A_gpu, B_gpu);
            cudaDeviceSynchronize();
            runJacobiCUDA_kernel2<<<grid, block>>>(N, A_gpu, B_gpu);
            cudaDeviceSynchronize();
        }
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        cudaEventRecord(start);

        for (int t = 0; t < TSTEPS; t++) {
            runJacobiCUDA_kernel1<<<grid, block>>>(N, A_gpu, B_gpu);
            cudaDeviceSynchronize();
            runJacobiCUDA_kernel2<<<grid, block>>>(N, A_gpu, B_gpu);
            cudaDeviceSynchronize();
        }

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

    // Bytes accessed per timestep:
    // kernel1: read 5 elements from A, write 1 to B per interior point
    // kernel2: read 1 from B, write 1 to A per interior point
    size_t interior_points = (N - 2) * (N - 2);
    result.bytes_accessed = (size_t)TSTEPS * interior_points * (5 + 1 + 1 + 1) * sizeof(DATA_TYPE);

    // Cleanup
    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

#endif // JACOBI2D_CUH
