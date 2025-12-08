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
		// Use size_t to avoid integer overflow for large grids (n > 46340)
		size_t idx = (size_t)i * n + j;
		B[idx] = 0.2f * (A[idx] + A[idx - 1] + A[idx + 1] + A[idx + n] + A[idx - n]);
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
	{
		// Use size_t to avoid integer overflow for large grids (n > 46340)
		size_t idx = (size_t)i * n + j;
		A[idx] = B[idx];
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

    // =========================================================================
    // 动态 grid 大小 + 固定迭代次数
    // - 根据 size_factor 调整 grid 大小，使数据量匹配 total_working_set
    // - size_factor=1.0 → grid ~64K×64K → ~32GB (2 arrays)
    // - 固定迭代次数产生 temporal locality
    // =========================================================================

    // Jacobi2D 需要 2 个 array: A, B
    // total_working_set = 2 * grid_size^2 * sizeof(float)
    // grid_size = sqrt(total_working_set / (2 * sizeof(float)))
    size_t elements_needed = total_working_set / (2 * sizeof(DATA_TYPE));
    int grid_size = (int)sqrt((double)elements_needed);

    // 对齐到 block 的倍数
    grid_size = (grid_size / DIM_THREAD_BLOCK_X) * DIM_THREAD_BLOCK_X;
    if (grid_size < 1024) grid_size = 1024;  // 最小 1024×1024

    int N = grid_size;
    size_t size = (size_t)N * N;
    size_t allocated_bytes = 2 * size * sizeof(DATA_TYPE);

    // 固定迭代次数，产生 temporal locality
    int TSTEPS = 100;

    DATA_TYPE *A_gpu, *B_gpu;

    // Allocate memory
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&A_gpu, sizeof(DATA_TYPE) * size));
        CUDA_CHECK(cudaMalloc(&B_gpu, sizeof(DATA_TYPE) * size));
        CUDA_CHECK(cudaMemset(A_gpu, 0, sizeof(DATA_TYPE) * size));
        CUDA_CHECK(cudaMemset(B_gpu, 0, sizeof(DATA_TYPE) * size));
    } else {
        CUDA_CHECK(cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * size));
        CUDA_CHECK(cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * size));

        // Use cudaMemset for large allocations to avoid integer overflow in CPU loops
        CUDA_CHECK(cudaMemset(A_gpu, 0, sizeof(DATA_TYPE) * size));
        CUDA_CHECK(cudaMemset(B_gpu, 0, sizeof(DATA_TYPE) * size));
    }

    // Apply UVM hints if needed
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        apply_uvm_hints(A_gpu, sizeof(DATA_TYPE) * size, mode, dev);
        apply_uvm_hints(B_gpu, sizeof(DATA_TYPE) * size, mode, dev);
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    fprintf(stderr, "Jacobi2D config: grid=%dx%d, iterations=%d\n", N, N, TSTEPS);
    fprintf(stderr, "  Allocated: %.1f MB, Total access: %.1f MB\n",
            allocated_bytes / (1024.0 * 1024.0),
            allocated_bytes * TSTEPS / (1024.0 * 1024.0));

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

    // Bytes accessed: allocated * iterations (approximate)
    result.bytes_accessed = allocated_bytes * TSTEPS;

    // Cleanup
    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

// ============================================================================
// Multi-Grid Oversub Variant - Tests eviction policy under competition
// ============================================================================
// Design: Allocate G grids where G * grid_size > HBM capacity
// In one kernel, different blocks process different grids simultaneously
// This creates multiple "hot regions" competing for HBM

__global__ void runJacobiCUDA_kernel1_multigrid(int n, DATA_TYPE **A_grids, DATA_TYPE **B_grids, int num_grids)
{
    int grid_id = blockIdx.z;
    if (grid_id >= num_grids) return;

    DATA_TYPE *A = A_grids[grid_id];
    DATA_TYPE *B = B_grids[grid_id];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
    {
        // Use size_t to avoid integer overflow for large grids
        size_t idx = (size_t)i * n + j;
        B[idx] = 0.2f * (A[idx] + A[idx - 1] + A[idx + 1] + A[idx + n] + A[idx - n]);
    }
}

__global__ void runJacobiCUDA_kernel2_multigrid(int n, DATA_TYPE **A_grids, DATA_TYPE **B_grids, int num_grids)
{
    int grid_id = blockIdx.z;
    if (grid_id >= num_grids) return;

    DATA_TYPE *A = A_grids[grid_id];
    DATA_TYPE *B = B_grids[grid_id];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
    {
        // Use size_t to avoid integer overflow for large grids
        size_t idx = (size_t)i * n + j;
        A[idx] = B[idx];
    }
}

inline void run_jacobi2d_multigrid(size_t total_working_set, const std::string& mode,
                                    size_t stride_bytes, int iterations,
                                    std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;

    // =========================================================================
    // Multi-Grid Oversub Design:
    // - Each grid is 2048x2048 (~32MB for 2 arrays) - smaller to fit more grids
    // - Allocate G grids where G * grid_size > total_working_set
    // - All grids processed in ONE kernel launch (blockIdx.z = grid_id)
    // - Creates true multi-region competition for HBM
    // =========================================================================

    const int GRID_SIZE = 2048;  // Smaller grid for more parallelism
    int N = GRID_SIZE;
    size_t single_grid_elements = (size_t)N * N;
    size_t single_grid_bytes = 2 * single_grid_elements * sizeof(DATA_TYPE);  // ~32MB per grid

    // Calculate number of grids to match total_working_set
    int num_grids = (total_working_set + single_grid_bytes - 1) / single_grid_bytes;
    if (num_grids < 2) num_grids = 2;

    size_t total_allocated = (size_t)num_grids * single_grid_bytes;

    fprintf(stderr, "Jacobi2D MultiGrid config: grid=%dx%d, num_grids=%d\n",
            N, N, num_grids);
    fprintf(stderr, "  Single grid: %.1f MB, Total allocation: %.1f MB\n",
            single_grid_bytes / (1024.0 * 1024.0),
            total_allocated / (1024.0 * 1024.0));

    // Allocate arrays of pointers for multi-grid
    DATA_TYPE **h_A = new DATA_TYPE*[num_grids];
    DATA_TYPE **h_B = new DATA_TYPE*[num_grids];

    // Device pointer arrays
    DATA_TYPE **d_A, **d_B;

    // Allocate each grid
    for (int g = 0; g < num_grids; g++) {
        if (mode == "device") {
            CUDA_CHECK(cudaMalloc(&h_A[g], sizeof(DATA_TYPE) * single_grid_elements));
            CUDA_CHECK(cudaMalloc(&h_B[g], sizeof(DATA_TYPE) * single_grid_elements));
            CUDA_CHECK(cudaMemset(h_A[g], 0, sizeof(DATA_TYPE) * single_grid_elements));
            CUDA_CHECK(cudaMemset(h_B[g], 0, sizeof(DATA_TYPE) * single_grid_elements));
        } else {
            CUDA_CHECK(cudaMallocManaged(&h_A[g], sizeof(DATA_TYPE) * single_grid_elements));
            CUDA_CHECK(cudaMallocManaged(&h_B[g], sizeof(DATA_TYPE) * single_grid_elements));

            // Initialize on CPU (ensures pages start on CPU for UVM)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    h_A[g][i * N + j] = ((DATA_TYPE)(i * (j + 2) + 10 + g)) / N;
                    h_B[g][i * N + j] = ((DATA_TYPE)((i - 4) * (j - 1) + 11 + g)) / N;
                }
            }
        }
    }

    // Allocate device pointer arrays
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(DATA_TYPE*) * num_grids));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(DATA_TYPE*) * num_grids));

    // Copy pointer arrays to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(DATA_TYPE*) * num_grids, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(DATA_TYPE*) * num_grids, cudaMemcpyHostToDevice));

    // Apply UVM hints if needed
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        for (int g = 0; g < num_grids; g++) {
            apply_uvm_hints(h_A[g], sizeof(DATA_TYPE) * single_grid_elements, mode, dev);
            apply_uvm_hints(h_B[g], sizeof(DATA_TYPE) * single_grid_elements, mode, dev);
        }
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // 3D grid: (blockCols, blockRows, num_grids)
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)N) / ((float)block.x)),
              (unsigned int)ceil(((float)N) / ((float)block.y)),
              num_grids);

    // Number of Jacobi timesteps per measured iteration
    int TSTEPS = 10;  // Fixed iterations per measurement

    // Warmup
    for (int w = 0; w < 2; w++) {
        for (int t = 0; t < TSTEPS; t++) {
            runJacobiCUDA_kernel1_multigrid<<<grid, block>>>(N, d_A, d_B, num_grids);
            cudaDeviceSynchronize();
            runJacobiCUDA_kernel2_multigrid<<<grid, block>>>(N, d_A, d_B, num_grids);
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
            runJacobiCUDA_kernel1_multigrid<<<grid, block>>>(N, d_A, d_B, num_grids);
            cudaDeviceSynchronize();
            runJacobiCUDA_kernel2_multigrid<<<grid, block>>>(N, d_A, d_B, num_grids);
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
    result.bytes_accessed = (size_t)num_grids * TSTEPS * interior_points * (5 + 1 + 1 + 1) * sizeof(DATA_TYPE);

    // Cleanup
    for (int g = 0; g < num_grids; g++) {
        cudaFree(h_A[g]);
        cudaFree(h_B[g]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
}

#endif // JACOBI2D_CUH
