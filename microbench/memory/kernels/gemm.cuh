/**
 * gemm.cuh - GEMM kernel from PolyBench/GPU
 *
 * Source: PolyBench/GPU 1.0
 * Original file: CUDA/GEMM/gemm.cu
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Web: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * This is an adapted standalone version for UVM benchmark integration.
 * The kernel computes: C = alpha * A * B + beta * C
 */

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

// ============================================================================
// Original PolyBench GEMM Kernel (adapted for standalone use)
// ============================================================================
// C = alpha * A * B + beta * C
// A: NI x NK, B: NK x NJ, C: NI x NJ

__global__ void gemm_kernel(int ni, int nj, int nk,
                            DATA_TYPE alpha, DATA_TYPE beta,
                            DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nj))
	{
		c[i * nj + j] *= beta;
		int k;
		for(k = 0; k < nk; k++)
		{
			c[i * nj + j] += alpha * a[i * nk + k] * b[k * nj + j];
		}
	}
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

inline void run_gemm(size_t total_working_set, const std::string& mode,
                     size_t stride_bytes, int iterations,
                     std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // GEMM uses matrix multiplication pattern

    // Calculate matrix sizes based on working set
    // Working set = A (NI*NK) + B (NK*NJ) + C (NI*NJ) * sizeof(DATA_TYPE)
    // For square matrices: NI = NJ = NK = N, so working_set = 3 * N^2 * sizeof(float)
    size_t N = (size_t)sqrt((double)total_working_set / (3.0 * sizeof(DATA_TYPE)));

    // Align to 64 for better memory access
    N = (N / 64) * 64;
    if (N < 64) N = 64;

    int ni = N, nj = N, nk = N;
    size_t size_A = ni * nk * sizeof(DATA_TYPE);
    size_t size_B = nk * nj * sizeof(DATA_TYPE);
    size_t size_C = ni * nj * sizeof(DATA_TYPE);

    DATA_TYPE *A, *B, *C;

    // Allocate based on mode
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&A, size_A));
        CUDA_CHECK(cudaMalloc(&B, size_B));
        CUDA_CHECK(cudaMalloc(&C, size_C));

        // Initialize on host and copy
        std::vector<DATA_TYPE> h_A(ni * nk), h_B(nk * nj), h_C(ni * nj);
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nk; j++) {
                h_A[i * nk + j] = ((DATA_TYPE)(i * j)) / ni;
            }
        }
        for (int i = 0; i < nk; i++) {
            for (int j = 0; j < nj; j++) {
                h_B[i * nj + j] = ((DATA_TYPE)(i * j)) / ni;
            }
        }
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nj; j++) {
                h_C[i * nj + j] = ((DATA_TYPE)(i * j)) / ni;
            }
        }
        CUDA_CHECK(cudaMemcpy(A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, h_B.data(), size_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C, h_C.data(), size_C, cudaMemcpyHostToDevice));
    } else {
        // UVM allocation
        CUDA_CHECK(cudaMallocManaged(&A, size_A));
        CUDA_CHECK(cudaMallocManaged(&B, size_B));
        CUDA_CHECK(cudaMallocManaged(&C, size_C));

        // Initialize data
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nk; j++) {
                A[i * nk + j] = ((DATA_TYPE)(i * j)) / ni;
            }
        }
        for (int i = 0; i < nk; i++) {
            for (int j = 0; j < nj; j++) {
                B[i * nj + j] = ((DATA_TYPE)(i * j)) / ni;
            }
        }
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nj; j++) {
                C[i * nj + j] = ((DATA_TYPE)(i * j)) / ni;
            }
        }
    }

    // Apply UVM hints
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        apply_uvm_hints(A, size_A, mode, dev);
        apply_uvm_hints(B, size_B, mode, dev);
        apply_uvm_hints(C, size_C, mode, dev);
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Launch configuration
    dim3 block(32, 32);
    dim3 grid((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);

    DATA_TYPE alpha = 1.0f;
    DATA_TYPE beta = 1.0f;

    auto launch = [&]() {
        gemm_kernel<<<grid, block>>>(ni, nj, nk, alpha, beta, A, B, C);
    };

    time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

    // Calculate bytes accessed
    // GEMM reads A once, B once per column of C, writes C once
    // For each element of C: read NK elements from A row, NK elements from B column
    // Total reads: NI*NJ*NK (A) + NI*NJ*NK (B) = 2*NI*NJ*NK
    // Total writes: NI*NJ (C)
    result.bytes_accessed = (2UL * ni * nj * nk + ni * nj) * sizeof(DATA_TYPE);

    // Cleanup
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
}

#endif // GEMM_CUH
