/**
 * kmeans.cuh - K-Means clustering kernel (Mixed access pattern)
 *
 * SOURCE: UVM Benchmark Suite
 * Original file: /memory/uvm_bench/UVM_benchmark/UVM_benchmarks/kmeans/kmeans_cuda.cu
 *
 * Access pattern: Mixed (sequential scan + random centroid access)
 * Representative of: Machine learning, clustering, data analytics
 *
 * This represents workloads with both sequential and irregular access.
 *
 * NOTE: Kernel code below is EXACT COPY from original source.
 */

#ifndef KMEANS_CUH
#define KMEANS_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// ============================================================================
// Original UVM Benchmark K-Means Macros and Kernels - EXACT COPY from:
// /memory/uvm_bench/UVM_benchmark/UVM_benchmarks/kmeans/kmeans_cuda.cu lines 16-78
// ============================================================================

#define I(row, col, ncols) (row * ncols + col)

__global__ void get_dst(float *dst, float *x, float *y,
			float *mu_x, float *mu_y){
  int i = blockIdx.x;
  int j = threadIdx.x;

  dst[I(i, j, blockDim.x)] = (x[i] - mu_x[j]) * (x[i] - mu_x[j]);
  dst[I(i, j, blockDim.x)] += (y[i] - mu_y[j]) * (y[i] - mu_y[j]);
}

__global__ void regroup(int *group, float *dst, int k){
  int i = blockIdx.x;
  int j;
  float min_dst;

  min_dst = dst[I(i, 0, k)];
  group[i] = 1;

  for(j = 1; j < k; ++j){
    if(dst[I(i, j, k)] < min_dst){
      min_dst = dst[I(i, j, k)];
      group[i] = j + 1;
    }
  }
}

__global__ void clear(float *sum_x, float *sum_y, int *nx, int *ny){
  int j = threadIdx.x;

  sum_x[j] = 0;
  sum_y[j] = 0;
  nx[j] = 0;
  ny[j] = 0;
}

__global__ void recenter_step1(float *sum_x, float *sum_y, int *nx, int *ny,
			       float *x, float *y, int *group, int n){
  int i;
  int j = threadIdx.x;

  for(i = 0; i < n; ++i){
    if(group[i] == (j + 1)){
      sum_x[j] += x[i];
      sum_y[j] += y[i];
      nx[j]++;
      ny[j]++;
    }
  }
}

__global__ void recenter_step2(float *mu_x, float *mu_y, float *sum_x,
			       float *sum_y, int *nx, int *ny){
  int j = threadIdx.x;

  mu_x[j] = sum_x[j]/nx[j];
  mu_y[j] = sum_y[j]/ny[j];
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

struct KmeansResult {
    size_t bytes_accessed;
    float median_ms;
    float min_ms;
    float max_ms;
};

inline void run_kmeans(size_t total_working_set, const std::string& mode,
                       size_t stride_bytes, int iterations,
                       std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // K-Means uses fixed clustering pattern

    // Calculate n (number of points) and k (number of clusters)
    // Working set ~= n * (x + y + group + dst*k) + k * (mu_x + mu_y + sum_x + sum_y + nx + ny)
    // Simplified: working_set ~= n * (2*float + int + k*float) + k * (4*float + 2*int)
    int k = 16;  // Number of clusters
    int n = (int)(total_working_set / (3 * sizeof(float) + sizeof(int) + k * sizeof(float)));
    if (n < 1000) n = 1000;

    // Allocate UVM memory
    int *group_d, *nx_d, *ny_d;
    float *x_d, *y_d, *mu_x_d, *mu_y_d, *sum_x_d, *sum_y_d, *dst_d;

    cudaMallocManaged(&group_d, n * sizeof(int));
    cudaMallocManaged(&nx_d, k * sizeof(int));
    cudaMallocManaged(&ny_d, k * sizeof(int));
    cudaMallocManaged(&x_d, n * sizeof(float));
    cudaMallocManaged(&y_d, n * sizeof(float));
    cudaMallocManaged(&mu_x_d, k * sizeof(float));
    cudaMallocManaged(&mu_y_d, k * sizeof(float));
    cudaMallocManaged(&sum_x_d, k * sizeof(float));
    cudaMallocManaged(&sum_y_d, k * sizeof(float));
    cudaMallocManaged(&dst_d, n * k * sizeof(float));

    // Initialize data with random points
    for (int i = 0; i < n; i++) {
        x_d[i] = (float)rand() / RAND_MAX * 100.0f;
        y_d[i] = (float)rand() / RAND_MAX * 100.0f;
    }

    // Initialize centroids
    for (int j = 0; j < k; j++) {
        mu_x_d[j] = (float)rand() / RAND_MAX * 100.0f;
        mu_y_d[j] = (float)rand() / RAND_MAX * 100.0f;
    }

    // Apply UVM hints if needed
    if (mode == "uvm_prefetch") {
        int dev;
        cudaGetDevice(&dev);
        cudaMemPrefetchAsync(x_d, n * sizeof(float), dev, 0);
        cudaMemPrefetchAsync(y_d, n * sizeof(float), dev, 0);
        cudaMemPrefetchAsync(mu_x_d, k * sizeof(float), dev, 0);
        cudaMemPrefetchAsync(mu_y_d, k * sizeof(float), dev, 0);
        cudaMemPrefetchAsync(dst_d, n * k * sizeof(float), dev, 0);
        cudaMemPrefetchAsync(group_d, n * sizeof(int), dev, 0);
        cudaDeviceSynchronize();
    }

    int nreps = 100;  // K-means iterations per run

    // Warmup
    for (int w = 0; w < 2; w++) {
        for (int rep = 0; rep < nreps; ++rep) {
            get_dst<<<n, k>>>(dst_d, x_d, y_d, mu_x_d, mu_y_d);
            regroup<<<n, 1>>>(group_d, dst_d, k);
            clear<<<1, k>>>(sum_x_d, sum_y_d, nx_d, ny_d);
            recenter_step1<<<1, k>>>(sum_x_d, sum_y_d, nx_d, ny_d, x_d, y_d, group_d, n);
            recenter_step2<<<1, k>>>(mu_x_d, mu_y_d, sum_x_d, sum_y_d, nx_d, ny_d);
        }
        cudaDeviceSynchronize();
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        // Re-initialize centroids
        for (int j = 0; j < k; j++) {
            mu_x_d[j] = (float)rand() / RAND_MAX * 100.0f;
            mu_y_d[j] = (float)rand() / RAND_MAX * 100.0f;
        }

        cudaEventRecord(start);

        for (int rep = 0; rep < nreps; ++rep) {
            get_dst<<<n, k>>>(dst_d, x_d, y_d, mu_x_d, mu_y_d);
            regroup<<<n, 1>>>(group_d, dst_d, k);
            clear<<<1, k>>>(sum_x_d, sum_y_d, nx_d, ny_d);
            recenter_step1<<<1, k>>>(sum_x_d, sum_y_d, nx_d, ny_d, x_d, y_d, group_d, n);
            recenter_step2<<<1, k>>>(mu_x_d, mu_y_d, sum_x_d, sum_y_d, nx_d, ny_d);
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

    // Estimate bytes accessed per iteration
    // get_dst: read x[n], y[n], mu_x[k], mu_y[k], write dst[n*k]
    // regroup: read dst[n*k], write group[n]
    // recenter_step1: read x[n], y[n], group[n], write sum_x[k], sum_y[k]
    // recenter_step2: read sum_x[k], sum_y[k], write mu_x[k], mu_y[k]
    result.bytes_accessed = (size_t)nreps * (
        2 * n * sizeof(float) +      // x, y read
        2 * k * sizeof(float) +      // mu_x, mu_y read
        n * k * sizeof(float) * 2 +  // dst write + read
        n * sizeof(int) * 2 +        // group write + read
        4 * k * sizeof(float)        // sum, mu operations
    );

    // Cleanup
    cudaFree(group_d);
    cudaFree(nx_d);
    cudaFree(ny_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(mu_x_d);
    cudaFree(mu_y_d);
    cudaFree(sum_x_d);
    cudaFree(sum_y_d);
    cudaFree(dst_d);
}

#endif // KMEANS_CUH
