//
// Common Kernels and Runners for CLC Benchmarks
//

#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "ai_workloads.cuh"
#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// ============================================
// Common Benchmark Result Struct
// ============================================

struct BenchmarkResult {
    float avg_time_ms;
    float avg_blocks;
    float avg_steals;
};

// ============================================
// Fixed Work Kernel and Runner
// ============================================

template<typename WorkloadType>
__global__ void kernel_fixed_work(float* data, int n, int* block_count, int prologue_complexity) {
    float weight = compute_prologue(prologue_complexity);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        process_workload(WorkloadType{}, data, i, n, weight);
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

template<typename WorkloadType>
BenchmarkResult run_fixed_work(float* d_data, int n, int blocks, int threads,
                                float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count;
    cudaMalloc(&d_block_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        kernel_fixed_work<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    float total_blocks = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_fixed_work<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        int h_blocks;
        cudaMemcpy(&h_blocks, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_blocks;
    }

    BenchmarkResult result = {total_time / runs, total_blocks / runs, 0};

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);

    return result;
}

// ============================================
// Fixed Blocks Kernel and Runner
// ============================================

template<typename WorkloadType>
__global__ void kernel_fixed_blocks(float* data, int n, int* block_count, int prologue_complexity) {
    float weight = compute_prologue(prologue_complexity);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        process_workload(WorkloadType{}, data, i, n, weight);
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

template<typename WorkloadType>
BenchmarkResult run_fixed_blocks(float* d_data, int n, int blocks, int threads,
                                  float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count;
    cudaMalloc(&d_block_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        kernel_fixed_blocks<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    float total_blocks = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_fixed_blocks<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, prologue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        int h_blocks;
        cudaMemcpy(&h_blocks, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_blocks;
    }

    BenchmarkResult result = {total_time / runs, total_blocks / runs, 0};

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);

    return result;
}

// ============================================
// Baseline CLC Kernel and Runner
// ============================================

template<typename WorkloadType>
__global__ void kernel_cluster_launch_control_baseline(float* data, int n, int* block_count, int* steal_count,
                                                       int prologue_complexity) {
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    float weight = compute_prologue(prologue_complexity);
    int bx = blockIdx.x;

    while (true) {
        __syncthreads();

        if (cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        int i = bx * blockDim.x + threadIdx.x;
        if (i < n) {
            process_workload(WorkloadType{}, data, i, n, weight);
        }

        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success) {
            break;
        }

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        if (threadIdx.x == 0) {
            atomicAdd(steal_count, 1);
        }

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}

template<typename WorkloadType>
BenchmarkResult run_clc_baseline(float* d_data, int n, int blocks, int threads,
                                 float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count, *d_steal_count;
    cudaMalloc(&d_block_count, sizeof(int));
    cudaMalloc(&d_steal_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));
        kernel_cluster_launch_control_baseline<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count, prologue);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f, total_blocks = 0.0f, total_steals = 0.0f;

    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));

        cudaEventRecord(start);
        kernel_cluster_launch_control_baseline<WorkloadType><<<blocks, threads>>>(d_data, n, d_block_count, d_steal_count, prologue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        int h_blocks, h_steals;
        cudaMemcpy(&h_blocks, d_block_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_steals, d_steal_count, sizeof(int), cudaMemcpyDeviceToHost);
        total_blocks += h_blocks;
        total_steals += h_steals;
    }

    BenchmarkResult result = {total_time / runs, total_blocks / runs, total_steals / runs};

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_count);
    cudaFree(d_steal_count);

    return result;
}
