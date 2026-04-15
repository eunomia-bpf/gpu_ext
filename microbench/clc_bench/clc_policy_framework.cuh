//
// CLC Policy Framework - Core Infrastructure
//
// This header contains the framework for CLC policy-based scheduling:
// - Template-based policy interface
// - Policy-aware CLC kernel orchestrator
// - Benchmark runner utilities
//
// Safe-by-construction design principles:
// - Framework holds policy state in __shared__ memory
// - Thread 0 evaluates policy decisions, broadcasts to all threads
// - Uniform control flow enforced (elect-and-broadcast pattern)
// - Policies cannot violate CLC hardware constraints
//

#pragma once

#include <cuda/ptx>
#include <cooperative_groups.h>
#include "benchmark_kernels.cuh"

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

namespace clc_policy {

// ============================================================================
// 1. Policy Interface Definition
// ============================================================================

/**
 * @brief Template-based CLC Scheduler Policy Interface
 *
 * Policies provide 2 static callbacks that control block participation in
 * work stealing. The CLC framework manages all hardware interactions and
 * enforces safety constraints.
 *
 * Key design principle: Policy defines State type, but the FRAMEWORK holds
 * it in __shared__ memory and passes it by reference to callbacks. This
 * enforces uniform control flow and prevents __shared__ static UB.
 *
 * Safety guarantees:
 * - Policy cannot submit requests after observing failure
 * - Policy cannot violate CLC memory ordering or synchronization
 * - Skipping steal attempts is always safe
 * - Framework enforces uniform control flow (elect-and-broadcast pattern)
 * - Barrier synchronization is always respected before policy decisions
 *
 * @tparam Policy The user-defined policy implementation
 */
template<typename Policy>
struct ClcSchedulerPolicy {
    // Policy must define this type (can be empty struct for stateless policies)
    using State = typename Policy::State;

    /**
     * @brief Initialize policy state.
     * Called once per thread block at kernel start, before any work stealing.
     *
     * @param s Policy state (held by framework in __shared__ memory)
     */
    __device__ static void init(State& s) {
        Policy::init(s);
    }

    /**
     * @brief Decide whether to submit a steal request.
     * Called BEFORE each steal attempt. Returning false safely exits the loop.
     *
     * CRITICAL: This is called by thread 0 only. Framework broadcasts result.
     *
     * @param s Policy state (held by framework in __shared__ memory)
     * @param current_block Current block ID being processed
     * @return true = submit steal request, false = exit loop
     */
    __device__ static bool should_try_steal(State& s, int current_block) {
        return Policy::should_try_steal(s, current_block);
    }
};


// ============================================================================
// 2. Policy Composition - AND Combinator
// ============================================================================

/**
 * @brief Combine multiple policies with AND logic
 *
 * Both policies must agree to continue. State is composed via nested structs.
 */
template<typename P1, typename P2>
struct AndPolicy {
    struct State {
        typename P1::State s1;
        typename P2::State s2;
    };

    __device__ static void init(State& s) {
        P1::init(s.s1);
        P2::init(s.s2);
    }

    __device__ static bool should_try_steal(State& s, int current_block) {
        return P1::should_try_steal(s.s1, current_block) && P2::should_try_steal(s.s2, current_block);
    }
};


// ============================================================================
// 3. Policy-Aware CLC Kernel Orchestrator
// ============================================================================

/**
 * @brief CLC kernel with policy-based scheduling
 *
 * This kernel orchestrates work-stealing using CLC hardware while respecting
 * policy decisions before each steal attempt.
 *
 * Framework guarantees:
 * - Uniform control flow (all threads take same path)
 * - Proper CLC barrier synchronization (wait before policy check)
 * - Safe exit on CLC failure
 * - Thread 0 evaluates policy, broadcasts decision
 *
 * @tparam WorkloadType The workload to process
 * @tparam Policy The scheduling policy
 */
template<typename WorkloadType, typename Policy>
__global__ void kernel_cluster_launch_control_policy(
    float* data, int n, int* block_count, int* steal_count, int prologue_complexity)
{
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    // Framework holds policy state in __shared__ memory
    __shared__ typename Policy::State policy_state;
    __shared__ int go;  // Broadcast flag for uniform control flow

    // Initialize the scheduler policy (thread 0 only, then sync)
    if (threadIdx.x == 0) {
        Policy::init(policy_state);
    }
    __syncthreads();

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    float weight = compute_prologue(prologue_complexity);
    int bx = blockIdx.x;

    while (true) {
        __syncthreads();

        // ELECT-AND-BROADCAST PATTERN: Thread 0 evaluates policy
        if (threadIdx.x == 0) {
            go = Policy::should_try_steal(policy_state, bx) ? 1 : 0;
        }
        __syncthreads();

        if (go && cg::thread_block::thread_rank() == 0) {
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

        if (!go) {
            break;  // Uniform exit - policy decided to stop
        }

        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success) {
            break;  // CLC failure - must exit immediately
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


// ============================================================================
// 4. Benchmark Runner Utility
// ============================================================================

/**
 * @brief Run a policy-based CLC benchmark
 *
 * Handles warmup, timing, and result collection for policy benchmarks.
 *
 * @tparam WorkloadType The workload to benchmark
 * @tparam Policy The scheduling policy
 * @return BenchmarkResult with average time, blocks, and steals
 */
template<typename WorkloadType, typename Policy>
BenchmarkResult run_clc_policy(float* d_data, int n, int blocks, int threads,
                                float* h_original, int prologue, int warmup, int runs) {
    int *d_block_count, *d_steal_count;
    cudaMalloc(&d_block_count, sizeof(int));
    cudaMalloc(&d_steal_count, sizeof(int));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_data, h_original, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_block_count, 0, sizeof(int));
        cudaMemset(d_steal_count, 0, sizeof(int));
        kernel_cluster_launch_control_policy<WorkloadType, Policy><<<blocks, threads>>>(
            d_data, n, d_block_count, d_steal_count, prologue);
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
        kernel_cluster_launch_control_policy<WorkloadType, Policy><<<blocks, threads>>>(
            d_data, n, d_block_count, d_steal_count, prologue);
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

} // namespace clc_policy
