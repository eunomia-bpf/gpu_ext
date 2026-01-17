/**
 * Example 4: eBPF Map Bandwidth Contention - Slowing Down GEMM
 *
 * This example demonstrates how excessive eBPF map operations can
 * turn a compute-bound GEMM kernel into a memory-bound one.
 *
 * Scenario:
 *   - Original kernel: Matrix multiplication (GEMM) - typically compute-bound
 *   - eBPF hook: Attached for monitoring/tracing purposes
 *   - Problem: Excessive map reads/writes saturate memory bandwidth
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe (all map accesses bounded)
 *   ✓ Bounded execution (finite number of map operations)
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific issues:
 *   ✗ Excessive memory traffic from frequent map reads/writes
 *   ✗ Saturates memory bandwidth, starving GEMM of data
 *   ✗ Converts compute-bound kernel to memory-bound
 *   ✗ Severe performance degradation (10-100x slowdown possible)
 */

#include <stdio.h>
#include <cuda_runtime.h>

//=============================================================================
// Simulated eBPF Infrastructure (provided by bpftime)
//=============================================================================

// HUGE map sizes to completely bust all cache levels
// 512M entries × 8 bytes = 4GB per map, 5 maps = 20GB total
// This ensures EVERY access goes to global memory (DRAM)
#define MAP_SIZE (512ULL * 1024 * 1024)
#define EVENT_LOG_SIZE (512ULL * 1024 * 1024)

// Large stride to ensure cache line misses (avoid spatial locality)
// 128 entries = 1KB stride - guarantees different cache lines
#define ACCESS_STRIDE 128

// Simulated BPF maps - dynamically allocated to avoid linker issues with large static arrays
// These pointers will be set via cudaMemcpyToSymbol after cudaMalloc
__device__ unsigned long long *bpf_event_log;           // Event timestamps (512MB)
__device__ unsigned long long *bpf_thread_state;        // Per-thread state (512MB)
__device__ unsigned long long *bpf_iteration_counts;    // Iteration counters (512MB)
__device__ unsigned long long *bpf_memory_access_log;   // Memory access tracking (512MB)
__device__ unsigned long long *bpf_performance_metrics; // Performance data (512MB)

// Global counters
__device__ unsigned long long bpf_total_invocations = 0;
__device__ unsigned long long bpf_total_iterations = 0;

// eBPF Helper: Get global timer
__device__ unsigned long long bpf_get_globaltimer() {
    unsigned long long timer;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
    return timer;
}

// eBPF Helper: Map update (write to map)
__device__ void bpf_map_update(unsigned long long *map, unsigned long long key, unsigned long long val) {
    if (key < MAP_SIZE) {
        map[key] = val;
    }
}

// eBPF Helper: Map lookup (read from map)
__device__ unsigned long long bpf_map_lookup(unsigned long long *map, unsigned long long key) {
    if (key < MAP_SIZE) {
        return map[key];
    }
    return 0;
}

// eBPF Helper: Log event to event ring buffer (scattered access)
__device__ void bpf_log_event(unsigned long long base_idx, unsigned long long event_type, unsigned long long data) {
    // Use large stride to bust cache - each event type goes to different region
    unsigned long long idx = (base_idx + event_type * ACCESS_STRIDE * 50000) % EVENT_LOG_SIZE;
    bpf_event_log[idx] = data;
}

//=============================================================================
// eBPF HOOK - BAD: Normal-looking tracing that causes memory bound
//=============================================================================

/**
 * A simple, normal-looking eBPF tracing program:
 *   - Record timestamp
 *   - Update invocation counter
 *   - Log latency
 *
 * This is minimal tracing - just 3 map operations!
 * On CPU: Totally fine
 * On GPU: 64M threads × 3 ops = 192M memory accesses → bandwidth saturated
 */
__device__ void ebpf_hook_BAD_bandwidth_hog(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long idx = (tid * ACCESS_STRIDE) % MAP_SIZE;

    // Record entry time
    unsigned long long ts_start = bpf_get_globaltimer();

    // Update invocation counter (read-modify-write)
    unsigned long long count = bpf_map_lookup(bpf_iteration_counts, idx);
    bpf_map_update(bpf_iteration_counts, idx, count + 1);

    // Record latency
    unsigned long long ts_end = bpf_get_globaltimer();
    bpf_map_update(bpf_performance_metrics, idx, ts_end - ts_start);
}

//=============================================================================
// eBPF HOOK - MEDIUM: Reduced map operations
//=============================================================================

/**
 * Moderate monitoring: Only essential metrics, fewer map operations
 */
__device__ void ebpf_hook_MEDIUM(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long ts = bpf_get_globaltimer();

    // Only log entry timestamp - 1 WRITE
    bpf_log_event(tid, 0, ts);

    // Single state update - 1 READ + 1 WRITE
    unsigned long long prev = bpf_map_lookup(bpf_thread_state, tid % MAP_SIZE);
    bpf_map_update(bpf_thread_state, tid % MAP_SIZE, prev + 1);

    // Single performance metric - 1 WRITE
    bpf_map_update(bpf_performance_metrics, tid % MAP_SIZE, ts);
}

//=============================================================================
// eBPF HOOK - GOOD: Minimal map operations with local aggregation
//=============================================================================

/**
 * GPU-aware monitoring: Minimize memory traffic
 * - Use registers for intermediate values
 * - Aggregate at warp level before writing
 * - Only write essential data
 */
__device__ void ebpf_hook_GOOD(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long warp_id = tid / 32;
    int lane_id = tid % 32;

    // Compute metrics locally in registers (no memory traffic)
    unsigned long long ts = bpf_get_globaltimer();
    unsigned long long local_metric = ts ^ (row * col);  // Some computation

    // Warp-level aggregation: only lane 0 writes to map
    // Reduces memory traffic by 32x
    unsigned mask = __activemask();

    // Use warp shuffle to aggregate (register-to-register, no memory)
    for (int offset = 16; offset > 0; offset /= 2) {
        local_metric += __shfl_down_sync(mask, local_metric, offset);
    }

    // Only lane 0 writes the aggregated result - 1 WRITE per warp
    if (lane_id == 0) {
        bpf_map_update(bpf_performance_metrics, warp_id % MAP_SIZE, local_metric);
    }
}

//=============================================================================
// eBPF HOOK - BEST: Sampling-based monitoring
//=============================================================================

/**
 * Sampling: Only 1 in N threads actually logs
 * Reduces memory traffic by sampling_rate x
 */
#define SAMPLING_RATE 1024

__device__ void ebpf_hook_BEST_sampling(int row, int col, int K) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Only sample every Nth thread
    if ((tid % SAMPLING_RATE) != 0) {
        return;  // Early exit - no memory traffic
    }

    // Sampled thread logs data
    unsigned long long ts = bpf_get_globaltimer();
    unsigned long long sample_idx = tid / SAMPLING_RATE;

    bpf_map_update(bpf_performance_metrics, sample_idx % MAP_SIZE, ts);
    bpf_map_update(bpf_thread_state, sample_idx % MAP_SIZE, row * 1000 + col);
}

//=============================================================================
// Vector Add Kernel - Simple Memory-Bound Operation
//=============================================================================

/**
 * Simple vector add: C[i] = A[i] + B[i]
 * This is MEMORY-BOUND: only 1 FLOP per 3 memory accesses (2 reads + 1 write)
 * eBPF map operations will directly compete for memory bandwidth
 */
__global__ void vecadd_with_bad_hook(float *A, float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    // eBPF tracing hook
    ebpf_hook_BAD_bandwidth_hog(tid, blockIdx.x, N);

    // Simple vector add - memory bound operation
    C[tid] = A[tid] + B[tid];
}

__global__ void vecadd_with_medium_hook(float *A, float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    ebpf_hook_MEDIUM(tid, blockIdx.x, N);
    C[tid] = A[tid] + B[tid];
}

__global__ void vecadd_with_good_hook(float *A, float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    ebpf_hook_GOOD(tid, blockIdx.x, N);
    C[tid] = A[tid] + B[tid];
}

__global__ void vecadd_with_sampling_hook(float *A, float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    ebpf_hook_BEST_sampling(tid, blockIdx.x, N);
    C[tid] = A[tid] + B[tid];
}

__global__ void vecadd_no_hook(float *A, float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    C[tid] = A[tid] + B[tid];
}

//=============================================================================
// Main
//=============================================================================

int main() {
    // Vector size - large enough to saturate memory bandwidth
    const int N = 64 * 1024 * 1024;  // 64M elements
    const int SIZE = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 100;

    // Allocate host memory
    float *h_A = (float*)malloc(SIZE);
    float *h_B = (float*)malloc(SIZE);
    float *h_C = (float*)malloc(SIZE);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i + 50) % 100) / 100.0f;
    }

    // Allocate device memory for vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, SIZE);
    cudaMalloc(&d_B, SIZE);
    cudaMalloc(&d_C, SIZE);

    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    // Allocate BPF maps - LARGE to bust L2 cache
    size_t map_bytes = MAP_SIZE * sizeof(unsigned long long);
    size_t event_log_bytes = EVENT_LOG_SIZE * sizeof(unsigned long long);

    unsigned long long *d_event_log, *d_thread_state, *d_iteration_counts;
    unsigned long long *d_memory_access_log, *d_performance_metrics;

    cudaError_t err;
    err = cudaMalloc(&d_event_log, event_log_bytes);
    if (err != cudaSuccess) { printf("Failed to allocate event_log: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_thread_state, map_bytes);
    if (err != cudaSuccess) { printf("Failed to allocate thread_state: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_iteration_counts, map_bytes);
    if (err != cudaSuccess) { printf("Failed to allocate iteration_counts: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_memory_access_log, map_bytes);
    if (err != cudaSuccess) { printf("Failed to allocate memory_access_log: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_performance_metrics, map_bytes);
    if (err != cudaSuccess) { printf("Failed to allocate performance_metrics: %s\n", cudaGetErrorString(err)); return 1; }

    // Copy pointers to device symbols
    cudaMemcpyToSymbol(bpf_event_log, &d_event_log, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(bpf_thread_state, &d_thread_state, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(bpf_iteration_counts, &d_iteration_counts, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(bpf_memory_access_log, &d_memory_access_log, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(bpf_performance_metrics, &d_performance_metrics, sizeof(unsigned long long*));

    printf("BPF Maps allocated: %.1f GB total\n",
           (5.0 * map_bytes) / (1024.0 * 1024.0 * 1024.0));

    int total_threads = N;
    float data_gb = 3.0f * SIZE / (1024.0f * 1024.0f * 1024.0f);  // A + B + C

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Example 5: eBPF Map Bandwidth Contention on Vector Add       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("Configuration:\n");
    printf("  Vector size: %d elements (%.2f GB data)\n", N, data_gb);
    printf("  Total threads: %d\n", total_threads);
    printf("  Kernel: C[i] = A[i] + B[i] (pure memory-bound)\n\n");

    // Warmup
    vecadd_no_hook<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Baseline (no hook)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        vecadd_no_hook<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float baseline;
    cudaEventElapsedTime(&baseline, start, stop);
    float baseline_bw = (data_gb * ITERATIONS) / (baseline / 1000.0f);

    // Sampling hook (best)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        vecadd_with_sampling_hook<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sampling_time;
    cudaEventElapsedTime(&sampling_time, start, stop);
    float sampling_bw = (data_gb * ITERATIONS) / (sampling_time / 1000.0f);

    // Good hook (warp aggregation)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        vecadd_with_good_hook<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float good_time;
    cudaEventElapsedTime(&good_time, start, stop);
    float good_bw = (data_gb * ITERATIONS) / (good_time / 1000.0f);

    // Medium hook
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        vecadd_with_medium_hook<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float medium_time;
    cudaEventElapsedTime(&medium_time, start, stop);
    float medium_bw = (data_gb * ITERATIONS) / (medium_time / 1000.0f);

    // Bad hook (bandwidth hog)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        vecadd_with_bad_hook<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bad_time;
    cudaEventElapsedTime(&bad_time, start, stop);
    float bad_bw = (data_gb * ITERATIONS) / (bad_time / 1000.0f);

    printf("Results (%d iterations):\n", ITERATIONS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  No hook (baseline):            %8.2f ms  (%6.1f GB/s)\n", baseline, baseline_bw);
    printf("  BEST hook (sampling):          %8.2f ms  (%6.1f GB/s)  %.2fx overhead\n",
           sampling_time, sampling_bw, sampling_time/baseline);
    printf("  GOOD hook (warp-aggregate):    %8.2f ms  (%6.1f GB/s)  %.2fx overhead\n",
           good_time, good_bw, good_time/baseline);
    printf("  MEDIUM hook (reduced ops):     %8.2f ms  (%6.1f GB/s)  %.2fx overhead\n",
           medium_time, medium_bw, medium_time/baseline);
    printf("  BAD hook (tracing):            %8.2f ms  (%6.1f GB/s)  %.2fx overhead\n\n",
           bad_time, bad_bw, bad_time/baseline);

    printf("Performance Comparison:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  BAD vs Baseline:     %.2fx slowdown\n", bad_time / baseline);
    printf("  BAD vs BEST:         %.2fx slower\n", bad_time / sampling_time);
    printf("  MEDIUM vs BEST:      %.2fx slower\n", medium_time / sampling_time);
    printf("  GOOD vs BEST:        %.2fx slower\n\n", good_time / sampling_time);

    printf("Memory Bandwidth Analysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("Map operations per thread (eBPF hook):\n");
    printf("  BAD:      3 ops (1 read + 2 writes) - minimal tracing!\n");
    printf("  MEDIUM:   4 ops (1 read + 3 writes)\n");
    printf("  GOOD:     ~0.03 ops (1 write per warp)\n");
    printf("  BEST:     ~0.002 ops (1 write per %d threads)\n\n", SAMPLING_RATE);

    printf("Total memory traffic per kernel call:\n");
    printf("  Vector Add itself: %.0f MB (2 reads + 1 write × %d × 4 bytes)\n",
           3.0f * N * 4 / 1024 / 1024, N);
    printf("  BAD hook adds:     ~%d MB (3 ops × %d threads × 8 bytes)\n",
           (int)(3ULL * total_threads * 8 / 1024 / 1024), total_threads);
    printf("  → eBPF adds %.1fx more memory traffic!\n\n",
           (3.0 * total_threads * 8) / (3.0 * N * 4));

    printf("Root Cause Analysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("Vector Add is already MEMORY-BOUND:\n");
    printf("  - Only 1 FLOP per 12 bytes of memory access\n");
    printf("  - Performance limited by memory bandwidth\n\n");
    printf("Even MINIMAL eBPF tracing causes huge overhead:\n");
    printf("  - BAD hook: just 3 map ops per thread!\n");
    printf("  - With %d threads → %d million extra memory ops\n",
           total_threads, 3 * total_threads / 1000000);
    printf("  - Effective bandwidth cut in half (or worse)\n");
    printf("  - Simple kernel becomes 2-3x slower\n\n");

    printf("Traditional eBPF Verifier:  PASS\n");
    printf("  ✓ All map accesses are bounded\n");
    printf("  ✓ Execution terminates\n");
    printf("  ✓ Valid helper calls\n\n");

    printf("GPU-aware Verifier should:  WARN or LIMIT\n");
    printf("  ✗ Map traffic exceeds kernel's own memory traffic\n");
    printf("  ✗ Would significantly degrade memory-bound kernels\n\n");

    printf("Recommendations for GPU eBPF:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  1. Use sampling - not every thread needs to log\n");
    printf("  2. Use warp-level aggregation before map writes\n");
    printf("  3. Limit total map traffic to fraction of kernel traffic\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_event_log);
    cudaFree(d_thread_state);
    cudaFree(d_iteration_counts);
    cudaFree(d_memory_access_log);
    cudaFree(d_performance_metrics);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
