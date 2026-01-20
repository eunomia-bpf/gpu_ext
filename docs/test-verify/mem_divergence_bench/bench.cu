/**
 * GPU Performance Bottleneck Formal Benchmark
 *
 * Evaluates four major GPU performance bottlenecks:
 * 1. Memory coalescing efficiency (stride access patterns)
 * 2. Thread divergence overhead (branch factor)
 * 3. Atomic operation contention (hardware lock serialization)
 * 4. Arithmetic intensity / Roofline (memory vs compute bound)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Configuration
#define DATA_SIZE (1 << 20)  // 1M elements
#define ITERATIONS 100
#define BLOCK_SIZE 256

// ==================== Memory Coalescing Kernel ====================
// Formal parameter: stride - controls memory access pattern
// stride=1: coalesced (optimal), stride=32: non-coalesced (worst)
__global__ void mem_coalescing_kernel(float* data, int stride, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid * stride) % n;

    // Read-modify-write to prevent optimization
    float val = data[idx];
    val = val * 1.01f + 0.01f;
    data[idx] = val;
}

// Random access kernel - worst case for memory coalescing
__global__ void mem_random_kernel(float* data, int* random_idx, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int idx = random_idx[tid];  // Random index lookup

    float val = data[idx];
    val = val * 1.01f + 0.01f;
    data[idx] = val;
}

// ==================== Thread Divergence Kernel ====================
// Formal parameter: div_factor - controls branch divergence
// div_factor=1: no divergence, div_factor=32: max divergence (32 paths)
__device__ float compute_path(int path, int work_amount) {
    float result = 0.0f;
    for (int i = 0; i < work_amount; i++) {
        switch (path) {
            case 0:  result += sinf((float)i * 0.01f); break;
            case 1:  result += cosf((float)i * 0.01f); break;
            case 2:  result += tanf((float)i * 0.001f); break;
            case 3:  result += expf((float)i * 0.0001f); break;
            case 4:  result += logf((float)(i + 1)); break;
            case 5:  result += sqrtf((float)(i + 1)); break;
            case 6:  result += sinf((float)i * 0.02f); break;
            case 7:  result += cosf((float)i * 0.02f); break;
            case 8:  result += sinf((float)i * 0.03f); break;
            case 9:  result += cosf((float)i * 0.03f); break;
            case 10: result += sinf((float)i * 0.04f); break;
            case 11: result += cosf((float)i * 0.04f); break;
            case 12: result += sinf((float)i * 0.05f); break;
            case 13: result += cosf((float)i * 0.05f); break;
            case 14: result += sinf((float)i * 0.06f); break;
            case 15: result += cosf((float)i * 0.06f); break;
            case 16: result += sinf((float)i * 0.07f); break;
            case 17: result += cosf((float)i * 0.07f); break;
            case 18: result += sinf((float)i * 0.08f); break;
            case 19: result += cosf((float)i * 0.08f); break;
            case 20: result += sinf((float)i * 0.09f); break;
            case 21: result += cosf((float)i * 0.09f); break;
            case 22: result += sinf((float)i * 0.10f); break;
            case 23: result += cosf((float)i * 0.10f); break;
            case 24: result += sinf((float)i * 0.11f); break;
            case 25: result += cosf((float)i * 0.11f); break;
            case 26: result += sinf((float)i * 0.12f); break;
            case 27: result += cosf((float)i * 0.12f); break;
            case 28: result += sinf((float)i * 0.13f); break;
            case 29: result += cosf((float)i * 0.13f); break;
            case 30: result += sinf((float)i * 0.14f); break;
            default: result += cosf((float)i * 0.15f); break;
        }
    }
    return result;
}

__global__ void thread_divergence_kernel(float* data, int div_factor, int work_amount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 32;  // Position within warp
    int path = lane_id % div_factor;  // Which path this thread takes

    data[tid] = compute_path(path, work_amount);
}

// ==================== Atomic Contention Kernel ====================
// Formal parameter: contention_factor - how many threads share one counter
// contention_factor=1: no contention (each thread has own counter)
// contention_factor=N: N threads compete for same counter
__global__ void atomic_contention_kernel(int* counters, int contention_factor, int ops_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int counter_idx = tid / contention_factor;  // Which counter this thread uses

    for (int i = 0; i < ops_per_thread; i++) {
        atomicAdd(&counters[counter_idx], 1);
    }
}

// ==================== Roofline/Arithmetic Intensity Kernel ====================
// Formal parameter: flops_per_elem - FLOPs to perform per memory element
// AI (Arithmetic Intensity) = flops_per_elem * 2 / 8 = flops_per_elem / 4
// Low AI = memory bound, High AI = compute bound
__global__ void roofline_kernel(float* data, int flops_per_elem, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = data[tid];  // 1 load (4 bytes)

    // Perform flops_per_elem iterations, each with 2 FLOPs (mul + add)
    for (int i = 0; i < flops_per_elem; i++) {
        val = val * 1.0001f + 0.0001f;
    }

    data[tid] = val;  // 1 store (4 bytes)
}

// ==================== Timing Utilities ====================
float run_memory_benchmark(float* d_data, int stride, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    mem_coalescing_kernel<<<blocks, BLOCK_SIZE>>>(d_data, stride, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        mem_coalescing_kernel<<<blocks, BLOCK_SIZE>>>(d_data, stride, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;  // Average time per iteration
}

float run_memory_random_benchmark(float* d_data, int* d_random_idx, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    mem_random_kernel<<<blocks, BLOCK_SIZE>>>(d_data, d_random_idx, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        mem_random_kernel<<<blocks, BLOCK_SIZE>>>(d_data, d_random_idx, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;
}

float run_divergence_benchmark(float* d_data, int div_factor, int work_amount) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    thread_divergence_kernel<<<blocks, BLOCK_SIZE>>>(d_data, div_factor, work_amount);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        thread_divergence_kernel<<<blocks, BLOCK_SIZE>>>(d_data, div_factor, work_amount);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;
}

float run_atomic_benchmark(int* d_counters, int contention_factor, int ops_per_thread, int num_threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Reset counters
    int num_counters = (num_threads + contention_factor - 1) / contention_factor;
    cudaMemset(d_counters, 0, num_counters * sizeof(int));

    // Warmup
    atomic_contention_kernel<<<blocks, BLOCK_SIZE>>>(d_counters, contention_factor, ops_per_thread);
    cudaDeviceSynchronize();

    cudaMemset(d_counters, 0, num_counters * sizeof(int));

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_contention_kernel<<<blocks, BLOCK_SIZE>>>(d_counters, contention_factor, ops_per_thread);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;
}

float run_roofline_benchmark(float* d_data, int flops_per_elem, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    roofline_kernel<<<blocks, BLOCK_SIZE>>>(d_data, flops_per_elem, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        roofline_kernel<<<blocks, BLOCK_SIZE>>>(d_data, flops_per_elem, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;
}

// ==================== Main ====================
int main(int argc, char** argv) {
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float theoretical_bandwidth = prop.memoryClockRate * 1000.0f * (prop.memoryBusWidth / 8) * 2 / 1e9;  // GB/s

    printf("GPU: %s\n", prop.name);
    printf("Theoretical Memory Bandwidth: %.1f GB/s\n", theoretical_bandwidth);
    printf("Data size: %d elements (%.2f MB)\n\n", DATA_SIZE, DATA_SIZE * sizeof(float) / 1e6);

    // Allocate memory
    float *d_data;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));
    cudaMemset(d_data, 0, DATA_SIZE * sizeof(float));

    // Open CSV file
    FILE* csv = fopen("results.csv", "w");
    fprintf(csv, "test_type,parameter,time_ms,bandwidth_gbps,efficiency\n");

    // ==================== Memory Coalescing Benchmark ====================
    printf("=== Memory Coalescing Benchmark ===\n");
    printf("%-10s %10s %15s %12s %10s\n", "pattern", "time(ms)", "bandwidth(GB/s)", "efficiency", "slowdown");
    printf("-------------------------------------------------------------\n");

    int strides[] = {1, 2, 4, 8, 16, 32};
    int num_strides = sizeof(strides) / sizeof(strides[0]);
    float mem_baseline = 0;

    for (int i = 0; i < num_strides; i++) {
        int stride = strides[i];
        float time_ms = run_memory_benchmark(d_data, stride, DATA_SIZE);

        // Calculate effective bandwidth (bytes transferred per second)
        // Each thread does 1 read + 1 write = 8 bytes
        float bytes_transferred = (float)DATA_SIZE * 2 * sizeof(float) * ITERATIONS;
        float bandwidth = bytes_transferred / (time_ms * ITERATIONS * 1e6);  // GB/s
        float efficiency = bandwidth / theoretical_bandwidth;

        if (i == 0) mem_baseline = time_ms;
        float slowdown = time_ms / mem_baseline;

        char label[16];
        snprintf(label, sizeof(label), "stride=%d", stride);
        printf("%-10s %10.3f %15.1f %11.1f%% %9.2fx\n", label, time_ms, bandwidth, efficiency * 100, slowdown);
        fprintf(csv, "memory,%d,%.4f,%.2f,%.4f\n", stride, time_ms, bandwidth, efficiency);
    }

    // Random access pattern (worst case)
    // Generate random indices on host and copy to device
    int* h_random_idx = (int*)malloc(DATA_SIZE * sizeof(int));
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < DATA_SIZE; i++) {
        h_random_idx[i] = rand() % DATA_SIZE;
    }
    int* d_random_idx;
    cudaMalloc(&d_random_idx, DATA_SIZE * sizeof(int));
    cudaMemcpy(d_random_idx, h_random_idx, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    float time_ms = run_memory_random_benchmark(d_data, d_random_idx, DATA_SIZE);
    float bytes_transferred = (float)DATA_SIZE * 2 * sizeof(float) * ITERATIONS;
    float bandwidth = bytes_transferred / (time_ms * ITERATIONS * 1e6);
    float efficiency = bandwidth / theoretical_bandwidth;
    float slowdown = time_ms / mem_baseline;

    printf("%-10s %10.3f %15.1f %11.1f%% %9.2fx\n", "random", time_ms, bandwidth, efficiency * 100, slowdown);
    fprintf(csv, "memory,random,%.4f,%.2f,%.4f\n", time_ms, bandwidth, efficiency);

    free(h_random_idx);
    cudaFree(d_random_idx);

    // ==================== Thread Divergence Benchmark ====================
    printf("\n=== Thread Divergence Benchmark ===\n");
    printf("%-10s %10s %12s\n", "div_factor", "time(ms)", "slowdown");
    printf("-------------------------------------\n");

    int div_factors[] = {1, 2, 4, 8, 16, 32};
    int num_divs = sizeof(div_factors) / sizeof(div_factors[0]);
    int work_amount = 100;  // Iterations per path
    float div_baseline = 0;

    for (int i = 0; i < num_divs; i++) {
        int div_factor = div_factors[i];
        float time_ms = run_divergence_benchmark(d_data, div_factor, work_amount);

        if (i == 0) div_baseline = time_ms;
        float slowdown = time_ms / div_baseline;

        printf("%-10d %10.3f %11.2fx\n", div_factor, time_ms, slowdown);
        fprintf(csv, "divergence,%d,%.4f,NA,%.4f\n", div_factor, time_ms, 1.0f / slowdown);
    }

    // ==================== Atomic Contention Benchmark ====================
    printf("\n=== Atomic Contention Benchmark ===\n");
    printf("%-12s %10s %15s %12s\n", "contention", "time(ms)", "throughput(M/s)", "slowdown");
    printf("----------------------------------------------------\n");

    // Allocate counters for atomic test
    int *d_counters;
    cudaMalloc(&d_counters, DATA_SIZE * sizeof(int));

    int contention_factors[] = {1, 32, 256, DATA_SIZE};  // DATA_SIZE = ALL threads on 1 counter
    int num_contentions = sizeof(contention_factors) / sizeof(contention_factors[0]);
    int ops_per_thread = 100;
    float atomic_baseline = 0;

    for (int i = 0; i < num_contentions; i++) {
        int contention = contention_factors[i];
        float time_ms = run_atomic_benchmark(d_counters, contention, ops_per_thread, DATA_SIZE);

        // Throughput = total ops / time
        float total_ops = (float)DATA_SIZE * ops_per_thread * ITERATIONS;
        float throughput = total_ops / (time_ms * ITERATIONS * 1e3);  // Million ops/sec

        if (i == 0) atomic_baseline = time_ms;
        float slowdown = time_ms / atomic_baseline;

        if (contention == DATA_SIZE) {
            printf("%-12s %10.3f %15.2f %11.1fx\n", "ALL", time_ms, throughput, slowdown);
            fprintf(csv, "atomic,ALL,%.4f,%.2f,%.4f\n", time_ms, throughput, 1.0f / slowdown);
        } else {
            printf("%-12d %10.3f %15.2f %11.1fx\n", contention, time_ms, throughput, slowdown);
            fprintf(csv, "atomic,%d,%.4f,%.2f,%.4f\n", contention, time_ms, throughput, 1.0f / slowdown);
        }
    }

    cudaFree(d_counters);

    // ==================== Roofline/Arithmetic Intensity Benchmark ====================
    printf("\n=== Roofline (Arithmetic Intensity) Benchmark ===\n");
    printf("%-10s %8s %12s %15s %10s %12s\n", "FLOPs/elem", "AI", "time(ms)", "GFLOPS", "slowdown", "bound");
    printf("------------------------------------------------------------------------\n");

    // More data points for complete roofline curve
    int flops_per_elem[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int num_flops = sizeof(flops_per_elem) / sizeof(flops_per_elem[0]);
    float roofline_baseline = 0;

    for (int i = 0; i < num_flops; i++) {
        int flops = flops_per_elem[i];
        float time_ms = run_roofline_benchmark(d_data, flops, DATA_SIZE);

        // Calculate metrics
        // Total FLOPs = DATA_SIZE * flops * 2 (mul + add per iteration)
        float total_flops = (float)DATA_SIZE * flops * 2 * ITERATIONS;
        float gflops = total_flops / (time_ms * ITERATIONS * 1e6);

        // Bandwidth = bytes transferred / time
        // Each element: 1 read (4B) + 1 write (4B) = 8 bytes
        float bytes = (float)DATA_SIZE * 8 * ITERATIONS;
        float bandwidth = bytes / (time_ms * ITERATIONS * 1e6);

        // Arithmetic intensity = FLOPs / bytes = flops * 2 / 8
        float ai = flops * 2.0f / 8.0f;

        if (i == 0) roofline_baseline = time_ms;
        float slowdown = time_ms / roofline_baseline;

        // Determine if memory or compute bound
        const char* bound = (bandwidth > theoretical_bandwidth * 0.3) ? "MEM" : "COMPUTE";

        printf("%-10d %8.2f %12.3f %12.1f %10.2fx %12s\n", flops, ai, time_ms, gflops, slowdown, bound);
        fprintf(csv, "roofline,%d,%.4f,%.2f,%.4f\n", flops, time_ms, gflops, ai);
    }

    fclose(csv);
    printf("\nResults saved to: results.csv\n");

    // Cleanup
    cudaFree(d_data);

    return 0;
}
