/**
 * pagerank.cuh - PageRank kernel (Iterative graph algorithm)
 *
 * SOURCE: Adapted from GPreempt/EMOGI PageRank implementation
 * Reference: gpu_ext_policy/docs/driver_docs/sched/GPreempt/src/workloads/graphcompute.cu
 *
 * Access pattern: Random (indirect access through adjacency list) + Iteration
 * Representative of: Graph algorithms, GNN, social network analysis, recommendation
 *
 * Design principle (matching GEMM/Hotspot/Jacobi):
 * - Fixed graph size (~300MB for 1M nodes, avgDegree=16)
 * - Control total work via PageRank iteration count
 * - Multiple iterations create temporal locality on adjacency list
 */

#ifndef PAGERANK_CUH
#define PAGERANK_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstdint>

// ============================================================================
// PageRank Constants (from EMOGI/GPreempt)
// ============================================================================
#define PR_WARP_SIZE 32
#define PR_WARP_SHIFT 5
#define PR_BLOCK_SIZE 256
#define PR_MEM_ALIGN 31  // For warp-aligned memory access

// ============================================================================
// PageRank Kernels - Adapted from GPreempt/EMOGI
// Source: graphcompute.cu lines 211-250
// ============================================================================

// Initialize PageRank values
__global__ void pr_initialize(bool *label, float *delta, float *residual, float *value,
                              uint64_t vertex_count, uint64_t *vertexList, float alpha) {
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        uint64_t degree = vertexList[tid + 1] - vertexList[tid];
        if (degree > 0) {
            delta[tid] = (1.0f - alpha) * alpha / degree;
        } else {
            delta[tid] = 0.0f;
        }
        residual[tid] = 0.0f;
        label[tid] = true;
    }
}

// Main PageRank kernel with warp-coalesced memory access
__global__ void pr_kernel_coalesce(bool *label, float *delta, float *residual,
                                   uint64_t vertex_count, uint64_t *vertexList, uint64_t *edgeList) {
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t warpIdx = tid >> PR_WARP_SHIFT;
    uint64_t laneIdx = tid & (PR_WARP_SIZE - 1);

    if (warpIdx < vertex_count && label[warpIdx]) {
        uint64_t start = vertexList[warpIdx];
        uint64_t shift_start = start & PR_MEM_ALIGN;
        uint64_t end = vertexList[warpIdx + 1];

        for (uint64_t i = shift_start + laneIdx; i < end; i += PR_WARP_SIZE) {
            if (i >= start) {
                atomicAdd(&residual[edgeList[i]], delta[warpIdx]);
            }
        }
        label[warpIdx] = false;
    }
}

// Update PageRank values based on residual
__global__ void pr_update(bool *label, float *delta, float *residual, float *value,
                          uint64_t vertex_count, uint64_t *vertexList,
                          float tolerance, float alpha, bool *changed) {
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        uint64_t degree = vertexList[tid + 1] - vertexList[tid];
        if (degree > 0) {
            delta[tid] = residual[tid] * alpha / degree;
        }
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
    }
}

// ============================================================================
// Graph structure for PageRank
// ============================================================================

struct PRGraph {
    uint64_t numNodes;
    uint64_t numEdges;
    uint64_t *vertexList;  // CSR offset array (size: numNodes + 1)
    uint64_t *edgeList;    // CSR adjacency array (size: numEdges)
};

// Generate random graph in CSR format
inline void generatePRGraph(PRGraph& g, uint64_t numNodes, int avgDegree, const std::string& mode) {
    g.numNodes = numNodes;

    // Allocate vertex list (CSR offsets)
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&g.vertexList, sizeof(uint64_t) * (numNodes + 1)));
    } else {
        CUDA_CHECK(cudaMallocManaged(&g.vertexList, sizeof(uint64_t) * (numNodes + 1)));
    }

    // Generate degrees and compute offsets on host
    std::vector<uint64_t> offsets(numNodes + 1);
    uint64_t totalEdges = 0;
    srand(42);  // Fixed seed for reproducibility

    offsets[0] = 0;
    for (uint64_t i = 0; i < numNodes; i++) {
        // Random degree between 1 and 2*avgDegree
        int degree = 1 + rand() % (2 * avgDegree);
        totalEdges += degree;
        offsets[i + 1] = totalEdges;
    }
    g.numEdges = totalEdges;

    // Copy offsets to device/managed memory
    if (mode == "device") {
        CUDA_CHECK(cudaMemcpy(g.vertexList, offsets.data(), sizeof(uint64_t) * (numNodes + 1), cudaMemcpyHostToDevice));
    } else {
        memcpy(g.vertexList, offsets.data(), sizeof(uint64_t) * (numNodes + 1));
    }

    // Allocate edge list
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&g.edgeList, sizeof(uint64_t) * totalEdges));
        // Generate edges on host and copy
        std::vector<uint64_t> edges(totalEdges);
        for (uint64_t i = 0; i < numNodes; i++) {
            for (uint64_t j = offsets[i]; j < offsets[i + 1]; j++) {
                edges[j] = rand() % numNodes;
            }
        }
        CUDA_CHECK(cudaMemcpy(g.edgeList, edges.data(), sizeof(uint64_t) * totalEdges, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMallocManaged(&g.edgeList, sizeof(uint64_t) * totalEdges));
        // Generate edges directly in managed memory
        for (uint64_t i = 0; i < numNodes; i++) {
            for (uint64_t j = offsets[i]; j < offsets[i + 1]; j++) {
                g.edgeList[j] = rand() % numNodes;
            }
        }
    }
}

inline void freePRGraph(PRGraph& g) {
    cudaFree(g.vertexList);
    cudaFree(g.edgeList);
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

inline void run_pagerank(size_t total_working_set, const std::string& mode,
                         size_t stride_bytes, int iterations,
                         std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // PageRank uses graph-based random access pattern

    // =========================================================================
    // Fixed graph size + iteration control (matching GEMM/Hotspot/Jacobi design)
    // - 1M nodes, avgDegree=16 -> ~300MB graph
    // - Control total work via PageRank iteration count
    // =========================================================================

    const uint64_t NUM_NODES = 1000000;  // 1M nodes
    const int AVG_DEGREE = 16;

    // Graph size: vertexList (8MB) + edgeList (~128MB for 16M edges)
    // Plus PageRank arrays: label (1MB) + delta (4MB) + residual (4MB) + value (4MB)
    // Total ~150MB per iteration access
    size_t graph_bytes = (NUM_NODES + 1) * sizeof(uint64_t) +  // vertexList
                         NUM_NODES * AVG_DEGREE * sizeof(uint64_t);  // edgeList (avg)
    size_t pagerank_arrays = NUM_NODES * (sizeof(bool) + 3 * sizeof(float));
    size_t single_iteration_bytes = graph_bytes + pagerank_arrays;

    // Calculate iteration count based on total_working_set
    int num_iterations = total_working_set / single_iteration_bytes;
    if (num_iterations < 10) num_iterations = 10;
    if (num_iterations > 1000) num_iterations = 1000;  // Reasonable upper limit

    // PageRank parameters
    float alpha = 0.85f;
    float tolerance = 1e-6f;

    // Generate graph
    PRGraph g;
    generatePRGraph(g, NUM_NODES, AVG_DEGREE, mode);

    // Allocate PageRank arrays
    bool *label;
    float *delta, *residual, *value;
    bool *changed_d;

    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&label, sizeof(bool) * NUM_NODES));
        CUDA_CHECK(cudaMalloc(&delta, sizeof(float) * NUM_NODES));
        CUDA_CHECK(cudaMalloc(&residual, sizeof(float) * NUM_NODES));
        CUDA_CHECK(cudaMalloc(&value, sizeof(float) * NUM_NODES));
        CUDA_CHECK(cudaMalloc(&changed_d, sizeof(bool)));
    } else {
        CUDA_CHECK(cudaMallocManaged(&label, sizeof(bool) * NUM_NODES));
        CUDA_CHECK(cudaMallocManaged(&delta, sizeof(float) * NUM_NODES));
        CUDA_CHECK(cudaMallocManaged(&residual, sizeof(float) * NUM_NODES));
        CUDA_CHECK(cudaMallocManaged(&value, sizeof(float) * NUM_NODES));
        CUDA_CHECK(cudaMallocManaged(&changed_d, sizeof(bool)));
    }

    // Apply UVM hints if needed
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        apply_uvm_hints(g.vertexList, sizeof(uint64_t) * (NUM_NODES + 1), mode, dev);
        apply_uvm_hints(g.edgeList, sizeof(uint64_t) * g.numEdges, mode, dev);
        apply_uvm_hints(label, sizeof(bool) * NUM_NODES, mode, dev);
        apply_uvm_hints(delta, sizeof(float) * NUM_NODES, mode, dev);
        apply_uvm_hints(residual, sizeof(float) * NUM_NODES, mode, dev);
        apply_uvm_hints(value, sizeof(float) * NUM_NODES, mode, dev);
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    fprintf(stderr, "PageRank config: nodes=%lu, edges=%lu, iterations=%d\n",
            NUM_NODES, g.numEdges, num_iterations);
    fprintf(stderr, "  Graph size: %.1f MB, Single iteration: %.1f MB, Total: %.1f MB\n",
            (graph_bytes) / (1024.0 * 1024.0),
            single_iteration_bytes / (1024.0 * 1024.0),
            single_iteration_bytes * num_iterations / (1024.0 * 1024.0));

    // Launch configuration
    int numBlocks_init = (NUM_NODES + PR_BLOCK_SIZE - 1) / PR_BLOCK_SIZE;
    int numBlocks_kernel = (NUM_NODES * PR_WARP_SIZE + PR_BLOCK_SIZE - 1) / PR_BLOCK_SIZE;

    // Lambda for PageRank iteration
    auto run_pagerank_iterations = [&]() {
        // Initialize
        pr_initialize<<<numBlocks_init, PR_BLOCK_SIZE>>>(
            label, delta, residual, value, NUM_NODES, g.vertexList, alpha);

        // Fixed number of iterations (not convergence-based for benchmarking)
        for (int iter = 0; iter < num_iterations; iter++) {
            pr_kernel_coalesce<<<numBlocks_kernel, PR_BLOCK_SIZE>>>(
                label, delta, residual, NUM_NODES, g.vertexList, g.edgeList);

            pr_update<<<numBlocks_init, PR_BLOCK_SIZE>>>(
                label, delta, residual, value, NUM_NODES, g.vertexList,
                tolerance, alpha, changed_d);
        }
    };

    // Warmup
    for (int w = 0; w < 2; w++) {
        run_pagerank_iterations();
        cudaDeviceSynchronize();
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        cudaEventRecord(start);

        run_pagerank_iterations();

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

    // Bytes accessed: each iteration reads graph + PageRank arrays
    result.bytes_accessed = (size_t)num_iterations * single_iteration_bytes;

    // Cleanup
    freePRGraph(g);
    cudaFree(label);
    cudaFree(delta);
    cudaFree(residual);
    cudaFree(value);
    cudaFree(changed_d);
}

#endif // PAGERANK_CUH
