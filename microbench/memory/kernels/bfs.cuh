/**
 * bfs.cuh - Breadth-First Search kernel (Graph traversal pattern)
 *
 * SOURCE: UVM Benchmark Suite
 * Original file: /memory/uvm_bench/UVM_benchmark/UVM_benchmarks/bfs/bfsCUDA.cu
 *
 * Access pattern: Random (indirect access through adjacency list)
 * Representative of: Graph algorithms, GNN, PageRank, SSSP
 *
 * This is a rand_stream equivalent with real graph traversal.
 *
 * NOTE: Kernel code below is EXACT COPY from original source.
 */

#ifndef BFS_CUH
#define BFS_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <climits>
#include <cstdlib>

// ============================================================================
// Original UVM Benchmark BFS Kernels - EXACT COPY from:
// /memory/uvm_bench/UVM_benchmark/UVM_benchmarks/bfs/bfsCUDA.cu lines 9-66
// ============================================================================

__global__
void simpleBfs(int N, int level, int *d_adjacencyList, int *d_edgesOffset,
               int *d_edgesSize, int *d_distance, int *d_parent, int *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;

    if (thid < N && d_distance[thid] == level) {
        int u = thid;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
                valueChange = 1;
            }
        }
    }

    if (valueChange) {
        *changed = valueChange;
    }
}

__global__
void queueBfs(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
              int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
                d_parent[v] = i;
                int position = atomicAdd(nextQueueSize, 1);
                d_nextQueue[position] = v;
            }
        }
    }
}

//Scan bfs
__global__
void nextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
               int queueSize, int *d_currentQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent,
                  int queueSize, int *d_currentQueue, int *d_degrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        int degree = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

// ============================================================================
// Synthetic graph generator for benchmark
// ============================================================================

struct Graph {
    int numNodes;
    int numEdges;
    int *adjacencyList;
    int *edgesOffset;
    int *edgesSize;
};

inline void generateRandomGraph(Graph& g, int numNodes, int avgDegree) {
    g.numNodes = numNodes;

    // Allocate offset and size arrays
    cudaMallocManaged(&g.edgesOffset, sizeof(int) * numNodes);
    cudaMallocManaged(&g.edgesSize, sizeof(int) * numNodes);

    // Generate random degrees
    std::vector<int> degrees(numNodes);
    int totalEdges = 0;
    for (int i = 0; i < numNodes; i++) {
        // Random degree between 1 and 2*avgDegree
        degrees[i] = 1 + rand() % (2 * avgDegree);
        g.edgesOffset[i] = totalEdges;
        g.edgesSize[i] = degrees[i];
        totalEdges += degrees[i];
    }
    g.numEdges = totalEdges;

    // Allocate adjacency list
    cudaMallocManaged(&g.adjacencyList, sizeof(int) * totalEdges);

    // Generate random edges
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < degrees[i]; j++) {
            // Random neighbor (not self)
            int neighbor = rand() % numNodes;
            while (neighbor == i) neighbor = rand() % numNodes;
            g.adjacencyList[g.edgesOffset[i] + j] = neighbor;
        }
    }
}

inline void freeGraph(Graph& g) {
    cudaFree(g.adjacencyList);
    cudaFree(g.edgesOffset);
    cudaFree(g.edgesSize);
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

struct BFSResult {
    size_t bytes_accessed;
    float median_ms;
    float min_ms;
    float max_ms;
};

inline void run_bfs(size_t total_working_set, const std::string& mode,
                    size_t stride_bytes, int iterations,
                    std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // BFS uses graph-based random access pattern

    // Estimate graph size from working set
    // Working set ~= adjacencyList + edgesOffset + edgesSize + distance + parent
    // Assume avgDegree = 16, so adjacencyList dominates
    int avgDegree = 16;
    int numNodes = (int)(total_working_set / (avgDegree * sizeof(int) + 4 * sizeof(int)));
    if (numNodes < 1000) numNodes = 1000;

    // Generate random graph
    Graph g;
    generateRandomGraph(g, numNodes, avgDegree);

    // Allocate BFS data structures
    int *d_distance, *d_parent, *d_changed;
    cudaMallocManaged(&d_distance, sizeof(int) * numNodes);
    cudaMallocManaged(&d_parent, sizeof(int) * numNodes);
    cudaMallocManaged(&d_changed, sizeof(int));

    // Apply UVM hints if needed
    if (mode == "uvm_prefetch") {
        int dev;
        cudaGetDevice(&dev);
        cudaMemPrefetchAsync(g.adjacencyList, sizeof(int) * g.numEdges, dev, 0);
        cudaMemPrefetchAsync(g.edgesOffset, sizeof(int) * numNodes, dev, 0);
        cudaMemPrefetchAsync(g.edgesSize, sizeof(int) * numNodes, dev, 0);
        cudaMemPrefetchAsync(d_distance, sizeof(int) * numNodes, dev, 0);
        cudaMemPrefetchAsync(d_parent, sizeof(int) * numNodes, dev, 0);
        cudaDeviceSynchronize();
    }

    int blockSize = 256;
    int numBlocks = (numNodes + blockSize - 1) / blockSize;

    // Warmup
    for (int w = 0; w < 2; w++) {
        // Initialize distance
        for (int i = 0; i < numNodes; i++) {
            d_distance[i] = INT_MAX;
            d_parent[i] = -1;
        }
        d_distance[0] = 0;  // Start from node 0

        int level = 0;
        int changed = 1;
        while (changed) {
            *d_changed = 0;
            simpleBfs<<<numBlocks, blockSize>>>(numNodes, level, g.adjacencyList,
                                                 g.edgesOffset, g.edgesSize,
                                                 d_distance, d_parent, d_changed);
            cudaDeviceSynchronize();
            changed = *d_changed;
            level++;
            if (level > 100) break;  // Safety limit
        }
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        // Reset distance
        for (int i = 0; i < numNodes; i++) {
            d_distance[i] = INT_MAX;
            d_parent[i] = -1;
        }
        d_distance[0] = 0;

        cudaEventRecord(start);

        int level = 0;
        int changed = 1;
        while (changed) {
            *d_changed = 0;
            simpleBfs<<<numBlocks, blockSize>>>(numNodes, level, g.adjacencyList,
                                                 g.edgesOffset, g.edgesSize,
                                                 d_distance, d_parent, d_changed);
            cudaDeviceSynchronize();
            changed = *d_changed;
            level++;
            if (level > 100) break;
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

    // Estimate bytes accessed (random access pattern)
    result.bytes_accessed = (size_t)g.numEdges * sizeof(int) * 2 +  // adjacencyList reads
                            (size_t)numNodes * sizeof(int) * 4;      // offset, size, distance, parent

    // Cleanup
    freeGraph(g);
    cudaFree(d_distance);
    cudaFree(d_parent);
    cudaFree(d_changed);
}

#endif // BFS_CUH
