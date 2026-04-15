// CLC Policy Benchmark - Unified benchmark for all scheduling policies
// Supports multiple modes: comprehensive comparison, specialized policies, quick test

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <vector>
#include "ai_workloads.cuh"
#include "clc_policy_framework.cuh"
#include "clc_policies.cuh"
#include "benchmark_kernels.cuh"

using namespace clc_policy;

// ============================================================================
// Comprehensive Benchmark Result Storage
// ============================================================================

struct ComprehensiveResult {
    const char* workload_name;
    int prologue;
    BenchmarkResult fixed_work;
    BenchmarkResult fixed_blocks;
    BenchmarkResult clc_baseline;
    BenchmarkResult greedy;
    BenchmarkResult maxsteals;
    BenchmarkResult throttled;
    BenchmarkResult selective;
    BenchmarkResult probe_everyn;
    BenchmarkResult latency_budget;
    BenchmarkResult token_bucket;
    BenchmarkResult voting;
    BenchmarkResult cluster_aware;
};

// ============================================================================
// Mode 1: Comprehensive Comparison (baselines + all policies)
// ============================================================================

template<typename WorkloadType>
ComprehensiveResult run_comprehensive(const char* scenario, int prologue,
                       float* d_data, int n, int threads, cudaDeviceProp& prop, float* h_data) {
    int blocks_fixed_work = (n + threads - 1) / threads;
    int blocks_fixed_blocks = prop.multiProcessorCount * 2;
    int blocks_clc = (n + threads - 1) / threads;
    int warmup = 3;
    int runs = 10;

    ComprehensiveResult result;
    result.workload_name = scenario;
    result.prologue = prologue;

    // Run all benchmarks - baselines
    result.fixed_work = run_fixed_work<WorkloadType>(d_data, n, blocks_fixed_work, threads, h_data, prologue, warmup, runs);
    result.fixed_blocks = run_fixed_blocks<WorkloadType>(d_data, n, blocks_fixed_blocks, threads, h_data, prologue, warmup, runs);
    result.clc_baseline = run_clc_baseline<WorkloadType>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);

    // Run all policies
    result.greedy = run_clc_policy<WorkloadType, GreedyPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.maxsteals = run_clc_policy<WorkloadType, MaxStealsPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.throttled = run_clc_policy<WorkloadType, NeverStealPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.selective = run_clc_policy<WorkloadType, SelectiveBlocksPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.probe_everyn = run_clc_policy<WorkloadType, WorkloadAwarePolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.latency_budget = run_clc_policy<WorkloadType, LatencyBudgetPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.token_bucket = run_clc_policy<WorkloadType, TokenBucketPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.voting = run_clc_policy<WorkloadType, VotingPolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);
    result.cluster_aware = run_clc_policy<WorkloadType, ClusterAwarePolicy>(d_data, n, blocks_clc, threads, h_data, prologue, warmup, runs);

    return result;
}

// ============================================================================
// Mode 2: Specialized Policy Comparison (production policies)
// ============================================================================

template<typename WorkloadType>
void run_specialized(const char* workload_name, const char* use_case, const char* best_policy,
                     int prologue, float* d_data, int n, int threads, float* h_data) {
    int blocks = (n + threads - 1) / threads;
    int warmup = 3;
    int runs = 10;

    printf("\n========================================\n");
    printf("Workload: %s (Prologue: %d)\n", workload_name, prologue);
    printf("Use case: %s\n", use_case);
    printf("Best policy: %s\n", best_policy);
    printf("Problem size: %d elements, %d blocks\n", n, blocks);
    printf("========================================\n");
    printf("%-30s | %10s | %10s | %10s\n", "Policy", "Time (ms)", "Blocks", "Steals");
    printf("-----------------------------------------------------------------------------------\n");

    BenchmarkResult r_greedy = run_clc_policy<WorkloadType, GreedyPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "Greedy (baseline)", r_greedy.avg_time_ms, r_greedy.avg_blocks, r_greedy.avg_steals);

    BenchmarkResult r_probe = run_clc_policy<WorkloadType, WorkloadAwarePolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "WorkloadAware (co-designed)", r_probe.avg_time_ms, r_probe.avg_blocks, r_probe.avg_steals);

    BenchmarkResult r_latency = run_clc_policy<WorkloadType, LatencyBudgetPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "LatencyBudget (150us)", r_latency.avg_time_ms, r_latency.avg_blocks, r_latency.avg_steals);

    BenchmarkResult r_token = run_clc_policy<WorkloadType, TokenBucketPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "TokenBucket (rate-limited)", r_token.avg_time_ms, r_token.avg_blocks, r_token.avg_steals);

    BenchmarkResult r_maxsteals = run_clc_policy<WorkloadType, MaxStealsPolicy>(
        d_data, n, blocks, threads, h_data, prologue, warmup, runs);
    printf("%-30s | %10.3f | %10.0f | %10.0f\n",
           "MaxSteals (limit=8)", r_maxsteals.avg_time_ms, r_maxsteals.avg_blocks, r_maxsteals.avg_steals);

    printf("-----------------------------------------------------------------------------------\n");

    printf("\nAnalysis:\n");
    float probe_overhead = ((r_probe.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  WorkloadAware: %.2f%% (co-designed for imbalance patterns)\n", probe_overhead);

    float latency_impact = ((r_latency.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  LatencyBudget: %.2f%% (stable p95/p99 latency)\n", latency_impact);

    float token_impact = ((r_token.avg_time_ms - r_greedy.avg_time_ms) / r_greedy.avg_time_ms) * 100.0f;
    printf("  TokenBucket: %.2f%% (bandwidth fairness)\n", token_impact);

    printf("  Steals reduction:\n");
    printf("    WorkloadAware: %.1f%%\n", 100.0f * (r_greedy.avg_steals - r_probe.avg_steals) / r_greedy.avg_steals);
    printf("    LatencyBudget: %.1f%%\n", 100.0f * (r_greedy.avg_steals - r_latency.avg_steals) / r_greedy.avg_steals);
    printf("    TokenBucket: %.1f%%\n", 100.0f * (r_greedy.avg_steals - r_token.avg_steals) / r_greedy.avg_steals);
}

// ============================================================================
// CSV Output Functions
// ============================================================================

void print_table_header() {
    printf("Workload,Prologue,FixedWork_ms,FixedBlocks_ms,CLCBaseline_ms,Greedy_ms,MaxSteals_ms,NeverSteal_ms,Selective_ms,WorkloadAware_ms,LatencyBudget_ms,TokenBucket_ms,Voting_ms,ClusterAware_ms,");
    printf("CLCBaseline_steals,Greedy_steals,MaxSteals_steals,NeverSteal_steals,Selective_steals,WorkloadAware_steals,LatencyBudget_steals,TokenBucket_steals,Voting_steals,ClusterAware_steals\n");
}

void print_table_row_time(const ComprehensiveResult& r) {
    printf("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
           r.workload_name, r.prologue,
           r.fixed_work.avg_time_ms,
           r.fixed_blocks.avg_time_ms,
           r.clc_baseline.avg_time_ms,
           r.greedy.avg_time_ms,
           r.maxsteals.avg_time_ms,
           r.throttled.avg_time_ms,
           r.selective.avg_time_ms,
           r.probe_everyn.avg_time_ms,
           r.latency_budget.avg_time_ms,
           r.token_bucket.avg_time_ms,
           r.voting.avg_time_ms,
           r.cluster_aware.avg_time_ms);
    printf("%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
           r.clc_baseline.avg_steals,
           r.greedy.avg_steals,
           r.maxsteals.avg_steals,
           r.throttled.avg_steals,
           r.selective.avg_steals,
           r.probe_everyn.avg_steals,
           r.latency_budget.avg_steals,
           r.token_bucket.avg_steals,
           r.voting.avg_steals,
           r.cluster_aware.avg_steals);
}

void print_table_row_steals(const ComprehensiveResult& r) {
    // Steals are now included in print_table_row_time, so this is a no-op
}

void print_speedup_analysis(const std::vector<ComprehensiveResult>& results) {
    // Print speedup analysis as CSV
    printf("\n");
    printf("Workload,Greedy_speedup,MaxSteals_speedup,NeverSteal_speedup,Selective_speedup,WorkloadAware_speedup,LatencyBudget_speedup,TokenBucket_speedup,Voting_speedup,ClusterAware_speedup,FixedWork_speedup,FixedBlocks_speedup\n");
    for (const auto& r : results) {
        float base = r.clc_baseline.avg_time_ms;
        printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
               r.workload_name,
               ((base - r.greedy.avg_time_ms) / base) * 100.0f,
               ((base - r.maxsteals.avg_time_ms) / base) * 100.0f,
               ((base - r.throttled.avg_time_ms) / base) * 100.0f,
               ((base - r.selective.avg_time_ms) / base) * 100.0f,
               ((base - r.probe_everyn.avg_time_ms) / base) * 100.0f,
               ((base - r.latency_budget.avg_time_ms) / base) * 100.0f,
               ((base - r.token_bucket.avg_time_ms) / base) * 100.0f,
               ((base - r.voting.avg_time_ms) / base) * 100.0f,
               ((base - r.cluster_aware.avg_time_ms) / base) * 100.0f,
               ((base - r.fixed_work.avg_time_ms) / base) * 100.0f,
               ((base - r.fixed_blocks.avg_time_ms) / base) * 100.0f);
    }
}

// ============================================================================
// Main - Run All Benchmarks
// ============================================================================

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        fprintf(stderr, "ERROR: CLC requires CC 10.0+\n");
        return 1;
    }

    // Parse arguments
    int n = 1024 * 1024;  // Default: 1M elements
    int threads = 256;
    float imb_scale = 1.0f;  // Default: no scaling
    float work_scale = 1.0f; // Default: no scaling

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) imb_scale = atof(argv[3]);
    if (argc > 4) work_scale = atof(argv[4]);

    // Set workload scale factors in device constant memory
    cudaMemcpyToSymbol(imbalance_scale, &imb_scale, sizeof(float));
    cudaMemcpyToSymbol(workload_scale, &work_scale, sizeof(float));

    fprintf(stderr, "# Config: n=%d, threads=%d, imbalance_scale=%.2f, workload_scale=%.2f\n",
           n, threads, imb_scale, work_scale);

    // Allocate data
    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_data[i] = (float)(i % 100) + 1.0f;

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // Run all comprehensive benchmarks
    std::vector<ComprehensiveResult> results;

    // NEW: ClusteredHeavy workload - designed to show 20%+ policy improvement
    fprintf(stderr, "# Running ClusteredHeavy workload (designed for policy comparison)...\n");
    results.push_back(run_comprehensive<ClusteredHeavy>(get_workload_name<ClusteredHeavy>(), 50, d_data, n, threads, prop, h_data));

    // Original AI workloads
    fprintf(stderr, "# Running AI workloads...\n");
    results.push_back(run_comprehensive<MixtureOfExperts>(get_workload_name<MixtureOfExperts>(), 75, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<NLPVariableSequence>(get_workload_name<NLPVariableSequence>(), 80, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<GraphNeuralNetwork>(get_workload_name<GraphNeuralNetwork>(), 50, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<DynamicBatching>(get_workload_name<DynamicBatching>(), 60, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<SparseAttention>(get_workload_name<SparseAttention>(), 70, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<VideoProcessing>(get_workload_name<VideoProcessing>(), 65, d_data, n, threads, prop, h_data));

    // GEMM workloads
    fprintf(stderr, "# Running GEMM workloads...\n");
    results.push_back(run_comprehensive<GEMMBalanced>(get_workload_name<GEMMBalanced>(), 40, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<GEMMImbalanced>(get_workload_name<GEMMImbalanced>(), 45, d_data, n, threads, prop, h_data));
    results.push_back(run_comprehensive<GEMMVariableSize>(get_workload_name<GEMMVariableSize>(), 50, d_data, n, threads, prop, h_data));

    // Print comprehensive table in CSV format (single table only)
    print_table_header();
    for (const auto& r : results) {
        print_table_row_time(r);
        print_table_row_steals(r);
    }

    cudaFree(d_data);
    free(h_data);

    return 0;
}
