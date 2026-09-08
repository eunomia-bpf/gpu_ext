/**
 * hotspot.cuh - Hotspot thermal simulation kernel (Stencil pattern)
 *
 * SOURCE: Rodinia Benchmark Suite
 * Original file: /memory/uvm_bench/UVM_benchmark/UVM_benchmarks/rodinia/hotspot/hotspot.cu
 *
 * Access pattern: Sequential (5-point stencil - N/S/E/W neighbors)
 * Representative of: CFD, thermal simulation, image processing
 *
 * This is a seq_stream equivalent with real computation.
 */

#ifndef HOTSPOT_CUH
#define HOTSPOT_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ============================================================================
// Original Rodinia Hotspot Parameters
// ============================================================================

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD (3.0e6)
/* required precision in degrees */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor */
#define FACTOR_CHIP 0.5

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

// ============================================================================
// Original Rodinia Hotspot Kernel (UNCHANGED from source)
// ============================================================================

__global__ void calculate_temp(int iteration,   // number of iteration
                               float *power,    // power input
                               float *temp_src, // temperature input/output
                               float *temp_dst, // temperature input/output
                               int grid_cols,   // Col of grid
                               int grid_rows,   // Row of grid
                               int border_cols, // border offset
                               int border_rows, // border offset
                               float Cap,       // Capacitance
                               float Rx, float Ry, float Rz, float step,
                               float time_elapsed) {

  __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_t[BLOCK_SIZE]
                         [BLOCK_SIZE]; // saving temparary temperature result

  float amb_temp = 80.0;
  float step_div_Cap;
  float Rx_1, Ry_1, Rz_1;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  step_div_Cap = step / Cap;

  Rx_1 = 1 / Rx;
  Ry_1 = 1 / Ry;
  Rz_1 = 1 / Rz;

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  int small_block_rows = BLOCK_SIZE - iteration * 2; // EXPAND_RATE
  int small_block_cols = BLOCK_SIZE - iteration * 2; // EXPAND_RATE

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkY = small_block_rows * by - border_rows;
  int blkX = small_block_cols * bx - border_cols;
  int blkYmax = blkY + BLOCK_SIZE - 1;
  int blkXmax = blkX + BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int yidx = blkY + ty;
  int xidx = blkX + tx;

  // load data if it is within the valid input range
  int loadYidx = yidx, loadXidx = xidx;
  // Use size_t to avoid integer overflow for large grids (grid_size > 46340)
  size_t index = (size_t)grid_cols * loadYidx + loadXidx;

  if (IN_RANGE(loadYidx, 0, grid_rows - 1) &&
      IN_RANGE(loadXidx, 0, grid_cols - 1)) {
    temp_on_cuda[ty][tx] = temp_src[index]; // Load the temperature data from
                                            // global memory to shared memory
    power_on_cuda[ty][tx] =
        power[index]; // Load the power data from global memory to shared memory
  }
  __syncthreads();

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validYmin = (blkY < 0) ? -blkY : 0;
  int validYmax = (blkYmax > grid_rows - 1)
                      ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1)
                      : BLOCK_SIZE - 1;
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > grid_cols - 1)
                      ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1)
                      : BLOCK_SIZE - 1;

  int N = ty - 1;
  int S = ty + 1;
  int W = tx - 1;
  int E = tx + 1;

  N = (N < validYmin) ? validYmin : N;
  S = (S > validYmax) ? validYmax : S;
  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
        IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
        IN_RANGE(tx, validXmin, validXmax) &&
        IN_RANGE(ty, validYmin, validYmax)) {
      computed = true;
      temp_t[ty][tx] =
          temp_on_cuda[ty][tx] +
          step_div_Cap * (power_on_cuda[ty][tx] +
                          (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] -
                           2.0 * temp_on_cuda[ty][tx]) *
                              Ry_1 +
                          (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] -
                           2.0 * temp_on_cuda[ty][tx]) *
                              Rx_1 +
                          (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
    }
    __syncthreads();
    if (i == iteration - 1)
      break;
    if (computed) // Assign the computation range
      temp_on_cuda[ty][tx] = temp_t[ty][tx];
    __syncthreads();
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    temp_dst[index] = temp_t[ty][tx];
  }
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

struct HotspotResult {
    size_t bytes_accessed;
    float median_ms;
    float min_ms;
    float max_ms;
};

inline void run_hotspot(size_t total_working_set, const std::string& mode,
                        size_t stride_bytes, int iterations,
                        std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;  // Hotspot uses fixed stencil pattern

    // =========================================================================
    // 动态 grid 大小 + 固定迭代次数
    // - 根据 size_factor 调整 grid 大小，使数据量匹配 total_working_set
    // - size_factor=1.0 → grid ~57K×57K → ~32GB (3 arrays)
    // - 固定迭代次数产生 temporal locality
    // =========================================================================

    // Hotspot 需要 3 个 array: temp[0], temp[1], power
    // total_working_set = 3 * grid_size^2 * sizeof(float)
    // grid_size = sqrt(total_working_set / (3 * sizeof(float)))
    size_t elements_needed = total_working_set / (3 * sizeof(float));
    int grid_size = (int)sqrt((double)elements_needed);

    // 对齐到 BLOCK_SIZE 的倍数，确保 kernel 正确工作
    grid_size = (grid_size / BLOCK_SIZE) * BLOCK_SIZE;
    if (grid_size < 1024) grid_size = 1024;  // 最小 1024×1024

    int grid_rows = grid_size;
    int grid_cols = grid_size;
    size_t size = (size_t)grid_rows * grid_cols;
    size_t allocated_bytes = 3 * size * sizeof(float);

    // 固定迭代次数，产生 temporal locality
    int total_iterations = 10;

    // Chip parameters
    float t_chip = 0.0005;
    float chip_height = 0.016;
    float chip_width = 0.016;

    float *MatrixTemp[2], *MatrixPower;

    // Allocate UVM memory (固定大小)
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&MatrixTemp[0], sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&MatrixTemp[1], sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&MatrixPower, sizeof(float) * size));
        CUDA_CHECK(cudaMemset(MatrixTemp[0], 0, sizeof(float) * size));
        CUDA_CHECK(cudaMemset(MatrixTemp[1], 0, sizeof(float) * size));
        CUDA_CHECK(cudaMemset(MatrixPower, 0, sizeof(float) * size));
    } else {
        CUDA_CHECK(cudaMallocManaged(&MatrixTemp[0], sizeof(float) * size));
        CUDA_CHECK(cudaMallocManaged(&MatrixTemp[1], sizeof(float) * size));
        CUDA_CHECK(cudaMallocManaged(&MatrixPower, sizeof(float) * size));

        // Use cudaMemset for large allocations to avoid CPU page faults
        // The kernel will work correctly with zero-initialized data
        CUDA_CHECK(cudaMemset(MatrixTemp[0], 0, sizeof(float) * size));
        CUDA_CHECK(cudaMemset(MatrixTemp[1], 0, sizeof(float) * size));
        CUDA_CHECK(cudaMemset(MatrixPower, 0, sizeof(float) * size));
    }

    // Apply UVM hints if needed
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        apply_uvm_hints(MatrixTemp[0], sizeof(float) * size, mode, dev);
        apply_uvm_hints(MatrixTemp[1], sizeof(float) * size, mode, dev);
        apply_uvm_hints(MatrixPower, sizeof(float) * size, mode, dev);
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    fprintf(stderr, "Hotspot config: grid=%dx%d, iterations=%d\n",
            grid_rows, grid_cols, total_iterations);
    fprintf(stderr, "  Allocated: %.1f MB, Total access: %.1f MB\n",
            allocated_bytes / (1024.0 * 1024.0),
            allocated_bytes * total_iterations / (1024.0 * 1024.0));

    // Pyramid parameters
    int pyramid_height = 1;

    #define EXPAND_RATE 2
    int borderCols = (pyramid_height) * EXPAND_RATE / 2;
    int borderRows = (pyramid_height) * EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
    int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);

    float grid_height = chip_height / grid_rows;
    float grid_width = chip_width / grid_cols;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float time_elapsed = 0.001;

    // Warmup
    for (int w = 0; w < 2; w++) {
        int src = 1, dst = 0;
        for (int t = 0; t < total_iterations; t += pyramid_height) {
            int temp = src; src = dst; dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, total_iterations - t), MatrixPower, MatrixTemp[src],
                MatrixTemp[dst], grid_cols, grid_rows, borderCols, borderRows, Cap, Rx, Ry, Rz,
                step, time_elapsed);
        }
        cudaDeviceSynchronize();
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        cudaEventRecord(start);

        int src = 1, dst = 0;
        for (int t = 0; t < total_iterations; t += pyramid_height) {
            int temp = src; src = dst; dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, total_iterations - t), MatrixPower, MatrixTemp[src],
                MatrixTemp[dst], grid_cols, grid_rows, borderCols, borderRows, Cap, Rx, Ry, Rz,
                step, time_elapsed);
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
    result.bytes_accessed = allocated_bytes * total_iterations;

    // Cleanup
    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);

    #undef EXPAND_RATE
}

// ============================================================================
// Multi-Grid Oversub Variant - Tests eviction policy under competition
// ============================================================================
// Design: Allocate G grids where G * grid_size > HBM capacity
// In one kernel, different blocks process different grids simultaneously
// This creates multiple "hot regions" competing for HBM

__global__ void calculate_temp_multigrid(int iteration,
                                         float **power_grids,
                                         float **temp_src_grids,
                                         float **temp_dst_grids,
                                         int num_grids,
                                         int grid_cols,
                                         int grid_rows,
                                         int border_cols,
                                         int border_rows,
                                         float Cap,
                                         float Rx, float Ry, float Rz, float step,
                                         float time_elapsed,
                                         int blocks_per_grid_x,
                                         int blocks_per_grid_y) {
    // Determine which grid this block belongs to
    int grid_id = blockIdx.z;
    if (grid_id >= num_grids) return;

    // Get pointers for this grid
    float *power = power_grids[grid_id];
    float *temp_src = temp_src_grids[grid_id];
    float *temp_dst = temp_dst_grids[grid_id];

    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];

    float amb_temp = 80.0;
    float step_div_Cap = step / Cap;
    float Rx_1 = 1 / Rx;
    float Ry_1 = 1 / Ry;
    float Rz_1 = 1 / Rz;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    #define EXPAND_RATE_MG 2
    int small_block_rows = BLOCK_SIZE - iteration * 2;
    int small_block_cols = BLOCK_SIZE - iteration * 2;

    int blkY = small_block_rows * by - border_rows;
    int blkX = small_block_cols * bx - border_cols;
    int blkYmax = blkY + BLOCK_SIZE - 1;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    int yidx = blkY + ty;
    int xidx = blkX + tx;

    int loadYidx = yidx, loadXidx = xidx;
    // Use size_t to avoid integer overflow for large grids
    size_t index = (size_t)grid_cols * loadYidx + loadXidx;

    if (IN_RANGE(loadYidx, 0, grid_rows - 1) &&
        IN_RANGE(loadXidx, 0, grid_cols - 1)) {
        temp_on_cuda[ty][tx] = temp_src[index];
        power_on_cuda[ty][tx] = power[index];
    }
    __syncthreads();

    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows - 1)
                        ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1)
                        : BLOCK_SIZE - 1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols - 1)
                        ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1)
                        : BLOCK_SIZE - 1;

    int N = ty - 1;
    int S = ty + 1;
    int W = tx - 1;
    int E = tx + 1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(tx, validXmin, validXmax) &&
            IN_RANGE(ty, validYmin, validYmax)) {
            computed = true;
            temp_t[ty][tx] =
                temp_on_cuda[ty][tx] +
                step_div_Cap * (power_on_cuda[ty][tx] +
                                (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] -
                                 2.0 * temp_on_cuda[ty][tx]) * Ry_1 +
                                (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] -
                                 2.0 * temp_on_cuda[ty][tx]) * Rx_1 +
                                (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
        }
        __syncthreads();
        if (i == iteration - 1) break;
        if (computed) temp_on_cuda[ty][tx] = temp_t[ty][tx];
        __syncthreads();
    }

    if (computed) {
        temp_dst[index] = temp_t[ty][tx];
    }
    #undef EXPAND_RATE_MG
}

inline void run_hotspot_multigrid(size_t total_working_set, const std::string& mode,
                                   size_t stride_bytes, int iterations,
                                   std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;

    // =========================================================================
    // Multi-Grid Oversub Design:
    // - Each grid is 2048x2048 (~50MB for 3 arrays) - smaller to fit more grids
    // - Allocate G grids where G * grid_size > total_working_set
    // - All grids processed in ONE kernel launch (blockIdx.z = grid_id)
    // - Creates true multi-region competition for HBM
    // =========================================================================

    const int GRID_SIZE = 2048;  // Smaller grid for more parallelism
    int grid_rows = GRID_SIZE;
    int grid_cols = GRID_SIZE;
    size_t single_grid_elements = (size_t)grid_rows * grid_cols;
    size_t single_grid_bytes = 3 * single_grid_elements * sizeof(float);  // ~50MB per grid

    // Calculate number of grids to match total_working_set
    int num_grids = (total_working_set + single_grid_bytes - 1) / single_grid_bytes;
    if (num_grids < 2) num_grids = 2;

    size_t total_allocated = (size_t)num_grids * single_grid_bytes;

    fprintf(stderr, "Hotspot MultiGrid config: grid=%dx%d, num_grids=%d\n",
            grid_rows, grid_cols, num_grids);
    fprintf(stderr, "  Single grid: %.1f MB, Total allocation: %.1f MB\n",
            single_grid_bytes / (1024.0 * 1024.0),
            total_allocated / (1024.0 * 1024.0));

    // Chip parameters
    float t_chip = 0.0005;
    float chip_height = 0.016;
    float chip_width = 0.016;

    // Allocate arrays of pointers for multi-grid
    float **h_MatrixTemp0 = new float*[num_grids];
    float **h_MatrixTemp1 = new float*[num_grids];
    float **h_MatrixPower = new float*[num_grids];

    // Device pointer arrays
    float **d_MatrixTemp0, **d_MatrixTemp1, **d_MatrixPower;

    // Allocate each grid
    for (int g = 0; g < num_grids; g++) {
        if (mode == "device") {
            CUDA_CHECK(cudaMalloc(&h_MatrixTemp0[g], sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMalloc(&h_MatrixTemp1[g], sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMalloc(&h_MatrixPower[g], sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMemset(h_MatrixTemp0[g], 0, sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMemset(h_MatrixTemp1[g], 0, sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMemset(h_MatrixPower[g], 0, sizeof(float) * single_grid_elements));
        } else {
            CUDA_CHECK(cudaMallocManaged(&h_MatrixTemp0[g], sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMallocManaged(&h_MatrixTemp1[g], sizeof(float) * single_grid_elements));
            CUDA_CHECK(cudaMallocManaged(&h_MatrixPower[g], sizeof(float) * single_grid_elements));

            // Initialize on CPU (ensures pages start on CPU for UVM)
            for (size_t i = 0; i < single_grid_elements; i++) {
                h_MatrixTemp0[g][i] = 80.0f + (float)((i + g * 1000) % 100) * 0.1f;
                h_MatrixTemp1[g][i] = h_MatrixTemp0[g][i];
                h_MatrixPower[g][i] = (float)((i + g * 500) % 50) * 0.01f;
            }
        }
    }

    // Allocate device pointer arrays
    CUDA_CHECK(cudaMalloc(&d_MatrixTemp0, sizeof(float*) * num_grids));
    CUDA_CHECK(cudaMalloc(&d_MatrixTemp1, sizeof(float*) * num_grids));
    CUDA_CHECK(cudaMalloc(&d_MatrixPower, sizeof(float*) * num_grids));

    // Copy pointer arrays to device
    CUDA_CHECK(cudaMemcpy(d_MatrixTemp0, h_MatrixTemp0, sizeof(float*) * num_grids, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MatrixTemp1, h_MatrixTemp1, sizeof(float*) * num_grids, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MatrixPower, h_MatrixPower, sizeof(float*) * num_grids, cudaMemcpyHostToDevice));

    // Apply UVM hints if needed
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        for (int g = 0; g < num_grids; g++) {
            apply_uvm_hints(h_MatrixTemp0[g], sizeof(float) * single_grid_elements, mode, dev);
            apply_uvm_hints(h_MatrixTemp1[g], sizeof(float) * single_grid_elements, mode, dev);
            apply_uvm_hints(h_MatrixPower[g], sizeof(float) * single_grid_elements, mode, dev);
        }
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Pyramid parameters
    int pyramid_height = 1;
    #define EXPAND_RATE 2
    int borderCols = (pyramid_height) * EXPAND_RATE / 2;
    int borderRows = (pyramid_height) * EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
    int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

    // 3D grid: (blockCols, blockRows, num_grids)
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows, num_grids);

    float grid_height = chip_height / grid_rows;
    float grid_width = chip_width / grid_cols;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float time_elapsed = 0.001;

    // Number of timesteps per measured iteration
    int total_iterations = 10;  // Fixed iterations per measurement

    // Warmup
    for (int w = 0; w < 2; w++) {
        for (int t = 0; t < total_iterations; t += pyramid_height) {
            calculate_temp_multigrid<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, total_iterations - t),
                d_MatrixPower, d_MatrixTemp0, d_MatrixTemp1,
                num_grids, grid_cols, grid_rows, borderCols, borderRows,
                Cap, Rx, Ry, Rz, step, time_elapsed, blockCols, blockRows);
            // Swap pointers for next iteration
            float **temp = d_MatrixTemp0;
            d_MatrixTemp0 = d_MatrixTemp1;
            d_MatrixTemp1 = temp;
        }
        cudaDeviceSynchronize();
    }

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < iterations; iter++) {
        cudaEventRecord(start);

        for (int t = 0; t < total_iterations; t += pyramid_height) {
            calculate_temp_multigrid<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, total_iterations - t),
                d_MatrixPower, d_MatrixTemp0, d_MatrixTemp1,
                num_grids, grid_cols, grid_rows, borderCols, borderRows,
                Cap, Rx, Ry, Rz, step, time_elapsed, blockCols, blockRows);
            float **temp = d_MatrixTemp0;
            d_MatrixTemp0 = d_MatrixTemp1;
            d_MatrixTemp1 = temp;
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

    // Bytes accessed: all grids * 3 arrays * iterations
    result.bytes_accessed = (size_t)num_grids * 3 * single_grid_elements * sizeof(float) * total_iterations;

    // Cleanup
    for (int g = 0; g < num_grids; g++) {
        cudaFree(h_MatrixTemp0[g]);
        cudaFree(h_MatrixTemp1[g]);
        cudaFree(h_MatrixPower[g]);
    }
    cudaFree(d_MatrixTemp0);
    cudaFree(d_MatrixTemp1);
    cudaFree(d_MatrixPower);
    delete[] h_MatrixTemp0;
    delete[] h_MatrixTemp1;
    delete[] h_MatrixPower;

    #undef EXPAND_RATE
}

#endif // HOTSPOT_CUH
