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
  int index = grid_cols * loadYidx + loadXidx;

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
    // 固定网格大小 + 迭代次数控制总工作量
    // - 4096×4096 网格 ≈ 200MB (3 arrays)，符合真实 CFD/热传导模拟
    // - 通过迭代次数控制总访问量，产生 temporal locality
    // =========================================================================

    // 固定合理的网格大小（类似真实 CFD 模拟）
    const int GRID_SIZE = 4096;  // 4096×4096，单次 kernel 访问 ~67MB per array
    int grid_rows = GRID_SIZE;
    int grid_cols = GRID_SIZE;
    size_t size = (size_t)grid_rows * grid_cols;
    size_t single_pass_bytes = 3 * size * sizeof(float);  // ~200MB

    // 根据 total_working_set 计算迭代次数
    int total_iterations = total_working_set / single_pass_bytes;
    if (total_iterations < 10) total_iterations = 10;
    if (total_iterations > 10000) total_iterations = 10000;  // 合理上限

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

        // Initialize data on CPU
        for (size_t i = 0; i < size; i++) {
            MatrixTemp[0][i] = 80.0f + (float)(i % 100) * 0.1f;
            MatrixTemp[1][i] = MatrixTemp[0][i];
            MatrixPower[i] = (float)(i % 50) * 0.01f;
        }
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
    fprintf(stderr, "  Single pass: %.1f MB, Total access: %.1f MB\n",
            single_pass_bytes / (1024.0 * 1024.0),
            single_pass_bytes * total_iterations / (1024.0 * 1024.0));

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
    result.bytes_accessed = 3 * size * sizeof(float) * total_iterations;

    // Cleanup
    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);

    #undef EXPAND_RATE
}

#endif // HOTSPOT_CUH
