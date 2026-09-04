#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef TRAMPOLINE_SCALING_MATRIX_HEADER
#define TRAMPOLINE_SCALING_MATRIX_HEADER "matrix.h"
#endif
#include TRAMPOLINE_SCALING_MATRIX_HEADER

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_check_error = (call);                            \
        if (cuda_check_error != cudaSuccess) {                                  \
            std::fprintf(stderr, "%s: %s\n", #call,                           \
                         cudaGetErrorString(cuda_check_error));                  \
            return 2;                                                           \
        }                                                                       \
    } while (0)

struct ScaleCell {
    unsigned id;
    unsigned blocks;
    unsigned threads_per_block;
    unsigned active_threads;
    unsigned counter_key;
};

#define CELL_ROW(id, blocks, threads, active, key) \
    {id##U, blocks##U, threads##U, active##U, key##U},
static constexpr ScaleCell kCells[] = {SCALE_CELL_LIST(CELL_ROW)};
#undef CELL_ROW

static_assert(sizeof(kCells) / sizeof(kCells[0]) == SCALE_CELL_COUNT,
              "matrix cell count drifted");

static constexpr uint64_t kCanary = UINT64_C(0xa5a5a5a5a5a5a5a5);
static constexpr uint64_t kMarker = UINT64_C(0x5a17000000000000);

extern "C" __device__ __noinline__ __attribute__((used)) void
__bpftime_cuda__kernel_trace(void)
{
    asm volatile("" ::: "memory");
}

extern "C" __global__ void trampoline_marker_kernel(uint64_t *output)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 32)
        output[index] = kMarker ^ index;
}

__host__ __device__ static uint64_t expected_value(uint64_t index,
                                                   uint64_t seed,
                                                   unsigned hook_repeats)
{
    uint64_t value = (seed ^ (index * UINT64_C(0x9e3779b97f4a7c15))) +
                     UINT64_C(0xd1b54a32d192ed03);
    for (unsigned repeat = 0; repeat < hook_repeats; ++repeat)
        value = (value ^ (value >> 29)) * UINT64_C(0x94d049bb133111eb) + repeat;
    return value ^ (index << 17);
}

extern "C" __global__ void trampoline_scale_kernel(uint64_t *output,
                                                    uint64_t active_threads,
                                                    uint64_t seed,
                                                    unsigned hook_repeats)
{
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= active_threads)
        return;

    uint64_t value = (seed ^ (index * UINT64_C(0x9e3779b97f4a7c15))) +
                     UINT64_C(0xd1b54a32d192ed03);
    #pragma unroll 1
    for (unsigned repeat = 0; repeat < hook_repeats; ++repeat) {
        value = (value ^ (value >> 29)) * UINT64_C(0x94d049bb133111eb) + repeat;
        /* Keep the function name on the call line for bpftime's native stub pass. */
        asm volatile("call.uni __bpftime_cuda__kernel_trace, ();" ::: "memory");
    }
    output[index] = value ^ (index << 17);
}

static bool parse_unsigned(const char *text, unsigned minimum,
                           unsigned maximum, unsigned *value)
{
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno || !end || *end || parsed < minimum || parsed > maximum)
        return false;
    *value = (unsigned)parsed;
    return true;
}

static const ScaleCell *find_cell(unsigned id)
{
    for (const auto &cell : kCells) {
        if (cell.id == id)
            return &cell;
    }
    return nullptr;
}

static bool parse_cells(const char *text, std::vector<const ScaleCell *> *cells)
{
    char *copy = ::strdup(text);
    if (!copy)
        return false;
    bool seen[SCALE_CELL_COUNT] = {};
    char *save = nullptr;
    for (char *token = ::strtok_r(copy, ",", &save); token;
         token = ::strtok_r(nullptr, ",", &save)) {
        unsigned id = 0;
        if (!parse_unsigned(token, 0, SCALE_CELL_COUNT - 1, &id) || seen[id]) {
            std::free(copy);
            return false;
        }
        const ScaleCell *cell = find_cell(id);
        if (!cell) {
            std::free(copy);
            return false;
        }
        seen[id] = true;
        cells->push_back(cell);
    }
    std::free(copy);
    return !cells->empty();
}

static void usage(const char *program)
{
    std::fprintf(stderr,
                 "usage: %s --cells IDS --warmup N --launches N "
                 "--hook-repeats N --run-id N\n",
                 program);
}

int main(int argc, char **argv)
{
    std::vector<const ScaleCell *> cells;
    unsigned warmup = 0, launches = 0, hook_repeats = 0, run_id = 0;
    bool have_cells = false, have_warmup = false, have_launches = false;
    bool have_repeats = false, have_run_id = false;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            usage(argv[0]);
            return 1;
        }
        if (std::strcmp(argv[index], "--cells") == 0) {
            have_cells = parse_cells(argv[index + 1], &cells);
        } else if (std::strcmp(argv[index], "--warmup") == 0) {
            have_warmup = parse_unsigned(argv[index + 1], 0, 1000, &warmup);
        } else if (std::strcmp(argv[index], "--launches") == 0) {
            have_launches = parse_unsigned(argv[index + 1], 1, 1000000, &launches);
        } else if (std::strcmp(argv[index], "--hook-repeats") == 0) {
            have_repeats = parse_unsigned(argv[index + 1], 1, 1000,
                                          &hook_repeats);
        } else if (std::strcmp(argv[index], "--run-id") == 0) {
            have_run_id = parse_unsigned(argv[index + 1], 0, 1000000, &run_id);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (!have_cells || !have_warmup || !have_launches || !have_repeats ||
        !have_run_id) {
        usage(argv[0]);
        return 1;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (std::strstr(properties.name, "RTX 5090") == nullptr ||
        properties.major != 12 || properties.minor != 0 ||
        properties.warpSize != 32 ||
        properties.maxThreadsPerBlock < (int)SCALE_MAX_THREADS_PER_BLOCK ||
        properties.maxGridSize[0] < 4096) {
        std::fprintf(stderr, "device does not satisfy frozen RTX 5090 matrix\n");
        return 3;
    }
    std::printf("{\"event\":\"device\",\"name\":\"%s\","
                "\"major\":%d,\"minor\":%d,\"warp_size\":%d,"
                "\"max_threads_per_block\":%d,\"max_grid_x\":%d}\n",
                properties.name, properties.major, properties.minor,
                properties.warpSize, properties.maxThreadsPerBlock,
                properties.maxGridSize[0]);

    uint64_t *device_output = nullptr;
    uint64_t *device_marker = nullptr;
    CUDA_CHECK(cudaMalloc(&device_output, SCALE_MAX_THREADS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&device_marker, 32 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(device_marker, 0, 32 * sizeof(uint64_t)));
    trampoline_marker_kernel<<<1, 32>>>(device_marker);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t marker[32] = {};
    CUDA_CHECK(cudaMemcpy(marker, device_marker, sizeof(marker),
                          cudaMemcpyDeviceToHost));
    for (unsigned index = 0; index < 32; ++index) {
        if (marker[index] != (kMarker ^ index)) {
            std::fprintf(stderr, "marker mismatch at %u\n", index);
            return 4;
        }
    }
    std::printf("{\"event\":\"marker\",\"threads\":32,\"mismatches\":0}\n");

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<uint64_t> host_output(SCALE_MAX_THREADS);

    for (const ScaleCell *cell : cells) {
        const uint64_t launched_threads =
            (uint64_t)cell->blocks * cell->threads_per_block;
        if (cell->active_threads > launched_threads ||
            launched_threads > SCALE_MAX_THREADS ||
            cell->threads_per_block == 0 ||
            cell->threads_per_block > SCALE_MAX_THREADS_PER_BLOCK ||
            cell->threads_per_block % properties.warpSize != 0) {
            std::fprintf(stderr, "cell %u violates launch bounds\n", cell->id);
            return 5;
        }
        CUDA_CHECK(cudaMemset(device_output, 0xa5,
                              SCALE_MAX_THREADS * sizeof(uint64_t)));
        const uint64_t seed = UINT64_C(0x1797000000000000) ^
                              ((uint64_t)run_id << 16) ^ cell->id;
        for (unsigned launch = 0; launch < warmup; ++launch) {
            trampoline_scale_kernel<<<cell->blocks, cell->threads_per_block>>>(
                device_output, cell->active_threads, seed, hook_repeats);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (unsigned launch = 0; launch < launches; ++launch) {
            trampoline_scale_kernel<<<cell->blocks, cell->threads_per_block>>>(
                device_output, cell->active_threads, seed, hook_repeats);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                              SCALE_MAX_THREADS * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));

        uint64_t mismatches = 0;
        for (uint64_t index = 0; index < SCALE_MAX_THREADS; ++index) {
            const uint64_t expected = index < cell->active_threads
                                          ? expected_value(index, seed, hook_repeats)
                                          : kCanary;
            if (host_output[index] != expected) {
                if (mismatches < 4) {
                    std::fprintf(stderr,
                                 "cell %u mismatch at %" PRIu64
                                 ": got=%" PRIu64 " expected=%" PRIu64 "\n",
                                 cell->id, index, host_output[index], expected);
                }
                ++mismatches;
            }
        }
        std::printf(
            "{\"event\":\"measurement\",\"cell\":%u,\"blocks\":%u,"
            "\"threads_per_block\":%u,\"launched_threads\":%" PRIu64 ","
            "\"active_threads\":%u,\"active_warps\":%u,\"counter_key\":%u,"
            "\"warmup\":%u,\"launches\":%u,\"hook_repeats\":%u,"
            "\"elapsed_ms\":%.9g,\"checked_values\":%u,"
            "\"mismatches\":%" PRIu64 "}\n",
            cell->id, cell->blocks, cell->threads_per_block, launched_threads,
            cell->active_threads, cell->active_threads / 32, cell->counter_key,
            warmup, launches, hook_repeats, elapsed_ms, SCALE_MAX_THREADS,
            mismatches);
        if (mismatches != 0)
            return 6;
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(device_marker));
    CUDA_CHECK(cudaFree(device_output));
    std::printf("{\"event\":\"complete\",\"cells\":%zu,\"run_id\":%u}\n",
                cells.size(), run_id);
    return 0;
}
