// SPDX-License-Identifier: MIT
// Short real-UVM workload for native-vs-gpubpf no-prefetch comparison.
// Reusable as a controlled pressure tenant via --passes/--pause-ms.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = (call);                                            \
        if (status_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(status_));                           \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void read_one_word_per_region(const uint32_t *data,
                                         uint32_t *observed,
                                         uint64_t regions,
                                         uint64_t stride_words)
{
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < regions)
        observed[index] = data[index * stride_words];
}
static bool parse_non_negative_u64(const char *text, uint64_t *value,
                                   bool positive)
{
    if (text[0] == '-' || text[0] == '+')
        return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || !end || *end != '\0' || (positive && parsed == 0))
        return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

static bool parse_positive_u64(const char *text, uint64_t *value)
{
    return parse_non_negative_u64(text, value, true);
}

static double median_of(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1)
        return values[n / 2];
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static int write_result(const std::string &path, uint64_t bytes,
                        uint64_t region_bytes, uint64_t regions,
                        uint64_t passes, uint64_t pause_ms,
                        const std::vector<float> &pass_ms,
                        uint64_t mismatches, uint64_t first_mismatch)
{
    std::ofstream output(path);
    if (!output) {
        std::fprintf(stderr, "failed to open result path: %s\n", path.c_str());
        return 1;
    }
    double total_ms = 0.0;
    double max_ms = 0.0;
    for (float value : pass_ms) {
        total_ms += value;
        max_ms = std::max(max_ms, static_cast<double>(value));
    }
    output << "{\n"
           << "  \"bytes\": " << bytes << ",\n"
           << "  \"region_bytes\": " << region_bytes << ",\n"
           << "  \"regions\": " << regions << ",\n"
           << "  \"passes\": " << passes << ",\n"
           << "  \"pause_ms\": " << pause_ms << ",\n"
           << "  \"kernel_ms\": " << pass_ms.back() << ",\n"
           << "  \"kernel_ms_per_pass\": [";
    for (std::size_t i = 0; i < pass_ms.size(); ++i) {
        if (i != 0)
            output << ", ";
        output << pass_ms[i];
    }
    output << "],\n"
           << "  \"kernel_ms_total\": " << total_ms << ",\n"
           << "  \"kernel_ms_median\": " << median_of(pass_ms) << ",\n"
           << "  \"kernel_ms_max\": " << max_ms << ",\n"
           << "  \"completed_passes\": " << pass_ms.size() << ",\n"
           << "  \"mismatches\": " << mismatches << ",\n"
           << "  \"first_mismatch\": ";
    if (mismatches == 0)
        output << "null\n";
    else
        output << first_mismatch << "\n";
    output << "}\n";
    return output.good() ? 0 : 1;
}

int main(int argc, char **argv)
{
    uint64_t gib = 8;
    uint64_t region_kib = 64;
    uint64_t passes = 1;
    uint64_t pause_ms = 0;
    bool wait_for_monitor = false;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gib") && i + 1 < argc) {
            if (!parse_positive_u64(argv[++i], &gib)) {
                std::fprintf(stderr, "invalid --gib value\n");
                return 2;
            }
        }
        else if (!std::strcmp(argv[i], "--region-kib") && i + 1 < argc) {
            if (!parse_positive_u64(argv[++i], &region_kib)) {
                std::fprintf(stderr, "invalid --region-kib value\n");
                return 2;
            }
        }
        else if (!std::strcmp(argv[i], "--passes") && i + 1 < argc) {
            if (!parse_positive_u64(argv[++i], &passes)) {
                std::fprintf(stderr, "invalid --passes value\n");
                return 2;
            }
        }
        else if (!std::strcmp(argv[i], "--pause-ms") && i + 1 < argc) {
            if (!parse_non_negative_u64(argv[++i], &pause_ms, false)) {
                std::fprintf(stderr, "invalid --pause-ms value\n");
                return 2;
            }
        }
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        }
        else if (!std::strcmp(argv[i], "--wait-for-monitor")) {
            wait_for_monitor = true;
        }
        else {
            std::fprintf(stderr,
                         "usage: %s [--gib N] [--region-kib N] [--passes N] "
                         "[--pause-ms N] [--wait-for-monitor] --output PATH\n",
                         argv[0]);
            return 2;
        }
    }
    if (output_path.empty()) {
        std::fprintf(stderr, "--output is required\n");
        return 2;
    }

    if (gib > std::numeric_limits<uint64_t>::max() / (UINT64_C(1) << 30) ||
        region_kib > std::numeric_limits<uint64_t>::max() / 1024) {
        std::fprintf(stderr, "requested size overflows\n");
        return 2;
    }
    const uint64_t bytes = gib * (UINT64_C(1) << 30);
    const uint64_t region_bytes = region_kib * 1024;
    if (region_bytes < sizeof(uint32_t) || bytes % region_bytes != 0 ||
        region_bytes % sizeof(uint32_t) != 0) {
        std::fprintf(stderr, "region size must divide allocation size\n");
        return 2;
    }
    const uint64_t regions = bytes / region_bytes;
    const uint64_t stride_words = region_bytes / sizeof(uint32_t);
    if (regions > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        std::fprintf(stderr, "result allocation overflows\n");
        return 2;
    }
    if (passes > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "passes value overflows\n");
        return 2;
    }
    if (pause_ms > std::numeric_limits<long long>::max()) {
        std::fprintf(stderr, "pause-ms value overflows\n");
        return 2;
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    uint32_t *data = nullptr;
    uint32_t *observed_device = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, static_cast<size_t>(bytes),
                                 cudaMemAttachGlobal));
    CUDA_CHECK(cudaMalloc(&observed_device,
                          static_cast<size_t>(regions * sizeof(uint32_t))));

    for (uint64_t i = 0; i < regions; ++i)
        data[i * stride_words] = static_cast<uint32_t>((i % 251) + 1);

    std::printf("READY pid=%ld gib=%" PRIu64 " regions=%" PRIu64
                " passes=%" PRIu64 " pause_ms=%" PRIu64 "\n",
                static_cast<long>(getpid()), gib, regions, passes, pause_ms);
    std::fflush(stdout);

    if (wait_for_monitor) {
        std::printf("MONITOR_PID: %ld\n", static_cast<long>(getpid()));
        std::printf("Press Enter after the UVM monitor is ready...\n");
        std::fflush(stdout);
        if (std::getchar() == EOF) {
            std::fprintf(stderr, "monitor wait ended without input\n");
            return 1;
        }
    }

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    const unsigned int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((regions + threads - 1) / threads);
    std::vector<float> pass_ms;
    pass_ms.reserve(static_cast<std::size_t>(passes));
    for (uint64_t pass = 0; pass < passes; ++pass) {
        CUDA_CHECK(cudaEventRecord(begin));
        read_one_word_per_region<<<blocks, threads>>>(data, observed_device,
                                                      regions, stride_words);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaDeviceSynchronize());
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, begin, end));
        pass_ms.push_back(kernel_ms);
        if (pass + 1 < passes && pause_ms > 0)
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<long long>(pause_ms)));
    }

    std::vector<uint32_t> observed(static_cast<size_t>(regions));
    CUDA_CHECK(cudaMemcpy(observed.data(), observed_device,
                          static_cast<size_t>(regions * sizeof(uint32_t)),
                          cudaMemcpyDeviceToHost));

    uint64_t mismatches = 0;
    uint64_t first_mismatch = 0;
    for (uint64_t i = 0; i < regions; ++i) {
        const uint32_t expected = static_cast<uint32_t>((i % 251) + 1);
        if (observed[static_cast<size_t>(i)] != expected) {
            if (mismatches == 0)
                first_mismatch = i;
            ++mismatches;
        }
    }

    const int result_status = write_result(output_path, bytes, region_bytes,
                                           regions, passes, pause_ms, pass_ms,
                                           mismatches, first_mismatch);
    std::printf("RESULT kernel_ms=%.6f regions=%" PRIu64
                " mismatches=%" PRIu64 "\n",
                pass_ms.back(), regions, mismatches);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    cudaFree(observed_device);
    cudaFree(data);
    if (result_status != 0 || mismatches != 0)
        return 1;
    return 0;
}
