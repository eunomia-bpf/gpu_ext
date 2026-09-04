// SPDX-License-Identifier: MIT
// Frozen alternating dense/sparse managed-memory workload for stale-state study.

#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <signal.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kAllocationBytes = UINT64_C(40) << 30;
constexpr uint64_t kRegionBytes = UINT64_C(64) << 10;
constexpr uint64_t kRegionWords = kRegionBytes / sizeof(uint32_t);
constexpr uint64_t kRegions = kAllocationBytes / kRegionBytes;
constexpr uint64_t kSparseStride = 32;
constexpr uint64_t kSparseRegions =
    (kRegions + kSparseStride - 1) / kSparseStride;
constexpr uint64_t kDenseLaunchRegions = 4096;
constexpr uint64_t kBootstrapNs = UINT64_C(1200) * 1000 * 1000;
constexpr uint64_t kPhaseNs = UINT64_C(2000) * 1000 * 1000;
constexpr uint64_t kMaximumBoundaryOverrunNs = UINT64_C(500) * 1000 * 1000;
constexpr unsigned int kMeasuredPhases = 6;
constexpr const char *kProtocol = "stale-cross-layer-575-v1";
constexpr const char *kTimeline = "alternating-dense-sparse-40g-v1";

static_assert(kAllocationBytes % kRegionBytes == 0,
              "region size must divide allocation");
static_assert(kRegions == 655360, "frozen region count changed");
static_assert(kSparseRegions == 20480, "frozen sparse region count changed");

struct PhaseResult {
    unsigned int measured_index;
    uint64_t sequence;
    const char *phase;
    uint64_t scheduled_offset_ns;
    uint64_t start_mono_ns;
    uint64_t end_mono_ns;
    uint64_t iterations;
    uint64_t checked_values;
    uint64_t mismatches;
    uint64_t first_mismatch;
    double kernel_ms;
};

__global__ void read_dense_regions(const uint32_t *data,
                                   uint32_t *observed,
                                   uint64_t total_regions,
                                   uint64_t first_region,
                                   uint64_t count)
{
    uint64_t item = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (item < count) {
        uint64_t region = (first_region + item) % total_regions;
        observed[item] = data[region * kRegionWords];
    }
}

__global__ void read_sparse_regions(const uint32_t *data,
                                    uint32_t *observed,
                                    uint64_t total_regions,
                                    uint64_t count)
{
    uint64_t item = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (item < count) {
        uint64_t region = item * kSparseStride;
        if (region < total_regions)
            observed[item] = data[region * kRegionWords];
    }
}

uint32_t expected_value(uint64_t region)
{
    return static_cast<uint32_t>((region % 251) + 1);
}

uint64_t monotonic_ns()
{
    struct timespec value = {};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        std::fprintf(stderr, "clock_gettime failed: %s\n", std::strerror(errno));
        std::exit(1);
    }
    return static_cast<uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
           static_cast<uint64_t>(value.tv_nsec);
}

bool wait_until(uint64_t deadline_ns)
{
    struct timespec deadline = {};
    deadline.tv_sec =
        static_cast<time_t>(deadline_ns / UINT64_C(1000000000));
    deadline.tv_nsec =
        static_cast<long>(deadline_ns % UINT64_C(1000000000));
    int status;
    do {
        status = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline,
                                 nullptr);
    } while (status == EINTR);
    if (status != 0) {
        std::fprintf(stderr, "clock_nanosleep failed: %s\n",
                     std::strerror(status));
        return false;
    }
    return true;
}

bool emit_line(std::ofstream &truth, FILE *relay, const std::string &line)
{
    truth << line << '\n';
    truth.flush();
    if (!truth.good()) {
        std::fprintf(stderr, "failed writing phase-truth file\n");
        return false;
    }
    if (std::fprintf(relay, "%s\n", line.c_str()) < 0 ||
        std::fflush(relay) != 0) {
        std::fprintf(stderr, "failed writing phase truth to snapshot relay\n");
        return false;
    }
    return true;
}

std::string phase_event(const char *event, uint64_t sequence,
                        const char *phase, bool measured,
                        uint64_t scheduled_offset_ns, uint64_t timestamp_ns)
{
    char buffer[512];
    int length = std::snprintf(
        buffer, sizeof(buffer),
        "{\"event\":\"%s\",\"sequence\":%" PRIu64
        ",\"phase\":\"%s\",\"measured\":%s,"
        "\"scheduled_offset_ns\":%" PRIu64 ",\"mono_ns\":%" PRIu64 "}",
        event, sequence, phase, measured ? "true" : "false",
        scheduled_offset_ns, timestamp_ns);
    if (length < 0 || static_cast<size_t>(length) >= sizeof(buffer)) {
        std::fprintf(stderr, "phase event formatting overflow\n");
        std::exit(1);
    }
    return std::string(buffer, static_cast<size_t>(length));
}

bool parse_fd(const char *text, int *value)
{
    char *end = nullptr;
    long parsed;
    errno = 0;
    parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max())
        return false;
    *value = static_cast<int>(parsed);
    return true;
}

bool cuda_ok(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool write_result(const std::filesystem::path &path, uint64_t epoch_ns,
                  const std::vector<PhaseResult> &phases,
                  uint64_t total_checked, uint64_t total_mismatches,
                  uint64_t first_mismatch, double total_kernel_ms,
                  uint64_t measured_end_ns)
{
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        std::fprintf(stderr, "failed to create result: %s\n",
                     path.c_str());
        return false;
    }
    const double end_to_end_ms =
        static_cast<double>(measured_end_ns - phases.front().start_mono_ns) /
        1.0e6;
    const double verified_words_per_second =
        static_cast<double>(total_checked) * 1000.0 / end_to_end_ms;

    output << "{\n"
           << "  \"protocol\": \"" << kProtocol << "\",\n"
           << "  \"timeline\": \"" << kTimeline << "\",\n"
           << "  \"allocation_bytes\": " << kAllocationBytes << ",\n"
           << "  \"region_bytes\": " << kRegionBytes << ",\n"
           << "  \"regions\": " << kRegions << ",\n"
           << "  \"sparse_stride_regions\": " << kSparseStride << ",\n"
           << "  \"sparse_regions\": " << kSparseRegions << ",\n"
           << "  \"dense_launch_regions\": " << kDenseLaunchRegions << ",\n"
           << "  \"bootstrap_ns\": " << kBootstrapNs << ",\n"
           << "  \"phase_ns\": " << kPhaseNs << ",\n"
           << "  \"measured_phases\": " << kMeasuredPhases << ",\n"
           << "  \"epoch_mono_ns\": " << epoch_ns << ",\n"
           << "  \"end_to_end_ms\": " << end_to_end_ms << ",\n"
           << "  \"total_kernel_ms\": " << total_kernel_ms << ",\n"
           << "  \"verified_words_per_second\": "
           << verified_words_per_second << ",\n"
           << "  \"checked_values\": " << total_checked << ",\n"
           << "  \"mismatches\": " << total_mismatches << ",\n"
           << "  \"first_mismatch\": ";
    if (total_mismatches == 0)
        output << "null,\n";
    else
        output << first_mismatch << ",\n";
    output << "  \"phases\": [\n";
    for (size_t index = 0; index < phases.size(); ++index) {
        const PhaseResult &phase = phases[index];
        const double wall_ms =
            static_cast<double>(phase.end_mono_ns - phase.start_mono_ns) /
            1.0e6;
        output << "    {\"measured_index\": " << phase.measured_index
               << ", \"sequence\": " << phase.sequence
               << ", \"phase\": \"" << phase.phase
               << "\", \"scheduled_offset_ns\": "
               << phase.scheduled_offset_ns
               << ", \"start_mono_ns\": " << phase.start_mono_ns
               << ", \"end_mono_ns\": " << phase.end_mono_ns
               << ", \"wall_ms\": " << wall_ms
               << ", \"kernel_ms\": " << phase.kernel_ms
               << ", \"iterations\": " << phase.iterations
               << ", \"checked_values\": " << phase.checked_values
               << ", \"mismatches\": " << phase.mismatches
               << ", \"first_mismatch\": ";
        if (phase.mismatches == 0)
            output << "null";
        else
            output << phase.first_mismatch;
        output << "}" << (index + 1 == phases.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.flush();
    return output.good();
}

}  // namespace

int main(int argc, char **argv)
{
    std::filesystem::path result_path;
    std::filesystem::path truth_path;
    int release_fd = -1;
    int truth_fd = -1;

    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--result") == 0 && index + 1 < argc)
            result_path = argv[++index];
        else if (std::strcmp(argv[index], "--truth") == 0 && index + 1 < argc)
            truth_path = argv[++index];
        else if (std::strcmp(argv[index], "--release-fd") == 0 &&
                 index + 1 < argc) {
            if (!parse_fd(argv[++index], &release_fd)) {
                std::fprintf(stderr, "invalid --release-fd\n");
                return 2;
            }
        }
        else if (std::strcmp(argv[index], "--truth-fd") == 0 &&
                 index + 1 < argc) {
            if (!parse_fd(argv[++index], &truth_fd)) {
                std::fprintf(stderr, "invalid --truth-fd\n");
                return 2;
            }
        }
        else {
            std::fprintf(stderr,
                         "usage: %s --result PATH --truth PATH "
                         "--release-fd FD --truth-fd FD\n",
                         argv[0]);
            return 2;
        }
    }
    if (result_path.empty() || truth_path.empty() || release_fd < 0 ||
        truth_fd < 0) {
        std::fprintf(stderr, "all output and pipe arguments are required\n");
        return 2;
    }
    if (std::filesystem::exists(result_path) ||
        std::filesystem::exists(truth_path)) {
        std::fprintf(stderr, "refusing to overwrite result or truth path\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    int relay_duplicate = dup(truth_fd);
    if (relay_duplicate < 0) {
        std::fprintf(stderr, "failed to duplicate truth fd: %s\n",
                     std::strerror(errno));
        return 1;
    }
    FILE *relay = fdopen(relay_duplicate, "w");
    if (relay == nullptr) {
        std::fprintf(stderr, "failed to open truth stream: %s\n",
                     std::strerror(errno));
        close(relay_duplicate);
        return 1;
    }
    setvbuf(relay, nullptr, _IOLBF, 0);
    std::ofstream truth(truth_path, std::ios::out | std::ios::trunc);
    if (!truth) {
        std::fprintf(stderr, "failed to create truth path: %s\n",
                     truth_path.c_str());
        fclose(relay);
        return 1;
    }

    uint32_t *data = nullptr;
    uint32_t *observed_device = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    int exit_status = 1;

    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice") ||
        !cuda_ok(cudaFree(nullptr), "cudaFree(nullptr)") ||
        !cuda_ok(cudaMallocManaged(&data, static_cast<size_t>(kAllocationBytes),
                                   cudaMemAttachGlobal),
                 "cudaMallocManaged") ||
        !cuda_ok(cudaMalloc(&observed_device,
                            static_cast<size_t>(kSparseRegions *
                                                sizeof(uint32_t))),
                 "cudaMalloc(observed)") ||
        !cuda_ok(cudaEventCreate(&begin), "cudaEventCreate(begin)") ||
        !cuda_ok(cudaEventCreate(&end), "cudaEventCreate(end)"))
        goto cleanup;

    for (uint64_t region = 0; region < kRegions; ++region)
        data[region * kRegionWords] = expected_value(region);

    {
        char ready[512];
        int length = std::snprintf(
            ready, sizeof(ready),
            "{\"event\":\"workload_ready\",\"pid\":%ld,"
            "\"protocol\":\"%s\",\"timeline\":\"%s\","
            "\"allocation_bytes\":%" PRIu64 ",\"regions\":%" PRIu64 "}",
            (long)getpid(), kProtocol, kTimeline, kAllocationBytes, kRegions);
        if (length < 0 || static_cast<size_t>(length) >= sizeof(ready) ||
            !emit_line(truth, relay,
                       std::string(ready, static_cast<size_t>(length))))
            goto cleanup;
    }

    {
        char release = 0;
        ssize_t count;
        do {
            count = read(release_fd, &release, 1);
        } while (count < 0 && errno == EINTR);
        if (count != 1 || release != 'R') {
            std::fprintf(stderr, "workload release gate did not receive R\n");
            goto cleanup;
        }
    }

    {
        const unsigned int threads = 256;
        const uint64_t epoch_ns = monotonic_ns();
        const uint64_t bootstrap_end_ns = epoch_ns + kBootstrapNs;
        uint64_t dense_cursor = 0;
        uint64_t total_checked = 0;
        uint64_t total_mismatches = 0;
        uint64_t total_first_mismatch = 0;
        double total_kernel_ms = 0.0;
        std::vector<uint32_t> observed(static_cast<size_t>(kSparseRegions));
        std::vector<PhaseResult> phases;

        if (!emit_line(truth, relay,
                       phase_event("phase_start", 1, "sparse", false, 0,
                                   epoch_ns)) ||
            !wait_until(bootstrap_end_ns) ||
            !emit_line(truth, relay,
                       phase_event("phase_end", 1, "sparse", false, 0,
                                   monotonic_ns())))
            goto cleanup;

        for (unsigned int phase_index = 0; phase_index < kMeasuredPhases;
             ++phase_index) {
            const bool dense = (phase_index % 2U) == 0;
            const char *phase_name = dense ? "dense" : "sparse";
            const uint64_t sequence = static_cast<uint64_t>(phase_index) + 2;
            const uint64_t scheduled_offset_ns =
                kBootstrapNs + static_cast<uint64_t>(phase_index) * kPhaseNs;
            const uint64_t scheduled_start_ns = epoch_ns + scheduled_offset_ns;
            const uint64_t scheduled_end_ns = scheduled_start_ns + kPhaseNs;
            PhaseResult phase = {};
            phase.measured_index = phase_index + 1;
            phase.sequence = sequence;
            phase.phase = phase_name;
            phase.scheduled_offset_ns = scheduled_offset_ns;

            if (!wait_until(scheduled_start_ns))
                goto cleanup;
            phase.start_mono_ns = monotonic_ns();
            if (phase.start_mono_ns >
                scheduled_start_ns + kMaximumBoundaryOverrunNs) {
                std::fprintf(stderr,
                             "phase %u started outside frozen schedule\n",
                             phase_index + 1);
                goto cleanup;
            }
            if (!emit_line(truth, relay,
                           phase_event("phase_start", sequence, phase_name,
                                       true, scheduled_offset_ns,
                                       phase.start_mono_ns)))
                goto cleanup;

            do {
                const uint64_t count = dense ? kDenseLaunchRegions
                                             : kSparseRegions;
                const unsigned int blocks = static_cast<unsigned int>(
                    (count + threads - 1) / threads);
                if (!cuda_ok(cudaEventRecord(begin), "cudaEventRecord(begin)"))
                    goto cleanup;
                if (dense) {
                    read_dense_regions<<<blocks, threads>>>(
                        data, observed_device, kRegions, dense_cursor, count);
                }
                else {
                    read_sparse_regions<<<blocks, threads>>>(
                        data, observed_device, kRegions, count);
                }
                if (!cuda_ok(cudaGetLastError(), "phase kernel launch") ||
                    !cuda_ok(cudaEventRecord(end), "cudaEventRecord(end)") ||
                    !cuda_ok(cudaEventSynchronize(end),
                             "cudaEventSynchronize(end)"))
                    goto cleanup;
                float launch_ms = 0.0f;
                if (!cuda_ok(cudaEventElapsedTime(&launch_ms, begin, end),
                             "cudaEventElapsedTime") ||
                    !cuda_ok(cudaMemcpy(observed.data(), observed_device,
                                        static_cast<size_t>(count *
                                                            sizeof(uint32_t)),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy(observed)"))
                    goto cleanup;

                for (uint64_t item = 0; item < count; ++item) {
                    const uint64_t region = dense
                        ? (dense_cursor + item) % kRegions
                        : item * kSparseStride;
                    if (observed[static_cast<size_t>(item)] !=
                        expected_value(region)) {
                        if (phase.mismatches == 0)
                            phase.first_mismatch = region;
                        if (total_mismatches == 0)
                            total_first_mismatch = region;
                        phase.mismatches++;
                        total_mismatches++;
                    }
                }
                phase.iterations++;
                phase.checked_values += count;
                phase.kernel_ms += static_cast<double>(launch_ms);
                total_checked += count;
                total_kernel_ms += static_cast<double>(launch_ms);
                if (dense)
                    dense_cursor = (dense_cursor + count) % kRegions;
            } while (monotonic_ns() < scheduled_end_ns);

            phase.end_mono_ns = monotonic_ns();
            if (phase.end_mono_ns >
                scheduled_end_ns + kMaximumBoundaryOverrunNs) {
                std::fprintf(stderr,
                             "phase %u exceeded frozen boundary by more than 500 ms\n",
                             phase_index + 1);
                goto cleanup;
            }
            if (!emit_line(truth, relay,
                           phase_event("phase_end", sequence, phase_name, true,
                                       scheduled_offset_ns,
                                       phase.end_mono_ns)))
                goto cleanup;
            phases.push_back(phase);
        }

        if (phases.size() != kMeasuredPhases || total_checked == 0 ||
            !write_result(result_path, epoch_ns, phases, total_checked,
                          total_mismatches, total_first_mismatch,
                          total_kernel_ms, phases.back().end_mono_ns))
            goto cleanup;
        std::printf("RESULT checked_values=%" PRIu64
                    " mismatches=%" PRIu64 " end_to_end_ms=%.3f\n",
                    total_checked, total_mismatches,
                    static_cast<double>(phases.back().end_mono_ns -
                                        phases.front().start_mono_ns) /
                        1.0e6);
        exit_status = total_mismatches == 0 ? 0 : 1;
    }

cleanup:
    if (end != nullptr)
        cudaEventDestroy(end);
    if (begin != nullptr)
        cudaEventDestroy(begin);
    if (observed_device != nullptr)
        cudaFree(observed_device);
    if (data != nullptr)
        cudaFree(data);
    truth.close();
    fclose(relay);
    return exit_status;
}
