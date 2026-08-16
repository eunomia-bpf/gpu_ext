#include <cuda_runtime.h>

#ifdef UVM_BASIC_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        cudaError_t error__ = (expr);                                                     \
        if (error__ != cudaSuccess) {                                                     \
            std::ostringstream message__;                                                \
            message__ << #expr << ": " << cudaGetErrorString(error__);                 \
            throw std::runtime_error(message__.str());                                   \
        }                                                                                 \
    } while (0)

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    size_t total_bytes = 0;
    size_t reserve_device_bytes = 0;
    size_t target_effective_gpu_bytes = 0;
    size_t guard_device_bytes = 0;
    size_t deprecated_safety_headroom_bytes = 0;
    double working_set_ratio = 0.80;
    double region_a_ratio = 0.50;
    int cycles = 1;
    int gpu_id = 0;
    bool verify = true;
    bool reserve_touch = true;
    bool reserve_verify = true;
    bool guard_touch = true;
    std::string output;
};

struct CapacityInfo {
    size_t gpu_total_bytes = 0;
    size_t gpu_free_initial = 0;
    size_t main_reserve_requested_bytes = 0;
    size_t main_reserve_allocated_bytes = 0;
    size_t gpu_free_after_main_reserve = 0;
    size_t guard_requested_bytes = 0;
    size_t guard_allocated_bytes = 0;
    size_t gpu_free_after_guard = 0;
    size_t effective_gpu_capacity_bytes = 0;
    size_t target_effective_gpu_bytes = 0;
    size_t managed_working_set_requested_bytes = 0;
    size_t managed_working_set_actual_bytes = 0;
    size_t region_a_bytes = 0;
    size_t region_b_bytes = 0;
    double requested_working_set_ratio = 0.0;
    double actual_working_set_ratio = 0.0;
    double capacity_target_relative_error = 0.0;
    double working_set_ratio_error = 0.0;
    bool reserve_touched = false;
    bool reserve_verified = false;
    bool guard_touched = false;
    bool guard_verified = false;
    bool capacity_target_pass = true;
    bool working_set_ratio_pass = true;
    bool reduced_capacity = false;

    const char *evidence_class() const
    {
        return reduced_capacity ? "PHYSICALLY_RESERVED_GUARD_MODEL"
                                : "NATURAL_GPU_CAPACITY";
    }
};

struct NvtxRange {
    explicit NvtxRange(const char *name)
    {
#ifdef UVM_BASIC_HAVE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~NvtxRange()
    {
#ifdef UVM_BASIC_HAVE_NVTX
        nvtxRangePop();
#endif
    }
};

size_t parse_size(const std::string &text)
{
    if (text == "auto")
        return 0;
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    std::string suffix = text.substr(consumed);
    for (char &c : suffix) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    unsigned long long multiplier = 1;
    if (suffix == "K" || suffix == "KB" || suffix == "KIB") multiplier = 1ULL << 10;
    else if (suffix == "M" || suffix == "MB" || suffix == "MIB") multiplier = 1ULL << 20;
    else if (suffix == "G" || suffix == "GB" || suffix == "GIB") multiplier = 1ULL << 30;
    else if (!suffix.empty()) throw std::invalid_argument("unsupported size suffix: " + suffix);
    if (value > std::numeric_limits<size_t>::max() / multiplier)
        throw std::overflow_error("total-bytes overflow");
    return static_cast<size_t>(value * multiplier);
}

bool yes_no(const std::string &value)
{
    if (value == "yes") return true;
    if (value == "no") return false;
    throw std::invalid_argument("expected yes or no: " + value);
}

Options parse_args(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return std::string(argv[i]);
        };
        if (arg == "--total-bytes") options.total_bytes = parse_size(next());
        else if (arg == "--reserve-device-bytes") options.reserve_device_bytes = parse_size(next());
        else if (arg == "--target-effective-gpu-bytes")
            options.target_effective_gpu_bytes = parse_size(next());
        else if (arg == "--guard-device-bytes")
            options.guard_device_bytes = parse_size(next());
        else if (arg == "--safety-headroom-bytes")
            options.deprecated_safety_headroom_bytes = parse_size(next());
        else if (arg == "--reserve-touch") options.reserve_touch = yes_no(next());
        else if (arg == "--reserve-verify") options.reserve_verify = yes_no(next());
        else if (arg == "--guard-touch") options.guard_touch = yes_no(next());
        else if (arg == "--working-set-ratio") options.working_set_ratio = std::stod(next());
        else if (arg == "--region-a-ratio") options.region_a_ratio = std::stod(next());
        else if (arg == "--cycles") options.cycles = std::stoi(next());
        else if (arg == "--gpu-id") options.gpu_id = std::stoi(next());
        else if (arg == "--verify") options.verify = yes_no(next());
        else if (arg == "--output") options.output = next();
        else if (arg == "--help" || arg == "-h") {
            std::cout << "uvm_phase_scan --total-bytes SIZE|auto --working-set-ratio R "
                         "--region-a-ratio R --cycles N --gpu-id N --verify yes|no "
                         "[--reserve-device-bytes SIZE | --target-effective-gpu-bytes SIZE] "
                         "[--guard-device-bytes SIZE] [--guard-touch yes|no] "
                         "[--reserve-touch yes|no] [--reserve-verify yes|no] "
                         "[--safety-headroom-bytes SIZE (deprecated guard alias)] "
                         "--output FILE\n";
            std::exit(0);
        }
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (!(options.working_set_ratio > 0.0 && options.working_set_ratio <= 1.25))
        throw std::invalid_argument("working-set-ratio must be in (0, 1.25]");
    if (!(options.region_a_ratio > 0.0 && options.region_a_ratio < 1.0))
        throw std::invalid_argument("region-a-ratio must be in (0, 1)");
    if (options.cycles <= 0) throw std::invalid_argument("cycles must be positive");
    if (options.reserve_device_bytes && options.target_effective_gpu_bytes)
        throw std::invalid_argument("reserve-device-bytes and target-effective-gpu-bytes are exclusive");
    if (options.deprecated_safety_headroom_bytes) {
        if (options.guard_device_bytes &&
            options.guard_device_bytes != options.deprecated_safety_headroom_bytes)
            throw std::invalid_argument("deprecated safety headroom conflicts with guard-device-bytes");
        if (!options.guard_device_bytes)
            options.guard_device_bytes = options.deprecated_safety_headroom_bytes;
        std::cerr << "warning: --safety-headroom-bytes is deprecated; "
                     "allocating it as a physical guard buffer\n";
    }
    if (options.reserve_verify && !options.reserve_touch &&
        (options.reserve_device_bytes || options.target_effective_gpu_bytes))
        throw std::invalid_argument("reserve-verify requires reserve-touch");
    if (options.reserve_verify && !options.guard_touch && options.guard_device_bytes)
        throw std::invalid_argument("reserve-verify requires guard-touch for a guard buffer");
    if (options.output.empty()) throw std::invalid_argument("--output is required");
    return options;
}

__global__ void mutate_region(float *buffer, size_t begin, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t offset = index; offset < count; offset += stride)
        buffer[begin + offset] += 1.0f;
}

__global__ void verify_samples(const float *buffer,
                               const size_t *indices,
                               size_t count,
                               float expected,
                               unsigned int *mismatches)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && fabsf(buffer[indices[index]] - expected) > 1.0e-5f)
        atomicAdd(mismatches, 1U);
}

std::string run_id()
{
    const auto value = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return std::to_string(value) + "-" + std::to_string(getpid());
}

class Recorder {
public:
    Recorder(const Options &options, const CapacityInfo &capacity, const std::string &id)
        : options_(options), capacity_(capacity), id_(id),
          output_(options.output, std::ios::app)
    {
        if (!output_) throw std::runtime_error("cannot open output: " + options.output);
    }

    void row(const std::string &phase, double elapsed_ms, size_t begin, size_t bytes, bool correct)
    {
        const auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count();
        const auto start_ns = end_ns - static_cast<long long>(elapsed_ms * 1.0e6);
        output_ << "{\"run_id\":\"" << id_ << "\",\"phase\":\"" << phase
                << "\",\"elapsed_ms\":" << std::fixed << std::setprecision(6) << elapsed_ms
                << ",\"monotonic_start_ns\":" << start_ns
                << ",\"monotonic_end_ns\":" << end_ns
                << ",\"region_begin\":" << begin << ",\"region_bytes\":" << bytes
                << ",\"total_working_set\":" << options_.total_bytes
                << ",\"requested_working_set_ratio\":"
                << capacity_.requested_working_set_ratio
                << ",\"actual_working_set_ratio\":"
                << capacity_.actual_working_set_ratio
                << ",\"usable_gpu_memory\":" << capacity_.effective_gpu_capacity_bytes
                << ",\"gpu_memory_total\":" << capacity_.gpu_total_bytes
                << ",\"capacity_model\":\"" << capacity_.evidence_class() << "\""
                << ",\"region_a_ratio\":" << options_.region_a_ratio
                << ",\"cycles\":" << options_.cycles
                << ",\"correct\":" << (correct ? "true" : "false")
                << ",\"evidence_class\":\"PROGRAM_TIMING\"}\n";
        output_.flush();
        std::cout << std::left << std::setw(28) << phase << std::right
                  << std::setw(12) << std::fixed << std::setprecision(3) << elapsed_ms
                  << " ms  " << (correct ? "PASS" : "FAIL") << '\n';
    }

    void capacity()
    {
        output_ << "{\"run_id\":\"" << id_
                << "\",\"phase\":\"capacity_manifest\""
                << ",\"gpu_total_bytes\":" << capacity_.gpu_total_bytes
                << ",\"gpu_free_initial\":" << capacity_.gpu_free_initial
                << ",\"target_effective_gpu_bytes\":" << capacity_.target_effective_gpu_bytes
                << ",\"main_reserve_requested_bytes\":"
                << capacity_.main_reserve_requested_bytes
                << ",\"main_reserve_allocated_bytes\":"
                << capacity_.main_reserve_allocated_bytes
                << ",\"gpu_free_after_main_reserve\":"
                << capacity_.gpu_free_after_main_reserve
                << ",\"guard_requested_bytes\":" << capacity_.guard_requested_bytes
                << ",\"guard_allocated_bytes\":" << capacity_.guard_allocated_bytes
                << ",\"gpu_free_after_guard\":" << capacity_.gpu_free_after_guard
                << ",\"effective_gpu_capacity_bytes\":"
                << capacity_.effective_gpu_capacity_bytes
                << ",\"managed_working_set_requested_bytes\":"
                << capacity_.managed_working_set_requested_bytes
                << ",\"managed_working_set_actual_bytes\":"
                << capacity_.managed_working_set_actual_bytes
                << ",\"managed_working_set_bytes\":"
                << capacity_.managed_working_set_actual_bytes
                << ",\"requested_working_set_ratio\":"
                << capacity_.requested_working_set_ratio
                << ",\"actual_working_set_ratio\":"
                << capacity_.actual_working_set_ratio
                << ",\"working_set_ratio\":" << capacity_.actual_working_set_ratio
                << ",\"region_a_bytes\":" << capacity_.region_a_bytes
                << ",\"region_b_bytes\":" << capacity_.region_b_bytes
                << ",\"capacity_target_relative_error\":"
                << capacity_.capacity_target_relative_error
                << ",\"working_set_ratio_error\":" << capacity_.working_set_ratio_error
                << ",\"capacity_target_pass\":"
                << (capacity_.capacity_target_pass ? "true" : "false")
                << ",\"working_set_ratio_pass\":"
                << (capacity_.working_set_ratio_pass ? "true" : "false")
                << ",\"reserve_touched\":" << (capacity_.reserve_touched ? "true" : "false")
                << ",\"reserve_verified\":" << (capacity_.reserve_verified ? "true" : "false")
                << ",\"guard_touched\":" << (capacity_.guard_touched ? "true" : "false")
                << ",\"guard_verified\":" << (capacity_.guard_verified ? "true" : "false")
                << ",\"deprecated_safety_headroom_bytes\":"
                << options_.deprecated_safety_headroom_bytes
                << ",\"correct\":"
                << ((capacity_.capacity_target_pass && capacity_.working_set_ratio_pass)
                        ? "true" : "false")
                << ",\"evidence_class\":\""
                << capacity_.evidence_class() << "\"}\n";
        output_.flush();
    }

    void memory_snapshot(const std::string &checkpoint, size_t free_bytes, size_t total_bytes)
    {
        output_ << "{\"run_id\":\"" << id_
                << "\",\"phase\":\"gpu_memory_snapshot\""
                << ",\"checkpoint\":\"" << checkpoint << "\""
                << ",\"gpu_free_bytes\":" << free_bytes
                << ",\"gpu_total_bytes\":" << total_bytes
                << ",\"correct\":true,\"evidence_class\":\"CUDA_MEM_GET_INFO\"}\n";
        output_.flush();
    }

    void allocation(const float *buffer)
    {
        const auto base = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buffer));
        output_ << "{\"run_id\":\"" << id_
                << "\",\"phase\":\"allocation_manifest\",\"buffer_base\":\"0x"
                << std::hex << base << "\",\"buffer_end\":\"0x" << base + options_.total_bytes
                << "\",\"buffer_base_u64\":" << std::dec << base
                << ",\"buffer_end_u64\":" << base + options_.total_bytes
                << ",\"total_working_set\":" << options_.total_bytes
                << ",\"correct\":true,\"evidence_class\":\"PROGRAM_ALLOCATION_RANGE\"}\n";
    }

private:
    const Options &options_;
    const CapacityInfo &capacity_;
    std::string id_;
    std::ofstream output_;
};

size_t aligned_down(size_t value, size_t alignment)
{
    return value / alignment * alignment;
}

size_t validate_target_capacity(size_t requested, size_t free_bytes, size_t guard_bytes)
{
    constexpr size_t gib = 1ULL << 30;
    if (requested < 4ULL * gib)
        throw std::runtime_error("target effective capacity is below the 4 GiB safety floor");
    if (free_bytes <= requested || free_bytes - requested <= guard_bytes)
        throw std::runtime_error("cannot allocate the physical reserve and guard for target capacity");
    return requested;
}

bool verify_device_bytes(unsigned char *buffer, size_t bytes, unsigned char expected)
{
    constexpr size_t sample_size = 64;
    unsigned char first[sample_size] = {};
    unsigned char last[sample_size] = {};
    CUDA_CHECK(cudaMemcpy(first, buffer, sample_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(last, buffer + bytes - sample_size, sample_size,
                          cudaMemcpyDeviceToHost));
    return std::all_of(std::begin(first), std::end(first),
                       [expected](unsigned char value) { return value == expected; }) &&
           std::all_of(std::begin(last), std::end(last),
                       [expected](unsigned char value) { return value == expected; });
}

void memory_info(size_t &free_bytes, size_t &total_bytes)
{
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

double run_phase(float *buffer, size_t begin, size_t count, cudaStream_t stream)
{
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<size_t>((count + threads - 1) / threads, 65535));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    mutate_region<<<blocks, threads, 0, stream>>>(buffer, begin, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    return milliseconds;
}

} // namespace

int main(int argc, char **argv)
{
    float *buffer = nullptr;
    unsigned char *main_reserve_buffer = nullptr;
    unsigned char *guard_buffer = nullptr;
    size_t *device_indices = nullptr;
    unsigned int *device_mismatches = nullptr;
    size_t verification_count = 0;
    cudaStream_t stream = nullptr;
    try {
        Options options = parse_args(argc, argv);
        CUDA_CHECK(cudaSetDevice(options.gpu_id));
        const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        if (page_size == 0) throw std::runtime_error("invalid system page size");

        CUDA_CHECK(cudaStreamCreate(&stream));

        // Fixed verifier allocations must not consume capacity after the reserve is established.
        if (options.verify) {
            CUDA_CHECK(cudaMalloc(&device_indices, 4096 * sizeof(size_t)));
            CUDA_CHECK(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)));
        }

        CapacityInfo capacity;
        memory_info(capacity.gpu_free_initial, capacity.gpu_total_bytes);
        const bool reduced_capacity = options.reserve_device_bytes != 0 ||
                                      options.target_effective_gpu_bytes != 0 ||
                                      options.guard_device_bytes != 0;
        capacity.reduced_capacity = reduced_capacity;
        capacity.target_effective_gpu_bytes = options.target_effective_gpu_bytes;
        capacity.guard_requested_bytes = aligned_down(options.guard_device_bytes, page_size);

        if (options.target_effective_gpu_bytes) {
            capacity.target_effective_gpu_bytes = validate_target_capacity(
                options.target_effective_gpu_bytes,
                capacity.gpu_free_initial,
                capacity.guard_requested_bytes);
            capacity.main_reserve_requested_bytes = aligned_down(
                capacity.gpu_free_initial - capacity.guard_requested_bytes -
                    capacity.target_effective_gpu_bytes,
                page_size);
        }
        else {
            capacity.main_reserve_requested_bytes = aligned_down(
                options.reserve_device_bytes, page_size);
        }

        if (capacity.main_reserve_requested_bytes) {
            if (capacity.main_reserve_requested_bytes < 64)
                throw std::invalid_argument("reserve-device-bytes must be at least 64 bytes");
            CUDA_CHECK(cudaMalloc(&main_reserve_buffer,
                                  capacity.main_reserve_requested_bytes));
            if (options.reserve_touch) {
                CUDA_CHECK(cudaMemsetAsync(main_reserve_buffer, 0xa5,
                                           capacity.main_reserve_requested_bytes, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                capacity.reserve_touched = true;
            }
            if (options.reserve_verify) {
                capacity.reserve_verified = verify_device_bytes(
                    main_reserve_buffer, capacity.main_reserve_requested_bytes, 0xa5);
                if (!capacity.reserve_verified)
                    throw std::runtime_error("reserve buffer verification failed");
            }
        }

        memory_info(capacity.gpu_free_after_main_reserve, capacity.gpu_total_bytes);
        capacity.main_reserve_allocated_bytes =
            capacity.gpu_free_initial > capacity.gpu_free_after_main_reserve
                ? capacity.gpu_free_initial - capacity.gpu_free_after_main_reserve
                : 0;

        if (capacity.guard_requested_bytes) {
            if (capacity.guard_requested_bytes < 64)
                throw std::invalid_argument("guard-device-bytes must be at least 64 bytes");
            CUDA_CHECK(cudaMalloc(&guard_buffer, capacity.guard_requested_bytes));
            if (options.guard_touch) {
                CUDA_CHECK(cudaMemsetAsync(guard_buffer, 0x5a,
                                           capacity.guard_requested_bytes, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                capacity.guard_touched = true;
            }
            if (options.reserve_verify) {
                capacity.guard_verified = verify_device_bytes(
                    guard_buffer, capacity.guard_requested_bytes, 0x5a);
                if (!capacity.guard_verified)
                    throw std::runtime_error("guard buffer verification failed");
            }
        }

        memory_info(capacity.gpu_free_after_guard, capacity.gpu_total_bytes);
        capacity.guard_allocated_bytes =
            capacity.gpu_free_after_main_reserve > capacity.gpu_free_after_guard
                ? capacity.gpu_free_after_main_reserve - capacity.gpu_free_after_guard
                : 0;
        capacity.effective_gpu_capacity_bytes = capacity.gpu_free_after_guard;
        if (reduced_capacity && capacity.effective_gpu_capacity_bytes < (4ULL << 30))
            throw std::runtime_error("effective GPU capacity is below the 4 GiB safety floor");

        const bool derived_working_set = options.total_bytes == 0;
        if (!options.total_bytes) {
            const long double requested =
                static_cast<long double>(capacity.effective_gpu_capacity_bytes) *
                options.working_set_ratio;
            if (requested > static_cast<long double>(std::numeric_limits<size_t>::max()))
                throw std::overflow_error("derived working set overflows size_t");
            options.total_bytes = aligned_down(static_cast<size_t>(requested), page_size);
        }
        if (options.total_bytes < 2 * page_size || options.total_bytes % sizeof(float))
            throw std::invalid_argument("total working set is too small or misaligned");

        capacity.managed_working_set_requested_bytes = options.total_bytes;
        capacity.requested_working_set_ratio = options.working_set_ratio;
        capacity.actual_working_set_ratio =
            static_cast<double>(options.total_bytes) /
            static_cast<double>(capacity.effective_gpu_capacity_bytes);
        capacity.working_set_ratio_error = std::fabs(
            capacity.actual_working_set_ratio - options.working_set_ratio);
        capacity.working_set_ratio_pass = !derived_working_set ||
                                          capacity.working_set_ratio_error <= 0.01;
        if (capacity.target_effective_gpu_bytes) {
            const long double difference = std::fabs(
                static_cast<long double>(capacity.effective_gpu_capacity_bytes) -
                static_cast<long double>(capacity.target_effective_gpu_bytes));
            capacity.capacity_target_relative_error = static_cast<double>(
                difference / static_cast<long double>(capacity.target_effective_gpu_bytes));
            capacity.capacity_target_pass = capacity.capacity_target_relative_error <= 0.02;
        }

        const size_t elements = options.total_bytes / sizeof(float);
        size_t region_a_elements = static_cast<size_t>(elements * options.region_a_ratio);
        region_a_elements = std::max<size_t>(1, std::min(region_a_elements, elements - 1));
        const size_t region_b_elements = elements - region_a_elements;
        const size_t region_b_begin = region_a_elements;
        capacity.region_a_bytes = region_a_elements * sizeof(float);
        capacity.region_b_bytes = region_b_elements * sizeof(float);

        if (!capacity.capacity_target_pass || !capacity.working_set_ratio_pass) {
            Recorder recorder(options, capacity, run_id());
            recorder.capacity();
            if (!capacity.capacity_target_pass)
                throw std::runtime_error("FAILED_EFFECTIVE_CAPACITY_TARGET");
            throw std::runtime_error("FAILED_WORKING_SET_RATIO");
        }

        CUDA_CHECK(cudaMallocManaged(&buffer, options.total_bytes));
        capacity.managed_working_set_actual_bytes = options.total_bytes;
        Recorder recorder(options, capacity, run_id());
        recorder.capacity();
        recorder.allocation(buffer);
        size_t free_bytes = 0, total_bytes = 0;
        memory_info(free_bytes, total_bytes);
        recorder.memory_snapshot("after_managed_allocation", free_bytes, total_bytes);

        if (options.verify) {
            std::vector<size_t> indices;
            verification_count = std::min<size_t>(4096, elements);
            indices.reserve(verification_count);
            std::mt19937_64 generator(0x535441474533ULL);
            std::uniform_int_distribution<size_t> distribution(0, elements - 1);
            for (size_t i = 0; i < verification_count; ++i)
                indices.push_back(distribution(generator));
            CUDA_CHECK(cudaMemcpy(device_indices, indices.data(), indices.size() * sizeof(size_t),
                                  cudaMemcpyHostToDevice));
        }

        {
            NvtxRange range("cpu_first_touch_entire_buffer");
            const auto start = Clock::now();
            for (size_t i = 0; i < elements; ++i) buffer[i] = 1.0f;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            recorder.row("cpu_first_touch", ms, 0, options.total_bytes, true);
        }
        memory_info(free_bytes, total_bytes);
        recorder.memory_snapshot("after_cpu_first_touch", free_bytes, total_bytes);

        for (int cycle = 0; cycle < options.cycles; ++cycle) {
            const std::string suffix = cycle == 0 ? "" : "_cycle_" + std::to_string(cycle);
            const std::string phases[] = {
                "phase_A_first" + suffix, "phase_B_first" + suffix,
                "phase_A_reuse" + suffix, "phase_B_reuse" + suffix,
            };
            const size_t begins[] = {0, region_b_begin, 0, region_b_begin};
            const size_t counts[] = {region_a_elements, region_b_elements,
                                     region_a_elements, region_b_elements};
            for (int phase = 0; phase < 4; ++phase) {
                NvtxRange range(phases[phase].c_str());
                const double ms = run_phase(buffer, begins[phase], counts[phase], stream);
                recorder.row(phases[phase], ms, begins[phase] * sizeof(float),
                             counts[phase] * sizeof(float), true);
                memory_info(free_bytes, total_bytes);
                recorder.memory_snapshot("after_" + phases[phase], free_bytes, total_bytes);
            }
        }

        bool correct = true;
        if (options.verify) {
            CUDA_CHECK(cudaMemset(device_mismatches, 0, sizeof(*device_mismatches)));
            constexpr unsigned int threads = 256;
            const unsigned int blocks = static_cast<unsigned int>(
                (verification_count + threads - 1) / threads);
            verify_samples<<<blocks, threads>>>(buffer, device_indices, verification_count,
                                                1.0f + 2.0f * options.cycles,
                                                device_mismatches);
            CUDA_CHECK(cudaGetLastError());
            unsigned int mismatches = 0;
            CUDA_CHECK(cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches),
                                  cudaMemcpyDeviceToHost));
            correct = mismatches == 0;
        }
        recorder.row("correctness", 0.0, 0, options.total_bytes, correct);
        if (!correct) throw std::runtime_error("phase scan verification failed");

        memory_info(free_bytes, total_bytes);
        recorder.memory_snapshot("before_cleanup", free_bytes, total_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(buffer));
        buffer = nullptr;
        if (guard_buffer) {
            CUDA_CHECK(cudaFree(guard_buffer));
            guard_buffer = nullptr;
        }
        if (main_reserve_buffer) {
            CUDA_CHECK(cudaFree(main_reserve_buffer));
            main_reserve_buffer = nullptr;
        }
        if (device_mismatches) {
            CUDA_CHECK(cudaFree(device_mismatches));
            device_mismatches = nullptr;
        }
        if (device_indices) {
            CUDA_CHECK(cudaFree(device_indices));
            device_indices = nullptr;
        }
        memory_info(free_bytes, total_bytes);
        recorder.memory_snapshot("after_cleanup", free_bytes, total_bytes);
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "uvm_phase_scan: " << error.what() << '\n';
        cudaDeviceSynchronize();
        if (guard_buffer) cudaFree(guard_buffer);
        if (buffer) cudaFree(buffer);
        if (main_reserve_buffer) cudaFree(main_reserve_buffer);
        if (device_mismatches) cudaFree(device_mismatches);
        if (device_indices) cudaFree(device_indices);
        if (stream) cudaStreamDestroy(stream);
        return 1;
    }
}
