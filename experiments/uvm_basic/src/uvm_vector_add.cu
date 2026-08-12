#include <cuda_runtime.h>

#ifdef UVM_BASIC_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        const cudaError_t cuda_check_error = (expr);                                      \
        if (cuda_check_error != cudaSuccess) {                                            \
            std::ostringstream cuda_check_stream;                                         \
            cuda_check_stream << #expr << " failed: "                                   \
                              << cudaGetErrorString(cuda_check_error);                     \
            throw std::runtime_error(cuda_check_stream.str());                            \
        }                                                                                 \
    } while (0)

namespace {

using Clock = std::chrono::steady_clock;

void cuda_cleanup(cudaError_t result, const char *operation) noexcept
{
    if (result != cudaSuccess)
        std::cerr << "CUDA cleanup warning: " << operation << ": " << cudaGetErrorString(result) << '\n';
}

struct Options {
    size_t bytes = 256ULL << 20;
    std::string allocation = "managed";
    int iterations = 1;
    std::string cpu_retouch = "none";
    bool gpu_prefetch = false;
    bool cpu_prefetch_before_retouch = false;
    bool verify = true;
    std::string output;
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

std::string json_escape(const std::string &value)
{
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
            else
                out << c;
        }
    }
    return out.str();
}

size_t parse_size(const std::string &text)
{
    if (text.empty())
        throw std::invalid_argument("empty size");
    size_t suffix_pos = text.size();
    while (suffix_pos > 0 && std::isalpha(static_cast<unsigned char>(text[suffix_pos - 1])))
        --suffix_pos;
    const std::string number = text.substr(0, suffix_pos);
    std::string suffix = text.substr(suffix_pos);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (number.empty())
        throw std::invalid_argument("size has no numeric component: " + text);
    size_t consumed = 0;
    const unsigned long long base = std::stoull(number, &consumed, 10);
    if (consumed != number.size())
        throw std::invalid_argument("invalid size: " + text);
    unsigned long long multiplier = 1;
    if (suffix == "K" || suffix == "KB" || suffix == "KIB") multiplier = 1ULL << 10;
    else if (suffix == "M" || suffix == "MB" || suffix == "MIB") multiplier = 1ULL << 20;
    else if (suffix == "G" || suffix == "GB" || suffix == "GIB") multiplier = 1ULL << 30;
    else if (!suffix.empty()) throw std::invalid_argument("unsupported size suffix: " + suffix);
    if (base > std::numeric_limits<size_t>::max() / multiplier)
        throw std::overflow_error("size overflows size_t");
    return static_cast<size_t>(base * multiplier);
}

bool parse_yes_no(const std::string &value)
{
    if (value == "yes") return true;
    if (value == "no") return false;
    throw std::invalid_argument("expected yes or no, got: " + value);
}

void usage(const char *program)
{
    std::cerr << "Usage: " << program
              << " --bytes SIZE --allocation managed|device --iterations N"
              << " --cpu-retouch none|page|full --gpu-prefetch yes|no"
              << " --cpu-prefetch-before-retouch yes|no --verify yes|no"
              << " --output FILE\n";
}

Options parse_args(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--bytes") options.bytes = parse_size(value());
        else if (arg == "--allocation") options.allocation = value();
        else if (arg == "--iterations") options.iterations = std::stoi(value());
        else if (arg == "--cpu-retouch") options.cpu_retouch = value();
        else if (arg == "--gpu-prefetch") options.gpu_prefetch = parse_yes_no(value());
        else if (arg == "--cpu-prefetch-before-retouch")
            options.cpu_prefetch_before_retouch = parse_yes_no(value());
        else if (arg == "--verify") options.verify = parse_yes_no(value());
        else if (arg == "--output") options.output = value();
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (options.allocation != "managed" && options.allocation != "device")
        throw std::invalid_argument("allocation must be managed or device");
    if (options.cpu_retouch != "none" && options.cpu_retouch != "page" &&
        options.cpu_retouch != "full")
        throw std::invalid_argument("cpu-retouch must be none, page, or full");
    if (options.iterations <= 0)
        throw std::invalid_argument("iterations must be positive");
    if (options.bytes < sizeof(float) || options.bytes % sizeof(float) != 0)
        throw std::invalid_argument("bytes must be a positive multiple of sizeof(float)");
    if (options.output.empty())
        throw std::invalid_argument("--output is required");
    if (options.allocation == "device" &&
        (options.cpu_retouch != "none" || options.cpu_prefetch_before_retouch || options.gpu_prefetch))
        throw std::invalid_argument("retouch and prefetch options apply only to managed allocation");
    return options;
}

std::string make_run_id()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return std::to_string(micros) + "-" + std::to_string(getpid());
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__global__ void vector_add(const float *a, const float *b, float *c, size_t elements)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = index; i < elements; i += stride)
        c[i] = a[i] + b[i];
}

__global__ void verify_samples(const float *a,
                               const float *b,
                               const float *c,
                               const size_t *indices,
                               size_t count,
                               unsigned int *mismatches)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t index = indices[i];
    if (fabsf(c[index] - (a[index] + b[index])) > 1.0e-5f)
        atomicAdd(mismatches, 1U);
}

class Recorder {
public:
    Recorder(const Options &options,
             std::string run_id,
             std::string device_name,
             int device,
             long page_size)
        : options_(options),
          run_id_(std::move(run_id)),
          device_name_(std::move(device_name)),
          device_(device),
          page_size_(page_size),
          output_(options.output, std::ios::app)
    {
        if (!output_) throw std::runtime_error("cannot open output: " + options.output);
        std::cout << std::left << std::setw(34) << "phase" << std::right
                  << std::setw(14) << "elapsed_ms" << std::setw(14) << "GB/s"
                  << std::setw(11) << "correct" << '\n';
    }

    void row(const std::string &phase,
             double milliseconds,
             double logical_bytes,
             bool correct,
             const std::string &message = "",
             size_t free_bytes = 0,
             size_t total_bytes = 0,
             bool skipped = false)
    {
        const double bandwidth = milliseconds > 0.0
                                     ? logical_bytes / (milliseconds * 1.0e6)
                                     : 0.0;
        output_ << "{\"run_id\":\"" << json_escape(run_id_)
                << "\",\"phase\":\"" << json_escape(phase)
                << "\",\"allocation\":\"" << options_.allocation
                << "\",\"bytes_per_array\":" << options_.bytes
                << ",\"elements\":" << options_.bytes / sizeof(float)
                << ",\"iterations\":" << options_.iterations
                << ",\"cpu_retouch\":\"" << options_.cpu_retouch
                << "\",\"gpu_prefetch\":" << (options_.gpu_prefetch ? "true" : "false")
                << ",\"cpu_prefetch_before_retouch\":"
                << (options_.cpu_prefetch_before_retouch ? "true" : "false")
                << ",\"elapsed_ms\":" << std::fixed << std::setprecision(6) << milliseconds
                << ",\"bandwidth_gbps\":" << bandwidth
                << ",\"logical_bytes\":" << std::fixed << std::setprecision(0) << logical_bytes
                << ",\"cuda_device_id\":" << device_
                << ",\"cuda_device\":\"" << json_escape(device_name_)
                << "\",\"cpu_page_size\":" << page_size_
                << ",\"correct\":" << (correct ? "true" : "false")
                << ",\"skipped\":" << (skipped ? "true" : "false")
                << ",\"cuda_mem_free\":" << free_bytes
                << ",\"cuda_mem_total\":" << total_bytes
                << ",\"error_message\":\"" << json_escape(message) << "\"}\n";
        output_.flush();
        std::cout << std::left << std::setw(34) << phase << std::right
                  << std::setw(14) << std::fixed << std::setprecision(3) << milliseconds
                  << std::setw(14) << std::setprecision(3) << bandwidth
                  << std::setw(11) << (skipped ? "SKIP" : (correct ? "yes" : "NO")) << '\n';
    }

private:
    const Options &options_;
    std::string run_id_;
    std::string device_name_;
    int device_;
    long page_size_;
    std::ofstream output_;
};

class CudaEventPair {
public:
    CudaEventPair()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaEventPair()
    {
        if (start_) cuda_cleanup(cudaEventDestroy(start_), "cudaEventDestroy(start)");
        if (stop_) cuda_cleanup(cudaEventDestroy(stop_), "cudaEventDestroy(stop)");
    }
    float measure(cudaStream_t stream, const std::function<void()> &operation)
    {
        CUDA_CHECK(cudaEventRecord(start_, stream));
        operation();
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        CUDA_CHECK(cudaDeviceSynchronize());
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }
private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

std::vector<size_t> sample_indices(size_t elements)
{
    std::vector<size_t> indices;
    const size_t span = std::min<size_t>(1024, elements);
    auto add_span = [&](size_t begin) {
        for (size_t i = 0; i < span && begin + i < elements; ++i)
            indices.push_back(begin + i);
    };
    add_span(0);
    add_span(elements / 2 > span / 2 ? elements / 2 - span / 2 : 0);
    add_span(elements > span ? elements - span : 0);
    std::mt19937_64 generator(0x55564d4241534943ULL);
    std::uniform_int_distribution<size_t> distribution(0, elements - 1);
    for (size_t i = 0; i < std::min<size_t>(256, elements); ++i)
        indices.push_back(distribution(generator));
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

bool verify_on_gpu(const float *a,
                   const float *b,
                   const float *c,
                   const size_t *device_indices,
                   size_t sample_count,
                   unsigned int *device_mismatches)
{
    CUDA_CHECK(cudaMemset(device_mismatches, 0, sizeof(*device_mismatches)));
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((sample_count + threads - 1) / threads);
    verify_samples<<<blocks, threads>>>(a, b, c, device_indices, sample_count, device_mismatches);
    CUDA_CHECK(cudaGetLastError());
    unsigned int mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    return mismatches == 0;
}

float launch_timed(const Options &options,
                   const float *a,
                   const float *b,
                   float *c,
                   size_t elements,
                   cudaStream_t stream)
{
    const unsigned int threads = 256;
    const size_t requested_blocks = (elements + threads - 1) / threads;
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>(requested_blocks, 65535));
    CudaEventPair events;
    return events.measure(stream, [&] {
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            vector_add<<<blocks, threads, 0, stream>>>(a, b, c, elements);
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

void run_managed(const Options &options, Recorder &recorder, int device, long page_size)
{
    float *a = nullptr;
    float *b = nullptr;
    float *c = nullptr;
    size_t *device_indices = nullptr;
    unsigned int *device_mismatches = nullptr;
    cudaStream_t stream = nullptr;
    const size_t elements = options.bytes / sizeof(float);
    const auto indices = sample_indices(elements);
    try {
        size_t free_before = 0, total_before = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));
        const auto allocation_start = Clock::now();
        CUDA_CHECK(cudaMallocManaged(&a, options.bytes));
        CUDA_CHECK(cudaMallocManaged(&b, options.bytes));
        CUDA_CHECK(cudaMallocManaged(&c, options.bytes));
        CUDA_CHECK(cudaMalloc(&device_indices, indices.size() * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMemcpy(device_indices, indices.data(), indices.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice));
        const auto allocation_end = Clock::now();
        size_t free_after = 0, total_after = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
        recorder.row("allocation", elapsed_ms(allocation_start, allocation_end),
                     0.0, true, "cudaMemGetInfo is auxiliary, not residency proof",
                     free_after, total_after);

        {
            NvtxRange range("cpu_first_touch");
            const auto start = Clock::now();
            for (size_t i = 0; i < elements; ++i) {
                a[i] = 1.0f;
                b[i] = 2.0f;
                c[i] = 0.0f;
            }
            const auto end = Clock::now();
            recorder.row("cpu_first_touch", elapsed_ms(start, end), 3.0 * options.bytes, true);
        }

        {
            NvtxRange range("kernel_1_demand");
            const float time = launch_timed(options, a, b, c, elements, stream);
            const bool correct = !options.verify || verify_on_gpu(a, b, c, device_indices,
                                                                  indices.size(), device_mismatches);
            recorder.row("kernel_1_demand", time, 3.0 * options.bytes * options.iterations, correct);
            if (!correct) throw std::runtime_error("kernel_1_demand verification failed");
        }
        {
            NvtxRange range("kernel_2_hot");
            const float time = launch_timed(options, a, b, c, elements, stream);
            const bool correct = !options.verify || verify_on_gpu(a, b, c, device_indices,
                                                                  indices.size(), device_mismatches);
            recorder.row("kernel_2_hot", time, 3.0 * options.bytes * options.iterations, correct);
            if (!correct) throw std::runtime_error("kernel_2_hot verification failed");
        }

        if (options.cpu_prefetch_before_retouch && options.cpu_retouch != "none") {
            NvtxRange range("cpu_prefetch_before_retouch");
            const auto start = Clock::now();
            CUDA_CHECK(cudaMemPrefetchAsync(a, options.bytes, cudaCpuDeviceId, stream));
            CUDA_CHECK(cudaMemPrefetchAsync(b, options.bytes, cudaCpuDeviceId, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto end = Clock::now();
            recorder.row("cpu_prefetch_before_retouch", elapsed_ms(start, end),
                         2.0 * options.bytes, true);
        }
        else {
            recorder.row("cpu_prefetch_before_retouch", 0.0, 0.0, true,
                         "skipped by option or no retouch", 0, 0, true);
        }

        {
            NvtxRange range("cpu_retouch");
            const auto start = Clock::now();
            size_t touched = 0;
            if (options.cpu_retouch == "page") {
                const size_t stride = std::max<size_t>(1, static_cast<size_t>(page_size) / sizeof(float));
                for (size_t i = 0; i < elements; i += stride) {
                    a[i] += 1.0f;
                    b[i] += 1.0f;
                    ++touched;
                }
            }
            else if (options.cpu_retouch == "full") {
                for (size_t i = 0; i < elements; ++i) {
                    a[i] += 1.0f;
                    b[i] += 1.0f;
                }
                touched = elements;
            }
            const auto end = Clock::now();
            recorder.row("cpu_retouch", elapsed_ms(start, end),
                         2.0 * touched * sizeof(float), true,
                         options.cpu_retouch == "none" ? "no CPU access requested" : "");
        }

        {
            NvtxRange range("kernel_3_after_cpu_touch");
            const float time = launch_timed(options, a, b, c, elements, stream);
            const bool correct = !options.verify || verify_on_gpu(a, b, c, device_indices,
                                                                  indices.size(), device_mismatches);
            recorder.row("kernel_3_after_cpu_touch", time,
                         3.0 * options.bytes * options.iterations, correct);
            if (!correct) throw std::runtime_error("kernel_3_after_cpu_touch verification failed");
        }

        if (options.gpu_prefetch) {
            {
                NvtxRange range("explicit_gpu_prefetch");
                const auto start = Clock::now();
                CUDA_CHECK(cudaMemPrefetchAsync(a, options.bytes, device, stream));
                CUDA_CHECK(cudaMemPrefetchAsync(b, options.bytes, device, stream));
                CUDA_CHECK(cudaMemPrefetchAsync(c, options.bytes, device, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto end = Clock::now();
                recorder.row("explicit_gpu_prefetch", elapsed_ms(start, end),
                             3.0 * options.bytes, true);
            }
            {
                NvtxRange range("kernel_4_after_gpu_prefetch");
                const float time = launch_timed(options, a, b, c, elements, stream);
                const bool correct = !options.verify || verify_on_gpu(a, b, c, device_indices,
                                                                      indices.size(), device_mismatches);
                recorder.row("kernel_4_after_gpu_prefetch", time,
                             3.0 * options.bytes * options.iterations, correct);
                if (!correct) throw std::runtime_error("kernel_4_after_gpu_prefetch verification failed");
            }
        }
        else {
            recorder.row("explicit_gpu_prefetch", 0.0, 0.0, true, "skipped by option", 0, 0, true);
            recorder.row("kernel_4_after_gpu_prefetch", 0.0, 0.0, true,
                         "skipped because GPU prefetch was disabled", 0, 0, true);
        }
    }
    catch (...) {
        if (stream) cuda_cleanup(cudaStreamDestroy(stream), "cudaStreamDestroy");
        if (device_mismatches) cuda_cleanup(cudaFree(device_mismatches), "cudaFree(mismatches)");
        if (device_indices) cuda_cleanup(cudaFree(device_indices), "cudaFree(indices)");
        if (c) cuda_cleanup(cudaFree(c), "cudaFree(C)");
        if (b) cuda_cleanup(cudaFree(b), "cudaFree(B)");
        if (a) cuda_cleanup(cudaFree(a), "cudaFree(A)");
        throw;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_mismatches));
    CUDA_CHECK(cudaFree(device_indices));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
}

void run_device(const Options &options, Recorder &recorder)
{
    float *a = nullptr;
    float *b = nullptr;
    float *c = nullptr;
    size_t *device_indices = nullptr;
    unsigned int *device_mismatches = nullptr;
    cudaStream_t stream = nullptr;
    const size_t elements = options.bytes / sizeof(float);
    const auto indices = sample_indices(elements);
    std::vector<float> host(elements);
    try {
        const auto start = Clock::now();
        CUDA_CHECK(cudaMalloc(&a, options.bytes));
        CUDA_CHECK(cudaMalloc(&b, options.bytes));
        CUDA_CHECK(cudaMalloc(&c, options.bytes));
        CUDA_CHECK(cudaMalloc(&device_indices, indices.size() * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMemcpy(device_indices, indices.data(), indices.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice));
        recorder.row("allocation", elapsed_ms(start, Clock::now()), 0.0, true);

        std::fill(host.begin(), host.end(), 1.0f);
        CudaEventPair h2d_a;
        const float a_ms = h2d_a.measure(stream, [&] {
            CUDA_CHECK(cudaMemcpyAsync(a, host.data(), options.bytes, cudaMemcpyHostToDevice, stream));
        });
        recorder.row("host_to_device_A", a_ms, options.bytes, true);
        std::fill(host.begin(), host.end(), 2.0f);
        CudaEventPair h2d_b;
        const float b_ms = h2d_b.measure(stream, [&] {
            CUDA_CHECK(cudaMemcpyAsync(b, host.data(), options.bytes, cudaMemcpyHostToDevice, stream));
        });
        recorder.row("host_to_device_B", b_ms, options.bytes, true);
        CUDA_CHECK(cudaMemset(c, 0, options.bytes));

        const float first = launch_timed(options, a, b, c, elements, stream);
        bool correct = !options.verify || verify_on_gpu(a, b, c, device_indices,
                                                        indices.size(), device_mismatches);
        recorder.row("kernel_1_device", first, 3.0 * options.bytes * options.iterations, correct);
        if (!correct) throw std::runtime_error("kernel_1_device verification failed");
        const float second = launch_timed(options, a, b, c, elements, stream);
        correct = !options.verify || verify_on_gpu(a, b, c, device_indices,
                                                   indices.size(), device_mismatches);
        recorder.row("kernel_2_device", second, 3.0 * options.bytes * options.iterations, correct);
        if (!correct) throw std::runtime_error("kernel_2_device verification failed");

        CudaEventPair d2h;
        const float d2h_ms = d2h.measure(stream, [&] {
            CUDA_CHECK(cudaMemcpyAsync(host.data(), c, options.bytes, cudaMemcpyDeviceToHost, stream));
        });
        bool host_correct = true;
        if (options.verify) {
            for (const size_t index : indices) {
                if (std::fabs(host[index] - 3.0f) > 1.0e-5f) {
                    host_correct = false;
                    break;
                }
            }
        }
        recorder.row("device_to_host_C", d2h_ms, options.bytes, host_correct);
        if (!host_correct) throw std::runtime_error("device_to_host_C verification failed");
    }
    catch (...) {
        if (stream) cuda_cleanup(cudaStreamDestroy(stream), "cudaStreamDestroy");
        if (device_mismatches) cuda_cleanup(cudaFree(device_mismatches), "cudaFree(mismatches)");
        if (device_indices) cuda_cleanup(cudaFree(device_indices), "cudaFree(indices)");
        if (c) cuda_cleanup(cudaFree(c), "cudaFree(C)");
        if (b) cuda_cleanup(cudaFree(b), "cudaFree(B)");
        if (a) cuda_cleanup(cudaFree(a), "cudaFree(A)");
        throw;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_mismatches));
    CUDA_CHECK(cudaFree(device_indices));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const Options options = parse_args(argc, argv);
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
        Recorder recorder(options, make_run_id(), properties.name, device, page_size);
        if (options.allocation == "managed")
            run_managed(options, recorder, device, page_size);
        else
            run_device(options, recorder);
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "uvm_vector_add: " << error.what() << '\n';
        return 1;
    }
}
