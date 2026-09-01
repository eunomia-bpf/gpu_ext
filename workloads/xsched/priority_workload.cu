#include <cuda_runtime.h>
#include <cupti.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct TaskStamp {
    unsigned long long entry_ns;
    unsigned long long exit_ns;
    unsigned int blocks_done;
    unsigned int started;
};

__device__ __forceinline__ unsigned long long globaltimer_ns()
{
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void compute_task(TaskStamp *stamps, float *sink, int task,
                             int grid_blocks, unsigned long long reps)
{
    TaskStamp *stamp = &stamps[task];
    if (threadIdx.x == 0) {
        atomicMin(&stamp->entry_ns, globaltimer_ns());
        __threadfence_system();
        stamp->started = 1;
        __threadfence_system();
    }

    float x = 1.0f + static_cast<float>((blockIdx.x * blockDim.x + threadIdx.x) & 31) * 0.001f;
    #pragma unroll 1
    for (unsigned long long i = 0; i < reps; ++i) {
        x = fmaf(x, 1.00000011920928955078125f, 0.00000095367431640625f);
    }
    sink[static_cast<size_t>(task) * grid_blocks * blockDim.x
         + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = x;
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int old = atomicAdd(&stamp->blocks_done, 1U);
        if (old + 1U == static_cast<unsigned int>(grid_blocks)) {
            stamp->exit_ns = globaltimer_ns();
            __threadfence_system();
        }
    }
}

__global__ void capture_globaltimer(unsigned long long *value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *value = globaltimer_ns();
        __threadfence_system();
    }
}

[[noreturn]] void fail(const char *what, const char *detail)
{
    std::fprintf(stderr, "fatal: %s: %s\n", what, detail);
    std::exit(2);
}

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) fail(what, cudaGetErrorString(status));
}

void check_cupti(CUptiResult status, const char *what)
{
    if (status == CUPTI_SUCCESS) return;
    const char *text = nullptr;
    cuptiGetResultString(status, &text);
    fail(what, text ? text : "unknown CUPTI error");
}

uint64_t monotonic_raw_ns()
{
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) fail("clock_gettime", std::strerror(errno));
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

unsigned long long parse_u64(const char *text, const char *name)
{
    char *end = nullptr;
    errno = 0;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0' || value == 0) fail(name, "expected a positive integer");
    return value;
}

int parse_positive_int(const char *text, const char *name)
{
    const unsigned long long value = parse_u64(text, name);
    if (value > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        fail(name, "value exceeds INT_MAX");
    return static_cast<int>(value);
}

size_t checked_multiply(size_t left, size_t right, const char *name)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
        fail(name, "size multiplication overflow");
    return left * right;
}

long long parse_i64(const char *text, const char *name)
{
    char *end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno || end == text || *end != '\0') fail(name, "expected an integer");
    return value;
}

uint64_t apply_clock_offset(uint64_t timestamp, long long offset)
{
    if (offset < 0) {
        const uint64_t magnitude = static_cast<uint64_t>(-(offset + 1)) + 1;
        if (magnitude > timestamp) fail("clock offset", "negative offset underflow");
        return timestamp - magnitude;
    }
    const uint64_t magnitude = static_cast<uint64_t>(offset);
    if (timestamp > std::numeric_limits<uint64_t>::max() - magnitude)
        fail("clock offset", "positive offset overflow");
    return timestamp + magnitude;
}

int run_clock_probe()
{
    check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    unsigned long long *host_value = nullptr;
    unsigned long long *device_value = nullptr;
    check_cuda(cudaHostAlloc(&host_value, sizeof(*host_value), cudaHostAllocMapped),
               "cudaHostAlloc clock probe");
    check_cuda(cudaHostGetDevicePointer(&device_value, host_value, 0),
               "cudaHostGetDevicePointer clock probe");
    check_cuda(cudaFree(nullptr), "CUDA context initialization");

    long long best_low = 0;
    long long best_high = 0;
    uint64_t best_width = std::numeric_limits<uint64_t>::max();
    for (int sample = 0; sample < 16; ++sample) {
        *host_value = 0;
        uint64_t before = 0;
        uint64_t after = 0;
        check_cupti(cuptiGetTimestamp(&before), "cuptiGetTimestamp before probe");
        capture_globaltimer<<<1, 1>>>(device_value);
        check_cuda(cudaPeekAtLastError(), "capture_globaltimer launch");
        check_cuda(cudaDeviceSynchronize(), "capture_globaltimer synchronize");
        check_cupti(cuptiGetTimestamp(&after), "cuptiGetTimestamp after probe");
        if (*host_value == 0 || after < before) fail("clock probe", "invalid timestamps");
        const long long low = static_cast<long long>(*host_value) - static_cast<long long>(after);
        const long long high = static_cast<long long>(*host_value) - static_cast<long long>(before);
        const uint64_t width = static_cast<uint64_t>(high - low);
        if (width < best_width) {
            best_low = low;
            best_high = high;
            best_width = width;
        }
    }
    const long long offset = best_low + (best_high - best_low) / 2;
    std::cout << "{\"event\":\"clock_probe\",\"offset_ns\":" << offset
              << ",\"offset_low_ns\":" << best_low
              << ",\"offset_high_ns\":" << best_high
              << ",\"uncertainty_ns\":" << (best_width + 1) / 2 << "}" << std::endl;
    check_cuda(cudaFreeHost(host_value), "cudaFreeHost clock probe");
    return 0;
}

std::string read_command()
{
    std::string command;
    if (!std::getline(std::cin, command)) fail("stdin", "orchestrator closed command pipe");
    return command;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--clock-probe") == 0) return run_clock_probe();
    if (argc != 10) {
        std::fprintf(stderr,
            "usage: %s ROLE PROCESS_ID STREAMS TASKS REPS BLOCKS THREADS "
            "WAIT_FOR_START CLOCK_OFFSET_NS\n", argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    if (role != "lc" && role != "be") fail("ROLE", "must be lc or be");
    const int process_id = parse_positive_int(argv[2], "PROCESS_ID");
    const int stream_count = parse_positive_int(argv[3], "STREAMS");
    const int tasks_per_stream = parse_positive_int(argv[4], "TASKS");
    const unsigned long long reps = parse_u64(argv[5], "REPS");
    const int grid_blocks = parse_positive_int(argv[6], "BLOCKS");
    const int threads = parse_positive_int(argv[7], "THREADS");
    if (std::strcmp(argv[8], "0") != 0 && std::strcmp(argv[8], "1") != 0)
        fail("WAIT_FOR_START", "must be 0 or 1");
    const bool wait_for_start = std::strcmp(argv[8], "1") == 0;
    const long long clock_offset_ns = parse_i64(argv[9], "CLOCK_OFFSET_NS");
    if (stream_count > 16 || tasks_per_stream > 1000 || threads > 1024) fail("shape", "out of range");

    check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != 1) fail("device count", "experiment requires exactly one visible GPU");
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    const int total_tasks = stream_count * tasks_per_stream;
    const size_t stamp_bytes = static_cast<size_t>(total_tasks) * sizeof(TaskStamp);
    TaskStamp *host_stamps = nullptr;
    TaskStamp *device_stamps = nullptr;
    check_cuda(cudaHostAlloc(&host_stamps, stamp_bytes, cudaHostAllocMapped), "cudaHostAlloc stamps");
    for (int i = 0; i < total_tasks; ++i) host_stamps[i] = {~0ULL, 0, 0, 0};
    check_cuda(cudaHostGetDevicePointer(&device_stamps, host_stamps, 0), "cudaHostGetDevicePointer");

    const size_t sink_count = checked_multiply(
        checked_multiply(static_cast<size_t>(total_tasks), static_cast<size_t>(grid_blocks),
                         "sink"),
        static_cast<size_t>(threads), "sink");
    if (sink_count > (1ULL << 29)) fail("sink", "allocation would exceed safety cap");
    float *device_sink = nullptr;
    check_cuda(cudaMalloc(&device_sink, sink_count * sizeof(float)), "cudaMalloc sink");

    std::vector<cudaStream_t> streams(stream_count);
    for (auto &stream : streams) check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check_cuda(cudaDeviceSynchronize(), "initial cudaDeviceSynchronize");

    std::cout << "{\"event\":\"ready\",\"role\":\"" << role
              << "\",\"process_id\":" << process_id
              << ",\"streams\":" << stream_count
              << ",\"tasks\":" << tasks_per_stream << "}" << std::endl;
    if (read_command() != "GO") fail("command", "expected GO");

    std::vector<uint64_t> submit_ns(total_tasks, 0);
    auto launch_one = [&](int stream_idx, int task_idx) {
        const int flat = stream_idx * tasks_per_stream + task_idx;
        check_cupti(cuptiGetTimestamp(&submit_ns[flat]), "cuptiGetTimestamp launch");
        compute_task<<<grid_blocks, threads, 0, streams[stream_idx]>>>(
            device_stamps, device_sink, flat, grid_blocks, reps);
        check_cuda(cudaPeekAtLastError(), "compute_task launch");
    };

    for (int stream = 0; stream < stream_count; ++stream) launch_one(stream, 0);
    if (wait_for_start) {
        for (;;) {
            const volatile unsigned int *started = &host_stamps[0].started;
            const volatile unsigned long long *finished = &host_stamps[0].exit_ns;
            if (*started != 0 && *finished == 0) break;
        }
        std::cout << "{\"event\":\"running\",\"role\":\"" << role
                  << "\",\"process_id\":" << process_id
                  << ",\"host_ns\":" << monotonic_raw_ns()
                  << ",\"stream0_task0_active\":true}" << std::endl;
    }
    for (int stream = 0; stream < stream_count; ++stream) {
        for (int task = 1; task < tasks_per_stream; ++task) launch_one(stream, task);
    }
    for (auto stream : streams) check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const uint64_t completion_host_ns = monotonic_raw_ns();

    std::vector<float> host_sink(sink_count);
    check_cuda(cudaMemcpy(host_sink.data(), device_sink, sink_count * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy sink");
    std::vector<float> expected_by_lane(32);
    for (size_t lane = 0; lane < expected_by_lane.size(); ++lane) {
        float value = 1.0f + static_cast<float>(lane) * 0.001f;
        for (unsigned long long i = 0; i < reps; ++i) {
            value = std::fmaf(value, 1.00000011920928955078125f, 0.00000095367431640625f);
        }
        expected_by_lane[lane] = value;
    }
    for (size_t index = 0; index < host_sink.size(); ++index) {
        const float value = host_sink[index];
        const size_t local_index = index % (static_cast<size_t>(grid_blocks) * threads);
        if (!std::isfinite(value) || value <= 0.0f
            || value != expected_by_lane[local_index & 31U]) {
            fail("correctness", "output differs from the per-lane recurrence");
        }
    }

    uint64_t min_queue_ns = std::numeric_limits<uint64_t>::max();
    uint64_t max_queue_ns = 0;
    std::vector<uint64_t> submit_gpu_ns(total_tasks);
    for (int i = 0; i < total_tasks; ++i) {
        const auto &stamp = host_stamps[i];
        submit_gpu_ns[i] = apply_clock_offset(submit_ns[i], clock_offset_ns);
        if (stamp.started != 1 || stamp.blocks_done != static_cast<unsigned int>(grid_blocks)
            || stamp.entry_ns < submit_gpu_ns[i] || stamp.exit_ns < stamp.entry_ns) {
            std::fprintf(stderr,
                "fatal: timestamp/correctness: task=%d submit_cupti_ns=%llu "
                "submit_gpu_ns=%llu entry_ns=%llu exit_ns=%llu started=%u "
                "blocks_done=%u expected_blocks=%d clock_offset_ns=%lld\n",
                i, static_cast<unsigned long long>(submit_ns[i]),
                static_cast<unsigned long long>(submit_gpu_ns[i]), stamp.entry_ns,
                stamp.exit_ns, stamp.started, stamp.blocks_done, grid_blocks,
                clock_offset_ns);
            return 2;
        }
    }
    std::cout << "{\"event\":\"result\",\"role\":\"" << role
              << "\",\"process_id\":" << process_id
              << ",\"completion_host_ns\":" << completion_host_ns
              << ",\"outputs_validated\":" << host_sink.size() << ",\"samples\":[";
    for (int i = 0; i < total_tasks; ++i) {
        const auto &stamp = host_stamps[i];
        const uint64_t queue_ns = stamp.entry_ns - submit_gpu_ns[i];
        min_queue_ns = std::min(min_queue_ns, queue_ns);
        max_queue_ns = std::max(max_queue_ns, queue_ns);
        if (i) std::cout << ',';
        std::cout << "{\"submit_ns\":" << submit_gpu_ns[i]
                  << ",\"submit_cupti_ns\":" << submit_ns[i]
                  << ",\"entry_ns\":" << stamp.entry_ns
                  << ",\"exit_ns\":" << stamp.exit_ns << '}';
    }
    std::cout << "],\"min_queue_ns\":" << min_queue_ns
              << ",\"max_queue_ns\":" << max_queue_ns
              << ",\"clock_offset_ns\":" << clock_offset_ns << "}" << std::endl;

    for (auto stream : streams) check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(device_sink), "cudaFree");
    check_cuda(cudaFreeHost(host_stamps), "cudaFreeHost");
    return 0;
}
