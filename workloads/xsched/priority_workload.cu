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

std::string read_command()
{
    std::string command;
    if (!std::getline(std::cin, command)) fail("stdin", "orchestrator closed command pipe");
    return command;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 9) {
        std::fprintf(stderr,
            "usage: %s ROLE PROCESS_ID STREAMS TASKS REPS BLOCKS THREADS WAIT_FOR_START\n", argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    if (role != "lc" && role != "be") fail("ROLE", "must be lc or be");
    const int process_id = static_cast<int>(parse_u64(argv[2], "PROCESS_ID"));
    const int stream_count = static_cast<int>(parse_u64(argv[3], "STREAMS"));
    const int tasks_per_stream = static_cast<int>(parse_u64(argv[4], "TASKS"));
    const unsigned long long reps = parse_u64(argv[5], "REPS");
    const int grid_blocks = static_cast<int>(parse_u64(argv[6], "BLOCKS"));
    const int threads = static_cast<int>(parse_u64(argv[7], "THREADS"));
    if (std::strcmp(argv[8], "0") != 0 && std::strcmp(argv[8], "1") != 0)
        fail("WAIT_FOR_START", "must be 0 or 1");
    const bool wait_for_start = std::strcmp(argv[8], "1") == 0;
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

    const size_t sink_count = static_cast<size_t>(total_tasks) * grid_blocks * threads;
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
    std::cout << "{\"event\":\"result\",\"role\":\"" << role
              << "\",\"process_id\":" << process_id
              << ",\"completion_host_ns\":" << completion_host_ns
              << ",\"outputs_validated\":" << host_sink.size() << ",\"samples\":[";
    for (int i = 0; i < total_tasks; ++i) {
        const auto &stamp = host_stamps[i];
        if (stamp.started != 1 || stamp.blocks_done != static_cast<unsigned int>(grid_blocks)
            || stamp.entry_ns < submit_ns[i] || stamp.exit_ns < stamp.entry_ns) {
            fail("timestamp/correctness", "invalid task stamp");
        }
        const uint64_t queue_ns = stamp.entry_ns - submit_ns[i];
        min_queue_ns = std::min(min_queue_ns, queue_ns);
        max_queue_ns = std::max(max_queue_ns, queue_ns);
        if (i) std::cout << ',';
        std::cout << "{\"submit_ns\":" << submit_ns[i]
                  << ",\"entry_ns\":" << stamp.entry_ns
                  << ",\"exit_ns\":" << stamp.exit_ns << '}';
    }
    std::cout << "],\"min_queue_ns\":" << min_queue_ns
              << ",\"max_queue_ns\":" << max_queue_ns << "}" << std::endl;

    for (auto stream : streams) check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(device_sink), "cudaFree");
    check_cuda(cudaFreeHost(host_stamps), "cudaFreeHost");
    return 0;
}
