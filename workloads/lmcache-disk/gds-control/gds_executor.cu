// SPDX-License-Identifier: GPL-2.0
// CUDA 12.9/cuFile compatibility-path benchmark for the GDS control policy.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _FILE_OFFSET_BITS 64

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kLogicalBytes = 24ULL * kMiB;
constexpr std::size_t kFirstBytes = 16ULL * kMiB;
constexpr std::size_t kSecondBytes = 8ULL * kMiB;
constexpr std::uint64_t kNsPerMs = 1000000ULL;
constexpr std::uint64_t kMaxDeferNs = 10ULL * kNsPerMs;

constexpr std::uint32_t kAbiVersion = 1;
constexpr std::uint32_t kOpRead = 0;
constexpr std::uint32_t kOpWrite = 1;
constexpr std::uint32_t kFlagDemand = 1U;
constexpr std::uint32_t kFlagSpeculative = 2U;
constexpr std::uint32_t kFlagRecomputable = 4U;
constexpr std::uint32_t kFlagSafeToDefer = 8U;
constexpr std::uint32_t kActionSubmitNow = 0;
constexpr std::uint32_t kActionDefer = 1;
constexpr std::uint32_t kActionRecompute = 2;
constexpr unsigned long kGpuStorageDecideIoctl = 82;

// Natural alignment is part of the userspace/kernel ABI. Do not pack this.
struct UvmGpuStorageDecideParams {
    std::uint32_t abiVersion;
    std::uint32_t op;
    std::uint32_t requestFlags;
    std::uint32_t inputPriority;
    std::uint64_t requestId;
    std::uint64_t objectId;
    std::uint64_t bytes;
    std::uint64_t tenantId;
    std::uint64_t callerHint;
    std::uint64_t deadlineNs;
    std::uint64_t slackNs;
    std::uint64_t estimatedTransferNs;
    std::uint64_t recomputeNs;
    std::uint32_t queueDepth;
    std::uint32_t hbmPressurePermille;
    std::uint32_t action;
    std::uint32_t outputPriority;
    std::uint64_t deferNs;
    std::uint32_t batchTarget;
    std::uint64_t callerTgid;
    std::uint32_t rmStatus;
};

static_assert(sizeof(UvmGpuStorageDecideParams) == 136,
              "UVM_GPU_STORAGE_DECIDE_PARAMS must be 136 bytes");
static_assert(alignof(UvmGpuStorageDecideParams) == 8, "UVM ABI alignment");
static_assert(offsetof(UvmGpuStorageDecideParams, requestId) == 16,
              "requestId ABI offset");
static_assert(offsetof(UvmGpuStorageDecideParams, action) == 96,
              "action ABI offset");
static_assert(offsetof(UvmGpuStorageDecideParams, deferNs) == 104,
              "deferNs ABI offset");
static_assert(offsetof(UvmGpuStorageDecideParams, callerTgid) == 120,
              "callerTgid ABI offset");
static_assert(offsetof(UvmGpuStorageDecideParams, rmStatus) == 128,
              "rmStatus ABI offset");

struct Decision {
    std::uint32_t action;
    std::uint64_t defer_ns;
    std::uint32_t priority;
    std::uint32_t batch_target;
};

// All four values are consumed/produced asynchronously by cuFile from device
// memory. There is one unique object per physical submission.
struct AsyncIoState {
    std::size_t size;
    off_t file_offset;
    off_t buffer_offset;
    ssize_t result;
};

static_assert(sizeof(AsyncIoState) == 32, "unexpected async metadata layout");

struct CompletionStamp {
    std::uint64_t submit_ns = 0;
    std::uint64_t complete_ns = 0;
    bool urgent_read = false;
};

struct PendingRequest {
    std::size_t sequence;
    UvmGpuStorageDecideParams request;
    std::uint64_t eligible_ns;
    std::uint32_t batch_target;
};

enum class Mode { Fifo, Native, Bpf };

[[noreturn]] void fail(const std::string &message)
{
    std::cerr << "gds_executor: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cufile(CUfileError_t status, const char *operation)
{
    if (status.err == CU_FILE_SUCCESS)
        return;
    std::string message = std::string(operation) + ": " +
                          cufileop_status_error(status.err);
    if (IS_CUDA_ERR(status))
        message += " (CUDA driver error " + std::to_string(status.cu_err) + ")";
    fail(message);
}

std::uint64_t monotonic_ns()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void CUDART_CB record_completion(void *opaque)
{
    static_cast<CompletionStamp *>(opaque)->complete_ns = monotonic_ns();
}

std::uint32_t clamp_priority(std::uint32_t priority)
{
    return std::min(priority, 7U);
}

Decision matched_native_decide(const UvmGpuStorageDecideParams &request)
{
    const std::uint32_t priority = clamp_priority(request.inputPriority);
    if (request.op == kOpRead && (request.requestFlags & kFlagDemand))
        return {kActionSubmitNow, 0, priority, 1};

    if (request.op == kOpRead &&
        (request.requestFlags & kFlagRecomputable) &&
        request.recomputeNs < request.estimatedTransferNs &&
        request.recomputeNs <= request.slackNs)
        return {kActionRecompute, 0, priority, 1};

    if (request.op == kOpRead &&
        (request.requestFlags & kFlagSpeculative) &&
        (request.requestFlags & kFlagSafeToDefer) &&
        request.hbmPressurePermille >= 800U)
        return {kActionDefer, std::min(request.slackNs, kMaxDeferNs),
                priority, 1};

    if (request.op == kOpWrite &&
        (request.requestFlags & kFlagSafeToDefer) &&
        request.hbmPressurePermille >= 600U) {
        const std::uint32_t batch =
            std::max(1U, std::min(request.queueDepth, 64U));
        return {kActionDefer, std::min(request.slackNs, kMaxDeferNs),
                priority, batch};
    }

    return {kActionSubmitNow, 0, priority, 1};
}

UvmGpuStorageDecideParams make_request(std::size_t sequence)
{
    UvmGpuStorageDecideParams request{};
    request.abiVersion = kAbiVersion;
    request.requestId = static_cast<std::uint64_t>(sequence + 1);
    request.objectId = static_cast<std::uint64_t>(sequence);
    request.bytes = kLogicalBytes;
    request.tenantId = 1;
    request.callerHint = sequence % 8;
    request.deadlineNs = 50ULL * kNsPerMs;

    // This exact eight-category cycle is shared unchanged by every mode.
    switch (sequence % 8) {
    case 0: // Demand wins over recompute and defer.
        request.op = kOpRead;
        request.requestFlags = kFlagDemand | kFlagSpeculative |
                               kFlagRecomputable | kFlagSafeToDefer;
        request.inputPriority = 7;
        request.slackNs = 10ULL * kNsPerMs;
        request.estimatedTransferNs = 5ULL * kNsPerMs;
        request.recomputeNs = 1ULL * kNsPerMs;
        request.queueDepth = 8;
        request.hbmPressurePermille = 900;
        break;
    case 1: // Recompute is cheaper and fits within slack.
        request.op = kOpRead;
        request.requestFlags = kFlagSpeculative | kFlagRecomputable |
                               kFlagSafeToDefer;
        request.inputPriority = 6;
        request.slackNs = 4ULL * kNsPerMs;
        request.estimatedTransferNs = 3ULL * kNsPerMs;
        request.recomputeNs = 1ULL * kNsPerMs;
        request.queueDepth = 8;
        request.hbmPressurePermille = 900;
        break;
    case 2: // Speculative safe read at high HBM pressure.
        request.op = kOpRead;
        request.requestFlags = kFlagSpeculative | kFlagSafeToDefer;
        request.inputPriority = 1;
        request.slackNs = 7ULL * kNsPerMs;
        request.estimatedTransferNs = 3ULL * kNsPerMs;
        request.recomputeNs = 5ULL * kNsPerMs;
        request.queueDepth = 8;
        request.hbmPressurePermille = 900;
        break;
    case 3: // Safe write above the write-defer threshold.
        request.op = kOpWrite;
        request.requestFlags = kFlagSafeToDefer;
        request.inputPriority = 2;
        request.slackNs = 6ULL * kNsPerMs;
        request.estimatedTransferNs = 2ULL * kNsPerMs;
        request.recomputeNs = 0;
        request.queueDepth = 32;
        request.hbmPressurePermille = 700;
        break;
    case 4: // Ordinary read defaults to submit now.
        request.op = kOpRead;
        request.requestFlags = 0;
        request.inputPriority = 5;
        request.slackNs = 2ULL * kNsPerMs;
        request.estimatedTransferNs = 3ULL * kNsPerMs;
        request.recomputeNs = 1ULL * kNsPerMs;
        request.queueDepth = 4;
        request.hbmPressurePermille = 500;
        break;
    case 5: // Recomputable read, but recompute is not cheaper.
        request.op = kOpRead;
        request.requestFlags = kFlagRecomputable;
        request.inputPriority = 4;
        request.slackNs = 10ULL * kNsPerMs;
        request.estimatedTransferNs = 3ULL * kNsPerMs;
        request.recomputeNs = 4ULL * kNsPerMs;
        request.queueDepth = 4;
        request.hbmPressurePermille = 500;
        break;
    case 6: // Safe speculative read just below its pressure threshold.
        request.op = kOpRead;
        request.requestFlags = kFlagSpeculative | kFlagSafeToDefer;
        request.inputPriority = 3;
        request.slackNs = 8ULL * kNsPerMs;
        request.estimatedTransferNs = 3ULL * kNsPerMs;
        request.recomputeNs = 5ULL * kNsPerMs;
        request.queueDepth = 8;
        request.hbmPressurePermille = 799;
        break;
    default: // Safe write just below its pressure threshold.
        request.op = kOpWrite;
        request.requestFlags = kFlagSafeToDefer;
        request.inputPriority = 0;
        request.slackNs = 5ULL * kNsPerMs;
        request.estimatedTransferNs = 2ULL * kNsPerMs;
        request.recomputeNs = 0;
        request.queueDepth = 16;
        request.hbmPressurePermille = 599;
        break;
    }
    return request;
}

Decision bpf_decide(int uvm_fd, UvmGpuStorageDecideParams request)
{
    if (ioctl(uvm_fd, kGpuStorageDecideIoctl, &request) != 0)
        fail(std::string("UVM_GPU_STORAGE_DECIDE ioctl: ") +
             std::strerror(errno));
    if (request.rmStatus != 0)
        fail("UVM_GPU_STORAGE_DECIDE rmStatus=" +
             std::to_string(request.rmStatus));
    if (request.action > kActionRecompute)
        fail("UVM_GPU_STORAGE_DECIDE returned unknown action " +
             std::to_string(request.action));
    return {request.action, request.deferNs, request.outputPriority,
            request.batchTarget};
}

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const std::size_t low = static_cast<std::size_t>(position);
    const std::size_t high = std::min(low + 1, values.size() - 1);
    const double weight = position - static_cast<double>(low);
    return values[low] * (1.0 - weight) + values[high] * weight;
}

const char *mode_name(Mode mode)
{
    switch (mode) {
    case Mode::Fifo: return "fifo";
    case Mode::Native: return "native";
    case Mode::Bpf: return "bpf";
    }
    return "unknown";
}

Mode parse_mode(const std::string &value)
{
    if (value == "fifo")
        return Mode::Fifo;
    if (value == "native")
        return Mode::Native;
    if (value == "bpf")
        return Mode::Bpf;
    fail("--mode must be fifo, native, or bpf");
}

std::size_t parse_requests(const char *text)
{
    if (!text || !*text || *text == '-')
        fail("--requests must be a positive integer");
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value == 0 ||
        value > std::numeric_limits<std::size_t>::max())
        fail("--requests must be a positive integer");
    return static_cast<std::size_t>(value);
}

struct Arguments {
    Mode mode = Mode::Fifo;
    bool have_mode = false;
    std::string file = "/tmp/gds-control-cufile.bin";
    std::size_t requests = 64;
};

Arguments parse_arguments(int argc, char **argv)
{
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--mode" && i + 1 < argc) {
            args.mode = parse_mode(argv[++i]);
            args.have_mode = true;
        }
        else if (option == "--file" && i + 1 < argc) {
            args.file = argv[++i];
            if (args.file.empty())
                fail("--file must not be empty");
        }
        else if (option == "--requests" && i + 1 < argc) {
            args.requests = parse_requests(argv[++i]);
        }
        else if (option == "--help" || option == "-h") {
            std::cout << "usage: " << argv[0]
                      << " --mode fifo|native|bpf [--file PATH]"
                         " [--requests COUNT]\n";
            std::exit(EXIT_SUCCESS);
        }
        else {
            fail("unknown or incomplete option: " + option);
        }
    }
    if (!args.have_mode)
        fail("--mode is required");
    if (args.requests >
        static_cast<std::size_t>(std::numeric_limits<off_t>::max() /
                                 static_cast<off_t>(kLogicalBytes)))
        fail("--requests makes the backing file too large");
    return args;
}

} // namespace

int main(int argc, char **argv)
{
    const Arguments args = parse_arguments(argc, argv);
    const off_t file_bytes = static_cast<off_t>(args.requests * kLogicalBytes);

    const int file_fd = open(args.file.c_str(), O_CREAT | O_RDWR | O_DIRECT |
                                                O_CLOEXEC, 0644);
    if (file_fd < 0)
        fail("open(" + args.file + "): " + std::strerror(errno));
    if (ftruncate(file_fd, file_bytes) != 0)
        fail("ftruncate(" + args.file + "): " + std::strerror(errno));
    const int allocation_status = posix_fallocate(file_fd, 0, file_bytes);
    if (allocation_status != 0)
        fail("posix_fallocate(" + args.file + "): " +
             std::strerror(allocation_status));

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "initialize CUDA context");

    // The transport is deliberately fixed across all modes so the only
    // experimental difference is the scheduling decision and stream route.
    check_cufile(cuFileSetParameterBool(CUFILE_PARAM_FORCE_COMPAT_MODE, true),
                 "force cuFile compatibility mode");
    check_cufile(cuFileDriverOpen(), "cuFileDriverOpen");

    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = file_fd;
    CUfileHandle_t file_handle = nullptr;
    check_cufile(cuFileHandleRegister(&file_handle, &descriptor),
                 "cuFileHandleRegister");

    void *urgent_buffers[2] = {nullptr, nullptr};
    void *background_buffers[2] = {nullptr, nullptr};
    constexpr std::size_t kPartSizes[2] = {kFirstBytes, kSecondBytes};
    for (std::size_t part = 0; part < 2; ++part) {
        check_cuda(cudaMalloc(&urgent_buffers[part], kPartSizes[part]),
                   "cudaMalloc urgent split buffer");
        check_cuda(cudaMalloc(&background_buffers[part], kPartSizes[part]),
                   "cudaMalloc background split buffer");
        check_cufile(cuFileBufRegister(urgent_buffers[part],
                                       kPartSizes[part], 0),
                     "cuFileBufRegister urgent split buffer");
        check_cufile(cuFileBufRegister(background_buffers[part],
                                       kPartSizes[part], 0),
                     "cuFileBufRegister background split buffer");
    }

    cudaStream_t urgent_stream = nullptr;
    cudaStream_t background_stream = nullptr;
    if (args.mode == Mode::Fifo) {
        check_cuda(cudaStreamCreateWithFlags(&urgent_stream,
                                              cudaStreamNonBlocking),
                   "create FIFO stream");
    }
    else {
        int least_priority = 0;
        int greatest_priority = 0;
        check_cuda(cudaDeviceGetStreamPriorityRange(&least_priority,
                                                     &greatest_priority),
                   "get CUDA stream priority range");
        check_cuda(cudaStreamCreateWithPriority(&urgent_stream,
                                                 cudaStreamNonBlocking,
                                                 greatest_priority),
                   "create urgent stream");
        check_cuda(cudaStreamCreateWithPriority(&background_stream,
                                                 cudaStreamNonBlocking,
                                                 least_priority),
                   "create background stream");
    }

    constexpr unsigned kStreamFlags = CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;
    check_cufile(cuFileStreamRegister(urgent_stream, kStreamFlags),
                 "cuFileStreamRegister urgent/FIFO");
    if (background_stream)
        check_cufile(cuFileStreamRegister(background_stream, kStreamFlags),
                     "cuFileStreamRegister background");

    for (std::size_t part = 0; part < 2; ++part) {
        check_cuda(cudaMemsetAsync(urgent_buffers[part], 0x5a,
                                   kPartSizes[part], urgent_stream),
                   "initialize urgent split buffer");
        if (background_stream)
            check_cuda(cudaMemsetAsync(background_buffers[part], 0xa5,
                                       kPartSizes[part], background_stream),
                       "initialize background split buffer");
    }
    check_cuda(cudaDeviceSynchronize(), "synchronize buffer initialization");

    if (args.requests > std::numeric_limits<std::size_t>::max() / 2)
        fail("--requests overflows the submission count");
    const std::size_t state_count = args.requests * 2;
    std::vector<AsyncIoState> initial_states(state_count);
    for (std::size_t i = 0; i < args.requests; ++i) {
        const off_t base = static_cast<off_t>(i * kLogicalBytes);
        initial_states[2 * i] =
            {kFirstBytes, base, 0, static_cast<ssize_t>(-1)};
        initial_states[2 * i + 1] =
            {kSecondBytes, base + static_cast<off_t>(kFirstBytes),
             0, static_cast<ssize_t>(-1)};
    }
    AsyncIoState *device_states = nullptr;
    // Managed CUDA storage is used because the stream API evaluates these
    // pointer arguments asynchronously but libcufile also inspects them while
    // setting up an operation. It is CUDA device-resident storage without the
    // invalid host dereference caused by passing a raw cudaMalloc pointer.
    check_cuda(cudaMallocManaged(&device_states,
                                 state_count * sizeof(AsyncIoState)),
               "cudaMallocManaged async submission state");
    check_cuda(cudaMemcpy(device_states, initial_states.data(),
                          state_count * sizeof(AsyncIoState),
                          cudaMemcpyHostToDevice),
               "copy async submission state to device");

    int uvm_fd = -1;
    if (args.mode == Mode::Bpf) {
        uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        if (uvm_fd < 0)
            fail(std::string("open(/dev/nvidia-uvm): ") +
                 std::strerror(errno));
    }

    std::vector<CompletionStamp> completion(args.requests);
    std::deque<PendingRequest> pending;
    std::uint64_t decision_total_ns = 0;
    std::size_t submit_count = 0;
    std::size_t defer_count = 0;
    std::size_t recompute_count = 0;
    std::size_t read_count = 0;
    std::size_t write_count = 0;
    std::size_t urgent_read_count = 0;
    std::size_t io_chunk_count = 0;

    auto issue_chunk = [&](std::size_t sequence,
                           const UvmGpuStorageDecideParams &request,
                           cudaStream_t stream, void *const buffers[2]) {
        completion[sequence].submit_ns = monotonic_ns();
        completion[sequence].urgent_read =
            request.op == kOpRead && (request.requestFlags & kFlagDemand);
        if (completion[sequence].urgent_read)
            ++urgent_read_count;

        for (std::size_t part = 0; part < 2; ++part) {
            const std::size_t index = 2 * sequence + part;
            AsyncIoState *state = device_states + index;
            CUfileError_t status;
            if (request.op == kOpRead) {
                status = cuFileReadAsync(file_handle, buffers[part],
                                         &state->size, &state->file_offset,
                                         &state->buffer_offset,
                                         &state->result, stream);
            }
            else {
                status = cuFileWriteAsync(file_handle, buffers[part],
                                          &state->size, &state->file_offset,
                                          &state->buffer_offset,
                                          &state->result, stream);
            }
            check_cufile(status, request.op == kOpRead ?
                         "cuFileReadAsync" : "cuFileWriteAsync");
        }
        check_cuda(cudaLaunchHostFunc(stream, record_completion,
                                      &completion[sequence]),
                   "enqueue logical-chunk completion timestamp");
    };

    // First release full batches whose members have all reached their bounded
    // defer time. Then release any individually expired entries as the
    // deadline fallback. This never sleeps per request; callers revisit the
    // queue while normal submission work proceeds and yield only while doing
    // the final drain.
    auto release_pending = [&]() {
        bool made_progress = true;
        while (made_progress && !pending.empty()) {
            made_progress = false;
            const std::uint64_t now = monotonic_ns();

            for (std::size_t start = 0; start < pending.size(); ++start) {
                const std::size_t target = pending[start].batch_target;
                if (target <= 1)
                    continue;
                std::array<std::size_t, 64> members{};
                std::size_t member_count = 0;
                for (std::size_t candidate = start;
                     candidate < pending.size() && member_count < target;
                     ++candidate) {
                    if (pending[candidate].batch_target == target &&
                        pending[candidate].request.op ==
                            pending[start].request.op)
                        members[member_count++] = candidate;
                }
                if (member_count < target)
                    continue;
                bool batch_is_eligible = true;
                for (std::size_t index = 0; index < member_count; ++index) {
                    const std::size_t member = members[index];
                    if (pending[member].eligible_ns > now) {
                        batch_is_eligible = false;
                        break;
                    }
                }
                if (!batch_is_eligible)
                    continue;
                for (std::size_t index = 0; index < member_count; ++index) {
                    const std::size_t member = members[index];
                    const PendingRequest &entry = pending[member];
                    issue_chunk(entry.sequence, entry.request,
                                background_stream, background_buffers);
                }
                for (std::size_t index = member_count; index > 0; --index)
                    pending.erase(pending.begin() + members[index - 1]);
                made_progress = true;
                break;
            }
            if (made_progress)
                continue;

            for (auto entry = pending.begin(); entry != pending.end(); ++entry) {
                if (entry->eligible_ns > now)
                    continue;
                issue_chunk(entry->sequence, entry->request,
                            background_stream, background_buffers);
                pending.erase(entry);
                made_progress = true;
                break;
            }
        }
    };

    const std::uint64_t elapsed_start_ns = monotonic_ns();
    for (std::size_t i = 0; i < args.requests; ++i) {
        const UvmGpuStorageDecideParams request = make_request(i);
        const std::uint64_t decision_start_ns = monotonic_ns();
        Decision decision{};
        if (args.mode == Mode::Fifo)
            decision = {kActionSubmitNow, 0,
                        clamp_priority(request.inputPriority), 1};
        else if (args.mode == Mode::Native)
            decision = matched_native_decide(request);
        else
            decision = bpf_decide(uvm_fd, request);
        decision_total_ns += monotonic_ns() - decision_start_ns;

        if (decision.action == kActionSubmitNow)
            ++submit_count;
        else if (decision.action == kActionDefer)
            ++defer_count;
        else
            ++recompute_count;

        if (decision.action == kActionRecompute)
            continue;

        ++io_chunk_count;
        if (request.op == kOpRead)
            ++read_count;
        else
            ++write_count;

        if (args.mode != Mode::Fifo && decision.action == kActionDefer) {
            pending.push_back({i, request,
                               monotonic_ns() +
                                   std::min(decision.defer_ns, kMaxDeferNs),
                               std::max(1U,
                                        std::min(decision.batch_target, 64U))});
            release_pending();
            continue;
        }
        issue_chunk(i, request, urgent_stream, urgent_buffers);
        release_pending();
    }

    while (!pending.empty()) {
        release_pending();
        if (!pending.empty())
            std::this_thread::yield();
    }

    check_cuda(cudaStreamSynchronize(urgent_stream),
               "synchronize urgent/FIFO stream");
    if (background_stream)
        check_cuda(cudaStreamSynchronize(background_stream),
                   "synchronize background stream");
    const std::uint64_t elapsed_end_ns = monotonic_ns();

    std::vector<double> urgent_read_us;
    urgent_read_us.reserve(urgent_read_count);
    for (const CompletionStamp &stamp : completion) {
        if (!stamp.urgent_read)
            continue;
        if (stamp.complete_ns < stamp.submit_ns)
            fail("invalid urgent-read completion timestamp");
        urgent_read_us.push_back(
            static_cast<double>(stamp.complete_ns - stamp.submit_ns) / 1000.0);
    }

    const double elapsed_s =
        static_cast<double>(elapsed_end_ns - elapsed_start_ns) / 1.0e9;
    const double io_gib =
        static_cast<double>(io_chunk_count * kLogicalBytes) /
        static_cast<double>(1ULL << 30);
    const double gib_per_s = elapsed_s > 0.0 ? io_gib / elapsed_s : 0.0;
    const double decision_total_us =
        static_cast<double>(decision_total_ns) / 1000.0;
    const double decision_mean_us =
        decision_total_us / static_cast<double>(args.requests);

    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"" << mode_name(args.mode)
              << "\",\"transport\":\"cufile_compatibility\""
              << ",\"requests\":" << args.requests
              << ",\"logical_chunk_bytes\":" << kLogicalBytes
              << ",\"split_bytes\":[" << kFirstBytes << ','
              << kSecondBytes << ']'
              << ",\"elapsed_s\":" << elapsed_s
              << ",\"gib_per_s\":" << gib_per_s
              << ",\"urgent_read_p50_us\":"
              << percentile(urgent_read_us, 0.50)
              << ",\"urgent_read_p99_us\":"
              << percentile(urgent_read_us, 0.99)
              << ",\"urgent_read_count\":" << urgent_read_count
              << ",\"submit_count\":" << submit_count
              << ",\"defer_count\":" << defer_count
              << ",\"recompute_count\":" << recompute_count
              << ",\"read_io_count\":" << read_count
              << ",\"write_io_count\":" << write_count
              << ",\"io_chunk_count\":" << io_chunk_count
              << ",\"io_submission_count\":" << io_chunk_count * 2
              << ",\"io_bytes\":" << io_chunk_count * kLogicalBytes
              << ",\"decision_total_us\":" << decision_total_us
              << ",\"decision_mean_us\":" << decision_mean_us
              << "}\n";

    check_cufile(cuFileStreamDeregister(urgent_stream),
                 "cuFileStreamDeregister urgent/FIFO");
    if (background_stream)
        check_cufile(cuFileStreamDeregister(background_stream),
                     "cuFileStreamDeregister background");
    for (std::size_t part = 0; part < 2; ++part) {
        check_cufile(cuFileBufDeregister(urgent_buffers[part]),
                     "cuFileBufDeregister urgent split buffer");
        check_cufile(cuFileBufDeregister(background_buffers[part]),
                     "cuFileBufDeregister background split buffer");
    }
    cuFileHandleDeregister(file_handle);
    check_cuda(cudaFree(device_states), "cudaFree async submission state");
    for (std::size_t part = 0; part < 2; ++part) {
        check_cuda(cudaFree(urgent_buffers[part]),
                   "cudaFree urgent split buffer");
        check_cuda(cudaFree(background_buffers[part]),
                   "cudaFree background split buffer");
    }
    check_cuda(cudaStreamDestroy(urgent_stream), "destroy urgent/FIFO stream");
    if (background_stream)
        check_cuda(cudaStreamDestroy(background_stream),
                   "destroy background stream");
    cuFileDriverClose();
    if (uvm_fd >= 0)
        close(uvm_fd);
    close(file_fd);
    return 0;
}
