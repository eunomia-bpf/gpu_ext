// SPDX-License-Identifier: GPL-2.0
// Standalone, bounded CUDA-context/owned-transport canary; never a performance
// result or a substitute for GPReempt's GDRCopy/model/hint workload.
#include <cuda.h>
#include "nvos.h"
#include "ctrl/ctrla06c.h"
#include "ctrl/ctrl2080/ctrl2080fifo.h"
#include "nv-gpreempt-transport.h"
#include "gpreempt_bridge.h"
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits.h>
#include <mutex>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;
using Channels = NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS;
constexpr unsigned long query_ioctl = 0xc0204660UL;
static_assert(sizeof(NVOS54_PARAMETERS) == 32 && sizeof(Channels) == 536 &&
              sizeof(NVA06C_CTRL_TIMESLICE_PARAMS) == 8, "575 transport ABI");
static_assert(offsetof(NVOS54_PARAMETERS, status) == 28 &&
              offsetof(Channels, hClientList) == 24 &&
              offsetof(Channels, hChannelList) == 280, "575 ABI offsets");

static std::mutex output_mutex, creation_mutex;
static std::atomic<unsigned> assertions{0}, negatives{0}, values{0};
static void require(bool condition, const char *message)
{ ++assertions; if (!condition) throw std::runtime_error(message); }
static void cuda(CUresult status, const char *operation)
{
    if (status == CUDA_SUCCESS) return;
    const char *name = nullptr;
    cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown CUDA error"));
}
struct Control {
    int fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    Control() { if (fd < 0) throw std::runtime_error("open /dev/nvidiactl failed"); }
    ~Control() { close(fd); }
};
struct Reply { int rc, error; NVOS54_PARAMETERS args; };
static Reply invoke(Control &control, NVOS54_PARAMETERS args, const char *label,
                    unsigned long command = query_ioctl)
{
    args.status = 0xffffffffU;
    errno = 0;
    int rc = ioctl(control.fd, command, &args), saved_errno = errno;
    std::lock_guard<std::mutex> guard(output_mutex);
    std::printf("{\"event\":\"transport\",\"test\":\"%s\",\"pid\":%d,\"tid\":%ld,"
                "\"ioctl_rc\":%d,\"errno\":%d,\"nvstatus\":%u,\"hclient\":%u,\"hobject\":%u}\n",
                label, getpid(), syscall(SYS_gettid), rc, saved_errno, args.status, args.hClient, args.hObject);
    std::fflush(stdout);
    return {rc, saved_errno, args};
}
static NVOS54_PARAMETERS query_args(unsigned tid, Channels &channels)
{
    NVOS54_PARAMETERS args{};
    args.hClient = tid;
    args.params = reinterpret_cast<NvP64>(&channels);
    args.paramsSize = sizeof(channels);
    return args;
}
static NVOS54_PARAMETERS timeslice_args(unsigned client, unsigned object, NvU64 &timeslice)
{
    NVOS54_PARAMETERS args{};
    args.hClient = client;
    args.hObject = object;
    args.flags = NV_GPREEMPT_V1_SET_TIMESLICE;
    args.cmd = NV_GPREEMPT_SET_TIMESLICE_CMD;
    args.params = reinterpret_cast<NvP64>(&timeslice);
    args.paramsSize = sizeof(timeslice);
    return args;
}
static void denied(const Reply &reply, bool query)
{
    require(reply.rc < 0 || reply.args.status != 0, "negative transport request unexpectedly succeeded");
    if (query && reply.rc == 0)
        require(!reply.args.hClient && !reply.args.hObject, "failed query published selected handles");
    ++negatives;
}
struct Identity {
    unsigned tid = 0, client = 0, tsg = 0, role = 0;
    gp_u64 context = 0;
    Channels channels{};
};
static Identity query(Control &control, unsigned tid, unsigned role, const char *label)
{
    Identity result{};
    result.tid = tid;
    result.role = role;
    Reply reply = invoke(control, query_args(tid, result.channels), label);
    require(reply.rc == 0 && reply.args.status == 0, "owned GR query failed");
    require(reply.args.hClient && reply.args.hObject, "owned query returned zero handles");
    require(result.channels.numChannels > 0 && result.channels.numChannels <= 64, "query channel bounds");
    for (unsigned i = 0; i < result.channels.numChannels; ++i) {
        require(result.channels.hClientList[i] == reply.args.hClient && result.channels.hChannelList[i],
                "query channel ownership/identity mismatch");
        for (unsigned j = 0; j < i; ++j)
            require(result.channels.hChannelList[i] != result.channels.hChannelList[j], "duplicate query channel");
    }
    result.client = reply.args.hClient;
    result.tsg = reply.args.hObject;
    return result;
}
static void control_timeslice(Control &control, const Identity &target, NvU64 time, const char *label)
{
    Reply reply = invoke(control, timeslice_args(target.client, target.tsg, time), label);
    require(reply.rc == 0 && reply.args.status == 0, "owned SET_TIMESLICE failed");
}

struct Context {
    CUcontext context = nullptr;
    ~Context() { if (context) cuCtxDestroy(context); }
    void create(CUdevice device) { cuda(cuCtxCreate(&context, 0, device), "cuCtxCreate"); }
    void destroy() { if (context) { cuda(cuCtxDestroy(context), "cuCtxDestroy"); context = nullptr; } }
};
static const char fill_ptx[] = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry gp_context_fill(.param .u64 output, .param .u32 count, .param .u32 value)
{
  .reg .pred %p;
  .reg .b32 %r<7>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [output];
  ld.param.u32 %r1, [count];
  ld.param.u32 %r2, [value];
  mov.u32 %r3, %tid.x;
  mov.u32 %r4, %ctaid.x;
  mov.u32 %r5, %ntid.x;
  mad.lo.u32 %r6, %r4, %r5, %r3;
  setp.ge.u32 %p, %r6, %r1;
  @%p bra done;
  mul.wide.u32 %rd2, %r6, 4;
  add.u64 %rd3, %rd1, %rd2;
  xor.b32 %r2, %r2, %r6;
  st.global.u32 [%rd3], %r2;
done:
  ret;
}
)ptx";
static void kernel_check(unsigned role)
{
    CUmodule module = nullptr;
    CUdeviceptr buffer = 0;
    CUstream stream = nullptr;
    try {
        cuda(cuModuleLoadData(&module, fill_ptx), "load tiny fill PTX");
        CUfunction function;
        cuda(cuModuleGetFunction(&function, module, "gp_context_fill"), "get fill kernel");
        cuda(cuMemAlloc(&buffer, 1024 * sizeof(unsigned)), "allocate canary buffer");
        cuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create canary stream");
        unsigned count = 1024, value = 0x13570000U + role;
        void *arguments[] = {&buffer, &count, &value};
        cuda(cuLaunchKernel(function, 4, 1, 1, 256, 1, 1, 0, stream, arguments, nullptr), "fill launch");
        cuda(cuStreamSynchronize(stream), "fill synchronize");
        std::array<unsigned, 1024> output{};
        cuda(cuMemcpyDtoH(output.data(), buffer, sizeof(output)), "copy canary output");
        for (unsigned i = 0; i < count; ++i) require(output[i] == (value ^ i), "canary numerical mismatch");
        values += count;
        cuda(cuStreamDestroy(stream), "destroy canary stream"); stream = nullptr;
        cuda(cuMemFree(buffer), "free canary buffer"); buffer = 0;
        cuda(cuModuleUnload(module), "unload canary module"); module = nullptr;
    } catch (...) {
        if (stream) cuStreamDestroy(stream);
        if (buffer) cuMemFree(buffer);
        if (module) cuModuleUnload(module);
        throw;
    }
}

static void negative_abi(Control &control, const Identity &target)
{
    Channels channels{};
    auto args = query_args(target.tid, channels);
    args.paramsSize--;
    denied(invoke(control, args, "bad_query_payload_size"), true);
    args = query_args(target.tid, channels); args.params = reinterpret_cast<NvP64>(1);
    denied(invoke(control, args, "bad_query_output_pointer"), true);
    args = query_args(target.tid, channels); args.params = nullptr;
    denied(invoke(control, args, "null_query_output_pointer"), true);
    args = query_args(target.tid, channels); args.hObject = target.tsg;
    denied(invoke(control, args, "nonzero_query_object"), true);
    args = query_args(target.tid, channels);
    denied(invoke(control, args, "bad_outer_size", 0xc0104660UL), false);
    NvU64 time = 1;
    args = timeslice_args(target.client, target.tsg, time); args.flags = 0x00020001;
    denied(invoke(control, args, "unsupported_control_version"), false);
    args = timeslice_args(target.client, target.tsg, time); args.cmd++;
    denied(invoke(control, args, "nonwhitelisted_control_command"), false);
    args = timeslice_args(target.client, target.tsg, time); args.paramsSize = 7;
    denied(invoke(control, args, "bad_control_payload_size"), false);
    args = timeslice_args(target.client, target.tsg, time); args.params = reinterpret_cast<NvP64>(1);
    denied(invoke(control, args, "bad_control_input_pointer"), false);
    time = 2;
    denied(invoke(control, timeslice_args(target.client, target.tsg, time), "nonwhitelisted_timeslice"), false);
    time = 1;
    denied(invoke(control, timeslice_args(target.client, target.channels.hChannelList[0], time),
                  "channel_is_not_gr_tsg"), false);
}

static int foreign(int argc, char **argv)
{
    require(argc == 5, "foreign-check needs TID client TSG");
    unsigned tid = std::stoul(argv[2]), client = std::stoul(argv[3]), tsg = std::stoul(argv[4]);
    Control control;
    Channels channels{};
    denied(invoke(control, query_args(tid, channels), "foreign_process_query"), true);
    NvU64 time = 1;
    denied(invoke(control, timeslice_args(client, tsg, time), "foreign_process_control"), false);
    std::printf("{\"event\":\"foreign_process_checks\",\"passed\":true,\"negatives\":%u}\n", negatives.load());
    return 0;
}
static void foreign_checks(const Identity &target)
{
    std::array<char, PATH_MAX> executable{};
    auto size = readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    require(size > 0 && size < static_cast<ssize_t>(executable.size() - 1), "resolve canary executable");
    std::string tid = std::to_string(target.tid), client = std::to_string(target.client), tsg = std::to_string(target.tsg);
    char option[] = "--foreign-check";
    char *arguments[] = {executable.data(), option, tid.data(), client.data(), tsg.data(), nullptr};
    pid_t child = -1;
    require(posix_spawn(&child, executable.data(), nullptr, nullptr, arguments, environ) == 0, "spawn fresh foreign process");
    int status = 0;
    while (waitpid(child, &status, 0) < 0) { if (errno != EINTR) throw std::runtime_error("wait foreign child failed"); }
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "foreign process gained access or failed to test");
    negatives += 2;
}

static void ambiguous_check(Control &control, CUdevice device)
{
    Context first, second;
    first.create(device);
    second.create(device);
    Channels channels{};
    Reply reply = invoke(control, query_args(syscall(SYS_gettid), channels), "same_creator_ambiguous_gr");
    denied(reply, true);
    require(reply.rc == 0 && reply.args.status == NV_ERR_INVALID_STATE, "ambiguity did not return explicit invalid-state");
    second.destroy();
    Identity remaining = query(control, syscall(SYS_gettid), 0, "unambiguous_after_one_destroy");
    require(remaining.client && remaining.tsg, "single remaining context query failed");
    first.destroy();
    denied(invoke(control, query_args(syscall(SYS_gettid), channels), "query_after_context_destruction"), true);
}

static void hint_check()
{
    require(gpreempt_hint_decide(GP_PREPROCESS, GP_LC, 0, 0, 1, 1) == (GP_RESET | GP_HINT), "reserved preprocess");
    require(gpreempt_hint_decide(GP_DUE, GP_LC, 999, 1000, 1, 1) == 0, "early hint");
    require(gpreempt_hint_decide(GP_DUE, GP_LC, 1000, 1000, 1, 1) == 0, "equal hint");
    require(gpreempt_hint_decide(GP_DUE, GP_LC, 1001, 1000, 1, 1) == GP_BLOCK, "late hint");
    require(gpreempt_hint_decide(GP_INFER, GP_LC, 0, 0, 1, 1) == GP_RELEASE, "LC release");
    require(gpreempt_hint_decide(GP_PREPROCESS, GP_BE, 0, 0, 1, 1) == 0, "BE unaffected");
}

int main(int argc, char **argv)
{
    try {
        if (argc > 1 && !std::strcmp(argv[1], "--foreign-check")) return foreign(argc, argv);
        if (argc == 2 && !std::strcmp(argv[1], "--help")) {
            std::puts("Usage: gpreempt_context_smoke (GPREEMPT_POLICY=original|bpf); GPU canary, use external timeout");
            return 0;
        }
        require(argc == 1, "unrecognized canary argument");
        cuda(cuInit(0), "cuInit");
        CUdevice device;
        cuda(cuDeviceGet(&device, 0), "get device zero");
        Control control;
        std::mutex mutex;
        std::condition_variable condition;
        unsigned ready = 0;
        bool release = false;
        std::string error;
        std::array<Identity, 2> identities{};
        std::array<std::thread, 2> workers;
        for (unsigned role = 0; role < 2; ++role) workers[role] = std::thread([&, role] {
            Context context;
            try {
                {
                    std::lock_guard<std::mutex> serialized(creation_mutex);
                    require(gpreempt_ctx_begin(role) == 0, "BPF scope begin");
                    context.create(device);
                    Control owned_fd;
                    identities[role] = query(owned_fd, syscall(SYS_gettid), role, "creator_thread_query");
                    identities[role].context = reinterpret_cast<gp_u64>(context.context);
                    if (!gpreempt_bpf_enabled())
                        control_timeslice(owned_fd, identities[role], role == GP_LC ? 1000000 : 1, "original_role_timeslice");
                    require(gpreempt_ctx_register(identities[role].context, identities[role].client,
                                                  identities[role].tsg, role) == 0, "BPF role registration");
                    require(gpreempt_ctx_end() == 0, "BPF scope end");
                    {
                        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
                        std::lock_guard<std::mutex> guard(output_mutex);
                        const auto &who = identities[role];
                        std::printf("{\"event\":\"role_context\",\"pid\":%d,\"tid\":%u,\"role\":%u,"
                                    "\"hclient\":%u,\"htsg\":%u,\"cuda_context\":%llu,"
                                    "\"channels\":%u,\"kernel_begin_ns\":%lld}\n", getpid(), who.tid, role,
                                    who.client, who.tsg, who.context, who.channels.numChannels,
                                    static_cast<long long>(now));
                        std::fflush(stdout);
                    }
                    kernel_check(role);
                }
                std::unique_lock<std::mutex> lock(mutex);
                ++ready; condition.notify_all();
                condition.wait(lock, [&] { return release; });
                lock.unlock();
                context.destroy();
            } catch (const std::exception &failure) {
                std::lock_guard<std::mutex> lock(mutex);
                if (error.empty()) error = failure.what();
                ++ready; condition.notify_all();
            }
        });
        try {
            std::unique_lock<std::mutex> lock(mutex);
            require(condition.wait_for(lock, std::chrono::seconds(30), [&] { return ready == 2 || !error.empty(); }), "context ready timeout");
            require(error.empty(), error.c_str());
            lock.unlock();
            require(identities[0].tid != identities[1].tid && identities[0].context != identities[1].context &&
                    (identities[0].client != identities[1].client || identities[0].tsg != identities[1].tsg), "roles alias");
            for (const auto &identity : identities) {
                Identity same_owner = query(control, identity.tid, identity.role, "same_process_different_fd_query");
                require(identity.client == same_owner.client && identity.tsg == same_owner.tsg, "different owned FD changed identity");
            }
            negative_abi(control, identities[0]);
            foreign_checks(identities[0]);
            ambiguous_check(control, device);
            hint_check();
        } catch (const std::exception &failure) {
            std::lock_guard<std::mutex> lock(mutex);
            if (error.empty()) error = failure.what();
        }
        { std::lock_guard<std::mutex> lock(mutex); release = true; condition.notify_all(); }
        for (auto &worker : workers) worker.join();
        require(error.empty(), error.c_str());
        for (const auto &identity : identities) {
            NvU64 time = 1;
            denied(invoke(control, timeslice_args(identity.client, identity.tsg, time), "stale_owned_tsg_control"), false);
        }
        std::printf("{\"event\":\"gpreempt_context_smoke\",\"passed\":true,\"policy\":\"%s\","
                    "\"roles\":2,\"validated_values\":%u,\"negative_cases\":%u,\"assertions\":%u,"
                    "\"gdr_actuator_tested\":false,\"performance_measured\":false}\n",
                    gpreempt_bpf_enabled() ? "bpf" : "original", values.load(), negatives.load(), assertions.load());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "gpreempt_context_smoke_failed: %s\n", error.what());
        return 1;
    }
}
