#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common.h"
#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"
#include "utils/utils.h"

#define CHANNEL_SIZE (1UL << 20)

enum class recv_state_t { INIT, WORKING, STOP, FINISHED };

struct context_state_t {
    ChannelDev* channel_dev = nullptr;
    ChannelHost channel_host;
    CUmodule tool_module = nullptr;
    CUfunction flush_channel_func = nullptr;
    volatile recv_state_t recv_state = recv_state_t::INIT;
    bool saw_selected_launch = false;
};

#include "tool_func/flush_channel.c"

static pthread_mutex_t state_mutex;
static pthread_mutex_t callback_mutex;
static std::unordered_map<CUcontext, context_state_t*> contexts;
static std::unordered_set<CUfunction> instrumented;
static std::unordered_set<std::string> reported_target_family;
static std::unordered_set<std::string> reported_launches;
static std::atomic<uint64_t> exit_records{0};
static std::atomic<uint64_t> nonzero_timestamps{0};
static std::atomic<uint64_t> selected_launches{0};
static bool skip_callback = false;

static observability_mode_t mode = OBS_KERNELRETSNOOP;
static std::string target_symbol;
static uint32_t thread_count = 1048576;
static uint64_t* thread_counters = nullptr;
static uint64_t* launch_histogram = nullptr;
static uint64_t* launch_samples = nullptr;
static uint64_t* launch_clock_errors = nullptr;
static bool trace_target_family = false;
static bool trace_launches = false;

static uint64_t realtime_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

static bool is_launch(nvbit_api_cuda_t cbid) {
    return cbid == API_CUDA_cuLaunchKernel ||
           cbid == API_CUDA_cuLaunchKernel_ptsz ||
           cbid == API_CUDA_cuLaunchCooperativeKernel ||
           cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
           cbid == API_CUDA_cuLaunchKernelEx ||
           cbid == API_CUDA_cuLaunchKernelEx_ptsz;
}

static CUfunction launch_function(nvbit_api_cuda_t cbid, void* params) {
    if (cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        return reinterpret_cast<cuLaunchKernelEx_params*>(params)->f;
    }
    return reinterpret_cast<cuLaunchKernel_params*>(params)->f;
}

static bool selected(CUcontext ctx, CUfunction func) {
    const char* name = nvbit_get_func_name(ctx, func, true);
    if (trace_launches && name != nullptr && reported_launches.size() < 256 &&
        reported_launches.insert(name).second) {
        fprintf(stderr, "NVBIT_OBS launched_symbol=%s\n", name);
    }
    if (trace_target_family && name != nullptr &&
        strstr(name, "rope_norm") != nullptr &&
        reported_target_family.insert(name).second) {
        fprintf(stderr, "NVBIT_OBS launched_target_family=%s\n", name);
    }
    return name != nullptr && target_symbol == name;
}

static void instrument_selected(CUcontext ctx, CUfunction func,
                                context_state_t* state) {
    if (!instrumented.insert(func).second) {
        return;
    }

    const std::vector<Instr*>& instructions = nvbit_get_instrs(ctx, func);
    if (instructions.empty()) {
        fprintf(stderr, "NVBIT_OBS error: selected function has no instructions\n");
        return;
    }

    if (mode == OBS_LAUNCHLATE) {
        Instr* first = instructions.front();
        nvbit_insert_call(first, "observe_entry", IPOINT_BEFORE);
        nvbit_add_call_arg_launch_val64(first, 0);
        nvbit_add_call_arg_const_val64(
            first, reinterpret_cast<uint64_t>(launch_histogram));
        nvbit_add_call_arg_const_val64(
            first, reinterpret_cast<uint64_t>(launch_samples));
        nvbit_add_call_arg_const_val64(
            first, reinterpret_cast<uint64_t>(launch_clock_errors));
        return;
    }

    uint32_t exits = 0;
    for (Instr* instruction : instructions) {
        if (strcmp(instruction->getOpcode(), "EXIT") != 0) {
            continue;
        }
        exits++;
        nvbit_insert_call(instruction, "observe_exit", IPOINT_BEFORE);
        // The injected call runs even when a predicated EXIT is not taken.
        // Count actual exits, matching the predicate-preserving PTX retprobe.
        nvbit_add_call_arg_guard_pred_val(instruction);
        nvbit_add_call_arg_const_val32(instruction, mode);
        nvbit_add_call_arg_const_val64(
            instruction, reinterpret_cast<uint64_t>(state->channel_dev));
        nvbit_add_call_arg_const_val64(
            instruction, reinterpret_cast<uint64_t>(thread_counters));
        nvbit_add_call_arg_const_val32(instruction, thread_count);
    }
    fprintf(stderr, "NVBIT_OBS instrumented_exits=%u symbol=%s\n", exits,
            target_symbol.c_str());
}

static void* receive_records(void* arg) {
    CUcontext ctx = reinterpret_cast<CUcontext>(arg);
    pthread_mutex_lock(&state_mutex);
    context_state_t* state = contexts.at(ctx);
    pthread_mutex_unlock(&state_mutex);

    char* buffer = reinterpret_cast<char*>(malloc(CHANNEL_SIZE));
    while (state->recv_state == recv_state_t::WORKING) {
        const uint32_t bytes = state->channel_host.recv(buffer, CHANNEL_SIZE);
        for (uint32_t offset = 0;
             offset + sizeof(exit_record_t) <= bytes;
             offset += sizeof(exit_record_t)) {
            const auto* record =
                reinterpret_cast<const exit_record_t*>(buffer + offset);
            exit_records.fetch_add(1, std::memory_order_relaxed);
            if (record->timestamp != 0) {
                nonzero_timestamps.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    free(buffer);
    state->recv_state = recv_state_t::FINISHED;
    return nullptr;
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    const char* mode_env = getenv("OBS_MODE");
    const char* target_env = getenv("OBS_TARGET_SYMBOL");
    if (target_env == nullptr || target_env[0] == '\0') {
        fprintf(stderr, "NVBIT_OBS error: OBS_TARGET_SYMBOL is required\n");
        abort();
    }
    target_symbol = target_env;
    if (mode_env == nullptr || strcmp(mode_env, "kernelretsnoop") == 0) {
        mode = OBS_KERNELRETSNOOP;
    } else if (strcmp(mode_env, "threadhist") == 0) {
        mode = OBS_THREADHIST;
    } else if (strcmp(mode_env, "launchlate") == 0) {
        mode = OBS_LAUNCHLATE;
    } else {
        fprintf(stderr, "NVBIT_OBS error: unknown OBS_MODE=%s\n", mode_env);
        abort();
    }
    if (const char* count_env = getenv("OBS_GPU_THREAD_COUNT")) {
        thread_count = static_cast<uint32_t>(strtoul(count_env, nullptr, 10));
    }
    trace_target_family = getenv("OBS_TRACE_TARGET_FAMILY") != nullptr;
    trace_launches = getenv("OBS_TRACE_LAUNCHES") != nullptr;

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&state_mutex, &attr);
    pthread_mutex_init(&callback_mutex, &attr);
    fprintf(stderr, "NVBIT_OBS mode=%u target=%s thread_count=%u\n", mode,
            target_symbol.c_str(), thread_count);
}

void nvbit_at_term() {
    fprintf(stderr, "NVBIT_OBS process_selected_launches=%llu\n",
            static_cast<unsigned long long>(selected_launches.load()));
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&state_mutex);
    auto* state = new context_state_t;
    contexts[ctx] = state;
    if (mode == OBS_KERNELRETSNOOP) {
        nvbit_load_tool_module(ctx, reinterpret_cast<const void*>(flush_channel_bin),
                               &state->tool_module);
        nvbit_find_function_by_name(ctx, state->tool_module, "flush_channel",
                                    &state->flush_channel_func);
    }
    pthread_mutex_unlock(&state_mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&state_mutex);
    context_state_t* state = contexts.at(ctx);
    if (mode == OBS_KERNELRETSNOOP) {
        CUDA_SAFECALL(cudaMallocManaged(&state->channel_dev, sizeof(ChannelDev)));
        state->recv_state = recv_state_t::WORKING;
        state->channel_host.init(static_cast<int>(contexts.size()) - 1,
                                 CHANNEL_SIZE, state->channel_dev,
                                 receive_records, ctx);
        nvbit_set_tool_pthread(state->channel_host.get_thread());
    } else if (mode == OBS_THREADHIST && thread_counters == nullptr) {
        CUDA_SAFECALL(cudaMallocManaged(&thread_counters,
                                       thread_count * sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMemset(thread_counters, 0,
                                thread_count * sizeof(uint64_t)));
    } else if (mode == OBS_LAUNCHLATE && launch_histogram == nullptr) {
        CUDA_SAFECALL(cudaMallocManaged(&launch_histogram,
                                       HIST_BINS * sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMallocManaged(&launch_samples, sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMallocManaged(&launch_clock_errors, sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMemset(launch_histogram, 0,
                                HIST_BINS * sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMemset(launch_samples, 0, sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMemset(launch_clock_errors, 0, sizeof(uint64_t)));
    }
    pthread_mutex_unlock(&state_mutex);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char*, void* params, CUresult*) {
    if (!is_launch(cbid)) {
        return;
    }
    pthread_mutex_lock(&callback_mutex);
    if (skip_callback || contexts.find(ctx) == contexts.end()) {
        pthread_mutex_unlock(&callback_mutex);
        return;
    }
    skip_callback = true;
    CUfunction func = launch_function(cbid, params);
    if (selected(ctx, func)) {
        context_state_t* state = contexts.at(ctx);
        if (!is_exit) {
            instrument_selected(ctx, func, state);
            if (mode == OBS_LAUNCHLATE) {
                nvbit_set_at_launch(ctx, func, realtime_ns());
            }
            nvbit_enable_instrumented(ctx, func, true, false);
            state->saw_selected_launch = true;
            selected_launches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    skip_callback = false;
    pthread_mutex_unlock(&callback_mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&state_mutex);
    skip_callback = true;
    context_state_t* state = contexts.at(ctx);
    if (state->saw_selected_launch) {
        CUDA_SAFECALL(cudaDeviceSynchronize());
    }

    if (mode == OBS_KERNELRETSNOOP &&
        state->recv_state != recv_state_t::INIT) {
        void* args[] = {&state->channel_dev};
        nvbit_launch_kernel(ctx, state->flush_channel_func, 1, 1, 1, 1, 1, 1,
                            0, nullptr, args, nullptr);
        CUDA_SAFECALL(cudaDeviceSynchronize());
        state->recv_state = recv_state_t::STOP;
        while (state->recv_state != recv_state_t::FINISHED) {
        }
        state->channel_host.destroy(false);
        CUDA_SAFECALL(cudaFree(state->channel_dev));
    }

    if (mode == OBS_THREADHIST && thread_counters != nullptr) {
        uint64_t nonzero = 0;
        uint64_t total = 0;
        for (uint32_t i = 0; i < thread_count; i++) {
            if (thread_counters[i] != 0) {
                nonzero++;
                total += thread_counters[i];
            }
        }
        fprintf(stderr,
                "NVBIT threadhist nonzero_threads=%llu total_exit_probes=%llu\n",
                static_cast<unsigned long long>(nonzero),
                static_cast<unsigned long long>(total));
    } else if (mode == OBS_LAUNCHLATE && launch_histogram != nullptr) {
        for (uint32_t i = 0; i < HIST_BINS; i++) {
            fprintf(stderr, "NVBIT launchlate bin_%u=%llu\n", i,
                    static_cast<unsigned long long>(launch_histogram[i]));
        }
        fprintf(stderr, "NVBIT launchlate samples=%llu clock_errors=%llu\n",
                static_cast<unsigned long long>(*launch_samples),
                static_cast<unsigned long long>(*launch_clock_errors));
    }
    fprintf(stderr, "NVBIT selected_launches=%llu\n",
            static_cast<unsigned long long>(selected_launches.load()));
    fprintf(stderr,
            "NVBIT kernelretsnoop events=%llu nonzero_timestamps=%llu\n",
            static_cast<unsigned long long>(exit_records.load()),
            static_cast<unsigned long long>(nonzero_timestamps.load()));

    delete state;
    contexts.erase(ctx);
    skip_callback = false;
    pthread_mutex_unlock(&state_mutex);
}
