#include <assert.h>
#include <errno.h>
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
#include <vector>

#include "clock_domain.h"
#include "common.h"
#include "rm_ptimer_575.h"
#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"
#include "utils/utils.h"

#define CHANNEL_SIZE (1UL << 20)
#define CALIBRATION_WARMUPS 4
#define CALIBRATION_TRIALS 32

enum class recv_state_t { INIT, WORKING, STOP, FINISHED };

struct context_state_t {
    ChannelDev* channel_dev = nullptr;
    ChannelHost channel_host;
    CUmodule tool_module = nullptr;
    CUfunction flush_channel_func = nullptr;
    clock_calibration_t* start_calibration = nullptr;
    struct rm_calibration_run* start_rm = nullptr;
    launch_pair_t* launch_pairs = nullptr;
    uint64_t* device_launch_entries = nullptr;
    uint64_t* launch_capture_errors = nullptr;
    uint64_t selected_launches = 0;
    uint64_t stored_launch_pairs = 0;
    uint64_t launch_pair_overflows = 0;
    bool selected_launch_counter_overflow = false;
    std::vector<exit_record_t> exit_events;
    uint64_t exit_bad_size_bytes = 0;
    volatile recv_state_t recv_state = recv_state_t::INIT;
    bool saw_selected_launch = false;
};

struct rm_calibration_run {
    uint64_t requested = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t cleanup_complete = 0;
    struct rm_ptimer_575_sample best = {};
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
static bool trace_target_family = false;
static bool trace_launches = false;

struct exit_validation_t {
    uint64_t extent_x = 0;
    uint64_t extent_y = 0;
    uint64_t extent_z = 0;
    uint64_t unique_coordinates = 0;
    uint64_t max_multiplicity = 0;
    uint64_t multiplicity_220 = 0;
    uint64_t multiplicity_44 = 0;
    uint64_t multiplicity_22 = 0;
    uint64_t other_multiplicity = 0;
    uint64_t segment_mismatches = 0;
    uint64_t invalid_coordinates = 0;
    bool complete = false;
};

static uint64_t expected_exit_multiplicity(uint64_t coordinate_x,
                                           uint64_t launches) {
    if (launches == 220) {
        if (coordinate_x < 4) return 220;
        if (coordinate_x < 8) return 44;
        return 22;
    }
    return launches;
}

static exit_validation_t validate_exit_events(const context_state_t* state,
                                              uint64_t launches) {
    exit_validation_t result;
    constexpr uint64_t extent_y = 256;
    if (state == nullptr || thread_count == 0 ||
        thread_count % extent_y != 0) {
        return result;
    }
    const uint64_t extent_x = thread_count / extent_y;
    std::vector<uint64_t> counts(thread_count, 0);
    for (const auto& event : state->exit_events) {
        if (event.coordinate_x >= extent_x ||
            event.coordinate_y >= extent_y || event.coordinate_z != 0) {
            result.invalid_coordinates++;
            continue;
        }
        const uint64_t index = event.coordinate_x * extent_y +
                               event.coordinate_y;
        counts[index]++;
        if (event.coordinate_x + 1 > result.extent_x)
            result.extent_x = event.coordinate_x + 1;
        if (event.coordinate_y + 1 > result.extent_y)
            result.extent_y = event.coordinate_y + 1;
        if (event.coordinate_z + 1 > result.extent_z)
            result.extent_z = event.coordinate_z + 1;
    }
    for (uint64_t index = 0; index < thread_count; index++) {
        const uint64_t multiplicity = counts[index];
        if (multiplicity == 0) continue;
        result.unique_coordinates++;
        if (multiplicity > result.max_multiplicity)
            result.max_multiplicity = multiplicity;
        if (multiplicity == 220)
            result.multiplicity_220++;
        else if (multiplicity == 44)
            result.multiplicity_44++;
        else if (multiplicity == 22)
            result.multiplicity_22++;
        else
            result.other_multiplicity++;
        const uint64_t coordinate_x = index / extent_y;
        if (multiplicity != expected_exit_multiplicity(coordinate_x, launches))
            result.segment_mismatches++;
    }
    result.complete = result.invalid_coordinates == 0 &&
                      result.unique_coordinates == thread_count &&
                      result.extent_x == extent_x &&
                      result.extent_y == extent_y && result.extent_z == 1;
    return result;
}

static void print_exit_validation(const context_state_t* state,
                                  uint64_t launches) {
    const exit_validation_t result = validate_exit_events(state, launches);
    const bool passed = state != nullptr && state->exit_bad_size_bytes == 0 &&
        result.complete && result.max_multiplicity == launches &&
        result.segment_mismatches == 0;
    fprintf(stderr, "NVBIT kernelretsnoop record_bytes=%zu\n",
            sizeof(exit_record_t));
    fprintf(stderr, "NVBIT kernelretsnoop bad_size_bytes=%llu\n",
            static_cast<unsigned long long>(state->exit_bad_size_bytes));
    fprintf(stderr, "NVBIT kernelretsnoop cartesian_launches=%llu\n",
            static_cast<unsigned long long>(result.max_multiplicity));
    fprintf(stderr, "NVBIT kernelretsnoop cartesian_coordinates=%llu\n",
            static_cast<unsigned long long>(result.unique_coordinates));
    fprintf(stderr, "NVBIT kernelretsnoop cartesian_complete=%u\n",
            result.complete ? 1U : 0U);
    fprintf(stderr, "NVBIT kernelretsnoop extent_x=%llu extent_y=%llu extent_z=%llu\n",
            static_cast<unsigned long long>(result.extent_x),
            static_cast<unsigned long long>(result.extent_y),
            static_cast<unsigned long long>(result.extent_z));
    fprintf(stderr, "NVBIT kernelretsnoop multiplicity_220=%llu multiplicity_44=%llu multiplicity_22=%llu multiplicity_other=%llu\n",
            static_cast<unsigned long long>(result.multiplicity_220),
            static_cast<unsigned long long>(result.multiplicity_44),
            static_cast<unsigned long long>(result.multiplicity_22),
            static_cast<unsigned long long>(result.other_multiplicity));
    fprintf(stderr, "NVBIT kernelretsnoop segment_mismatches=%llu invalid_coordinates=%llu unique_coordinates=%llu\n",
            static_cast<unsigned long long>(result.segment_mismatches),
            static_cast<unsigned long long>(result.invalid_coordinates),
            static_cast<unsigned long long>(result.unique_coordinates));
    fprintf(stderr, "NVBIT kernelretsnoop collector_gate_passed=%u\n",
            passed ? 1U : 0U);
}

static bool monotonic_raw_ns(uint64_t* value) {
    struct timespec ts;
    if (value == nullptr || clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0 ||
        ts.tv_sec < 0 || ts.tv_nsec < 0 || ts.tv_nsec >= 1000000000L ||
        static_cast<uint64_t>(ts.tv_sec) >
            UINT64_MAX / 1000000000ULL) {
        return false;
    }
    *value = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
             static_cast<uint64_t>(ts.tv_nsec);
    return true;
}

static bool calibrate_gpu_clock(clock_calibration_t* calibration,
                                rm_calibration_run* run) {
    if (calibration == nullptr || run == nullptr) {
        return false;
    }
    *calibration = {};
    *run = {};
    run->requested = CALIBRATION_TRIALS;
    rm_ptimer_575_client client;
    rm_ptimer_575_client_init(&client);
    if (rm_ptimer_575_open(&client) != 0) {
        return false;
    }
    for (uint32_t trial = 0; trial < CALIBRATION_WARMUPS; trial++) {
        struct rm_ptimer_575_sample sample = {};
        if (rm_ptimer_575_sample(&client, &sample) != 0) {
            run->cleanup_complete = rm_ptimer_575_close(&client) == 0;
            return false;
        }
    }
    for (uint32_t trial = 0; trial < CALIBRATION_TRIALS; trial++) {
        struct rm_ptimer_575_sample sample = {};
        if (rm_ptimer_575_sample(&client, &sample) != 0) {
            run->rejected++;
            run->cleanup_complete = rm_ptimer_575_close(&client) == 0;
            return false;
        }
        run->accepted++;
        if (run->accepted == 1 ||
            sample.bracket_width_ns < run->best.bracket_width_ns) {
            run->best = sample;
            calibration->offset_low_ns = sample.offset_low_ns;
            calibration->offset_high_ns = sample.offset_high_ns;
            calibration->uncertainty_ns = sample.bracket_width_ns / 2 +
                                          sample.bracket_width_ns % 2;
            calibration->host_anchor_ns = sample.cpu_before_raw_ns +
                                          sample.selected_gap_ns / 2;
            calibration->valid = 1;
        }
    }
    run->cleanup_complete = rm_ptimer_575_close(&client) == 0;
    return run->cleanup_complete && run->accepted == run->requested &&
           run->rejected == 0 && clock_calibration_valid(*calibration);
}

static bool wait_for_minimum_clock_span(
    const clock_calibration_t& start_calibration) {
    uint64_t deadline_ns;
    if (!minimum_end_calibration_deadline(start_calibration, &deadline_ns)) {
        return false;
    }
    for (;;) {
        uint64_t now_ns;
        if (!monotonic_raw_ns(&now_ns)) return false;
        if (now_ns >= deadline_ns) return true;
        const uint64_t remaining = deadline_ns - now_ns;
        struct timespec delay = {
            static_cast<time_t>(remaining / 1000000000ULL),
            static_cast<long>(remaining % 1000000000ULL),
        };
        while (nanosleep(&delay, &delay) != 0) {
            if (errno != EINTR) return false;
        }
    }
}

static void print_clock_calibration(
    const char* phase, const clock_calibration_t& calibration,
    const rm_calibration_run& run) {
    fprintf(stderr, "NVBIT launchlate %s_clock_offset_lower_ns=%lld\n", phase,
            static_cast<long long>(calibration.offset_low_ns));
    fprintf(stderr, "NVBIT launchlate %s_clock_offset_upper_ns=%lld\n", phase,
            static_cast<long long>(calibration.offset_high_ns));
    fprintf(stderr, "NVBIT launchlate %s_clock_uncertainty_ns=%llu\n", phase,
            static_cast<unsigned long long>(calibration.uncertainty_ns));
    fprintf(stderr, "NVBIT launchlate %s_clock_host_anchor_ns=%llu\n", phase,
            static_cast<unsigned long long>(calibration.host_anchor_ns));
    fprintf(stderr, "NVBIT launchlate %s_clock_calibration_valid=%llu\n", phase,
            static_cast<unsigned long long>(calibration.valid));
    fprintf(stderr, "NVBIT launchlate %s_rm_samples_requested=%llu\n", phase,
            static_cast<unsigned long long>(run.requested));
    fprintf(stderr, "NVBIT launchlate %s_rm_samples_accepted=%llu\n", phase,
            static_cast<unsigned long long>(run.accepted));
    fprintf(stderr, "NVBIT launchlate %s_rm_samples_rejected=%llu\n", phase,
            static_cast<unsigned long long>(run.rejected));
    fprintf(stderr, "NVBIT launchlate %s_rm_outer_before_raw_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.outer_before_raw_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_cpu_before_raw_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.cpu_before_raw_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_gpu_ptimer_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.gpu_ptimer_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_cpu_after_raw_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.cpu_after_raw_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_outer_after_raw_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.outer_after_raw_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_outer_width_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.outer_width_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_selected_gap_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.selected_gap_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_bracket_width_ns=%llu\n", phase,
            static_cast<unsigned long long>(run.best.bracket_width_ns));
    fprintf(stderr, "NVBIT launchlate %s_rm_status=%u\n", phase,
            run.best.rm_status);
    fprintf(stderr, "NVBIT launchlate %s_rm_cleanup_complete=%llu\n", phase,
            static_cast<unsigned long long>(run.cleanup_complete));
}

static bool exact_sample_accounting(uint64_t selected, uint64_t classified,
                                    uint64_t uncertain,
                                    uint64_t clock_errors) {
    return classified <= selected && uncertain <= selected - classified &&
           clock_errors == selected - classified - uncertain;
}

static void print_launchlate_results(
    const context_state_t* state, const clock_calibration_t& end_calibration) {
    uint64_t histogram[HIST_BINS] = {};
    uint64_t classified = 0;
    uint64_t uncertain = 0;
    uint64_t clock_errors = state->launch_pair_overflows;
    const uint64_t stored = state->stored_launch_pairs;
    const uint64_t readable =
        stored <= LAUNCH_PAIR_CAPACITY ? stored : LAUNCH_PAIR_CAPACITY;

    for (uint64_t index = 0; index < readable; index++) {
        const launch_pair_t& pair = state->launch_pairs[index];
        uint32_t bin = 0;
        if (pair.sequence != index + 1) {
            clock_errors++;
            continue;
        }
        const launch_sample_status_t status = classify_affine_launch_latency(
            pair.host_raw_ns, pair.gpu_entry_ns, *state->start_calibration,
            end_calibration, &bin);
        if (status == LAUNCH_SAMPLE_CLASSIFIED) {
            histogram[bin]++;
            classified++;
        } else if (status == LAUNCH_SAMPLE_UNCERTAIN) {
            uncertain++;
        } else {
            clock_errors++;
        }
    }
    if (stored > readable) {
        clock_errors += stored - readable;
    }

    uint64_t histogram_total = 0;
    for (uint32_t index = 0; index < HIST_BINS; index++) {
        histogram_total += histogram[index];
        fprintf(stderr, "NVBIT launchlate bin_%u=%llu\n", index,
                static_cast<unsigned long long>(histogram[index]));
    }

    const uint64_t selected = state->selected_launches;
    const uint64_t device_entries = *state->device_launch_entries;
    const uint64_t capture_errors = *state->launch_capture_errors;
    const bool storage_complete =
        stored <= LAUNCH_PAIR_CAPACITY && stored <= selected &&
        state->launch_pair_overflows == selected - stored;
    const bool accounting_complete =
        !state->selected_launch_counter_overflow && storage_complete &&
        device_entries == selected &&
        histogram_total == classified &&
        exact_sample_accounting(selected, classified, uncertain, clock_errors);

    fprintf(stderr, "NVBIT launchlate pair_capacity=%llu\n",
            static_cast<unsigned long long>(LAUNCH_PAIR_CAPACITY));
    fprintf(stderr, "NVBIT launchlate stored_pairs=%llu\n",
            static_cast<unsigned long long>(stored));
    fprintf(stderr, "NVBIT launchlate device_entries=%llu\n",
            static_cast<unsigned long long>(device_entries));
    fprintf(stderr, "NVBIT launchlate pair_overflows=%llu\n",
            static_cast<unsigned long long>(state->launch_pair_overflows));
    fprintf(stderr, "NVBIT launchlate capture_errors=%llu\n",
            static_cast<unsigned long long>(capture_errors));
    fprintf(stderr, "NVBIT launchlate selected_counter_overflow=%u\n",
            state->selected_launch_counter_overflow ? 1U : 0U);
    fprintf(stderr, "NVBIT launchlate uncertain_samples=%llu\n",
            static_cast<unsigned long long>(uncertain));
    fprintf(stderr,
            "NVBIT launchlate samples=%llu clock_errors=%llu\n",
            static_cast<unsigned long long>(classified),
            static_cast<unsigned long long>(clock_errors));
    fprintf(stderr, "NVBIT launchlate accounting_complete=%u\n",
            accounting_complete ? 1U : 0U);
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
            first, reinterpret_cast<uint64_t>(state->device_launch_entries));
        nvbit_add_call_arg_const_val64(
            first, reinterpret_cast<uint64_t>(state->launch_capture_errors));
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
        state->exit_bad_size_bytes += bytes % sizeof(exit_record_t);
        for (uint32_t offset = 0;
             offset + sizeof(exit_record_t) <= bytes;
             offset += sizeof(exit_record_t)) {
            const auto* record =
                reinterpret_cast<const exit_record_t*>(buffer + offset);
            exit_records.fetch_add(1, std::memory_order_relaxed);
            if (record->timestamp != 0) {
                nonzero_timestamps.fetch_add(1, std::memory_order_relaxed);
            }
            state->exit_events.push_back(*record);
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
    }
    if (mode == OBS_KERNELRETSNOOP) {
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
    } else if (mode == OBS_LAUNCHLATE) {
        CUDA_SAFECALL(cudaMallocManaged(
            &state->launch_pairs,
            LAUNCH_PAIR_CAPACITY * sizeof(launch_pair_t)));
        CUDA_SAFECALL(cudaMallocManaged(&state->device_launch_entries,
                                       sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMallocManaged(&state->launch_capture_errors,
                                       sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMemset(
            state->launch_pairs, 0,
            LAUNCH_PAIR_CAPACITY * sizeof(launch_pair_t)));
        CUDA_SAFECALL(cudaMemset(state->device_launch_entries, 0,
                                sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMemset(state->launch_capture_errors, 0,
                                sizeof(uint64_t)));
        CUDA_SAFECALL(cudaMallocManaged(&state->start_calibration,
                                       sizeof(clock_calibration_t)));
        state->start_rm = new rm_calibration_run;
        const bool calibrated =
            calibrate_gpu_clock(state->start_calibration, state->start_rm);
        fprintf(stderr,
                "NVBIT launchlate clock_calibration_method="
                "rm_endpoints_v1_PTIMER_against_CLOCK_MONOTONIC_RAW_"
                "with_affine_interpolation_and_drift_bound\n");
        print_clock_calibration("start", *state->start_calibration,
                                *state->start_rm);
        if (!calibrated) {
            fprintf(stderr,
                    "NVBIT_OBS error: launchlate start clock calibration "
                    "failed\n");
        }
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
                uint64_t launch_raw_ns = 0;
                uint64_t pair_ptr = 0;
                if (!monotonic_raw_ns(&launch_raw_ns)) {
                    fprintf(stderr,
                            "NVBIT_OBS error: CLOCK_MONOTONIC_RAW read failed\n");
                }
                if (state->selected_launches == UINT64_MAX) {
                    state->selected_launch_counter_overflow = true;
                } else {
                    const uint64_t index = state->selected_launches++;
                    if (index < LAUNCH_PAIR_CAPACITY) {
                        launch_pair_t* pair = &state->launch_pairs[index];
                        pair->host_raw_ns = launch_raw_ns;
                        pair->gpu_entry_ns = 0;
                        pair->sequence = index + 1;
                        state->stored_launch_pairs++;
                        pair_ptr = reinterpret_cast<uint64_t>(pair);
                    } else {
                        state->launch_pair_overflows++;
                        if (index == LAUNCH_PAIR_CAPACITY) {
                            fprintf(stderr,
                                    "NVBIT_OBS error: launchlate raw-pair "
                                    "capacity exceeded\n");
                        }
                    }
                }
                nvbit_set_at_launch(ctx, func, pair_ptr);
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
        print_exit_validation(state, selected_launches.load());
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
    } else if (mode == OBS_LAUNCHLATE && state->launch_pairs != nullptr) {
        clock_calibration_t end_calibration = {};
        rm_calibration_run end_rm = {};
        clock_drift_t drift = {};
        const bool minimum_span_ready =
            state->start_calibration != nullptr &&
            wait_for_minimum_clock_span(*state->start_calibration);
        const bool end_calibrated = minimum_span_ready &&
            calibrate_gpu_clock(&end_calibration, &end_rm);
        const bool drift_measured =
            end_calibrated && state->start_calibration != nullptr &&
            clock_calibration_drift(*state->start_calibration,
                                    end_calibration, &drift);
        const bool clock_model_bounded =
            drift_measured &&
            drift.elapsed_ns >= CLOCK_MIN_CALIBRATION_SPAN_NS &&
            drift.bounded;
        print_clock_calibration("end", end_calibration, end_rm);
        fprintf(stderr,
                "NVBIT launchlate clock_offset_change_lower_ns=%lld\n",
                static_cast<long long>(drift.offset_change_low_ns));
        fprintf(stderr,
                "NVBIT launchlate clock_offset_change_upper_ns=%lld\n",
                static_cast<long long>(drift.offset_change_high_ns));
        fprintf(stderr, "NVBIT launchlate clock_calibration_elapsed_ns=%llu\n",
                static_cast<unsigned long long>(drift.elapsed_ns));
        fprintf(stderr, "NVBIT launchlate clock_drift_rate_bound_ppb=%llu\n",
                static_cast<unsigned long long>(drift.rate_bound_ppb));
        fprintf(stderr, "NVBIT launchlate clock_drift_limit_ppb=%llu\n",
                static_cast<unsigned long long>(CLOCK_DRIFT_LIMIT_PPB));
        fprintf(stderr, "NVBIT launchlate clock_drift_bounded=%u\n",
                clock_model_bounded ? 1U : 0U);
        if (!minimum_span_ready) {
            fprintf(stderr,
                    "NVBIT_OBS error: launchlate minimum clock calibration "
                    "span wait failed\n");
        }
        if (!clock_model_bounded) {
            fprintf(stderr,
                    "NVBIT_OBS error: launchlate clock drift exceeds bound\n");
        }
        print_launchlate_results(state, end_calibration);
    }
    fprintf(stderr, "NVBIT selected_launches=%llu\n",
            static_cast<unsigned long long>(
                mode == OBS_LAUNCHLATE ? state->selected_launches
                                       : selected_launches.load()));
    fprintf(stderr,
            "NVBIT kernelretsnoop events=%llu nonzero_timestamps=%llu\n",
            static_cast<unsigned long long>(exit_records.load()),
            static_cast<unsigned long long>(nonzero_timestamps.load()));

    if (state->start_calibration != nullptr) {
        CUDA_SAFECALL(cudaFree(state->start_calibration));
    }
    delete state->start_rm;
    if (state->launch_pairs != nullptr) {
        CUDA_SAFECALL(cudaFree(state->launch_pairs));
    }
    if (state->device_launch_entries != nullptr) {
        CUDA_SAFECALL(cudaFree(state->device_launch_entries));
    }
    if (state->launch_capture_errors != nullptr) {
        CUDA_SAFECALL(cudaFree(state->launch_capture_errors));
    }
    delete state;
    contexts.erase(ctx);
    skip_callback = false;
    pthread_mutex_unlock(&state_mutex);
}
