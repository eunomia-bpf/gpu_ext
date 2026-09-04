#include <stdint.h>

#include "clock_domain.h"
#include "common.h"
#include "utils/channel.hpp"

static __device__ __forceinline__ uint64_t read_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

extern "C" __device__ __noinline__ void observe_exit(
    int predicate, uint32_t mode, uint64_t channel_ptr, uint64_t counters_ptr,
    uint32_t counter_count) {
    if (!predicate) return;
    if (mode == OBS_KERNELRETSNOOP) {
        exit_record_t record = {
            static_cast<uint64_t>(blockIdx.x),
            static_cast<uint64_t>(blockIdx.y),
            static_cast<uint64_t>(blockIdx.z),
            static_cast<uint64_t>(threadIdx.x),
            static_cast<uint64_t>(threadIdx.y),
            static_cast<uint64_t>(threadIdx.z),
            read_globaltimer_ns(),
        };
        reinterpret_cast<ChannelDev*>(channel_ptr)->push(
            &record, sizeof(exit_record_t));
        return;
    }

    if (mode == OBS_THREADHIST) {
        const uint64_t block_linear =
            blockIdx.x + static_cast<uint64_t>(gridDim.x) *
                (blockIdx.y + static_cast<uint64_t>(gridDim.y) * blockIdx.z);
        const uint64_t threads_per_block =
            static_cast<uint64_t>(blockDim.x) * blockDim.y * blockDim.z;
        const uint64_t thread_linear =
            threadIdx.x + static_cast<uint64_t>(blockDim.x) *
                (threadIdx.y + static_cast<uint64_t>(blockDim.y) * threadIdx.z);
        const uint64_t global_thread =
            block_linear * threads_per_block + thread_linear;
        if (global_thread < counter_count) {
            // Match gpubpf's ordinary per-thread map increment. Within one
            // launch each logical thread has a unique array slot.
            reinterpret_cast<uint64_t*>(counters_ptr)[global_thread] += 1;
        }
    }
}

extern "C" __device__ __noinline__ void observe_entry(
    uint64_t host_mono_ns, uint64_t calibration_ptr, uint64_t histogram_ptr,
    uint64_t sample_count_ptr, uint64_t uncertain_count_ptr,
    uint64_t clock_error_ptr) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    const uint64_t gpu_ns = read_globaltimer_ns();
    if (calibration_ptr == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(clock_error_ptr), 1ULL);
        return;
    }
    uint32_t bin = 0;
    const launch_sample_status_t status = classify_launch_latency(
        host_mono_ns, gpu_ns,
        *reinterpret_cast<const clock_calibration_t*>(calibration_ptr), &bin);
    if (status == LAUNCH_SAMPLE_CLOCK_ERROR) {
        atomicAdd(reinterpret_cast<unsigned long long*>(clock_error_ptr), 1ULL);
        return;
    }
    if (status == LAUNCH_SAMPLE_UNCERTAIN) {
        atomicAdd(reinterpret_cast<unsigned long long*>(uncertain_count_ptr),
                  1ULL);
        return;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(histogram_ptr) + bin, 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(sample_count_ptr), 1ULL);
}
