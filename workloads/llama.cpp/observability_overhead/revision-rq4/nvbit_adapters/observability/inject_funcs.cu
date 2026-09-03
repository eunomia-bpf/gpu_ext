#include <stdint.h>

#include "common.h"
#include "utils/channel.hpp"

static __device__ __forceinline__ uint64_t read_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ __forceinline__ uint32_t launch_hist_bin(uint64_t delta_ns) {
    if (delta_ns < 100) return 0;
    if (delta_ns < 1000) return 1;
    if (delta_ns < 10000) return 2;
    if (delta_ns < 100000) return 3;
    if (delta_ns < 1000000) return 4;
    if (delta_ns < 10000000) return 5;
    if (delta_ns < 100000000) return 6;
    if (delta_ns < 1000000000) return 7;
    if (delta_ns < 10000000000ULL) return 8;
    return 9;
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
    uint64_t host_launch_ns, uint64_t histogram_ptr, uint64_t sample_count_ptr,
    uint64_t clock_error_ptr) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    const uint64_t gpu_ns = read_globaltimer_ns();
    if (gpu_ns < host_launch_ns) {
        atomicAdd(reinterpret_cast<unsigned long long*>(clock_error_ptr), 1ULL);
        return;
    }
    const uint32_t bin = launch_hist_bin(gpu_ns - host_launch_ns);
    atomicAdd(reinterpret_cast<unsigned long long*>(histogram_ptr) + bin, 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(sample_count_ptr), 1ULL);
}
