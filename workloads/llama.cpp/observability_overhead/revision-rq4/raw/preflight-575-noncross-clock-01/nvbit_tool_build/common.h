#pragma once

#include <stdint.h>

enum observability_mode_t : uint32_t {
    OBS_KERNELRETSNOOP = 1,
    OBS_THREADHIST = 2,
    OBS_LAUNCHLATE = 3,
};

struct exit_record_t {
    uint64_t block_x;
    uint64_t block_y;
    uint64_t block_z;
    uint64_t thread_x;
    uint64_t thread_y;
    uint64_t thread_z;
    uint64_t timestamp;
};

static constexpr uint32_t HIST_BINS = 10;
static constexpr uint64_t LAUNCH_PAIR_CAPACITY = 65536ULL;

struct launch_pair_t {
    uint64_t host_mono_ns;
    uint64_t gpu_entry_ns;
    uint64_t sequence;
};

static_assert(sizeof(launch_pair_t) == 3 * sizeof(uint64_t),
              "launch_pair_t ABI must remain three packed 64-bit fields");
