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
