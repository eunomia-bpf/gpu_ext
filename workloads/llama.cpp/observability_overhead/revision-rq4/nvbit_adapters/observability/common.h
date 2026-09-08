#pragma once

#include <stdint.h>

enum observability_mode_t : uint32_t {
    OBS_KERNELRETSNOOP = 1,
    OBS_THREADHIST = 2,
    OBS_LAUNCHLATE = 3,
    OBS_KERNELRETSNOOP_WARP_ARRAY = 4,
};

struct exit_record_t {
    uint64_t coordinate_x;
    uint64_t coordinate_y;
    uint64_t coordinate_z;
    uint64_t timestamp;
};

static_assert(sizeof(exit_record_t) == 4 * sizeof(uint64_t),
              "exit_record_t ABI must remain four packed 64-bit fields");

// Frozen matched-granularity warp-exit array, byte-for-byte identical to the
// gpubpf onevalue-array candidate value (onevalue_array_value.h): 16384 warps,
// 44 events per warp, 32 bytes per event, 23199768 bytes total. The device
// stores exit_record_t events directly; total_committed is reserved (never
// written by the device, derived host-side as the sum of warp_counters).
#define WARP_ARRAY_MAX_WARPS 16384ULL
#define WARP_ARRAY_MAX_EVENTS_PER_WARP 44ULL

struct warp_array_value_t {
    uint64_t total_committed;
    uint64_t total_overflow;
    uint64_t total_out_of_range;
    uint64_t warp_counters[WARP_ARRAY_MAX_WARPS];
    exit_record_t events[WARP_ARRAY_MAX_WARPS * WARP_ARRAY_MAX_EVENTS_PER_WARP];
};

static_assert(sizeof(warp_array_value_t) == 23199768ULL,
              "warp_array_value_t must stay the frozen 23199768-byte payload");

static constexpr uint32_t HIST_BINS = 10;
static constexpr uint64_t LAUNCH_PAIR_CAPACITY = 65536ULL;

struct launch_pair_t {
    uint64_t host_raw_ns;
    uint64_t gpu_entry_ns;
    uint64_t sequence;
};

static_assert(sizeof(launch_pair_t) == 3 * sizeof(uint64_t),
              "launch_pair_t ABI must remain three packed 64-bit fields");
