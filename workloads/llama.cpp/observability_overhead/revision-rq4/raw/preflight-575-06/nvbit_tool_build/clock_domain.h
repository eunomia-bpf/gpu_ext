#pragma once

#include <limits.h>
#include <stdint.h>

struct clock_calibration_t {
    int64_t offset_low_ns;
    int64_t offset_high_ns;
    uint64_t uncertainty_ns;
    uint64_t valid;
};

enum launch_sample_status_t : uint32_t {
    LAUNCH_SAMPLE_CLASSIFIED = 0,
    LAUNCH_SAMPLE_UNCERTAIN = 1,
    LAUNCH_SAMPLE_CLOCK_ERROR = 2,
};

#if defined(__CUDACC__)
#define OBS_HOST_DEVICE __host__ __device__
#else
#define OBS_HOST_DEVICE
#endif

OBS_HOST_DEVICE static inline uint32_t launch_hist_bin(uint64_t delta_ns) {
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

OBS_HOST_DEVICE static inline bool subtract_i64(int64_t left, int64_t right,
                                                int64_t* result) {
    if ((right > 0 && left < INT64_MIN + right) ||
        (right < 0 && left > INT64_MAX + right)) {
        return false;
    }
    *result = left - right;
    return true;
}

OBS_HOST_DEVICE static inline uint64_t calibration_width(
    const clock_calibration_t& calibration) {
    return static_cast<uint64_t>(calibration.offset_high_ns) -
           static_cast<uint64_t>(calibration.offset_low_ns);
}

OBS_HOST_DEVICE static inline bool clock_calibration_valid(
    const clock_calibration_t& calibration) {
    if (calibration.valid != 1 ||
        calibration.offset_low_ns > calibration.offset_high_ns) {
        return false;
    }
    const uint64_t width = calibration_width(calibration);
    return calibration.uncertainty_ns == width / 2 + width % 2;
}

static inline bool consider_clock_calibration_sample(
    clock_calibration_t* best, uint64_t gpu_ns, uint64_t host_before_ns,
    uint64_t host_after_ns) {
    if (best == nullptr || gpu_ns == 0 || host_before_ns == 0 ||
        host_after_ns < host_before_ns || gpu_ns > INT64_MAX ||
        host_before_ns > INT64_MAX || host_after_ns > INT64_MAX) {
        return false;
    }

    const int64_t low = static_cast<int64_t>(gpu_ns) -
                        static_cast<int64_t>(host_after_ns);
    const int64_t high = static_cast<int64_t>(gpu_ns) -
                         static_cast<int64_t>(host_before_ns);
    const uint64_t width = host_after_ns - host_before_ns;
    if (low > high) {
        return false;
    }
    if (!clock_calibration_valid(*best) ||
        width < calibration_width(*best)) {
        best->offset_low_ns = low;
        best->offset_high_ns = high;
        best->uncertainty_ns = width / 2 + width % 2;
        best->valid = 1;
    }
    return true;
}

static inline bool clock_calibration_intersection(
    const clock_calibration_t& first, const clock_calibration_t& second,
    int64_t* low, int64_t* high) {
    if (low == nullptr || high == nullptr ||
        !clock_calibration_valid(first) || !clock_calibration_valid(second)) {
        return false;
    }
    *low = first.offset_low_ns > second.offset_low_ns
               ? first.offset_low_ns
               : second.offset_low_ns;
    *high = first.offset_high_ns < second.offset_high_ns
                ? first.offset_high_ns
                : second.offset_high_ns;
    return *low <= *high;
}

OBS_HOST_DEVICE static inline launch_sample_status_t classify_launch_latency(
    uint64_t host_mono_ns, uint64_t gpu_ns,
    const clock_calibration_t& calibration, uint32_t* bin) {
    if (bin == nullptr || host_mono_ns == 0 || gpu_ns == 0 ||
        host_mono_ns > INT64_MAX || gpu_ns > INT64_MAX ||
        !clock_calibration_valid(calibration)) {
        return LAUNCH_SAMPLE_CLOCK_ERROR;
    }

    const int64_t observed_ns = static_cast<int64_t>(gpu_ns) -
                                static_cast<int64_t>(host_mono_ns);
    int64_t latency_low_ns;
    int64_t latency_high_ns;
    if (!subtract_i64(observed_ns, calibration.offset_high_ns,
                      &latency_low_ns) ||
        !subtract_i64(observed_ns, calibration.offset_low_ns,
                      &latency_high_ns) ||
        latency_high_ns < 0) {
        return LAUNCH_SAMPLE_CLOCK_ERROR;
    }
    if (latency_low_ns < 0) {
        return LAUNCH_SAMPLE_UNCERTAIN;
    }

    const uint32_t low_bin =
        launch_hist_bin(static_cast<uint64_t>(latency_low_ns));
    const uint32_t high_bin =
        launch_hist_bin(static_cast<uint64_t>(latency_high_ns));
    if (low_bin != high_bin) {
        return LAUNCH_SAMPLE_UNCERTAIN;
    }
    *bin = low_bin;
    return LAUNCH_SAMPLE_CLASSIFIED;
}

#undef OBS_HOST_DEVICE
