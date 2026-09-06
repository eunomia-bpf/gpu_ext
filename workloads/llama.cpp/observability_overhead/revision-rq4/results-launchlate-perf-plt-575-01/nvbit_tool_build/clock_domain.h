#pragma once

#include <limits.h>
#include <stdint.h>

struct clock_calibration_t {
    int64_t offset_low_ns;
    int64_t offset_high_ns;
    uint64_t uncertainty_ns;
    uint64_t host_anchor_ns;
    uint64_t valid;
};

struct clock_drift_t {
    int64_t offset_change_low_ns;
    int64_t offset_change_high_ns;
    uint64_t elapsed_ns;
    uint64_t rate_bound_ppb;
    uint64_t bounded;
};

// Retained for a comparable slope diagnostic; it is not an admission limit.
static constexpr uint64_t CLOCK_DRIFT_LIMIT_PPB = 10000ULL;
static constexpr uint64_t CLOCK_CALIBRATION_TRIALS = 32ULL;
static constexpr uint64_t CLOCK_MAX_ANCHOR_BRACKET_NS = 1500ULL;
// Keep the held-out validation anchor distinct from measurement_end.  A full
// second prevents the validation from being an effectively coincident repeat.
static constexpr uint64_t CLOCK_MIN_CALIBRATION_SPAN_NS = 1000000000ULL;

struct clock_anchor_quality_t {
    uint64_t requested;
    uint64_t accepted;
    uint64_t rejected;
    uint64_t bracket_width_ns;
    uint64_t cleanup_complete;
};

struct held_out_clock_validation_t {
    int64_t predicted_low_ns;
    int64_t predicted_high_ns;
    int64_t overlap_low_ns;
    int64_t overlap_high_ns;
    uint64_t validation_span_ns;
    uint64_t passed;
};

enum launch_sample_status_t : uint32_t {
    LAUNCH_SAMPLE_CLASSIFIED = 0,
    LAUNCH_SAMPLE_UNCERTAIN = 1,
    LAUNCH_SAMPLE_CLOCK_ERROR = 2,
};

#if defined(__CUDACC__)
#define OBS_HOST_DEVICE __host__ __device__
#define OBS_HOST __host__
#else
#define OBS_HOST_DEVICE
#define OBS_HOST
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
        calibration.host_anchor_ns == 0 ||
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
        best->host_anchor_ns = host_before_ns + width / 2;
        best->valid = 1;
    }
    return true;
}

static inline uint64_t abs_i64(int64_t value) {
    return value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1
                     : static_cast<uint64_t>(value);
}

static inline bool minimum_end_calibration_deadline(
    const clock_calibration_t& start, uint64_t* deadline_ns) {
    if (deadline_ns == nullptr || !clock_calibration_valid(start) ||
        start.host_anchor_ns >
            UINT64_MAX - CLOCK_MIN_CALIBRATION_SPAN_NS) {
        return false;
    }
    *deadline_ns = start.host_anchor_ns + CLOCK_MIN_CALIBRATION_SPAN_NS;
    return true;
}

static inline bool clock_calibration_drift(
    const clock_calibration_t& start, const clock_calibration_t& end,
    clock_drift_t* drift) {
    if (drift == nullptr) {
        return false;
    }
    *drift = {};
    if (!clock_calibration_valid(start) || !clock_calibration_valid(end) ||
        end.host_anchor_ns <= start.host_anchor_ns ||
        !subtract_i64(end.offset_low_ns, start.offset_high_ns,
                      &drift->offset_change_low_ns) ||
        !subtract_i64(end.offset_high_ns, start.offset_low_ns,
                      &drift->offset_change_high_ns)) {
        return false;
    }
    drift->elapsed_ns = end.host_anchor_ns - start.host_anchor_ns;
    uint64_t largest_change = abs_i64(drift->offset_change_low_ns);
    const uint64_t high_change = abs_i64(drift->offset_change_high_ns);
    if (high_change > largest_change) largest_change = high_change;
    if (largest_change > UINT64_MAX / 1000000000ULL) return false;
    const uint64_t scaled = largest_change * 1000000000ULL;
    drift->rate_bound_ppb = scaled / drift->elapsed_ns +
                            (scaled % drift->elapsed_ns != 0);
    drift->bounded = drift->rate_bound_ppb <= CLOCK_DRIFT_LIMIT_PPB;
    return true;
}

OBS_HOST_DEVICE static inline launch_sample_status_t classify_launch_latency(
    uint64_t host_raw_ns, uint64_t gpu_ns,
    const clock_calibration_t& calibration, uint32_t* bin) {
    if (bin == nullptr || host_raw_ns == 0 || gpu_ns == 0 ||
        host_raw_ns > INT64_MAX || gpu_ns > INT64_MAX ||
        !clock_calibration_valid(calibration)) {
        return LAUNCH_SAMPLE_CLOCK_ERROR;
    }

    const int64_t observed_ns = static_cast<int64_t>(gpu_ns) -
                                static_cast<int64_t>(host_raw_ns);
    int64_t latency_low_ns;
    int64_t latency_high_ns;
    if (!subtract_i64(observed_ns, calibration.offset_high_ns,
                      &latency_low_ns) ||
        !subtract_i64(observed_ns, calibration.offset_low_ns,
                      &latency_high_ns)) {
        return LAUNCH_SAMPLE_CLOCK_ERROR;
    }
    if (latency_high_ns < 0) {
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

OBS_HOST static inline __int128 divide_floor_i128(__int128 numerator,
                                                  uint64_t denominator) {
    __int128 quotient = numerator / denominator;
    if (numerator % denominator < 0) quotient--;
    return quotient;
}

OBS_HOST static inline __int128 divide_ceil_i128(__int128 numerator,
                                                 uint64_t denominator) {
    __int128 quotient = numerator / denominator;
    if (numerator % denominator > 0) quotient++;
    return quotient;
}

OBS_HOST static inline bool affine_clock_offset_interval(
    uint64_t host_ns, const clock_calibration_t& start,
    const clock_calibration_t& end, int64_t* offset_low_ns,
    int64_t* offset_high_ns) {
    if (offset_low_ns == nullptr || offset_high_ns == nullptr ||
        !clock_calibration_valid(start) || !clock_calibration_valid(end) ||
        end.host_anchor_ns <= start.host_anchor_ns ||
        host_ns < start.host_anchor_ns || host_ns > end.host_anchor_ns) {
        return false;
    }

    const uint64_t elapsed_ns = end.host_anchor_ns - start.host_anchor_ns;
    const uint64_t position_ns = host_ns - start.host_anchor_ns;
    const __int128 low = static_cast<__int128>(start.offset_low_ns) +
        divide_floor_i128(
            (static_cast<__int128>(end.offset_low_ns) -
             static_cast<__int128>(start.offset_low_ns)) * position_ns,
            elapsed_ns);
    const __int128 high = static_cast<__int128>(start.offset_high_ns) +
        divide_ceil_i128(
            (static_cast<__int128>(end.offset_high_ns) -
             static_cast<__int128>(start.offset_high_ns)) * position_ns,
            elapsed_ns);
    if (low < INT64_MIN || low > INT64_MAX || high < INT64_MIN ||
        high > INT64_MAX || low > high) {
        return false;
    }
    *offset_low_ns = static_cast<int64_t>(low);
    *offset_high_ns = static_cast<int64_t>(high);
    return true;
}

static inline bool clock_anchor_quality_valid(
    const clock_calibration_t& calibration,
    const clock_anchor_quality_t& quality) {
    return clock_calibration_valid(calibration) &&
           quality.requested == CLOCK_CALIBRATION_TRIALS &&
           quality.accepted == quality.requested && quality.rejected == 0 &&
           quality.cleanup_complete == 1 && quality.bracket_width_ns > 0 &&
           quality.bracket_width_ns <= CLOCK_MAX_ANCHOR_BRACKET_NS &&
           calibration_width(calibration) == quality.bracket_width_ns;
}

/*
 * Validate the start-to-measurement affine model against a third, held-out
 * anchor.  Only start and measurement_end may classify samples.  The later
 * validation_end anchor is an admission check: the affine interval from
 * start to validation_end must overlap the measured middle interval.
 */
static inline bool held_out_affine_clock_validation(
    const clock_calibration_t& start,
    const clock_calibration_t& measurement_end,
    const clock_calibration_t& validation_end,
    const clock_anchor_quality_t& start_quality,
    const clock_anchor_quality_t& measurement_end_quality,
    const clock_anchor_quality_t& validation_end_quality,
    held_out_clock_validation_t* result) {
    if (result == nullptr) return false;
    *result = {};
    if (!clock_anchor_quality_valid(start, start_quality) ||
        !clock_anchor_quality_valid(measurement_end,
                                    measurement_end_quality) ||
        !clock_anchor_quality_valid(validation_end, validation_end_quality) ||
        start.host_anchor_ns >= measurement_end.host_anchor_ns ||
        measurement_end.host_anchor_ns >= validation_end.host_anchor_ns ||
        validation_end.host_anchor_ns - measurement_end.host_anchor_ns <
            CLOCK_MIN_CALIBRATION_SPAN_NS ||
        !affine_clock_offset_interval(measurement_end.host_anchor_ns, start,
                                      validation_end,
                                      &result->predicted_low_ns,
                                      &result->predicted_high_ns)) {
        return false;
    }
    result->validation_span_ns = validation_end.host_anchor_ns -
                                 measurement_end.host_anchor_ns;
    result->overlap_low_ns =
        result->predicted_low_ns > measurement_end.offset_low_ns
            ? result->predicted_low_ns
            : measurement_end.offset_low_ns;
    result->overlap_high_ns =
        result->predicted_high_ns < measurement_end.offset_high_ns
            ? result->predicted_high_ns
            : measurement_end.offset_high_ns;
    if (result->overlap_low_ns > result->overlap_high_ns) return false;
    result->passed = 1;
    return true;
}

OBS_HOST static inline launch_sample_status_t classify_affine_launch_latency(
    uint64_t host_raw_ns, uint64_t gpu_ns,
    const clock_calibration_t& start, const clock_calibration_t& end,
    uint32_t* bin) {
    if (bin == nullptr || host_raw_ns == 0 || gpu_ns == 0 ||
        host_raw_ns > INT64_MAX || gpu_ns > INT64_MAX) {
        return LAUNCH_SAMPLE_CLOCK_ERROR;
    }

    int64_t offset_low_ns;
    int64_t offset_high_ns;
    if (!affine_clock_offset_interval(host_raw_ns, start, end,
                                      &offset_low_ns, &offset_high_ns)) {
        return LAUNCH_SAMPLE_CLOCK_ERROR;
    }

    const int64_t observed_ns = static_cast<int64_t>(gpu_ns) -
                                static_cast<int64_t>(host_raw_ns);
    int64_t latency_low_ns;
    int64_t latency_high_ns;
    if (!subtract_i64(observed_ns, offset_high_ns, &latency_low_ns) ||
        !subtract_i64(observed_ns, offset_low_ns, &latency_high_ns) ||
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

#undef OBS_HOST
#undef OBS_HOST_DEVICE
