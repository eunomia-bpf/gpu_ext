#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "clock_domain.h"
#include "rm_ptimer_575.h"

int main() {
    assert(rm_ptimer_575_self_test() == 0);
    clock_calibration_t calibration = {};
    assert(consider_clock_calibration_sample(&calibration, 1250, 1000, 1100));
    assert(calibration.offset_low_ns == 150);
    assert(calibration.offset_high_ns == 250);
    assert(calibration.uncertainty_ns == 50);
    assert(calibration.host_anchor_ns == 1050);
    assert(clock_calibration_valid(calibration));

    uint64_t deadline_ns = 0;
    assert(minimum_end_calibration_deadline(calibration, &deadline_ns));
    assert(deadline_ns == calibration.host_anchor_ns +
                              CLOCK_MIN_CALIBRATION_SPAN_NS);
    assert(!minimum_end_calibration_deadline(calibration, nullptr));
    clock_calibration_t overflow_deadline = calibration;
    overflow_deadline.host_anchor_ns =
        UINT64_MAX - CLOCK_MIN_CALIBRATION_SPAN_NS + 1;
    assert(!minimum_end_calibration_deadline(overflow_deadline,
                                             &deadline_ns));

    assert(consider_clock_calibration_sample(&calibration, 2210, 2000, 2040));
    assert(calibration.offset_low_ns == 170);
    assert(calibration.offset_high_ns == 210);
    assert(calibration.uncertainty_ns == 20);
    assert(consider_clock_calibration_sample(&calibration, 1040, 800, 840));
    assert(calibration.offset_low_ns == 170);
    assert(calibration.offset_high_ns == 210);

    uint32_t bin = UINT32_MAX;
    assert(classify_launch_latency(5000, 5400, calibration, &bin) ==
           LAUNCH_SAMPLE_CLASSIFIED);
    assert(bin == 1);

    clock_calibration_t later = {};
    assert(consider_clock_calibration_sample(&later, 3200, 3000, 3040));
    assert(later.host_anchor_ns == 3020);

    clock_calibration_t drift_start = {100, 120, 10, 1000000000ULL, 1};
    clock_calibration_t drift_end = {300, 340, 20, 1100000000ULL, 1};
    clock_drift_t drift = {};
    assert(clock_calibration_drift(drift_start, drift_end, &drift));
    assert(drift.offset_change_low_ns == 180);
    assert(drift.offset_change_high_ns == 240);
    assert(drift.elapsed_ns == 100000000ULL);
    assert(drift.rate_bound_ppb == 2400);
    assert(drift.bounded == 1);

    int64_t affine_low = 0;
    int64_t affine_high = 0;
    assert(affine_clock_offset_interval(1050000000ULL, drift_start, drift_end,
                                        &affine_low, &affine_high));
    assert(affine_low == 200);
    assert(affine_high == 230);
    assert(classify_affine_launch_latency(1050000000ULL, 1050000720ULL,
                                          drift_start, drift_end, &bin) ==
           LAUNCH_SAMPLE_CLASSIFIED);
    assert(bin == 1);
    // The interpolated interval crosses 1 us and must remain uncertain.
    assert(classify_affine_launch_latency(1050000000ULL, 1050001210ULL,
                                          drift_start, drift_end, &bin) ==
           LAUNCH_SAMPLE_UNCERTAIN);
    // A complete negative interval is causally impossible.
    assert(classify_affine_launch_latency(1050000000ULL, 1050000100ULL,
                                          drift_start, drift_end, &bin) ==
           LAUNCH_SAMPLE_CLOCK_ERROR);
    // Samples outside the two calibration anchors fail closed.
    assert(classify_affine_launch_latency(999999999ULL, 1000000719ULL,
                                          drift_start, drift_end, &bin) ==
           LAUNCH_SAMPLE_CLOCK_ERROR);

    clock_calibration_t negative_offset = {};
    assert(consider_clock_calibration_sample(&negative_offset, 900, 1000,
                                             1040));
    assert(classify_launch_latency(2000, 2025, negative_offset, &bin) ==
           LAUNCH_SAMPLE_CLASSIFIED);
    assert(bin == 1);

    // A complete negative interval is causally impossible, not uncertainty.
    assert(classify_launch_latency(2000, 1800, negative_offset, &bin) ==
           LAUNCH_SAMPLE_CLOCK_ERROR);
    // Intervals that merely might be negative are rejected as uncertain.
    assert(classify_launch_latency(2000, 1880, negative_offset, &bin) ==
           LAUNCH_SAMPLE_UNCERTAIN);
    // A calibration interval spanning a histogram boundary is also uncertain.
    assert(classify_launch_latency(5000, 5290, calibration, &bin) ==
           LAUNCH_SAMPLE_UNCERTAIN);

    clock_calibration_t invalid = calibration;
    invalid.uncertainty_ns++;
    assert(classify_launch_latency(5000, 5400, invalid, &bin) ==
           LAUNCH_SAMPLE_CLOCK_ERROR);
    assert(classify_launch_latency(UINT64_MAX, 5400, calibration, &bin) ==
           LAUNCH_SAMPLE_CLOCK_ERROR);

    // A large endpoint change crosses the retained diagnostic threshold.
    drift_end.offset_low_ns = 2000100;
    drift_end.offset_high_ns = 2000140;
    assert(clock_calibration_drift(drift_start, drift_end, &drift));
    assert(drift.bounded == 0);
    assert(drift.rate_bound_ppb > CLOCK_DRIFT_LIMIT_PPB);

    // A 22 ppm affine clock passes when the observed middle anchor overlaps
    // the start-to-held-out-validation interpolation.  Slope is diagnostic.
    const clock_calibration_t held_start = {
        980, 1020, 20, 1000000000ULL, 1};
    const clock_calibration_t held_measurement_end = {
        9760, 9840, 40, 1400000000ULL, 1};
    const clock_calibration_t held_validation_end = {
        31780, 31820, 20, 2400000000ULL, 1};
    const clock_anchor_quality_t narrow_quality = {
        CLOCK_CALIBRATION_TRIALS, CLOCK_CALIBRATION_TRIALS, 0, 40, 1};
    const clock_anchor_quality_t middle_quality = {
        CLOCK_CALIBRATION_TRIALS, CLOCK_CALIBRATION_TRIALS, 0, 80, 1};
    held_out_clock_validation_t held_validation = {};
    assert(clock_calibration_drift(held_start, held_validation_end, &drift));
    assert(drift.rate_bound_ppb == 22029);
    assert(drift.bounded == 0);
    assert(held_out_affine_clock_validation(
        held_start, held_measurement_end, held_validation_end, narrow_quality,
        middle_quality, narrow_quality, &held_validation));
    assert(held_validation.predicted_low_ns == 9780);
    assert(held_validation.predicted_high_ns == 9820);
    assert(held_validation.overlap_low_ns == 9780);
    assert(held_validation.overlap_high_ns == 9820);
    assert(held_validation.validation_span_ns == 1000000000ULL);
    assert(held_validation.passed == 1);
    assert(classify_affine_launch_latency(
               1200000000ULL, 1200006000ULL, held_start,
               held_measurement_end, &bin) == LAUNCH_SAMPLE_CLASSIFIED);
    assert(bin == 1);

    clock_calibration_t malformed_middle = held_measurement_end;
    malformed_middle.uncertainty_ns++;
    assert(!held_out_affine_clock_validation(
        held_start, malformed_middle, held_validation_end, narrow_quality,
        middle_quality, narrow_quality, &held_validation));
    clock_calibration_t nonoverlap_middle = held_measurement_end;
    nonoverlap_middle.offset_low_ns = 11000;
    nonoverlap_middle.offset_high_ns = 11080;
    assert(!held_out_affine_clock_validation(
        held_start, nonoverlap_middle, held_validation_end, narrow_quality,
        middle_quality, narrow_quality, &held_validation));
    clock_calibration_t unordered_middle = held_measurement_end;
    unordered_middle.host_anchor_ns = held_start.host_anchor_ns;
    assert(!held_out_affine_clock_validation(
        held_start, unordered_middle, held_validation_end, narrow_quality,
        middle_quality, narrow_quality, &held_validation));
    clock_calibration_t unordered_validation = held_validation_end;
    unordered_validation.host_anchor_ns =
        held_measurement_end.host_anchor_ns;
    assert(!held_out_affine_clock_validation(
        held_start, held_measurement_end, unordered_validation,
        narrow_quality, middle_quality, narrow_quality, &held_validation));
    clock_calibration_t short_validation = held_validation_end;
    short_validation.host_anchor_ns = 2399999999ULL;
    assert(!held_out_affine_clock_validation(
        held_start, held_measurement_end, short_validation, narrow_quality,
        middle_quality, narrow_quality, &held_validation));
    clock_anchor_quality_t rejected_quality = narrow_quality;
    rejected_quality.rejected = 1;
    assert(!held_out_affine_clock_validation(
        held_start, held_measurement_end, held_validation_end,
        rejected_quality, middle_quality, narrow_quality, &held_validation));
    clock_anchor_quality_t incomplete_quality = narrow_quality;
    incomplete_quality.accepted--;
    assert(!held_out_affine_clock_validation(
        held_start, held_measurement_end, held_validation_end,
        incomplete_quality, middle_quality, narrow_quality,
        &held_validation));
    clock_anchor_quality_t wide_quality = middle_quality;
    wide_quality.bracket_width_ns = CLOCK_MAX_ANCHOR_BRACKET_NS + 1;
    clock_calibration_t wide_middle = held_measurement_end;
    wide_middle.offset_high_ns =
        wide_middle.offset_low_ns + wide_quality.bracket_width_ns;
    wide_middle.uncertainty_ns =
        wide_quality.bracket_width_ns / 2 +
        wide_quality.bracket_width_ns % 2;
    assert(!held_out_affine_clock_validation(
        held_start, wide_middle, held_validation_end, narrow_quality,
        wide_quality, narrow_quality, &held_validation));
    clock_anchor_quality_t dirty_quality = narrow_quality;
    dirty_quality.cleanup_complete = 0;
    assert(!held_out_affine_clock_validation(
        held_start, held_measurement_end, held_validation_end, narrow_quality,
        middle_quality, dirty_quality, &held_validation));
    assert(!held_out_affine_clock_validation(
        held_start, held_measurement_end, held_validation_end, narrow_quality,
        middle_quality, narrow_quality, nullptr));

    // Invalid raw inputs remain true clock errors.
    assert(classify_launch_latency(0, 5400, calibration, &bin) ==
           LAUNCH_SAMPLE_CLOCK_ERROR);

    puts("NVBit launchlate clock-domain CPU test: PASS");
    return 0;
}
