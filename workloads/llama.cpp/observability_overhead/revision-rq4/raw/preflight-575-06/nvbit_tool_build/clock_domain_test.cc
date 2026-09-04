#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "clock_domain.h"

int main() {
    clock_calibration_t calibration = {};
    assert(consider_clock_calibration_sample(&calibration, 1250, 1000, 1100));
    assert(calibration.offset_low_ns == 150);
    assert(calibration.offset_high_ns == 250);
    assert(calibration.uncertainty_ns == 50);
    assert(clock_calibration_valid(calibration));

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
    int64_t intersection_low;
    int64_t intersection_high;
    assert(clock_calibration_intersection(calibration, later,
                                          &intersection_low,
                                          &intersection_high));
    assert(intersection_low == 170);
    assert(intersection_high == 200);

    clock_calibration_t negative_offset = {};
    assert(consider_clock_calibration_sample(&negative_offset, 900, 1000,
                                             1040));
    assert(classify_launch_latency(2000, 2025, negative_offset, &bin) ==
           LAUNCH_SAMPLE_CLASSIFIED);
    assert(bin == 1);

    // Entirely negative intervals are clock errors, never zero-latency samples.
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

    clock_calibration_t disjoint = {};
    assert(consider_clock_calibration_sample(&disjoint, 4100, 4000, 4040));
    assert(!clock_calibration_intersection(calibration, disjoint,
                                           &intersection_low,
                                           &intersection_high));

    puts("NVBit launchlate clock-domain CPU test: PASS");
    return 0;
}
