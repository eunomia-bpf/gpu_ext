// Calibration control only: this program does not produce a launch-latency
// or performance result.

#include <cuda_runtime.h>

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../nvbit_adapters/observability/rm_ptimer_575.h"

namespace {

constexpr unsigned int kDefaultSamples = 200;
constexpr unsigned int kMaxSamples = 100000;

__global__ void read_globaltimer(uint64_t* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint64_t value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        *output = value;
        __threadfence_system();
    }
}

bool raw_ns(uint64_t* value) {
    struct timespec now = {};
    if (value == nullptr || clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0 ||
        now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L) {
        return false;
    }
    *value = static_cast<uint64_t>(now.tv_sec) * 1000000000ULL +
             static_cast<uint64_t>(now.tv_nsec);
    return true;
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool sample_well_formed(const struct rm_ptimer_575_sample& sample) {
    return sample.outer_before_raw_ns != 0 && sample.gpu_ptimer_ns != 0 &&
           sample.outer_before_raw_ns <= sample.cpu_before_raw_ns &&
           sample.cpu_before_raw_ns <= sample.cpu_after_raw_ns &&
           sample.cpu_after_raw_ns <= sample.outer_after_raw_ns &&
           sample.outer_after_raw_ns - sample.outer_before_raw_ns ==
               sample.outer_width_ns &&
           sample.cpu_after_raw_ns - sample.cpu_before_raw_ns ==
               sample.selected_gap_ns &&
           sample.offset_low_ns <= sample.offset_high_ns &&
           static_cast<uint64_t>(sample.offset_high_ns) -
                   static_cast<uint64_t>(sample.offset_low_ns) ==
               sample.bracket_width_ns &&
           sample.bracket_width_ns ==
               sample.selected_gap_ns + 2 * RM_PTIMER_QUANTIZATION_NS &&
           sample.outer_width_ns < RM_PTIMER_MAX_OUTER_NS &&
           sample.rm_status == 0;
}

bool identity_trial_valid(const struct rm_ptimer_575_sample& before,
                          uint64_t kernel_before_raw_ns,
                          uint64_t device_globaltimer_ns,
                          uint64_t kernel_after_raw_ns,
                          const struct rm_ptimer_575_sample& after) {
    return sample_well_formed(before) && sample_well_formed(after) &&
           before.outer_after_raw_ns <= kernel_before_raw_ns &&
           kernel_before_raw_ns <= kernel_after_raw_ns &&
           kernel_after_raw_ns <= after.outer_before_raw_ns &&
           before.gpu_ptimer_ns <= device_globaltimer_ns &&
           device_globaltimer_ns <= after.gpu_ptimer_ns;
}

int self_test() {
    struct rm_ptimer_575_sample before = {};
    before.outer_before_raw_ns = 900;
    before.cpu_before_raw_ns = 1000;
    before.cpu_after_raw_ns = 1100;
    before.outer_after_raw_ns = 1200;
    before.gpu_ptimer_ns = 2000;
    before.outer_width_ns = 300;
    before.selected_gap_ns = 100;
    before.offset_low_ns = 868;
    before.offset_high_ns = 1032;
    before.bracket_width_ns = 164;
    struct rm_ptimer_575_sample after = before;
    after.outer_before_raw_ns = 1500;
    after.cpu_before_raw_ns = 1600;
    after.cpu_after_raw_ns = 1700;
    after.outer_after_raw_ns = 1800;
    after.gpu_ptimer_ns = 2600;
    after.offset_low_ns = 868;
    after.offset_high_ns = 1032;
    if (!identity_trial_valid(before, 1300, 2300, 1400, after) ||
        identity_trial_valid(before, 1300, 1999, 1400, after) ||
        identity_trial_valid(before, 1300, 2601, 1400, after) ||
        identity_trial_valid(before, 1199, 2300, 1400, after) ||
        identity_trial_valid(before, 1300, 2300, 1501, after)) {
        return 1;
    }
    puts("rm_globaltimer_identity self-test: PASS");
    return 0;
}

bool parse_samples(const char* text, unsigned int* value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno || end == nullptr || *end != '\0' || parsed == 0 ||
        parsed > kMaxSamples) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) return self_test();
    unsigned int requested = kDefaultSamples;
    if (argc == 3 && strcmp(argv[1], "--samples") == 0) {
        if (!parse_samples(argv[2], &requested)) return 2;
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [--samples N] | --self-test\n", argv[0]);
        return 2;
    }

    rm_ptimer_575_client client;
    rm_ptimer_575_client_init(&client);
    uint64_t* device_value = nullptr;
    unsigned int attempted = 0;
    unsigned int accepted = 0;
    unsigned int rejected = 0;
    unsigned int containment_failures = 0;
    unsigned int raw_regressions = 0;
    unsigned int ptimer_regressions = 0;
    unsigned int cuda_errors = 0;
    uint64_t previous_raw = 0;
    uint64_t previous_ptimer = 0;
    bool setup_complete = rm_ptimer_575_open(&client) == 0;
    if (setup_complete) {
        setup_complete = cuda_ok(cudaMallocManaged(&device_value,
                                                   sizeof(*device_value)),
                                 "cudaMallocManaged");
    }

    for (unsigned int trial = 0; setup_complete && trial < requested; ++trial) {
        attempted++;
        struct rm_ptimer_575_sample before = {};
        struct rm_ptimer_575_sample after = {};
        uint64_t kernel_before_raw_ns = 0;
        uint64_t kernel_after_raw_ns = 0;
        bool valid = cuda_ok(cudaMemset(device_value, 0,
                                       sizeof(*device_value)), "cudaMemset") &&
                     cuda_ok(cudaDeviceSynchronize(), "pre-synchronize") &&
                     rm_ptimer_575_sample(&client, &before) == 0 &&
                     raw_ns(&kernel_before_raw_ns);
        if (valid) {
            read_globaltimer<<<1, 1>>>(device_value);
            valid = cuda_ok(cudaGetLastError(), "read_globaltimer launch") &&
                    cuda_ok(cudaDeviceSynchronize(), "kernel synchronize") &&
                    raw_ns(&kernel_after_raw_ns) &&
                    rm_ptimer_575_sample(&client, &after) == 0;
        }
        if (!valid) cuda_errors++;
        const bool raw_regression = previous_raw != 0 &&
            (before.cpu_before_raw_ns < previous_raw ||
             after.cpu_before_raw_ns < before.cpu_before_raw_ns);
        const bool ptimer_regression = previous_ptimer != 0 &&
            (before.gpu_ptimer_ns < previous_ptimer ||
             after.gpu_ptimer_ns < before.gpu_ptimer_ns);
        const bool contained = valid && identity_trial_valid(
            before, kernel_before_raw_ns, *device_value, kernel_after_raw_ns,
            after);
        if (raw_regression) raw_regressions++;
        if (ptimer_regression) ptimer_regressions++;
        if (valid && !contained) containment_failures++;
        const bool admitted = valid && contained && !raw_regression &&
                              !ptimer_regression;
        if (admitted) accepted++; else rejected++;
        printf("{\"type\":\"identity_sample\",\"trial\":%u,"
               "\"rm_before_outer_before_raw_ns\":%" PRIu64 ","
               "\"rm_before_cpu_before_raw_ns\":%" PRIu64 ","
               "\"rm_before_gpu_ptimer_ns\":%" PRIu64 ","
               "\"rm_before_cpu_after_raw_ns\":%" PRIu64 ","
               "\"rm_before_outer_after_raw_ns\":%" PRIu64 ","
               "\"rm_before_offset_low_ns\":%" PRId64 ","
               "\"rm_before_offset_high_ns\":%" PRId64 ","
               "\"kernel_before_raw_ns\":%" PRIu64 ","
               "\"device_globaltimer_ns\":%" PRIu64 ","
               "\"kernel_after_raw_ns\":%" PRIu64 ","
               "\"rm_after_outer_before_raw_ns\":%" PRIu64 ","
               "\"rm_after_cpu_before_raw_ns\":%" PRIu64 ","
               "\"rm_after_gpu_ptimer_ns\":%" PRIu64 ","
               "\"rm_after_cpu_after_raw_ns\":%" PRIu64 ","
               "\"rm_after_outer_after_raw_ns\":%" PRIu64 ","
               "\"rm_after_offset_low_ns\":%" PRId64 ","
               "\"rm_after_offset_high_ns\":%" PRId64 ","
               "\"before_bracket_width_ns\":%" PRIu64 ","
               "\"after_bracket_width_ns\":%" PRIu64 ","
               "\"contained\":%s,\"accepted\":%s}\n",
               trial, before.outer_before_raw_ns, before.cpu_before_raw_ns,
               before.gpu_ptimer_ns, before.cpu_after_raw_ns,
               before.outer_after_raw_ns, before.offset_low_ns,
               before.offset_high_ns, kernel_before_raw_ns, *device_value,
               kernel_after_raw_ns, after.outer_before_raw_ns,
               after.cpu_before_raw_ns, after.gpu_ptimer_ns,
               after.cpu_after_raw_ns, after.outer_after_raw_ns,
               after.offset_low_ns, after.offset_high_ns,
               before.bracket_width_ns, after.bracket_width_ns,
               contained ? "true" : "false", admitted ? "true" : "false");
        previous_raw = after.cpu_before_raw_ns;
        previous_ptimer = after.gpu_ptimer_ns;
        if (!valid) break;
    }

    bool cleanup_complete = true;
    if (device_value != nullptr && !cuda_ok(cudaFree(device_value), "cudaFree"))
        cleanup_complete = false;
    if (setup_complete && !cuda_ok(cudaDeviceReset(), "cudaDeviceReset"))
        cleanup_complete = false;
    if (rm_ptimer_575_close(&client) != 0) cleanup_complete = false;
    const bool passed = setup_complete && attempted == requested &&
        accepted == requested && rejected == 0 && containment_failures == 0 &&
        raw_regressions == 0 && ptimer_regressions == 0 && cuda_errors == 0 &&
        cleanup_complete && !ferror(stdout);
    printf("{\"type\":\"identity_summary\",\"requested\":%u,"
           "\"attempted\":%u,\"accepted\":%u,\"rejected\":%u,"
           "\"containment_failures\":%u,\"raw_regressions\":%u,"
           "\"ptimer_regressions\":%u,\"cuda_errors\":%u,"
           "\"setup_complete\":%s,\"cleanup_complete\":%s,"
           "\"gate_passed\":%s}\n",
           requested, attempted, accepted, rejected, containment_failures,
           raw_regressions, ptimer_regressions, cuda_errors,
           setup_complete ? "true" : "false",
           cleanup_complete ? "true" : "false", passed ? "true" : "false");
    return passed ? 0 : 1;
}
