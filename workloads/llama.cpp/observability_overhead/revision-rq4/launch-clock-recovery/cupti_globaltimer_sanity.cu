// Diagnostic only: CUPTI does not document cuptiGetTimestamp as raw PTIMER.
// Compiling this probe is safe; running it uses a GPU and is a separate action.

#include <cuda_runtime.h>
#include <cupti.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

namespace {

constexpr unsigned int kTrials = 32;

__global__ void sample_globaltimer(uint64_t* output) {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = value;
        __threadfence_system();
    }
}

bool monotonic_ns(uint64_t* value) {
    struct timespec now = {};
    if (value == nullptr || clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
        now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L) {
        return false;
    }
    *value = static_cast<uint64_t>(now.tv_sec) * 1000000000ULL +
             static_cast<uint64_t>(now.tv_nsec);
    return true;
}

bool cupti_timestamp_with_host_bracket(uint64_t* host_before_ns,
                                       uint64_t* cupti_ns,
                                       uint64_t* host_after_ns) {
    return monotonic_ns(host_before_ns) &&
           cuptiGetTimestamp(cupti_ns) == CUPTI_SUCCESS &&
           monotonic_ns(host_after_ns) &&
           *host_after_ns >= *host_before_ns;
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    uint64_t* raw_globaltimer = nullptr;
    if (!cuda_ok(cudaMallocManaged(&raw_globaltimer, sizeof(*raw_globaltimer)),
                 "cudaMallocManaged")) {
        return 1;
    }

    unsigned int same_domain_observations = 0;
    for (unsigned int trial = 0; trial < kTrials; ++trial) {
        if (!cuda_ok(cudaMemset(raw_globaltimer, 0,
                               sizeof(*raw_globaltimer)),
                     "cudaMemset") ||
            !cuda_ok(cudaDeviceSynchronize(), "initial synchronize")) {
            cudaFree(raw_globaltimer);
            return 1;
        }

        uint64_t mono_before_first = 0;
        uint64_t cupti_before = 0;
        uint64_t mono_after_first = 0;
        if (!cupti_timestamp_with_host_bracket(
                &mono_before_first, &cupti_before, &mono_after_first)) {
            fprintf(stderr, "failed to bracket first CUPTI timestamp\n");
            cudaFree(raw_globaltimer);
            return 1;
        }

        sample_globaltimer<<<1, 1>>>(raw_globaltimer);
        if (!cuda_ok(cudaGetLastError(), "sample_globaltimer launch") ||
            !cuda_ok(cudaDeviceSynchronize(), "sample synchronize")) {
            cudaFree(raw_globaltimer);
            return 1;
        }

        uint64_t mono_before_second = 0;
        uint64_t cupti_after = 0;
        uint64_t mono_after_second = 0;
        if (!cupti_timestamp_with_host_bracket(
                &mono_before_second, &cupti_after, &mono_after_second)) {
            fprintf(stderr, "failed to bracket second CUPTI timestamp\n");
            cudaFree(raw_globaltimer);
            return 1;
        }

        const bool ordered = cupti_before <= *raw_globaltimer &&
                             *raw_globaltimer <= cupti_after;
        same_domain_observations += ordered ? 1U : 0U;
        const int64_t candidate_offset_low =
            static_cast<int64_t>(cupti_before) -
            static_cast<int64_t>(mono_after_first);
        const int64_t candidate_offset_high =
            static_cast<int64_t>(cupti_before) -
            static_cast<int64_t>(mono_before_first);

        printf("trial=%u cupti_host_bracket_ns=%" PRIu64
               " candidate_offset_low_ns=%" PRId64
               " candidate_offset_high_ns=%" PRId64
               " raw_between_cupti=%u second_host_bracket_ns=%" PRIu64
               "\n",
               trial, mono_after_first - mono_before_first,
               candidate_offset_low, candidate_offset_high,
               ordered ? 1U : 0U,
               mono_after_second - mono_before_second);
    }

    cudaFree(raw_globaltimer);
    printf("diagnostic_only=1 raw_between_cupti_trials=%u total_trials=%u\n",
           same_domain_observations, kTrials);
    if (same_domain_observations != kTrials) {
        fprintf(stderr,
                "CUPTI/raw-globaltimer same-domain assumption rejected\n");
        return 2;
    }
    fprintf(stderr,
            "local ordering observed, but the CUPTI API does not guarantee "
            "the raw clock domain; do not enable production calibration\n");
    return 0;
}
