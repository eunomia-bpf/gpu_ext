#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct raw_record {
    uint64_t sequence;
    uint64_t block_x;
    uint64_t block_y;
    uint64_t block_z;
    uint64_t thread_x;
    uint64_t thread_y;
    uint64_t thread_z;
};

static_assert(sizeof(raw_record) == 56, "raw-record ABI changed");

#define CUDA_CHECK(call) do {                                                    \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error__)); \
        return 2;                                                               \
    }                                                                           \
} while (0)

extern "C" __global__ void raw_map_kernel(raw_record *records,
                                            uint64_t sequence)
{
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x
                          + threadIdx.x;
    records[linear] = {
        sequence,
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
    };
}

static bool parse_positive(const char *text, uint64_t *value)
{
    errno = 0;
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || !end || *end != '\0' || parsed == 0)
        return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

int main(int argc, char **argv)
{
    constexpr uint64_t block_dim = 128;
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s THREADS LAUNCHES\n", argv[0]);
        return 64;
    }
    uint64_t threads = 0, launches = 0;
    if (!parse_positive(argv[1], &threads) || !parse_positive(argv[2], &launches)
            || threads % block_dim || threads > (1ULL << 20) || launches > 1024) {
        std::fprintf(stderr, "invalid geometry\n");
        return 65;
    }
    const uint64_t blocks = threads / block_dim;
    if (launches > SIZE_MAX / threads
            || launches * threads > SIZE_MAX / sizeof(raw_record)) {
        std::fprintf(stderr, "record allocation overflow\n");
        return 65;
    }

    raw_record *device = nullptr;
    std::vector<raw_record> host(static_cast<size_t>(threads * launches));
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(raw_record)));
    for (uint64_t launch = 0; launch < launches; ++launch) {
        raw_map_kernel<<<static_cast<unsigned>(blocks),
                         static_cast<unsigned>(block_dim)>>>(
            device + launch * threads, launch + 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(raw_record),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));

    uint64_t checked = 0;
    for (uint64_t launch = 0; launch < launches; ++launch) {
        for (uint64_t linear = 0; linear < threads; ++linear) {
            const raw_record &record = host[launch * threads + linear];
            const raw_record expected = {
                launch + 1, linear / block_dim, 0, 0, linear % block_dim, 0, 0,
            };
            if (record.sequence != expected.sequence
                    || record.block_x != expected.block_x
                    || record.block_y != expected.block_y
                    || record.block_z != expected.block_z
                    || record.thread_x != expected.thread_x
                    || record.thread_y != expected.thread_y
                    || record.thread_z != expected.thread_z) {
                std::fprintf(stderr,
                    "truth mismatch launch=%" PRIu64 " linear=%" PRIu64 "\n",
                    launch, linear);
                return 3;
            }
            ++checked;
            std::printf(
                "{\"event\":\"cuda_truth\",\"sequence\":%" PRIu64
                ",\"block_x\":%" PRIu64 ",\"block_y\":%" PRIu64
                ",\"block_z\":%" PRIu64 ",\"thread_x\":%" PRIu64
                ",\"thread_y\":%" PRIu64 ",\"thread_z\":%" PRIu64 "}\n",
                record.sequence, record.block_x, record.block_y, record.block_z,
                record.thread_x, record.thread_y, record.thread_z);
        }
    }
    std::printf(
        "{\"event\":\"cuda_summary\",\"threads\":%" PRIu64
        ",\"blocks\":%" PRIu64 ",\"threads_per_block\":%" PRIu64
        ",\"launches\":%" PRIu64 ",\"truth_records\":%" PRIu64
        ",\"checked_records\":%" PRIu64 ",\"mismatches\":0}\n",
        threads, blocks, block_dim, launches, checked, checked);
    return 0;
}

