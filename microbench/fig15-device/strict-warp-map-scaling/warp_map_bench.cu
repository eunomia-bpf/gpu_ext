#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                        \
	do {                                                                      \
		const cudaError_t cuda_error = (call);                               \
		if (cuda_error != cudaSuccess) {                                     \
			std::fprintf(stderr, "%s: %s\n", #call,                       \
				     cudaGetErrorString(cuda_error));                    \
			return 2;                                                       \
		}                                                                     \
	} while (0)

static constexpr unsigned kWarpSize = 32;
static constexpr unsigned kMaximumThreads = 1024;
static constexpr uint64_t kSeedBase = UINT64_C(0x1797575000000000);

extern "C" __device__ __noinline__ __attribute__((used)) void
__bpftime_cuda__kernel_trace(void)
{
	asm volatile("" ::: "memory");
}

__host__ __device__ static uint64_t expected_value(unsigned thread,
						   uint64_t seed)
{
	uint64_t value = seed ^ ((uint64_t)thread * UINT64_C(0x9e3779b97f4a7c15));
	value = (value ^ (value >> 29)) * UINT64_C(0x94d049bb133111eb);
	return value ^ ((uint64_t)thread << 17);
}

extern "C" __global__ void fig15_warp_map_kernel(uint64_t *output,
						 uint64_t seed)
{
	const unsigned thread = threadIdx.x;
	asm volatile("call.uni __bpftime_cuda__kernel_trace, ();" ::: "memory");
	output[thread] = expected_value(thread, seed);
}

static bool parse_unsigned(const char *text, unsigned minimum,
			   unsigned maximum, unsigned *value)
{
	char *end = nullptr;
	errno = 0;
	const unsigned long parsed = std::strtoul(text, &end, 10);
	if (errno || !end || *end || parsed < minimum || parsed > maximum)
		return false;
	*value = (unsigned)parsed;
	return true;
}

static void usage(const char *program)
{
	std::fprintf(stderr,
		     "usage: %s --threads N --warmup N --launches N --run-id N\n",
		     program);
}

int main(int argc, char **argv)
{
	unsigned threads = 0, warmup = 0, launches = 0, run_id = 0;
	bool have_threads = false, have_warmup = false;
	bool have_launches = false, have_run_id = false;
	for (int index = 1; index < argc; index += 2) {
		if (index + 1 >= argc) {
			usage(argv[0]);
			return 1;
		}
		if (std::strcmp(argv[index], "--threads") == 0)
			have_threads = parse_unsigned(argv[index + 1], kWarpSize,
						      kMaximumThreads, &threads);
		else if (std::strcmp(argv[index], "--warmup") == 0)
			have_warmup = parse_unsigned(argv[index + 1], 0, 1000, &warmup);
		else if (std::strcmp(argv[index], "--launches") == 0)
			have_launches = parse_unsigned(argv[index + 1], 1, 1000000,
						       &launches);
		else if (std::strcmp(argv[index], "--run-id") == 0)
			have_run_id = parse_unsigned(argv[index + 1], 0, 1000000,
						     &run_id);
		else {
			usage(argv[0]);
			return 1;
		}
	}
	if (!have_threads || !have_warmup || !have_launches || !have_run_id ||
	    threads % kWarpSize != 0) {
		usage(argv[0]);
		return 1;
	}

	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
	if (std::strstr(properties.name, "RTX 5090") == nullptr ||
	    properties.major != 12 || properties.minor != 0 ||
	    properties.warpSize != (int)kWarpSize ||
	    threads > (unsigned)properties.maxThreadsPerBlock) {
		std::fprintf(stderr, "device or launch shape does not match the frozen plan\n");
		return 3;
	}
	std::printf("FIG15_DEVICE\t%s\t%d\t%d\t%d\n", properties.name,
		    properties.major, properties.minor, properties.warpSize);
	std::printf("FIG15_WARP_SHAPE\t%u\t%u\n", threads,
		    threads / kWarpSize);

	uint64_t *device_output = nullptr;
	CUDA_CHECK(cudaMalloc(&device_output, threads * sizeof(uint64_t)));
	const uint64_t seed = kSeedBase ^ ((uint64_t)run_id << 16) ^ threads;

	for (unsigned index = 0; index < warmup; ++index)
		fig15_warp_map_kernel<<<1, threads>>>(device_output, seed);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaEvent_t start = nullptr, stop = nullptr;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));
	for (unsigned index = 0; index < launches; ++index)
		fig15_warp_map_kernel<<<1, threads>>>(device_output, seed);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float elapsed_ms = 0.0F;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

	std::vector<uint64_t> host_output(threads);
	CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
			      host_output.size() * sizeof(uint64_t),
			      cudaMemcpyDeviceToHost));
	unsigned mismatches = 0;
	for (unsigned thread = 0; thread < threads; ++thread) {
		const uint64_t expected = expected_value(thread, seed);
		if (host_output[thread] != expected) {
			if (mismatches < 4)
				std::fprintf(stderr,
					     "output mismatch thread=%u got=%" PRIu64
					     " expected=%" PRIu64 "\n",
					     thread, host_output[thread], expected);
			++mismatches;
		}
	}
	std::printf("FIG15_MEASUREMENT\t%u\t%u\t%.9g\n", warmup, launches,
		    elapsed_ms);
	std::printf("FIG15_CORRECT\t%u\t%u\n", threads, mismatches);

	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	cudaFree(device_output);
	return mismatches == 0 ? 0 : 4;
}
