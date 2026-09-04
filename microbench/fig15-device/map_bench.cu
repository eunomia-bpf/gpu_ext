#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                        \
	do {                                                                      \
		const cudaError_t cuda_error = (call);                               \
		if (cuda_error != cudaSuccess) {                                     \
			std::fprintf(stderr, "%s: %s\n", #call,                       \
				     cudaGetErrorString(cuda_error));                    \
			return 2;                                                       \
		}                                                                     \
	} while (0)

static constexpr unsigned kThreads = 32;
static constexpr uint64_t kSeedBase = UINT64_C(0x1797000000000000);

extern "C" __device__ __noinline__ __attribute__((used)) void
__bpftime_cuda__kernel_trace(void)
{
	asm volatile("" ::: "memory");
}

__host__ __device__ static uint64_t expected_value(unsigned lane, uint64_t seed)
{
	uint64_t value = seed ^ ((uint64_t)lane * UINT64_C(0x9e3779b97f4a7c15));
	value = (value ^ (value >> 29)) * UINT64_C(0x94d049bb133111eb);
	return value ^ ((uint64_t)lane << 17);
}

extern "C" __global__ void fig15_map_kernel(uint64_t *output, uint64_t seed)
{
	const unsigned lane = threadIdx.x;
	asm volatile("call.uni __bpftime_cuda__kernel_trace, ();" ::: "memory");
	output[lane] = expected_value(lane, seed);
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
		     "usage: %s --warmup N --launches N --run-id N\n", program);
}

int main(int argc, char **argv)
{
	unsigned warmup = 0, launches = 0, run_id = 0;
	bool have_warmup = false, have_launches = false, have_run_id = false;
	for (int index = 1; index < argc; index += 2) {
		if (index + 1 >= argc) {
			usage(argv[0]);
			return 1;
		}
		if (std::strcmp(argv[index], "--warmup") == 0)
			have_warmup = parse_unsigned(argv[index + 1], 0, 1000, &warmup);
		else if (std::strcmp(argv[index], "--launches") == 0)
			have_launches = parse_unsigned(argv[index + 1], 1, 1000000,
						       &launches);
		else if (std::strcmp(argv[index], "--run-id") == 0)
			have_run_id = parse_unsigned(argv[index + 1], 0, 1000000, &run_id);
		else {
			usage(argv[0]);
			return 1;
		}
	}
	if (!have_warmup || !have_launches || !have_run_id) {
		usage(argv[0]);
		return 1;
	}

	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
	if (std::strstr(properties.name, "RTX 5090") == nullptr ||
	    properties.major != 12 || properties.minor != 0 ||
	    properties.warpSize != (int)kThreads) {
		std::fprintf(stderr, "device does not match the frozen RTX 5090 plan\n");
		return 3;
	}
	std::printf("FIG15_DEVICE\t%s\t%d\t%d\t%d\n", properties.name,
		    properties.major, properties.minor, properties.warpSize);

	uint64_t *device_output = nullptr;
	CUDA_CHECK(cudaMalloc(&device_output, kThreads * sizeof(uint64_t)));
	const uint64_t seed = kSeedBase ^ ((uint64_t)run_id << 16);

	for (unsigned index = 0; index < warmup; ++index)
		fig15_map_kernel<<<1, kThreads>>>(device_output, seed);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaEvent_t start = nullptr, stop = nullptr;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));
	for (unsigned index = 0; index < launches; ++index)
		fig15_map_kernel<<<1, kThreads>>>(device_output, seed);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float elapsed_ms = 0.0F;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

	uint64_t host_output[kThreads] = {};
	CUDA_CHECK(cudaMemcpy(host_output, device_output, sizeof(host_output),
			      cudaMemcpyDeviceToHost));
	unsigned mismatches = 0;
	for (unsigned lane = 0; lane < kThreads; ++lane) {
		const uint64_t expected = expected_value(lane, seed);
		if (host_output[lane] != expected) {
			if (mismatches < 4)
				std::fprintf(stderr,
					     "output mismatch lane=%u got=%" PRIu64
					     " expected=%" PRIu64 "\n",
					     lane, host_output[lane], expected);
			++mismatches;
		}
	}
	std::printf("FIG15_MEASUREMENT\t%u\t%u\t%.9g\n", warmup, launches,
		    elapsed_ms);
	std::printf("FIG15_CORRECT\t%u\t%u\n", kThreads, mismatches);

	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	cudaFree(device_output);
	return mismatches == 0 ? 0 : 4;
}
