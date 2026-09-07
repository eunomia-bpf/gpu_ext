#include "sk_bpf.h"

extern "C" __device__ __noinline__ void sk_bpf_trampoline(int32_t guard,
							  uint64_t slots,
							  uint64_t capacity)
{
	if (guard == 0) {
		return;
	}
	const uint64_t block_linear =
		static_cast<uint64_t>(blockIdx.x) +
		static_cast<uint64_t>(blockIdx.y) * gridDim.x +
		static_cast<uint64_t>(blockIdx.z) * gridDim.x * gridDim.y;
	const uint64_t threads_per_block =
		static_cast<uint64_t>(blockDim.x) * blockDim.y * blockDim.z;
	const uint64_t thread_linear =
		static_cast<uint64_t>(threadIdx.x) +
		static_cast<uint64_t>(threadIdx.y) * blockDim.x +
		static_cast<uint64_t>(threadIdx.z) * blockDim.x * blockDim.y;
	const uint64_t slot = block_linear * threads_per_block + thread_linear;
	if (slot >= capacity) {
		return;
	}
	bpf_exit(slots + slot * 8, 8);
}
