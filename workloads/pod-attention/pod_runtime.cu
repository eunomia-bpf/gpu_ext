#include "pod_runtime.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <map>

namespace {
thread_local unsigned selected_mode = 0;
thread_local bool selected_trace = false;
thread_local std::map<int, unsigned> identifier_bounds;
thread_local PodWorkspace last;

__global__ void pod_read_identifier_bound(unsigned *out) {
    unsigned nsmid;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
    *out = nsmid;
}

unsigned identifier_bound(cudaStream_t stream) {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    auto known = identifier_bounds.find(device);
    if (known != identifier_bounds.end()) return known->second;
    auto tmp = at::empty({1}, at::TensorOptions().device(at::kCUDA, device).dtype(at::kInt));
    pod_read_identifier_bound<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned *>(tmp.data_ptr<int>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    unsigned value = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(&value, tmp.data_ptr(), sizeof(value),
                                  cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(value > 0 && value < (1u << 20), "invalid device %nsmid bound");
    identifier_bounds[device] = value;
    return value;
}
} // namespace

void pod_configure(const std::string &mode, bool trace) {
    TORCH_CHECK(mode == "inline" || mode == "cuda" || mode == "bpf", "invalid POD selector mode");
    selected_mode = mode == "inline" ? 0 : mode == "cuda" ? 1 : 2;
    selected_trace = trace;
}

PodWorkspace pod_workspace(unsigned grid, unsigned prefill_blocks,
                           unsigned decode_blocks, unsigned factor_p,
                           unsigned factor_d, unsigned smem_bytes,
                           unsigned threads, unsigned fused_op,
                           cudaStream_t stream) {
    TORCH_CHECK(grid && grid < 0x3fffffffu && prefill_blocks && decode_blocks,
                "invalid POD work extent");
    TORCH_CHECK(factor_p && factor_d, "invalid POD logical CTA factors");
    TORCH_CHECK(!(fused_op & 128), "device selector experiment excludes unexposed persistent path");
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(stream == at::cuda::getCurrentCUDAStream().stream(),
                "POD workspace must use the current allocation stream");
    unsigned bound = identifier_bound(stream);
    const auto options = at::TensorOptions().device(at::kCUDA, device);
    PodWorkspace workspace;
    workspace.counters = at::empty({static_cast<int64_t>(bound + 2)}, options.dtype(at::kInt));
    // Context workspace is identical in all POD arms. Inline perf mode does
    // not touch it; diagnostic mode saves its actual local decision here.
    workspace.contexts = at::empty({static_cast<int64_t>(grid),
                                    static_cast<int64_t>(sizeof(PodSelectorContext))},
                                   options.dtype(at::kByte));
    workspace.errors = at::empty({1}, options.dtype(at::kInt));
    C10_CUDA_CHECK(cudaMemsetAsync(workspace.counters.data_ptr(), 0,
                                  (bound + 2) * sizeof(unsigned), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(workspace.errors.data_ptr(), 0, sizeof(unsigned), stream));
    workspace.view = {reinterpret_cast<PodSelectorContext *>(workspace.contexts.data_ptr()),
                      reinterpret_cast<unsigned *>(workspace.errors.data_ptr()), bound,
                      selected_mode, selected_trace ? 1u : 0u, grid};
    // CPU metadata is not a device measurement. Actual SM/tickets come from contexts.
    workspace.metadata = at::tensor({static_cast<int64_t>(bound),
        static_cast<int64_t>(grid), static_cast<int64_t>(prefill_blocks),
        static_cast<int64_t>(decode_blocks), static_cast<int64_t>(factor_p),
        static_cast<int64_t>(factor_d), static_cast<int64_t>(smem_bytes),
        static_cast<int64_t>(threads), static_cast<int64_t>(fused_op),
        static_cast<int64_t>(selected_mode), static_cast<int64_t>(selected_trace)},
        at::TensorOptions().dtype(at::kLong));
    // Tensor ownership replaces the upstream cudaMalloc leak. Allocations and
    // uses share the current stream; PyTorch's stream-aware allocator controls
    // reuse, and retaining the last launch permits post-timing diagnostics.
    last = workspace;
    return workspace;
}

std::vector<at::Tensor> pod_last_launch() {
    TORCH_CHECK(last.counters.defined(), "no POD launch recorded");
    return {last.metadata, last.counters, last.contexts, last.errors};
}
