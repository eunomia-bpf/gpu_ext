// CPU-only fake CUDA/GDR APIs: no CUDA library is linked or device contacted.
#include "flag_transport.h"
#include <cstdlib>
#include <string>

namespace {
alignas(GPU_PAGE_SIZE) unsigned char storage[GPU_PAGE_SIZE * 2];
unsigned host_allocs = 0, host_frees = 0, gpu_allocs = 0, gpu_frees = 0;
unsigned gdr_opens = 0, unmaps = 0, unpins = 0, closes = 0, copies = 0, checks = 0;
unsigned flags_seen = 0;
bool fail_pin = false, ready = true;
CUcontext current = reinterpret_cast<CUcontext>(0x1110);
CUdeviceptr pinned = 0;
void require(bool value, const char *message) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::abort(); }
}
}

extern "C" {
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *out) { *out = current; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxSetCurrent(CUcontext value) { current = value; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxGetDevice(CUdevice *out) { *out = 0; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGetAttribute(int *out, CUdevice_attribute attribute, CUdevice) {
    require(attribute == CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, "wrong capability query");
    *out = 1; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemHostAlloc(void **out, size_t bytes, unsigned flags) {
    require(bytes == GPU_PAGE_SIZE, "unexpected host allocation");
    *out = storage; ++host_allocs; flags_seen = flags; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *out, void *pointer, unsigned flags) {
    require(pointer == storage && flags == 0, "wrong mapped pointer lookup");
    *out = reinterpret_cast<CUdeviceptr>(pointer); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemFreeHost(void *pointer) {
    require(pointer == storage, "free not allocation base"); ++host_frees; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAlloc(CUdeviceptr *out, size_t bytes) {
    require(bytes == 2 * GPU_PAGE_SIZE, "unexpected GPU allocation");
    *out = reinterpret_cast<CUdeviceptr>(storage); ++gpu_allocs; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemFree(CUdeviceptr pointer) {
    require(pointer == reinterpret_cast<CUdeviceptr>(storage), "GPU free not original base");
    ++gpu_frees; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventCreate(CUevent *out, unsigned flags) {
    require(flags == CU_EVENT_DISABLE_TIMING, "cleanup event timing enabled");
    *out = reinterpret_cast<CUevent>(0x2220); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecord(CUevent, CUstream) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventQuery(CUevent) { return ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY; }
CUresult CUDAAPI cuEventDestroy(CUevent) { return CUDA_SUCCESS; }
gdr_t gdr_open(void) { ++gdr_opens; return reinterpret_cast<gdr_t>(0x3330); }
int gdr_close(gdr_t) { ++closes; return 0; }
int gdr_pin_buffer(gdr_t, unsigned long address, size_t, uint64_t, uint32_t, gdr_mh_t *handle) {
    if (fail_pin) return -1;
    pinned = address; handle->h = 1; return 0;
}
int gdr_map(gdr_t, gdr_mh_t, void **out, size_t) { *out = storage; return 0; }
int gdr_get_info(gdr_t, gdr_mh_t, gdr_info_t *info) {
    info->va = pinned; info->mapped_size = GPU_PAGE_SIZE; return 0;
}
int gdr_unmap(gdr_t, gdr_mh_t, void *base, size_t size) {
    require(base == storage && size == GPU_PAGE_SIZE, "unmap is not exact original range");
    ++unmaps; return 0;
}
int gdr_unpin_buffer(gdr_t, gdr_mh_t) { ++unpins; return 0; }
int gdr_copy_to_mapping(gdr_mh_t, void *out, const void *in, size_t size) {
    require(size == sizeof(int), "wrong flag write width");
    std::memcpy(out, in, size); ++copies; return 0;
}
}

int main() {
    using gpreempt_artifact::FlagPool;
    CUdeviceptr device;
    void *host;
    {
        FlagPool pool;
        require(std::string(pool.name()) == "gdr" && pool.context_flags() == 0, "default is not original GDR");
        require(pool.configure("automatic") < 0, "silent fallback mode accepted");
        require(pool.configure("host_mapped") == 0, "explicit host selection failed");
        require(pool.configure("gdr") < 0, "transport changed after selection");
        require(pool.context_flags() == CU_CTX_MAP_HOST, "mapped context missing flag");
        require(pool.allocate(&device, &host) == 0, "host allocation failed");
        require(flags_seen == (CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP), "allocation not portable mapped");
        require(gdr_opens == 0 && gpu_allocs == 0, "host path touched GDR or GPU allocation");
        require(*static_cast<int *>(host) == 1, "initial flag not released");
        require(pool.store(host, 0) == 0 && *static_cast<int *>(host) == 0, "reset failed");
        require(pool.store(host, 1) == 0 && *static_cast<int *>(host) == 1, "release failed");
        require(pool.store(static_cast<char *>(host) + 1, 1) < 0, "unaligned store accepted");
        require(pool.store(static_cast<char *>(host) + 4, 1) < 0, "unallocated slot accepted");
        require(pool.store(host, 2) < 0, "invalid flag value accepted");
        pool.track_stream(reinterpret_cast<CUstream>(0x4440));
        require(pool.cleanup() == 0 && host_frees == 1, "host cleanup failed");
        require(pool.cleanup() == 0 && host_frees == 1, "cleanup not idempotent");
        require(pool.store(host, 0) < 0 && pool.allocate(&device, &host) < 0, "freed pool reused");
    }
    {
        FlagPool pool;
        require(pool.configure("host_mapped") == 0 && pool.allocate(&device, &host) == 0,
                "timeout fixture allocation failed");
        pool.track_stream(reinterpret_cast<CUstream>(0x4440));
        ready = false;
        const auto begin = std::chrono::steady_clock::now();
        require(pool.cleanup(0) < 0 && host_frees == 1, "inflight flag prematurely freed");
        require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1), "cleanup not bounded");
        require(*static_cast<int *>(host) == 1, "cleanup did not unblock kernel first");
        ready = true;
        require(pool.cleanup() == 0 && host_frees == 2, "later completed cleanup failed");
    }
    {
        FlagPool pool;
        require(pool.allocate(&device, &host) == 0, "GDR allocation failed");
        require(pool.store(host, 0) == 0 && pool.store(host, 1) == 0, "GDR stores failed");
        require(copies == 3, "GDR mapped writes did not use barrier-aware helper");
        require(pool.cleanup() == 0 && unmaps == 1 && unpins == 1 && closes == 1 && gpu_frees == 1,
                "GDR cleanup order/ownership failed");
    }
    {
        FlagPool pool;
        fail_pin = true;
        require(pool.allocate(&device, &host) < 0, "GDR pin failure did not propagate");
        require(host_allocs == 2, "GDR failure silently switched to host allocation");
        require(pool.cleanup() == 0 && gpu_frees == 2 && closes == 2, "partial GDR failure leaked owned resources");
    }
    std::printf("flag_transport CPU mocks: %u checks passed; no CUDA/GDR runtime linked\n", checks);
}
