// Explicit transport compatibility, not a replacement for the GPreempt policy.
#pragma once
#include <cuda.h>
#include <gdrapi.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace gpreempt_artifact {
class FlagPool {
public:
    int configure(const char *name) {
        if (owner_ || configured_ || !name ||
            (std::strcmp(name, "gdr") && std::strcmp(name, "host_mapped"))) return -1;
        mapped_ = !std::strcmp(name, "host_mapped");
        configured_ = true;
        return 0;
    }
    const char *name() const { return mapped_ ? "host_mapped" : "gdr"; }
    unsigned context_flags() const { return mapped_ ? CU_CTX_MAP_HOST : 0; }
    int allocate(CUdeviceptr *device, void **host) {
        if (cleaned_ || !device || !host || used_ >= bytes_ / sizeof(int)) return -1;
        if (!host_) {
            if (cuda(cuCtxGetCurrent(&owner_), "cuCtxGetCurrent") || !owner_) return -1;
            if (mapped_) {
                CUdevice gpu;
                int capable = 0;
                if (cuda(cuCtxGetDevice(&gpu), "cuCtxGetDevice") ||
                    cuda(cuDeviceGetAttribute(&capable, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, gpu),
                         "cuDeviceGetAttribute") || !capable) return error("host mapping unsupported");
                if (cuda(cuMemHostAlloc(&host_, bytes_, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
                         "cuMemHostAlloc") ||
                    cuda(cuMemHostGetDevicePointer(&device_, host_, 0), "cuMemHostGetDevicePointer"))
                    return -1;
            } else {
                gdr_ = gdr_open();
                if (!gdr_) return error("gdr_open failed; no transport fallback");
                if (cuda(cuMemAlloc(&allocation_, bytes_ * 2), "cuMemAlloc")) return -1;
                device_ = (allocation_ + bytes_ - 1) & ~(CUdeviceptr(bytes_) - 1);
                if (gdr_pin_buffer(gdr_, device_, bytes_, 0, 0, &mapping_))
                    return error("gdr_pin_buffer failed; no transport fallback");
                pinned_ = true;
                if (gdr_map(gdr_, mapping_, &map_base_, bytes_)) return error("gdr_map failed");
                gdr_info_t info{};
                if (gdr_get_info(gdr_, mapping_, &info) || info.va > device_ ||
                    device_ - info.va + bytes_ > info.mapped_size) return error("invalid GDR mapping");
                host_ = static_cast<char *>(map_base_) + (device_ - info.va);
            }
            std::printf("gpreempt_flag_transport: transport=%s portable=%d original_gdr=%d\n",
                        name(), mapped_ ? 1 : 0, mapped_ ? 0 : 1);
            std::fflush(stdout);
        }
        *device = device_ + used_ * sizeof(int);
        *host = static_cast<int *>(host_) + used_;
        ++used_;
        return store(*host, 1);
    }
    int store(void *pointer, int value) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        const auto begin = reinterpret_cast<std::uintptr_t>(host_);
        if (cleaned_ || !host_ || !pointer || (value != 0 && value != 1) ||
            address < begin || address - begin >= used_ * sizeof(int) || (address - begin) % sizeof(int))
            return error("invalid flag store");
        if (mapped_) __atomic_store_n(static_cast<int *>(pointer), value, __ATOMIC_RELEASE);
        else if (gdr_copy_to_mapping(mapping_, pointer, &value, sizeof(value)))
            return error("gdr_copy_to_mapping failed");
        return 0;
    }
    void track_stream(CUstream stream) { streams_.push_back(stream); }
    int cleanup(unsigned timeout_ms = 5000) {
        if (cleaned_) return 0;
        CUcontext previous = nullptr;
        if (cuda(cuCtxGetCurrent(&previous), "cleanup cuCtxGetCurrent")) return -1;
        if (owner_ && cuda(cuCtxSetCurrent(owner_), "cleanup cuCtxSetCurrent")) return -1;
        int result = 0;
        for (unsigned i = 0; i < used_; ++i)
            if (store(static_cast<int *>(host_) + i, 1)) result = -1;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (CUstream stream : streams_) {
            CUevent done = nullptr;
            if (cuda(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING), "cleanup cuEventCreate")) { result = -1; break; }
            CUresult status = cuEventRecord(done, stream);
            if (status == CUDA_SUCCESS) {
                while ((status = cuEventQuery(done)) == CUDA_ERROR_NOT_READY &&
                       std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (cuda(status, "cleanup event completion")) result = -1;
            if (cuda(cuEventDestroy(done), "cleanup cuEventDestroy")) result = -1;
            if (result) break;
        }
        // Do not free a flag that a still-active original blocking kernel may read.
        // The external runner bounds and cleans the owning process on any failure.
        if (!result) {
            if (mapped_ && host_ && cuda(cuMemFreeHost(host_), "cuMemFreeHost")) result = -1;
            if (!mapped_ && map_base_ && gdr_unmap(gdr_, mapping_, map_base_, bytes_)) result = error("gdr_unmap failed");
            if (!mapped_ && pinned_ && gdr_unpin_buffer(gdr_, mapping_)) result = error("gdr_unpin_buffer failed");
            if (gdr_ && gdr_close(gdr_)) result = error("gdr_close failed");
            if (allocation_ && cuda(cuMemFree(allocation_), "cuMemFree")) result = -1;
        }
        if (cuda(cuCtxSetCurrent(previous), "restore cuCtxSetCurrent")) result = -1;
        std::printf("gpreempt_flag_cleanup: transport=%s status=%s slots=%u\n",
                    name(), result ? "failed" : "passed", used_);
        std::fflush(stdout);
        cleaned_ = !result;
        return result;
    }
private:
    int error(const char *message) const {
        std::fprintf(stderr, "gpreempt_flag_error: transport=%s %s\n", name(), message);
        return -1;
    }
    int cuda(CUresult status, const char *operation) const {
        if (status == CUDA_SUCCESS) return 0;
        std::fprintf(stderr, "gpreempt_flag_error: transport=%s operation=%s cuda_status=%d\n",
                     name(), operation, int(status));
        return -1;
    }
    static constexpr unsigned bytes_ = GPU_PAGE_SIZE;
    bool mapped_ = false, configured_ = false, pinned_ = false, cleaned_ = false;
    unsigned used_ = 0;
    CUcontext owner_ = nullptr;
    CUdeviceptr device_ = 0, allocation_ = 0;
    void *host_ = nullptr, *map_base_ = nullptr;
    gdr_t gdr_ = nullptr;
    gdr_mh_t mapping_{};
    std::vector<CUstream> streams_;
};
}
