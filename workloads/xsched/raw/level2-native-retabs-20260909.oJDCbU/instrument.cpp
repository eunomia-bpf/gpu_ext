#include <mutex>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <dlfcn.h>
#include "xsched/cuda/hal/common/options.h"
#include <cuxtra/cuxtra.h>

#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/level2/instrument.h"

#define XG_TOOL_CTX_BYTES 28 /* level2/xsched_guardian_abi.h arg block */
#define XG_TOOL_CTX_STRIDE 32
#define XG_TOOL_CTX_SLOTS 8192
#define XG_TOOL_CTX_POOL_BYTES (XG_TOOL_CTX_STRIDE * XG_TOOL_CTX_SLOTS)

/* Publish/unpublish the context block of the launch running on this thread.
 * The NVBit gate tool exports xg_host_publish and consumes the pointer in
 * its at-launch callback (one-shot per launch on the same thread). The tool
 * must be preloaded ahead of libhalcuda when the tool actuator is enabled. */
static void ToolPublish(CUdeviceptr ctx_dev)
{
    typedef void (*PublishFn)(uint64_t);
    static PublishFn publish = nullptr;
    static bool looked_up = false;
    if (!looked_up) {
        publish = (PublishFn)dlsym(RTLD_DEFAULT, "xg_host_publish");
        looked_up = true;
    }
    XASSERT(publish != nullptr,
            "Level-2 tool actuator enabled but xg_host_publish was not found; "
            "preload the NVBit instrument tool first");
    publish((uint64_t)ctx_dev);
}

using namespace xsched::cuda;
using namespace xsched::preempt;

InstrumentContext::InstrumentContext(CUcontext ctx): kCtx(ctx), kToolActuator(GetLevel2ToolActuator())
{
    // check current context
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx,
            "create InstrumentContext failed: current context (%p) does not match context (%p)",
            current_ctx, ctx);

    // create a non-blocking operation stream with highest priority
    int lp, hp;
    CUDA_ASSERT(Driver::CtxGetStreamPriorityRange(&lp, &hp));
    CUDA_ASSERT(Driver::StreamCreateWithPriority(&op_stream_, CU_STREAM_NON_BLOCKING, hp));

    CUdevice dev;
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));
    preempt_buf_ = std::make_unique<ResizableBuffer>(dev);
    // clear the preempt buffer
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_->Ptr(), 0, preempt_buf_->Size(), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

    if (kToolActuator) {
        // NVBit tool actuator mode (Level-2 sm_120 port): the gate tool
        // injects the guardian and resume trampolines at kernel entry at
        // launch time, so no binary surgery happens here. Neither
        // Guardian::Instance nor the cuXtra InstrMemAllocator is touched
        // on this path. Per-command context blocks live in a pinned mapped
        // pool and are published to the tool on the launching thread right
        // before the launch.
        CUDA_ASSERT(Driver::MemHostAlloc((void **)&tool_ctx_pool_host_,
                                         XG_TOOL_CTX_POOL_BYTES,
                                         CU_MEMHOSTALLOC_DEVICEMAP));
        CUDA_ASSERT(Driver::MemHostGetDevicePointer_v2(&tool_ctx_pool_dev_,
                                                       tool_ctx_pool_host_, 0));
        memset(tool_ctx_pool_host_, 0, XG_TOOL_CTX_POOL_BYTES);
        return;
    }

    guardian_ = Guardian::Instance(dev);
    XASSERT(guardian_ != nullptr,
            "no Level-2 blob guardian for this architecture; use the NVBit tool "
            "actuator instead (XSCHED_LEVEL2_TOOL_ACTUATOR=1)");
    instr_mem_ = std::make_unique<InstrMemAllocator>(kCtx, dev);

    // prepare the resume instructions
    size_t resume_size;
    const void *resume_host;
    guardian_->GetResumeInstructions(&resume_host, &resume_size);
    entry_point_resume_ = instr_mem_->Alloc(ROUND_UP(resume_size, 256));
    cuXtraInstrMemcpyHtoD(entry_point_resume_, resume_host, resume_size, op_stream_);
    // native-700 probe 1 (instr-memory hypothesis): this copy is the
    // ctor's only never-synchronized device operation; sync it here so an
    // instruction-memory fault (bad BlockAlloc VA on sm_120, or invalid
    // HtoD write) surfaces at queue creation instead of being deferred to
    // the first launch's EventSynchronize, where it becomes
    // indistinguishable from launch-side corruption.
    XINFO("native-700 probe 1: resume blob VA 0x%llx size %llu, syncing",
          (unsigned long long)entry_point_resume_,
          (unsigned long long)resume_size);
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
    XINFO("native-700 probe 1: resume blob copy synced clean");
}

std::shared_ptr<InstrumentContext> InstrumentContext::Instance(CUcontext ctx)
{
    static std::mutex ctx_mtx;
    static std::map<CUcontext, std::shared_ptr<InstrumentContext>> ctx_map;

    std::lock_guard<std::mutex> lock(ctx_mtx);
    auto it = ctx_map.find(ctx);
    if (it != ctx_map.end()) return it->second;

    auto instr_ctx = std::make_shared<InstrumentContext>(ctx);
    ctx_map[ctx] = instr_ctx;
    return instr_ctx;
}

CUresult InstrumentContext::Launch(std::shared_ptr<CudaKernelCommand> kernel,
                                   CUstream stream, LaunchType type)
{
    char args_buf[28];
    uint64_t *preempt_buf = (uint64_t *)(args_buf +  0); // 1st arg: preempt buffer addr
    uint64_t *guardian    = (uint64_t *)(args_buf +  8); // 2nd arg: guardian entry point
    int64_t  *kernel_idx  = (int64_t  *)(args_buf + 16); // 3rd arg: kernel index
    uint32_t *killable    = (uint32_t *)(args_buf + 24); // 4th arg: killable flag

    if (type == kKernelLaunchOriginal) {
        size_t block_cnt = kernel->BlockCnt();
        size_t buf_size = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t) * block_cnt;
        preempt_buf_->ExpandTo(buf_size, op_stream_);

        if (kToolActuator) {
            // tool actuator: publish the default (empty-flag) preempt
            // buffer with launch type 0; the trampoline falls through
            // into the kernel body without actuating
            char *args = (char *)tool_ctx_pool_host_ +
                         kernel->tool_ctx * XG_TOOL_CTX_STRIDE;
            memset(args, 0, XG_TOOL_CTX_BYTES);
            *(uint64_t *)(args + 0) = (uint64_t)preempt_buf_->Ptr();
            *(uint32_t *)(args + 24) = kKernelLaunchOriginal;
            ToolPublish(tool_ctx_pool_dev_ +
                        kernel->tool_ctx * XG_TOOL_CTX_STRIDE);
            CUresult ret = kernel->LaunchWrapper(stream);
            ToolPublish(0);
            return ret;
        }

        // only preempt_buf is useful in this case
        memset(args_buf, 0, sizeof(args_buf));
        *preempt_buf = preempt_buf_->Ptr(); // use default (empty) preempt buffer

        launch_mtx_.lock();
        cuXtraSetDebuggerParams(kernel->kFuncHandle, args_buf, sizeof(args_buf));
        CUresult ret = kernel->LaunchWrapper(stream);
        launch_mtx_.unlock();
        return ret;
    }

    *preempt_buf = kernel->preempt_buffer;
    *guardian    = kernel->entry_point_guardian;
    *kernel_idx  = kernel->GetIdx();
    *killable    = kernel->killable;

    if (kToolActuator) {
        // tool actuator: the injected trampoline implements the guardian
        // (type 1) and resume (type 2) decisions from this block; the
        // entry slot stays the (unused) guardian entry of the command
        char *args = (char *)tool_ctx_pool_host_ +
                     kernel->tool_ctx * XG_TOOL_CTX_STRIDE;
        memset(args, 0, XG_TOOL_CTX_BYTES);
        *(uint64_t *)(args + 0) = (uint64_t)kernel->preempt_buffer;
        *(uint64_t *)(args + 8) = (uint64_t)kernel->entry_point_guardian;
        *(int64_t *)(args + 16) = (int64_t)kernel->GetIdx();
        *(uint32_t *)(args + 24) = (uint32_t)type;
        ToolPublish(tool_ctx_pool_dev_ +
                    kernel->tool_ctx * XG_TOOL_CTX_STRIDE);
        CUresult ret = kernel->LaunchWrapper(stream);
        ToolPublish(0);
        return ret;
    }

    CUdeviceptr entry_point = type == kKernelLaunchResume
                            ? entry_point_resume_ // launch to resume
                            : kernel->entry_point_guardian;
    
    launch_mtx_.lock();
    cuXtraSetDebuggerParams(kernel->kFuncHandle, args_buf, sizeof(args_buf));
    // native-700 probe 2 (debugger-window + launch-shape hypothesis): the
    // failing native cell launched the GUARDIAN through this common
    // branch. Log the launch shape, then read the 28-byte block back
    // through the offset-explicit getter to test whether the implicit
    // setter window (0x1880 on sm_70/86) round-trips on driver 575 for
    // compute_task whose declared c[0x0] extent is only 0x3a0 bytes.
    // Logging only: launch behavior is unchanged.
    XINFO("native-700 probe 2: type=%d preempt_buf=0x%llx guardian=0x%llx"
          " entry=0x%llx idx=%lld killable=%u",
          (int)type, *(unsigned long long *)preempt_buf,
          *(unsigned long long *)guardian, (unsigned long long)entry_point,
          (long long)*kernel_idx, *killable);
    char dbg_readback[28];
    memset(dbg_readback, 0, sizeof(dbg_readback));
    cuXtraGetDebuggerParams(kernel->kFuncHandle, dbg_readback, 0,
                            sizeof(args_buf));
    XINFO("native-700 probe 2: readback equal=%d byte0=0x%02x"
          " byte8=0x%02x byte16=0x%02x",
          memcmp(args_buf, dbg_readback, sizeof(args_buf)) == 0 ? 1 : 0,
          (unsigned)dbg_readback[0], (unsigned)dbg_readback[8],
          (unsigned)dbg_readback[16]);
    // sm_120 window-args relay: adopt a launch blob carrying the kernel's
    // real params and the 28-byte window args (absolute c[0x0][0x1880],
    // blob-relative 0x1500 on this sm_120 parameter base) so the
    // guardian/resume prefix reads real values through the normal parameter
    // upload, independently of the debugger-window staging. The launch form
    // (extra/params pointers) is adopted under this launch-mutex critical
    // section and restored right after the launch.
    void *relay_token = kernel->AdoptWindowRelayExtra(args_buf);
    if (std::getenv("XG_NATIVE_ORIGINAL_ENTRY_CONTROL") == nullptr)
        cuXtraSetEntryPoint(kernel->kFuncHandle, entry_point);
    CUresult ret = kernel->LaunchWrapper(stream);
    kernel->RestoreWindowRelayExtra(relay_token);
    XINFO("native-700 probe 2: launch ret=%d", (int)ret);
    cuXtraSetEntryPoint(kernel->kFuncHandle, kernel->entry_point_original);
    launch_mtx_.unlock();

    return ret;
}

void InstrumentContext::NotifyTrapInstrumented()
{
    std::lock_guard<std::mutex> lock(launch_mtx_);
    trap_instrumented_ = true;
}

void InstrumentContext::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    CUfunction func = kernel->kFuncHandle;

    if (kToolActuator) {
        // tool actuator: no binary instrumentation; assign one publish
        // pool slot per command (slot 0 is reserved, never assigned)
        std::lock_guard<std::mutex> lock(instrument_mtx_);
        uint64_t slot = ++tool_ctx_count_;
        XASSERT(slot < XG_TOOL_CTX_SLOTS,
                "tool actuator context pool exhausted (%u slots)",
                (unsigned int)XG_TOOL_CTX_SLOTS);
        kernel->tool_ctx = slot;
        memset((char *)tool_ctx_pool_host_ + slot * XG_TOOL_CTX_STRIDE, 0,
               XG_TOOL_CTX_BYTES);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(instrument_mtx_);
        auto it = kernels_.find(func);
        if (it != kernels_.end()) {
            // the kernel has been instrumented
            kernel->entry_point_original = it->second.entry_point_original;
            kernel->entry_point_guardian = it->second.entry_point_guardian;
            return;
        }
    }

    // the kernel has not been instrumented, instrument it
    launch_mtx_.lock();
    // get the original entry point of the kernel
    CUdeviceptr ep_orig = cuXtraGetEntryPoint(func);
    launch_mtx_.unlock();

    instrument_mtx_.lock();

    size_t check_size, kernel_size;
    const void *check_host, *kernel_host;
    guardian_->GetGuardianInstructions(&check_host, &check_size);
    cuXtraGetBinary(kCtx, func, &kernel_host, &kernel_size, false);

    // allocate memory for the instrumented kernel, return ptr is entry point
    CUdeviceptr ep_inst = instr_mem_->Alloc(kernel_size + check_size);
    // the instrumented kernel starts with the guardian instructions
    cuXtraInstrMemcpyHtoD(ep_inst, check_host, check_size, op_stream_);
    // followed by the original kernel instructions
    cuXtraInstrMemcpyHtoD(ep_inst + check_size, kernel_host, kernel_size, op_stream_);
    
    // the guardian instructions will use 32 regs per thread
    size_t reg_cnt = cuXtraGetLocalRegsPerThread(func);
    if (reg_cnt < 32) cuXtraSetLocalRegsPerThread(func, 32);

    // the guardian instructions will use 1 barrier
    size_t barrier_cnt = cuXtraGetBarrierCnt(func);
    if (barrier_cnt < 1) cuXtraSetBarrierCnt(func, 1);

    // native-700 probe 3 (surgery hypothesis): record exactly what the
    // binary surgery produced on sm_120 before the first guardian launch:
    // the kernel byte size returned by cuXtraGetBinary (raw vs loaded
    // layout), the guardian size, the instrumented entry VA, and whether
    // the register/barrier bumps round-trip on driver 575.
    size_t reg_after = cuXtraGetLocalRegsPerThread(func);
    size_t bar_after = cuXtraGetBarrierCnt(func);
    XINFO("native-700 probe 3: binary kernel=%llu B guardian=%llu B ep_inst=0x%llx"
          " regs=%llu->%llu barriers=%llu->%llu",
          (unsigned long long)kernel_size, (unsigned long long)check_size,
          (unsigned long long)ep_inst, (unsigned long long)reg_cnt,
          (unsigned long long)reg_after, (unsigned long long)barrier_cnt,
          (unsigned long long)bar_after);
    // flush instruction cache to take effect
    cuXtraInvalInstrCache(kCtx);
    
    // update the instrumented kernel map
    kernels_[func] = InstrumentedKernel {
        .func = func,
        .entry_point_original = ep_orig,
        .entry_point_guardian = ep_inst,
    };

    instrument_mtx_.unlock();

    kernel->entry_point_original = ep_orig;
    kernel->entry_point_guardian = ep_inst;
}


InstrumentManager::InstrumentManager(CUcontext ctx, CUdevice dev)
{
    preempt_buf_ = std::make_unique<ResizableBuffer>(dev);
    preempt_buf_ptr_ = preempt_buf_->Ptr();
    instrument_ctx_ = InstrumentContext::Instance(ctx);
    op_stream_ = instrument_ctx_->OpStream();
}

/* preempt buffer layout: (see also tools/instrument/inject.cu)
 * |<------- uint32 ------->|<--- 32 bits -->|<---- uint64 ----->|<--------- uint32 --------->|<----------- uint32 ---------->|<------ ... ------>|
 * |<-- global_exit_flag -->|<-- reserved -->|<-- preempt_idx -->|<-- exit_flag_of_block_0 -->|<-- restore_flag_of_block_0 -->|<-- block_1 ... -->|
 */
void InstrumentManager::Deactivate()
{
    // set global_exit_flag to 1
    CUDA_ASSERT(Driver::MemsetD32Async(preempt_buf_ptr_, 1, 1, op_stream_));
}

uint64_t InstrumentManager::Reactivate()
{
    // read preempt_idx from preempt buffer
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(&preempt_idx_, preempt_buf_ptr_ + sizeof(uint64_t),
                                           sizeof(uint64_t), op_stream_));
    // clear the header of preempt buffer
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_ptr_, 0, 2 * sizeof(uint64_t), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

#define MAX_DEBUG_BLOCK_SIZE    1024
#define PREEMPT_BUFFER_DEBUG    false

#if PREEMPT_BUFFER_DEBUG
    XINFO("preempt idx: %" FMT_64U, preempt_idx_);
    uint32_t buffer_host[MAX_DEBUG_BLOCK_SIZE * 2 + 4];
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(buffer_host, preempt_buf_ptr_,
                                           sizeof(buffer_host), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
    for (size_t i = 0; i < MAX_DEBUG_BLOCK_SIZE; ++i) {
        XINFO("block[%ld]:\t%d,\t%d", i, buffer_host[2*i+4], buffer_host[2*i+5]);
    }
#endif

    return preempt_idx_;
}

void InstrumentManager::Launch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream, XPreemptLevel level)
{
    LaunchType launch_type = kKernelLaunchOriginal;
    if (level >= kPreemptLevelDeactivate) {
        launch_type = (int64_t)preempt_idx_ == kernel->GetIdx()
                    ? kKernelLaunchResume // the first preempted kernel
                    : kKernelLaunchGuardian;
    }
    // A failed XQueue launch must fail fast: the launch never reaches the
    // device, so any wait that relies on the started stamp (device-written
    // at kernel entry) would spin forever with no launch error reported.
    CUDA_ASSERT(instrument_ctx_->Launch(kernel, stream, launch_type));
}

void InstrumentManager::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    size_t block_cnt = kernel->BlockCnt();
    size_t buf_size = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t) * block_cnt;
    preempt_buf_->ExpandTo(buf_size, op_stream_);
    instrument_ctx_->Instrument(kernel);
    kernel->preempt_buffer = preempt_buf_->Ptr();
}
