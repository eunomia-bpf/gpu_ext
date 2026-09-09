#include <mutex>
#include <atomic>
#include <cstring>
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

/* Arm the context block of the launch running on this thread in the NVBit
 * gate tool. The tool exports xg_host_prepare and arms
 * nvbit_set_at_launch directly with the provided (context, function), so
 * value delivery does not depend on NVBit pre-callbacks, which are not
 * delivered on every thread that issues a launch. All actuator launches
 * serialize on launch_mtx_, so the armed value always belongs to the launch
 * that follows it on the same thread. The trailing zero call clears the
 * tool's one-shot marker only: the armed value must remain in place until
 * the next launch overwrites it, because CTAs may enter after the launch
 * call returns. The tool must be preloaded ahead of libhalcuda when the
 * tool actuator is enabled. */
static void ToolPrepare(CUcontext ctx, CUfunction func, CUdeviceptr ctx_dev)
{
    typedef void (*PrepareFn)(CUcontext, CUfunction, uint64_t);
    static PrepareFn prepare = nullptr;
    static bool looked_up = false;
    if (!looked_up) {
        prepare = (PrepareFn)dlsym(RTLD_DEFAULT, "xg_host_prepare");
        looked_up = true;
    }
    XASSERT(prepare != nullptr,
            "Level-2 tool actuator enabled but xg_host_prepare was not found; "
            "preload the NVBit instrument tool first");
    prepare(ctx, func, (uint64_t)ctx_dev);
}

/* Level-2 tool actuator launch diagnostics (bounded: first 64 publications
 * per process). Dumps, per actuator launch: (1) the param count the command
 * ctor observed from the live CUfunction metadata, (2) the deep-copied task
 * (param offset 16) and reps (offset 24) values of the target's 32-byte
 * layout, and (3) the kernel_idx / launch_type published into the context
 * block. Read-only; changes no delivered value and no policy. */
using namespace xsched::cuda;
using namespace xsched::preempt;

/* Level-2 tool actuator launch diagnostics (bounded: first 64 publications
 * per process). Dumps, per actuator launch: (1) the param count the command
 * ctor observed from the live CUfunction metadata, (2) the deep-copied task
 * (param offset 16) and reps (offset 24) values of the target's 32-byte
 * layout, and (3) the kernel_idx / launch_type published into the context
 * block. Read-only; changes no delivered value and no policy. pub_block must
 * be the host-mapped 32-byte-stride block for this command's slot. */
static void XgDumpLaunch(const CudaKernelCommand &kernel, unsigned int type,
                         const char *pub_block)
{
    static std::atomic<uint64_t> dump_cnt{0};
    const uint64_t n = dump_cnt.fetch_add(1);
    if (n >= 64) return;
    const size_t pc = kernel.ParamCnt();
    const char *pd = kernel.ParamData();
    int task = 0;
    uint64_t reps = 0;
    if (pd != nullptr && pc >= 5) {
        memcpy(&task, pd + 16, sizeof(task));
        memcpy(&reps, pd + 24, sizeof(reps));
    }
    uint64_t pub_idx = 0;
    uint32_t pub_type = 0;
    if (pub_block != nullptr) {
        memcpy(&pub_idx, pub_block + 16, sizeof(pub_idx));
        memcpy(&pub_type, pub_block + 24, sizeof(pub_type));
    }
    XINFO("XG4 args cmd=" FMT_64D " type=%u param_cnt=%zu"
          " task=%d reps=" FMT_64U " pub_idx=" FMT_64U " pub_type=%u"
          " slot=" FMT_64U,
          (int64_t)kernel.GetIdx(), type, pc, task, reps,
          pub_idx, pub_type, (uint64_t)kernel.tool_ctx);
}

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
    entry_point_resume_ = instr_mem_->Alloc(resume_size);
    cuXtraInstrMemcpyHtoD(entry_point_resume_, resume_host, resume_size, op_stream_);
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
            XgDumpLaunch(*kernel, kKernelLaunchOriginal, args);
            launch_mtx_.lock();
            ToolPrepare(kCtx, kernel->kFuncHandle,
                        tool_ctx_pool_dev_ +
                        kernel->tool_ctx * XG_TOOL_CTX_STRIDE);
            CUresult ret = kernel->LaunchWrapper(stream);
            ToolPrepare(kCtx, kernel->kFuncHandle, 0);
            launch_mtx_.unlock();
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
        XgDumpLaunch(*kernel, (unsigned int)type, args);
        {
            static std::atomic<uint64_t> xg2_pub_cnt{0};
            if (xg2_pub_cnt.fetch_add(1) < 512) {
                XINFO("XG2 toolpub slot=" FMT_64U " buf=%p cmd_idx=" FMT_64D
                      " type=%u ctx_dev=%p",
                      (uint64_t)kernel->tool_ctx, (void *)kernel->preempt_buffer,
                      (int64_t)kernel->GetIdx(), (unsigned int)type,
                      (void *)(tool_ctx_pool_dev_ +
                               kernel->tool_ctx * XG_TOOL_CTX_STRIDE));
            }
        }
        launch_mtx_.lock();
        ToolPrepare(kCtx, kernel->kFuncHandle,
                    tool_ctx_pool_dev_ +
                    kernel->tool_ctx * XG_TOOL_CTX_STRIDE);
        CUresult ret = kernel->LaunchWrapper(stream);
        {
            static std::atomic<uint64_t> xg3_submit_cnt{0};
            if (xg3_submit_cnt.fetch_add(1) < 512) {
                XINFO("XG3 submit slot=" FMT_64U " cmd_idx=" FMT_64D
                      " type=%u ret=%d",
                      (uint64_t)kernel->tool_ctx, (int64_t)kernel->GetIdx(),
                      (unsigned int)type, (int)ret);
            }
        }
        ToolPrepare(kCtx, kernel->kFuncHandle, 0);
        launch_mtx_.unlock();
        return ret;
    }

    CUdeviceptr entry_point = type == kKernelLaunchResume
                            ? entry_point_resume_ // launch to resume
                            : kernel->entry_point_guardian;
    
    launch_mtx_.lock();
    cuXtraSetDebuggerParams(kernel->kFuncHandle, args_buf, sizeof(args_buf));
    cuXtraSetEntryPoint(kernel->kFuncHandle, entry_point);
    CUresult ret = kernel->LaunchWrapper(stream);
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

    // ResizableBuffer does not zero its initial PM chunk (ExpandTo only
    // clears newly expanded regions). Clear it before first use, mirroring
    // the InstrumentContext constructor: the Level-2 protocol requires the
    // header (global exit flag, preempt_idx) to start clean, otherwise the
    // first aborted leader can read a stale nonzero preempt_idx, skip
    // recording, and its command is never relaunched by Reactivate.
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_ptr_, 0,
                                      preempt_buf_->Size(), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
}

/* preempt buffer layout: (see also tools/instrument/inject.cu)
 * |<------- uint32 ------->|<--- 32 bits -->|<---- uint64 ----->|<--------- uint32 --------->|<----------- uint32 ---------->|<------ ... ------>|
 * |<-- global_exit_flag -->|<-- reserved -->|<-- preempt_idx -->|<-- exit_flag_of_block_0 -->|<-- restore_flag_of_block_0 -->|<-- block_1 ... -->|
 */
void InstrumentManager::Deactivate()
{
    XINFO("XG2 Deactivate mgr=%p buf=%p", (void *)this, (void *)preempt_buf_ptr_);
    // set global_exit_flag to 1
    CUDA_ASSERT(Driver::MemsetD32Async(preempt_buf_ptr_, 1, 1, op_stream_));
}

uint64_t InstrumentManager::Reactivate()
{
    XINFO("XG2 Reactivate in mgr=%p", (void *)this);
    // read preempt_idx from preempt buffer
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(&preempt_idx_, preempt_buf_ptr_ + sizeof(uint64_t),
                                           sizeof(uint64_t), op_stream_));
    // clear the header of preempt buffer
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_ptr_, 0, 2 * sizeof(uint64_t), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

    XINFO("XG2 Reactivate out idx=" FMT_64U " mgr=%p",
          preempt_idx_, (void *)this);

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
    {
        static std::atomic<uint64_t> xg2_log_cnt{0};
        if (xg2_log_cnt.fetch_add(1) < 512) {
            XINFO("XG2 Launch mgr=%p cmd_idx=" FMT_64D " type=%d level=%d"
                  " preempt_idx=" FMT_64U,
                  (void *)this, (int64_t)kernel->GetIdx(), (int)launch_type,
                  (int)level, preempt_idx_);
        }
    }
    instrument_ctx_->Launch(kernel, stream, launch_type);
}

void InstrumentManager::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    size_t block_cnt = kernel->BlockCnt();
    size_t buf_size = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t) * block_cnt;
    preempt_buf_->ExpandTo(buf_size, op_stream_);
    instrument_ctx_->Instrument(kernel);
    kernel->preempt_buffer = preempt_buf_->Ptr();
}
