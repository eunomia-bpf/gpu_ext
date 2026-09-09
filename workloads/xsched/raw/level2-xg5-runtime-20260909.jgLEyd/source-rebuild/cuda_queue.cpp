#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

CudaQueueLv2::CudaQueueLv2(CUstream stream): CudaQueueLv1(stream)
{
    instrument_manager_ = std::make_unique<InstrumentManager>(context_, cudevice_);
}

void CudaQueueLv2::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) return instrument_manager_->Launch(kernel, kStream, level_);
    
    auto cuda_cmd = std::dynamic_pointer_cast<CudaCommand>(hw_cmd);
    XASSERT(cuda_cmd != nullptr, "hw_cmd is not a CudaCommand");
    CUDA_ASSERT(cuda_cmd->LaunchWrapper(kStream));
}

void CudaQueueLv2::Synchronize()
{
    CudaQueueLv1::Synchronize();
    instrument_manager_->Xg5Drain("queue-synchronized");
}

void CudaQueueLv2::Deactivate()
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Deactivate() not supported on level-%d", level_);
    instrument_manager_->Deactivate();
}

void CudaQueueLv2::Reactivate(const preempt::CommandLog &log)
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Reactivate() not supported on level-%d", level_);

    // Reactivate() runs on the caller's thread (e.g. the scheduler), which
    // unlike worker launch threads may have no current CUDA context (workers
    // get one via OnXQueueCreate()). NVBit's interception of the replayed
    // launches keys off the current context: nvbit_is_func_kernel(ctx, f)
    // filters the pre-launch callback and the set_at_launch value is associated
    // per-context. Push the queue's context for the replay, then restore the
    // caller's context, so the resume launch is intercepted on the same context
    // the value was armed on.
    CUcontext popped_ctx = nullptr;
    CUDA_ASSERT(Driver::CtxPushCurrent_v2(context_));

    this->Synchronize();

    uint64_t resume_cmd_idx = instrument_manager_->Reactivate();
    if (resume_cmd_idx != 0) {
        for (auto cmd : log) {
            if (cmd->GetIdx() < (int64_t)resume_cmd_idx) continue;
            this->Launch(cmd);
        }
    }

    CUDA_ASSERT(Driver::CtxPopCurrent_v2(&popped_ctx));
}

void CudaQueueLv2::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelDeactivate, "unsupported level: %d", level);
    level_ = level;
}

void CudaQueueLv2::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    this->CudaQueueLv1::OnHwCommandSubmit(hw_cmd);
    if (level_ < kPreemptLevelDeactivate) return;

    if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
        // do nothing here, will automatically fallback to wait-based preemption
        static bool warned = false;
        if (!warned) {
            warned = true;
            XWARN("CUDA graph cannot support level-2 preemption, falling back to level-1");
        }
        return;
    }
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) instrument_manager_->Instrument(kernel);
}

CUresult CudaQueueLv2::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel,
                                    CUcontext ctx, CUstream stream)
{
    auto instrument_ctx = InstrumentContext::Instance(ctx);
    return instrument_ctx->Launch(kernel, stream, kKernelLaunchOriginal);
}
