#include "xsched/cuda/hal/arch/sm120.h"

// The guardian/resume stub SASS bytes are NOT hardcoded here. They are
// re-encoded from real sm_120 cubins by the Level-2 build's ldc-patch step
// (workloads/xsched/level2/native/ldc_patcher.cpp), which emits
// xg_sm120_guardian_arrays.h. The build wires that header in through
// XG_SM120_GENERATED_HEADER (see platforms/cuda/CMakeLists.txt). Without
// it, this translation unit is empty and arch 120 has no blob guardian.
#ifdef XG_SM120_ARRAYS_HEADER
#include XG_SM120_ARRAYS_HEADER

using namespace xsched::cuda;

void GuardianSM120::GetGuardianInstructions(const void **guardian_instr, size_t *size)
{
    *guardian_instr = xg_sm120_check_preempt;
    *size = sizeof(xg_sm120_check_preempt);
}

void GuardianSM120::GetResumeInstructions(const void **resume_instr, size_t *size)
{
    *resume_instr = xg_sm120_restore_exec;
    *size = sizeof(xg_sm120_restore_exec);
}

#endif /* XG_SM120_ARRAYS_HEADER */
