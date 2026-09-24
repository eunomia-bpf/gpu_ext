#pragma once

#include "xsched/cuda/hal/level2/guardian.h"

namespace xsched::cuda
{

class GuardianSM120 : public Guardian
{
public:
    GuardianSM120() = default;
    virtual ~GuardianSM120() = default;

    virtual void GetGuardianInstructions(const void **guardian_instr, size_t *size) override;
    virtual void GetResumeInstructions(const void **resume_instr, size_t *size) override;
};

} // namespace xsched::cuda
