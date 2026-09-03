#pragma once
#include "selector_abi.h"

struct PodLaunchView {
    PodSelectorContext *contexts;
    pod_u32 *errors;
    pod_u32 nsmid;
    pod_u32 mode;       // 0 original inline, 1 typed CUDA, 2 typed device-BPF
    pod_u32 trace;
    pod_u32 grid_ctas;
};
