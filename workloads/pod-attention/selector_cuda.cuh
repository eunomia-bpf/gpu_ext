#pragma once
#include "selector_abi.h"
#define POD_POLICY_INLINE static __device__ __forceinline__
#define POD_FETCH_ADD(ptr) atomicAdd((ptr), 1u)
#include "selector_policy.h"

/* A real two-argument device call. The scoped PTX adapter replaces only this
 * callee; output fields, not eBPF r0, select the real attention branch. */
extern "C" __device__ __noinline__ void pod_device_selector(
    PodSelectorContext *ctx, pod_u64 len) {
    pod_select_policy(ctx, len, POD_ENGINE_CUDA);
}
