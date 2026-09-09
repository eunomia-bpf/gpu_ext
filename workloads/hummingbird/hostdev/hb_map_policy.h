// hb_map_policy.h: the one mapping algorithm, shared by the native CUDA
// device engine and the eBPF device engine.
//
//     out_{x,y,z} = block_{x,y,z} + off_{x,y,z}
//
// Keeping a single body guarantees the two engines are bit-identical for
// identical inputs; only the execution mechanism (native PTX vs compiled
// eBPF) differs. The entry points in hb_map_cuda.cuh and hb_map.bpf.c wrap
// this with their engine id and the shared HbMapContext ABI.
//
// No branch, no nested pointer dereference, and all accesses inside the
// 48-byte context, which is the shape the ctx48 exporter verifies against.

#ifndef HB_MAP_POLICY_H
#define HB_MAP_POLICY_H

#include "hb_map_abi.h"

#if defined(__CUDACC__)
#define HB_MAP_INLINE static __device__ __forceinline__
#else
#define HB_MAP_INLINE static inline
#endif

// len is carried in the device ABI so the wrapper and the BPF program keep the
// same two-parameter (context_ptr, context_length) call shape as the POD
// selector; the wrapper always materializes a full 48-byte context, so the
// policy does not re-validate len on device.
HB_MAP_INLINE void hb_map_policy(struct HbMapContext *ctx, hb_u64 len,
                                 hb_u32 engine)
{
    (void)len;
    ctx->out_x = ctx->block_x + ctx->off_x;
    ctx->out_y = ctx->block_y + ctx->off_y;
    ctx->out_z = ctx->block_z + ctx->off_z;
    ctx->status = HB_MAP_OK;
    ctx->engine = engine;
}

#endif // HB_MAP_POLICY_H
