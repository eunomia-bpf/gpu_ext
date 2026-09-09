// hb_map_cuda.cuh: native CUDA device engine for the Hummingbird coordinate
// mapping. This is the existing device path expressed through the shared
// HbMapContext ABI so the BPF engine can drop in with an identical call.
//
//     struct HbMapContext c;   // in kernel local memory
//     c.block_x = blockIdx.x; ... c.off_x = hb_offset_x; ...
//     hb_device_map(&c, sizeof(c));   // <- patched to the BPF engine in arm 3
//     bx = c.out_x; by = c.out_y; bz = c.out_z;
//
// `__noinline__` forces a real `call` in the generated PTX so the PTX patcher
// can route it to the compiled eBPF function in the BPF-device build.

#ifndef HB_MAP_CUDA_CUH
#define HB_MAP_CUDA_CUH

#include "hb_map_policy.h"

extern "C" __device__ __noinline__ void
hb_device_map(struct HbMapContext *ctx, unsigned long len)
{
    hb_map_policy(ctx, (unsigned long)len, HB_ENGINE_NATIVE);
}

#endif // HB_MAP_CUDA_CUH
