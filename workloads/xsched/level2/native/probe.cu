/* SPDX-License-Identifier: GPL-2.0
 * LDC consumer probe for the sm_120 guardian adapter. The patcher derives
 * the actual parameter-region base and the constant-bank offset encoding
 * from the GENERATED cubin of this probe (nvdisasm text plus raw
 * instruction bytes); no assumed offsets enter the ported blob. The sink
 * writes through a device global so ptxas cannot eliminate the parameter
 * loads. */
#include <stdint.h>

__device__ uint64_t xg_probe_sink;

__global__ void xg_ldc_probe(uint64_t a, uint64_t b, uint64_t c)
{
    /* Three consumed 64-bit parameters force constant-bank consumers at
     * the parameter base offsets used by the guardian stubs (base+0,
     * +8, +16 halves). */
    xg_probe_sink = a * 3u + b + c;
}
