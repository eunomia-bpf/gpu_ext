/* SPDX-License-Identifier: GPL-2.0
 * LDC consumer specimen for the sm_120 guardian adapter. Three consumed
 * u64 parameters reproduce the region the guardian stubs read (base+0,
 * +8, +16). A 0x1500-byte unused pad struct parameter positions a consumed
 * tail parameter at relative offset 0x1518 (absolute c[0x0][0x1898]), so
 * the generated cubin demonstrates the immediate bits the patcher needs to
 * re-encode stub consumers onto the 0x1880 debugger region no matter which
 * LDC/LDCU form ptxas selects. The patcher derives the parameter region
 * from EIATTR_PARAM_CBANK metadata and the offset encoding from the
 * generated text plus raw instruction words; no assumed offsets or
 * encodings enter the ported blob. The sink writes through a device global
 * so ptxas cannot eliminate the parameter loads. */
#include <stdint.h>

__device__ uint64_t xg_probe_sink;

/* Passed by value without array decay; reserves 0x1500 bytes in the
 * parameter block so the tail parameter sits past byte offset 0x1880. */
struct XgProbePad {
    long long bytes[672];
};

extern "C" __global__ void xg_ldc_probe(uint64_t a, uint64_t b, uint64_t c,
                                        XgProbePad pad, uint64_t d)
{
    (void)pad;
    xg_probe_sink = a * 3u + b + c + d;
}
