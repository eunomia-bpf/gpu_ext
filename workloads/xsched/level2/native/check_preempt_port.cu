/* SPDX-License-Identifier: GPL-2.0
 *
 * sm_120 port of the ORIGINAL check_preempt guardian from
 * platforms/cuda/hal/inject/inject.cu. Statement-for-statement follower of
 * the original: CTA-leader-only decision, per-block exit/restore flags,
 * first preempt_idx recording, trusted threadfence/barrier actuation, and
 * per-thread cooperative exit.
 *
 * A leading unused 0x1500-byte pad struct (device of proof: native/probe.cu,
 * which shows a by-value pad moving the next consumed parameter exactly
 * 0x1500 bytes past the 0x380 parameter-region start) places the real
 * arguments in the upstream debugger-parameter window that instrument.cpp
 * fills via cuXtraSetDebuggerParams. The compiler therefore emits this
 * kernel's actual parameter LDC/LDCU consumers directly on
 * c[0x0][0x1880..]: param0 preempt_buffer @+0, param1 entry placeholder
 * @+8, param2 kernel_idx @+16. No binary immediate re-encoding of the
 * generated cubin is needed. The unused param1 keeps the debugger-region
 * slot aligned with the original 2nd argument word pair; the blob never
 * reads the pad or param1 (matching the original check_preempt prototype).
 */
#include <stdint.h>

// Build-only retention marker proposed by the local GLM implementation.
// The prefix extractor must exclude this store and the final kernel EXIT.
__device__ uint32_t xg_check_exit_marker;

/* Passed by value without array decay; reserves 0x1500 bytes at the start
 * of the parameter block so the real arguments land in the upstream
 * debugger-parameter window (see native/probe.cu for the device of proof). */
struct XgCheckPad {
    long long bytes[672];
};

static __device__ __forceinline__ uint32_t xg_blockid()
{
    return blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
}

static __device__ __forceinline__ void xg_thread_exit()
{
    asm volatile("exit;");
}

extern "C" __global__ void check_preempt_port(XgCheckPad pad,
                                              uint64_t preempt_buffer,
                                              uint64_t entry_point_reserved,
                                              uint64_t kernel_idx)
{
    (void)pad; /* parameter-region padding, never read */
    (void)entry_point_reserved; /* slot alignment with the original args */

    uint32_t block_idx = xg_blockid();
    uint32_t *global_exit_flag = (uint32_t *)preempt_buffer;
    uint64_t *preempt_idx_ptr = (uint64_t *)&((uint32_t *)preempt_buffer)[2];
    uint32_t *block_exit_flag =
        &((uint32_t *)preempt_buffer)[2 * block_idx + 4];
    uint32_t *block_restore_flag = block_exit_flag + 1;
    uint64_t preempt_idx = 0; /* declared before the gotos, as upstream */

    if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0) goto sync;

    if (!*global_exit_flag) {
        *block_exit_flag = 0;
        goto fence;
    }

    *block_exit_flag = 1;
    preempt_idx = *preempt_idx_ptr; /* single read, as upstream */
    if (preempt_idx == 0) {
        *preempt_idx_ptr = kernel_idx;
        *block_restore_flag = 1;
        goto fence;
    }
    if (preempt_idx == kernel_idx) *block_restore_flag = 1;

fence:
    __threadfence_block();

sync:
    __syncthreads();
    if (*block_exit_flag != 0) xg_thread_exit();
    xg_check_exit_marker = 0x1badb002u;
}
