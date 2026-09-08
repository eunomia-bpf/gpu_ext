/* SPDX-License-Identifier: GPL-2.0
 *
 * sm_120 port of the ORIGINAL restore_exec resume snippet from
 * platforms/cuda/hal/inject/inject.cu: per-block restore flag check, exit
 * for blocks that completed before deactivation, barrier, flag clear, and
 * transfer into the instrumented entry point held in the argument word
 * pair that the original reads from c[0x0][0x1888/0x188c].
 *
 * A leading unused 0x1500-byte pad struct (device of proof: native/probe.cu)
 * places the real arguments directly in that upstream debugger-parameter
 * window (preempt buffer @ c[0x0][0x1880/0x1884], entry point @
 * c[0x0][0x1888/0x188c]), so the compiler emits the actual consumers with
 * no binary immediate re-encoding of the generated cubin.
 *
 * The transfer uses a device function-pointer call at the runtime address;
 * the emitted instruction is taken from the actual generated cubin (the
 * patcher asserts a register-indirect CALL exists before vendoring the
 * arrays). The called instrumented image ends with the original kernel
 * EXIT, so the call never returns to the stub.
 */
#include <stdint.h>

/* Passed by value without array decay; reserves 0x1500 bytes at the start
 * of the parameter block so the real arguments land in the upstream
 * debugger-parameter window (see native/probe.cu for the device of proof). */
struct XgRestorePad {
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

typedef void (*xg_target_fn)();

extern "C" __global__ void restore_exec_port(XgRestorePad pad,
                                             uint64_t preempt_buffer,
                                             uint64_t entry_point)
{
    (void)pad; /* parameter-region padding, never read */
    uint32_t block_idx = xg_blockid();
    uint32_t *block_restore_flag =
        &((uint32_t *)preempt_buffer)[2 * block_idx + 5];

    if (*block_restore_flag == 0) xg_thread_exit();

    __syncthreads();
    *block_restore_flag = 0; /* all threads, as in the original */

    ((xg_target_fn)entry_point)();
}
