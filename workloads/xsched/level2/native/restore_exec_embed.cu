/* SPDX-License-Identifier: GPL-2.0
 *
 * sm_120 native embed-port variant of restore_exec_port.cu for the
 * per-command resume image: identical per-block restore logic, but the
 * per-command values arrive as patched immediate operands instead of the
 * unavailable debugger-parameter window. Placeholders are patched by the
 * HAL at instruction-copy time; each value is emitted by the compiler as a
 * single MOV.64 immediate located by nvdisasm and verified by the
 * extraction step. The transfer keeps the device function-pointer call at
 * the runtime address; the called instrumented image ends with the
 * original kernel EXIT, so the call never returns to the stub.
 */
#include <stdint.h>

#define XG_EMBED_RESTORE_PREEMPT 0xdf9f00030badc0e0ULL
#define XG_EMBED_RESTORE_ENTRY   0xdf9f00040badc0e1ULL

static __device__ __forceinline__ uint32_t xg_blockid()
{
    return blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
}

static __device__ __forceinline__ void xg_thread_exit()
{
    asm volatile("exit;");
}

typedef void (*xg_target_fn)();

extern "C" __global__ void restore_exec_embed()
{
    const uint64_t preempt_buffer = XG_EMBED_RESTORE_PREEMPT;  // patched per command
    const uint64_t entry_point    = XG_EMBED_RESTORE_ENTRY;    // patched per command

    uint32_t block_idx = xg_blockid();
    uint32_t *block_restore_flag =
        &((uint32_t *)preempt_buffer)[2 * block_idx + 5];

    if (*block_restore_flag == 0) xg_thread_exit();

    __syncthreads();
    *block_restore_flag = 0; /* all threads, as in the original */

    ((xg_target_fn)entry_point)();
}
