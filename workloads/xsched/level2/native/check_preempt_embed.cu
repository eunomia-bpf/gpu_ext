/* SPDX-License-Identifier: GPL-2.0
 *
 * sm_120 native embed-port variant of check_preempt_port.cu for the
 * per-command fused image: identical preemption decision logic, but the
 * per-command values arrive as patched immediate operands instead of the
 * unavailable debugger-parameter window. Placeholders below are patched by
 * the HAL at instruction-copy time; each 64-bit value is emitted by the
 * compiler as two distinct 32-bit halves whose patterns are located by
 * nvdisasm and verified by the extraction step.
 */
#include <stdint.h>

// Build-only retention marker, as upstream port: keeps the conditional
// exit predicated and separates it from the final standalone EXIT.
__device__ uint32_t xg_check_exit_marker_embed;

#define XG_EMBED_PREEMPT_BUFFER 0xdf9f00010badc0deULL
#define XG_EMBED_KERNEL_IDX     0xdf9f00020badc0dfULL

static __device__ __forceinline__ uint32_t xg_blockid()
{
    return blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
}

static __device__ __forceinline__ void xg_thread_exit()
{
    asm volatile("exit;");
}

extern "C" __global__ void check_preempt_embed()
{
    const uint64_t preempt_buffer = XG_EMBED_PREEMPT_BUFFER;   // patched per command
    const uint64_t kernel_idx     = XG_EMBED_KERNEL_IDX;      // patched per command

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
    xg_check_exit_marker_embed = 0x1badb002u;
}
