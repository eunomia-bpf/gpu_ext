/* SPDX-License-Identifier: GPL-2.0
 *
 * sm_120 port of the ORIGINAL check_preempt guardian from
 * platforms/cuda/hal/inject/inject.cu. Statement-for-statement follower of
 * the original: CTA-leader-only decision, per-block exit/restore flags,
 * first preempt_idx recording, trusted threadfence/barrier actuation, and
 * per-thread cooperative exit.
 *
 * Arguments occupy the first three 28-byte actuator slots (param0
 * preempt_buffer @+0, param1 entry placeholder @+8, param2 kernel_idx
 * @+16) so the patcher can re-encode this kernel's parameter LDC consumers
 * from the derived parameter base of the generated cubin onto the upstream
 * debugger region c[0x0][0x1880..] that instrument.cpp fills via
 * cuXtraSetDebuggerParams. The unused param1 keeps the signature-sized slot
 * aligned with the original 2nd argument word pair; the ported blob never
 * reads it (matching the original check_preempt prototype).
 */
#include <stdint.h>

static __device__ __forceinline__ uint32_t xg_blockid()
{
    return blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
}

static __device__ __forceinline__ void xg_thread_exit()
{
    asm volatile("exit;");
}

extern "C" __global__ void check_preempt_port(uint64_t preempt_buffer,
                                   uint64_t entry_point_reserved,
                                   uint64_t kernel_idx)
{
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
}
