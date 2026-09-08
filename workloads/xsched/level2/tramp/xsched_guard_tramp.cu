/* SPDX-License-Identifier: GPL-2.0
 *
 * Trusted entry/abort/replay glue of the XSched Level-2 guardian, shared by
 * the matched native and BPF ports. This translation unit is compiled into
 * the astoolspatch cubin that NVBit patches into every target kernel entry.
 *
 * The glue reproduces the original guardian execution protocol from
 * inject.cu:
 *
 *   guardian launch (check_preempt): one leader per CTA obtains the bounded
 *   decision (always one consistent reader of the mutable global
 *   deactivation flag), the leader performs the record writes of the
 *   decision (block exit flag, preempt_idx, block restore flag) followed by
 *   __threadfence_block, then __syncthreads, then every thread of the CTA
 *   acts uniformly on the recorded flag and exits before any kernel work
 *   when the flag says abort.
 *
 *   resume launch (restore_exec): every thread obtains the decision
 *   (read-only per-block restore flag, as in the original), blocks whose
 *   restore flag is zero exit, the others synchronize, clear the flag and
 *   fall through into the kernel body (command replay continues).
 *
 * The ONLY difference between the matched arms is which function supplies
 * the bounded decision, selected at insertion time (decision_mode):
 *
 *   XG_DECISION_NATIVE: xg_native_decision - the original decision logic
 *     re-expressed in compiled device C, reading the same context and
 *     preempt buffer bytes.
 *
 *   XG_DECISION_BPF: xg_bpf_decision - calls xsched_guardian, the real
 *     compiled eBPF program exported by ptxpass to a device .func, and
 *     consumes the decision from its context write-back (the exported BPF
 *     ABI has no return value; the BPF program records the decision word in
 *     the context, exactly like the verified sass-kretprobe result channel).
 */
#include <stdint.h>

#include "../xsched_guardian_abi.h"

extern "C" __device__ void xsched_guardian(unsigned long long ctx,
					   unsigned long long ctx_len);

struct XgArgs {
    uint64_t preempt_buf;
    uint64_t entry_point; /* native arm artifact; unused in BPF mode */
    int64_t kernel_idx;
    uint32_t launch_type; /* XG_LAUNCH_* */
    uint32_t reserved;
};

static __device__ __forceinline__ uint32_t xg_blockid()
{
    return blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
}

static __device__ __forceinline__ void xg_exit()
{
    asm volatile("exit;");
}

__device__ __noinline__ unsigned long long xg_native_decision(
    const struct XgGuardianCallCtx *ctx)
{
    const uint64_t buf = ctx->preempt_buffer;
    const uint64_t kernel_idx = ctx->kernel_idx;
    const uint64_t block_idx = ctx->block_and_type & 0xffffffffULL;
    const uint64_t launch_type = ctx->block_and_type >> 32;

    if (launch_type == XG_LAUNCH_RESUME) {
        const uint32_t *restore =
            (const uint32_t *)(buf + 4ULL * (2ULL * block_idx + 5ULL));
        return (*restore == 0) ? XG_D_RESUME_SKIP : XG_D_RESUME_CLEAR;
    }
    const uint32_t *global_exit_flag = (const uint32_t *)buf;
    if (*global_exit_flag == 0) return XG_D_CLEAR_EXIT;
    const unsigned long long *preempt_idx =
        (const unsigned long long *)(buf + 8ULL);
    unsigned long long decision = XG_D_ABORT;
    const unsigned long long recorded = *preempt_idx;
    if (recorded == 0) {
        decision |= XG_D_RECORD_IDX | XG_D_RECORD_RESTORE;
    } else if (recorded == kernel_idx) {
        decision |= XG_D_RECORD_RESTORE;
    }
    return decision;
}

__device__ __noinline__ unsigned long long xg_bpf_decision(
    const struct XgGuardianCallCtx *ctx)
{
    xsched_guardian((unsigned long long)ctx, (unsigned long long)sizeof(*ctx));
    /* The decision is consumed from the BPF program's context write-back;
     * the exported device BPF ABI returns no value. */
    return ctx->decision;
}

extern "C" __device__ __noinline__ void xg_tramp(int32_t guard,
                                                 uint64_t args_dev,
                                                 uint32_t decision_mode)
{
    if (guard == 0) return;

    /* Un-published launches (args_dev == 0) never actuate: a target kernel
     * launched outside the XSched queue must run without the guardian. */
    if (args_dev == 0) return;

    const XgArgs *args = (const XgArgs *)args_dev;
    const uint64_t buf = args->preempt_buf;
    if (buf == 0 || args->launch_type == XG_LAUNCH_ORIGINAL) return;

    struct XgGuardianCallCtx ctx_local;
    ctx_local.preempt_buffer = buf;
    ctx_local.kernel_idx = (unsigned long long)args->kernel_idx;
    ctx_local.block_and_type =
        (unsigned long long)xg_blockid() |
        ((unsigned long long)args->launch_type << 32);
    ctx_local.decision = 0;

    if (args->launch_type == XG_LAUNCH_RESUME) {
        uint32_t *restore =
            (uint32_t *)(buf + 4ULL * (2ULL * (ctx_local.block_and_type & 0xffffffffULL) + 5ULL));
        const unsigned long long d =
            decision_mode == 1u ? xg_bpf_decision(&ctx_local)
                                : xg_native_decision(&ctx_local);
        if (d & XG_D_RESUME_SKIP) xg_exit();
        __syncthreads();
        *restore = 0; /* all threads, as in restore_exec */
        return;       /* fall through into the kernel body */
    }

    uint32_t *block_exit =
        (uint32_t *)(buf + 4ULL *
                                (2ULL * (ctx_local.block_and_type & 0xffffffffULL) + 4ULL));
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const unsigned long long d =
            decision_mode == 1u ? xg_bpf_decision(&ctx_local)
                                : xg_native_decision(&ctx_local);
        if (d & XG_D_CLEAR_EXIT) *block_exit = 0;
        if (d & XG_D_ABORT) *block_exit = 1;
        if (d & XG_D_RECORD_IDX)
            *(unsigned long long *)(buf + 8ULL) = ctx_local.kernel_idx;
        if (d & XG_D_RECORD_RESTORE)
            *(uint32_t *)(buf + 4ULL *
                                    (2ULL * (ctx_local.block_and_type & 0xffffffffULL) + 5ULL)) = 1;
        __threadfence_block();
    }
    __syncthreads();
    if (*block_exit != 0) xg_exit(); /* whole CTA aborts, per-CTA record kept */
}
