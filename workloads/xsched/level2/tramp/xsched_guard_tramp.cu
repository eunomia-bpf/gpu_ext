/* SPDX-License-Identifier: GPL-2.0
 *
 * Trusted entry/abort/replay glue of the XSched Level-2 guardian, shared by
 * the matched native and BPF ports. This translation unit is compiled into
 * the astoolspatch cubin that NVBit patches into every target kernel entry.
 *
 * The glue reproduces the original guardian execution protocol from
 * inject.cu:
 *
 *   guardian launch (check_preempt): one leader per CTA materializes the
 *   bounded snapshot of the mutable global deactivation flag and the
 *   recorded preempt_idx (single consistent reader), takes the decision,
 *   performs the record writes of the decision (block exit flag, preempt
 *   idx, block restore flag) followed by __threadfence_block, then
 *   __syncthreads, then every thread of the CTA acts uniformly on the
 *   recorded flag and exits before any kernel work when the flag says
 *   abort.
 *
 *   resume launch (restore_exec): every thread materializes the per-block
 *   snapshot of the restore flag (read-only) and takes the decision;
 *   blocks whose restore flag is zero exit, the others synchronize, clear
 *   the flag, and re-enter the guardian entry (check_preempt) exactly as
 *   the original restore_exec transfers to func_entry_point; only after
 *   that re-check does execution fall through into the kernel body.
 *
 *   All reads of the externally-mutable argument block and the
 *   preempt-buffer flag/index words are volatile, requesting explicit
 *   accesses rather than compiler reuse. This is not a general guarantee
 *   of cross-device coherence or freshness.
 *
 * The ONLY difference between the matched arms is which function supplies
 * the bounded decision, selected at insertion time (decision_mode); BOTH
 * decisions consume the SAME bounded snapshot context (6 x u64 = 48 bytes,
 * level2/xsched_guardian_abi.h) that this glue materializes from the real
 * device words, and neither decision dereferences a device pointer:
 *
 *   decision_mode 0 = xg_native_decision: the original decision logic
 *     re-expressed in compiled device C over the snapshot.
 *
 *   decision_mode 1 = xg_bpf_decision: calls xsched_guardian, the real
 *     compiled eBPF program exported through the context-adapted
 *     workloads/sass-kretprobe bpf_to_ptx exporter (verify-time context
 *     descriptor 48 bytes, level2/bpf/bpf_to_ptx_ctx48.patch), and
 *     consumes the decision from its context write-back (the exported BPF
 *     ABI has no return value; the program records the decision word in
 *     the context, exactly like the verified sass-kretprobe result
 *     channel).
 */
#include <stdint.h>

#include "../xsched_guardian_abi.h"

extern "C" __device__ void xsched_guardian(unsigned long long ctx,
					   unsigned long long ctx_len);

struct XgArgs {
    uint64_t preempt_buf;
    uint64_t entry_point; /* reserved for the blob arm; unused by this glue */
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

/* Materialize the bounded snapshot from the device preempt buffer. Resume
 * launches read only the per-block restore flag; guardian launches read
 * the global deactivation flag and the recorded preempt_idx. */
static __device__ __forceinline__ void xg_snapshot_fill(
    struct XgSnapshotCtx *snap, const volatile XgArgs *args, uint64_t buf,
    uint64_t block_idx, uint64_t launch_type)
{
    snap->kernel_idx = (unsigned long long)args->kernel_idx;
    snap->block_and_type = block_idx | (launch_type << 32);
    snap->decision = 0;
    if (launch_type == XG_LAUNCH_RESUME) {
        snap->global_exit_flag = 0;
        snap->preempt_idx = 0;
        snap->block_restore_flag =
            (unsigned long long)(*(const volatile uint32_t *)
                                 (buf + 4ULL * (2ULL * block_idx + 5ULL)));
        return;
    }
    snap->block_restore_flag = 0;
    snap->global_exit_flag =
        (unsigned long long)(*(const volatile uint32_t *)buf);
    snap->preempt_idx =
        (*(const volatile unsigned long long *)(buf + 8ULL));
}

__device__ __noinline__ unsigned long long xg_native_decision(
    const struct XgSnapshotCtx *snap)
{
    if ((snap->block_and_type >> 32) == XG_LAUNCH_RESUME) {
        return (snap->block_restore_flag == 0) ? XG_D_RESUME_SKIP
                                              : XG_D_RESUME_CLEAR;
    }

    /* check_preempt leader decision from the same snapshot the compiled
     * eBPF program consumes */
    if (snap->global_exit_flag == 0) return XG_D_CLEAR_EXIT;

    unsigned long long decision = XG_D_ABORT;
    if (snap->preempt_idx == 0) {
        decision |= XG_D_RECORD_IDX | XG_D_RECORD_RESTORE;
    } else if (snap->preempt_idx == snap->kernel_idx) {
        decision |= XG_D_RECORD_RESTORE;
    }
    return decision;
}

__device__ __noinline__ unsigned long long xg_bpf_decision(
    struct XgSnapshotCtx *snap)
{
    xsched_guardian((unsigned long long)snap,
                    (unsigned long long)sizeof(*snap));
    /* The decision is consumed from the BPF program's context write-back;
     * the exported device BPF ABI returns no value. */
    return snap->decision;
}

extern "C" __device__ __noinline__ void xg_tramp(int32_t guard,
                                                 uint64_t args_dev,
                                                 uint32_t decision_mode)
{
    if (guard == 0) return;

    /* Un-published launches (args_dev == 0) never actuate: a target kernel
     * launched outside the XSched queue must run without the guardian. */
    if (args_dev == 0) return;

    /* The argument block is written by the host actuator before each launch
     * and the preempt-buffer words are written by other launches/threads;
     * read them volatile to request explicit accesses. The protocol still
     * relies on its existing host/device ordering and synchronization. */
    const volatile XgArgs *args = (const volatile XgArgs *)args_dev;
    const uint64_t buf = args->preempt_buf;
    const uint32_t launch_type = args->launch_type;
    if (buf == 0 || launch_type == XG_LAUNCH_ORIGINAL) return;

    const uint64_t block_idx = (uint64_t)xg_blockid();

    if (launch_type == XG_LAUNCH_RESUME) {
        /* restore_exec: every thread materializes the per-block restore flag
         * (read-only snapshot) and takes the decision; blocks whose restore
         * flag is zero exit, the others synchronize, clear the flag, and then
         * re-enter the guardian entry below (as the original restore_exec
         * transfers to func_entry_point) rather than returning straight to
         * the application body. */
        volatile uint32_t *restore =
            (volatile uint32_t *)(buf + 4ULL * (2ULL * block_idx + 5ULL));
        struct XgSnapshotCtx snap;
        xg_snapshot_fill(&snap, args, buf, block_idx, (uint64_t)launch_type);
        const unsigned long long d =
            decision_mode == 1u ? xg_bpf_decision(&snap)
                                : xg_native_decision(&snap);
        if (d & XG_D_RESUME_SKIP) xg_exit();
        __syncthreads();
        *restore = 0; /* all threads, as in restore_exec */
        /* fall through to the guardian re-entry below */
    }

    /* check_preempt (guardian entry): taken both by direct GUARDIAN launches
     * and by the resume re-entry above. The leader materializes an explicit
     * GUARDIAN snapshot (global exit flag + recorded preempt idx), not the
     * resume snapshot. */
    volatile uint32_t *block_exit =
        (volatile uint32_t *)(buf + 4ULL * (2ULL * block_idx + 4ULL));
    volatile uint32_t *block_restore =
        (volatile uint32_t *)(buf + 4ULL * (2ULL * block_idx + 5ULL));
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        struct XgSnapshotCtx snap;
        xg_snapshot_fill(&snap, args, buf, block_idx, XG_LAUNCH_GUARDIAN);
        const unsigned long long d =
            decision_mode == 1u ? xg_bpf_decision(&snap)
                                : xg_native_decision(&snap);
        if (d & XG_D_CLEAR_EXIT) *block_exit = 0;
        if (d & XG_D_ABORT) *block_exit = 1;
        if (d & XG_D_RECORD_IDX)
            *(volatile unsigned long long *)(buf + 8ULL) = snap.kernel_idx;
        if (d & XG_D_RECORD_RESTORE) *block_restore = 1;
        __threadfence_block();
    }
    __syncthreads();
    if (*block_exit != 0) xg_exit(); /* whole CTA aborts, per-CTA record kept */
}
