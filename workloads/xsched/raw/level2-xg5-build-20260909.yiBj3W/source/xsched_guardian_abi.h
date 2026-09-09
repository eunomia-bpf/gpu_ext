/* SPDX-License-Identifier: GPL-2.0
 *
 * Shared ABI of the XSched Level-2 guardian on sm_120.
 *
 * The 28-byte argument block reproduces the upstream debugger-parameter
 * layout that instrument.cpp passes via cuXtraSetDebuggerParams
 * (platforms/cuda/hal/src/level2/instrument.cpp) and that the original
 * sm_35/70/86 guardian SASS reads from c[0x0][0x1880..]:
 *
 *   offset  0  u64  preempt buffer device address
 *   offset  8  u64  instrumented entry point (native arm; unused in BPF arm)
 *   offset 16  i64  command index (CudaKernelCommand::GetIdx())
 *   offset 24  u32  killable flag (native arm) / launch type (BPF arm)
 *
 * Preempt buffer layout (upstream, uint32 units; see inject.cu and
 * instrument.cpp "preempt buffer layout" comment):
 *
 *   [0]                  u32  global exit flag (deactivation flag)
 *   [1]                  u32  reserved
 *   [2..4)               u64  preempt_idx (first aborted command index)
 *   [4 + 2*b]            u32  exit_flag_of_block_b
 *   [5 + 2*b]            u32  restore_flag_of_block_b
 */
#ifndef XSCHED_GUARDIAN_ABI_H
#define XSCHED_GUARDIAN_ABI_H

#define XG_ARGS_BYTES 28
#define XG_ARG_PREEMPT_BUF_OFF 0
#define XG_ARG_ENTRY_OFF 8
#define XG_ARG_KERNEL_IDX_OFF 16
#define XG_ARG_TAIL_OFF 24

/* Launch types carried in the arg block tail field (u32 @ offset 24). */
#define XG_LAUNCH_ORIGINAL 0u /* trampoline must not actuate */
#define XG_LAUNCH_GUARDIAN 1u /* check_preempt semantics */
#define XG_LAUNCH_RESUME 2u   /* restore_exec semantics */

/* Upstream debugger-parameter region bases used by the original guardian
 * blobs (the sm_35/70/86 SASS reads c[0x0][0x1880..]). These are the
 * ORIGINAL consumer offsets; they are not a compatibility proof. The
 * sm_120 native adapter must derive the parameter base of its own stub
 * cubins from the generated artifact (nvdisasm) and re-encode its LDC
 * consumers onto the debugger region, see level2/native/ldc_patcher.cpp. */
#define XG_PARAM_BASE 0x180
#define XG_DEBUGGER_BASE 0x1880
#define XG_DEBUGGER_SHIFT (XG_DEBUGGER_BASE - XG_PARAM_BASE)

/* Decision bits returned by the device BPF guardian program. The trusted
 * actuator (trampoline or native guardian glue) executes these; the BPF
 * program itself never issues barriers or thread exits. */
#define XG_D_ABORT (1ULL << 0)          /* leader writes block exit flag = 1 */
#define XG_D_CLEAR_EXIT (1ULL << 1)     /* leader writes block exit flag = 0 */
#define XG_D_RECORD_IDX (1ULL << 2)     /* leader writes preempt_idx      */
#define XG_D_RECORD_RESTORE (1ULL << 3) /* leader writes block restore flag = 1 */
#define XG_D_RESUME_SKIP (1ULL << 4)    /* resume: block never ran, exit   */
#define XG_D_RESUME_CLEAR (1ULL << 5)   /* resume: clear restore flag, run */

/* Device BPF program symbol and section. */
#define XG_BPF_SYMBOL xsched_guardian
#define XG_BPF_SECTION "cuda__/xsched_guardian"

/* Bounded snapshot decision context consumed by BOTH decisions (the
 * native-C decision and the compiled eBPF program). The trusted
 * trampoline materializes the slots from the real device words of the
 * preempt buffer; neither decision dereferences a device pointer, and the
 * BPF program touches only these six slots.
 *
 * Context-size contract (explicit): the exported device-BPF ABI passes
 * (context_ptr, context_length) with context_length == 48 bytes, and the
 * strict PREVAIL verify-time context descriptor in the adapted exporter
 * (level2/bpf/bpf_to_ptx_ctx48.patch) grants exactly 48 read/write bytes
 * at context_ptr. Granted size == used size == 6 x u64. */
#define XG_SNAPSHOT_BYTES 48
#define XG_SNAPSHOT_SLOTS 6
struct XgSnapshotCtx {
    unsigned long long kernel_idx;         /* slot 0 */
    unsigned long long block_and_type;     /* slot 1: low u32 block, high u32 type */
    unsigned long long global_exit_flag;   /* slot 2: u32 snapshot, zero-extended */
    unsigned long long preempt_idx;        /* slot 3: recorded preempt_idx */
    unsigned long long block_restore_flag; /* slot 4: u32 snapshot, zero-extended */
    unsigned long long decision;           /* slot 5: write-back (BPF) */
};

#endif /* XSCHED_GUARDIAN_ABI_H */
