/* SPDX-License-Identifier: GPL-2.0
 * Device-side constants for the BPF compilation of the guardian rule
 * (bpf/xsched_guardian.bpf.c). Kept separate from
 * level2/xsched_guardian_abi.h because the BPF toolchain compiles this
 * translation unit alone. */
#ifndef XG_BPF_ABI_H
#define XG_BPF_ABI_H

#define XG_LAUNCH_ORIGINAL 0ull
#define XG_LAUNCH_GUARDIAN 1ull
#define XG_LAUNCH_RESUME 2ull

#define XG_D_ABORT (1ULL << 0)
#define XG_D_CLEAR_EXIT (1ULL << 1)
#define XG_D_RECORD_IDX (1ULL << 2)
#define XG_D_RECORD_RESTORE (1ULL << 3)
#define XG_D_RESUME_SKIP (1ULL << 4)
#define XG_D_RESUME_CLEAR (1ULL << 5)

#define XG_BPF_SECTION "cuda__/xsched_guardian"

#endif /* XG_BPF_ABI_H */
