/* SPDX-License-Identifier: GPL-2.0 */

#ifndef REVISION_PMM_FIXTURE
#error "REVISION_PMM_FIXTURE must select one fixture"
#endif

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "uvm_types.h"

#define BPF_NO_KFUNC_PROTOTYPES
#include "bpf_testmod.h"

#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif

extern int bpf_gpu_request_reorder(uvm_bpf_pmm_decision_ctx_t *decision_ctx,
                                   __u64 destination,
                                   __u64 position) __ksym;

char LICENSE[] SEC("license") = "GPL";

SEC("struct_ops/gpu_block_access")
int BPF_PROG(revision_pmm_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    (void)pmm;
    (void)chunk;

#if REVISION_PMM_FIXTURE == 1
    decision_ctx->request.destination = NV_GPU_PMM_DESTINATION_USED;
    return 0;
#elif REVISION_PMM_FIXTURE == 2
    return bpf_gpu_request_reorder(decision_ctx,
                                   NV_GPU_PMM_DESTINATION_USED,
                                   NV_GPU_PMM_POSITION_HEAD);
#else
#error "unknown PMM verifier fixture"
#endif
}

SEC(".struct_ops")
struct gpu_mem_ops revision_pmm_ops = {
    .gpu_block_access = (void *)revision_pmm_block_access,
};
