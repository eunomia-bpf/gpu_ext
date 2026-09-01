/* SPDX-License-Identifier: GPL-2.0 */

#ifndef REVISION_SCHED_FIXTURE
#error "REVISION_SCHED_FIXTURE must select one fixture"
#endif

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct nv_gpu_task_init_ctx {
    __u64 tsg_id;
    __u32 engine_type;
    __u64 default_timeslice;
    __u32 default_interleave;
    __u32 runlist_id;
};

struct nv_gpu_transition_u64_request {
    __u8 attempted;
    __u8 conflict;
    __u64 value;
};

struct nv_gpu_transition_u32_request {
    __u8 attempted;
    __u8 conflict;
    __u32 value;
};

struct nv_gpu_task_init_decision_ctx {
    struct nv_gpu_task_init_ctx input;
    struct nv_gpu_transition_u64_request timeslice_request;
    struct nv_gpu_transition_u32_request interleave_request;
};

struct nv_gpu_bind_ctx {
    __u64 tsg_id;
    __u32 runlist_id;
    __u32 channel_count;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 allow;
};

struct nv_gpu_task_destroy_ctx {
    __u64 tsg_id;
};

struct nv_gpu_sched_ops {
    int (*on_task_init)(struct nv_gpu_task_init_ctx *ctx);
    int (*on_bind)(struct nv_gpu_bind_ctx *ctx);
    int (*on_task_destroy)(struct nv_gpu_task_destroy_ctx *ctx);
};

_Static_assert(sizeof(struct nv_gpu_task_init_ctx) == 32,
               "scheduler fixture input ABI");
_Static_assert(sizeof(struct nv_gpu_transition_u64_request) == 16,
               "u64 request ABI");
_Static_assert(__builtin_offsetof(struct nv_gpu_transition_u64_request, value) == 8,
               "u64 request value offset");
_Static_assert(sizeof(struct nv_gpu_transition_u32_request) == 8,
               "u32 request ABI");
_Static_assert(__builtin_offsetof(struct nv_gpu_transition_u32_request, value) == 4,
               "u32 request value offset");
_Static_assert(__builtin_offsetof(struct nv_gpu_task_init_decision_ctx,
                                  timeslice_request) == 32,
               "timeslice request offset");
_Static_assert(__builtin_offsetof(struct nv_gpu_task_init_decision_ctx,
                                  interleave_request) == 48,
               "interleave request offset");
_Static_assert(sizeof(struct nv_gpu_task_init_decision_ctx) == 56,
               "scheduler decision wrapper ABI");

#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif

extern int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *ctx,
                                    __u64 timeslice_us) __ksym;
extern int bpf_nv_gpu_set_interleave(struct nv_gpu_task_init_ctx *ctx,
                                     __u32 interleave_level) __ksym;

char LICENSE[] SEC("license") = "GPL";

SEC("struct_ops/on_task_init")
int BPF_PROG(revision_sched_on_task_init, struct nv_gpu_task_init_ctx *policy_ctx)
{
#if REVISION_SCHED_FIXTURE == 1
    policy_ctx->default_timeslice = 1;
    return 0;
#elif REVISION_SCHED_FIXTURE == 2
    struct nv_gpu_task_init_decision_ctx *decision =
        (struct nv_gpu_task_init_decision_ctx *)policy_ctx;

    decision->timeslice_request.attempted = 1;
    return 0;
#elif REVISION_SCHED_FIXTURE == 3
    volatile __u64 observed = policy_ctx->tsg_id + policy_ctx->default_timeslice +
                              policy_ctx->runlist_id;
    return observed == ~0ULL;
#elif REVISION_SCHED_FIXTURE == 4
    return bpf_nv_gpu_set_timeslice(policy_ctx, 100);
#elif REVISION_SCHED_FIXTURE == 5
    return bpf_nv_gpu_set_interleave(policy_ctx, 0);
#else
#error "unknown scheduler verifier fixture"
#endif
}

SEC(".struct_ops")
struct nv_gpu_sched_ops revision_sched_ops = {
    .on_task_init = (void *)revision_sched_on_task_init,
};
