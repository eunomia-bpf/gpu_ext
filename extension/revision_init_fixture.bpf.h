/* SPDX-License-Identifier: GPL-2.0 */
#ifndef REVISION_INIT_FIXTURE
#error "REVISION_INIT_FIXTURE must select one bounded init request sequence"
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
struct nv_gpu_bind_ctx {
    __u64 tsg_id;
    __u32 runlist_id;
    __u32 channel_count;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 allow;
};
struct nv_gpu_task_destroy_ctx { __u64 tsg_id; };
/* As in the existing load-only fixtures, the optional fourth callback is
 * absent. libbpf maps members by name and leaves it null in the kernel map. */
struct nv_gpu_sched_ops {
    int (*on_task_init)(struct nv_gpu_task_init_ctx *ctx);
    int (*on_bind)(struct nv_gpu_bind_ctx *ctx);
    int (*on_task_destroy)(struct nv_gpu_task_destroy_ctx *ctx);
};

_Static_assert(sizeof(struct nv_gpu_task_init_ctx) == 32, "575 init input ABI");
_Static_assert(__builtin_offsetof(struct nv_gpu_task_init_ctx,
                                 default_timeslice) == 16, "575 timeslice ABI");
_Static_assert(__builtin_offsetof(struct nv_gpu_task_init_ctx,
                                 runlist_id) == 28, "575 runlist ABI");

extern int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *, __u64) __ksym;
extern int bpf_nv_gpu_set_interleave(struct nv_gpu_task_init_ctx *, __u32) __ksym;
#include "revision_init_requests.h"

/* Unconfigured fixtures affect nobody. The live loader sets the exact gated
 * target TGID before object load, never its PID or a process-name match. */
const volatile __u32 target_tgid = 0;

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, struct revision_init_key);
    __type(value, struct revision_init_record);
} init_requests SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, INIT_STAT_COUNT);
    __type(key, __u32);
    __type(value, __u64);
} init_stats SEC(".maps");

static __always_inline void init_count(__u32 key)
{
    __u64 *value = bpf_map_lookup_elem(&init_stats, &key);
    if (value)
        __sync_fetch_and_add(value, 1);
}

SEC("struct_ops/on_task_init")
int BPF_PROG(revision_init_request, struct nv_gpu_task_init_ctx *policy_ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct revision_init_record initial = {0};
    struct revision_init_record *record;
    struct revision_init_key key = {0};

    if (!target_tgid || (id >> 32) != target_tgid)
        return 0;
    init_count(INIT_SEEN);
    key.pid_tgid = id;
    key.tsg_id = policy_ctx->tsg_id;
    key.runlist_id = policy_ctx->runlist_id;
    initial.input.tsg_id = policy_ctx->tsg_id;
    initial.input.engine_type = policy_ctx->engine_type;
    initial.input.default_timeslice = policy_ctx->default_timeslice;
    initial.input.default_interleave = policy_ctx->default_interleave;
    initial.input.runlist_id = policy_ctx->runlist_id;
    initial.timestamp_ns = bpf_ktime_get_ns();
    initial.fixture = REVISION_INIT_FIXTURE;

    /* Never overwrite a reused identity or silently lose an applied request.
     * Failure to reserve an observation leaves this target's defaults alone. */
    if (bpf_map_update_elem(&init_requests, &key, &initial, BPF_NOEXIST)) {
        init_count(INIT_RECORD_ERROR);
        return 0;
    }
    record = bpf_map_lookup_elem(&init_requests, &key);
    if (!record) {
        init_count(INIT_RECORD_ERROR);
        return 0;
    }
    revision_init_issue_requests(REVISION_INIT_FIXTURE, policy_ctx, &record->requests);
    record->complete = 1;
    init_count(INIT_RECORDED);
    return 0;
}

SEC(".struct_ops")
struct nv_gpu_sched_ops revision_init_ops = {
    .on_task_init = (void *)revision_init_request,
};

char LICENSE[] SEC("license") = "GPL";
