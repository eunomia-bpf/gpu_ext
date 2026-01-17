/* SPDX-License-Identifier: GPL-2.0 */
/*
 * GPU Scheduler struct_ops BPF program
 *
 * This program allows setting custom timeslices based on process name.
 * Userspace can configure policies via the process_timeslice map.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "gpu_sched_set_timeslices.h"

char _license[] SEC("license") = "GPL";

#define TASK_COMM_LEN 16

/* Map: process name -> timeslice (in microseconds) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, char[TASK_COMM_LEN]);
    __type(value, __u64);
} process_timeslice SEC(".maps");

/* Statistics map */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, __u32);
    __type(value, __u64);
} stats SEC(".maps");

#define STAT_TASK_INIT      0
#define STAT_BIND           1
#define STAT_TASK_DESTROY   2
#define STAT_TIMESLICE_MOD  3
#define STAT_POLICY_HIT     4
#define STAT_POLICY_MISS    5

/* Trace events for debugging */
struct gpu_sched_trace_event {
    __u64 tsg_id;
    __u64 timeslice_us;
    __u64 default_timeslice;
    char comm[TASK_COMM_LEN];
    __u8 policy_applied;  /* 1 if policy was applied, 0 otherwise */
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);  /* 256KB ring buffer */
} gpu_trace_events SEC(".maps");

/* Last N timeslice modifications for inspection */
struct timeslice_record {
    __u64 tsg_id;
    __u64 old_timeslice;
    __u64 new_timeslice;
    char comm[TASK_COMM_LEN];
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 32);
    __type(key, __u32);
    __type(value, struct timeslice_record);
} timeslice_history SEC(".maps");

/* Index for circular history buffer */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} history_index SEC(".maps");

static __always_inline void inc_stat(__u32 key)
{
    __u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline void record_timeslice_change(__u64 tsg_id, __u64 old_ts, __u64 new_ts, char *comm)
{
    __u32 zero = 0;
    __u32 *idx = bpf_map_lookup_elem(&history_index, &zero);
    if (!idx)
        return;

    __u32 slot = *idx % 32;
    struct timeslice_record rec = {};
    rec.tsg_id = tsg_id;
    rec.old_timeslice = old_ts;
    rec.new_timeslice = new_ts;
    __builtin_memcpy(rec.comm, comm, TASK_COMM_LEN);

    bpf_map_update_elem(&timeslice_history, &slot, &rec, BPF_ANY);
    __sync_fetch_and_add(idx, 1);
}

/*
 * on_task_init - Apply timeslice policy based on process name
 */
SEC("struct_ops/on_task_init")
int BPF_PROG(on_task_init, struct nv_gpu_task_init_ctx *init_ctx)
{
    char comm[TASK_COMM_LEN];
    __u64 *timeslice;

    inc_stat(STAT_TASK_INIT);

    if (!init_ctx)
        return 0;

    /* Get current process name */
    bpf_get_current_comm(&comm, sizeof(comm));

    bpf_printk("GPU sched: TSG %llu init, comm=%s, default_ts=%llu\n",
               init_ctx->tsg_id, comm, init_ctx->default_timeslice);

    /* Look up timeslice for this process */
    timeslice = bpf_map_lookup_elem(&process_timeslice, comm);
    if (timeslice && *timeslice > 0) {
        __u64 old_ts = init_ctx->default_timeslice;
        __u64 new_ts = *timeslice;

        bpf_nv_gpu_set_timeslice(init_ctx, new_ts);
        inc_stat(STAT_TIMESLICE_MOD);
        inc_stat(STAT_POLICY_HIT);

        /* Record the change */
        record_timeslice_change(init_ctx->tsg_id, old_ts, new_ts, comm);

        bpf_printk("GPU sched: TSG %llu MODIFIED timeslice %llu -> %llu us for %s\n",
                   init_ctx->tsg_id, old_ts, new_ts, comm);
        return 1;
    }

    inc_stat(STAT_POLICY_MISS);
    return 0;
}

/*
 * on_bind - Admission control hook
 */
SEC("struct_ops/on_bind")
int BPF_PROG(on_bind, struct nv_gpu_bind_ctx *bind_ctx)
{
    char comm[TASK_COMM_LEN];

    inc_stat(STAT_BIND);

    if (!bind_ctx)
        return 0;

    bpf_get_current_comm(&comm, sizeof(comm));
    bpf_printk("GPU sched: TSG %llu bind, comm=%s, channels=%u, ts=%llu\n",
               bind_ctx->tsg_id, comm, bind_ctx->channel_count, bind_ctx->timeslice_us);

    return 0;
}

/*
 * on_task_destroy - Cleanup hook
 */
SEC("struct_ops/on_task_destroy")
int BPF_PROG(on_task_destroy, struct nv_gpu_task_destroy_ctx *destroy_ctx)
{
    inc_stat(STAT_TASK_DESTROY);

    if (!destroy_ctx)
        return 0;

    bpf_printk("GPU sched: TSG %llu destroyed\n", destroy_ctx->tsg_id);

    return 0;
}

/* Register the struct_ops */
SEC(".struct_ops")
struct nv_gpu_sched_ops gpu_sched_ops = {
    .on_task_init = (void *)on_task_init,
    .on_bind = (void *)on_bind,
    .on_task_destroy = (void *)on_task_destroy,
};
