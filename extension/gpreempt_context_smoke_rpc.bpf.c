/* SPDX-License-Identifier: GPL-2.0 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "gpreempt_context_smoke_rpc.h"

const volatile __u32 target_pid = 0;
struct gp_rpc_pending { struct gp_rpc_event event; __u32 nested; };
struct {
    __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 128);
    __type(key, __u64); __type(value, struct gp_rpc_pending);
} rpc_calls SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF); __uint(max_entries, 65536);
} events SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 4);
    __type(key, __u32); __type(value, __u64);
} stats SEC(".maps");
static __always_inline void count(__u32 key)
{
    __u64 *value = bpf_map_lookup_elem(&stats, &key);
    if (value) __sync_fetch_and_add(value, 1);
}

/* Source-confirmed 575 physical-RMAPI GSP entry, not the vGPU wrapper.
 * Only observe: never write driver fields or modify a return value. */
SEC("kprobe/rpcRmApiControl_GSP")
int rpc_enter(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    if (target_pid && (id >> 32) != target_pid) return 0;
    __u32 command = PT_REGS_PARM4(ctx);
    struct gp_rpc_pending *outer = bpf_map_lookup_elem(&rpc_calls, &id);
    if (outer) {
        outer->nested++;
        if (command == 0xa06c0103) count(2);
        return 0;
    }
    if (command != 0xa06c0103) return 0;
    struct gp_rpc_pending pending = { .event = {
        .pid_tgid = id, .entered_ns = bpf_ktime_get_ns(),
        .hclient = PT_REGS_PARM2(ctx), .hobject = PT_REGS_PARM3(ctx),
        .command = command, .params_size = PT_REGS_PARM6(ctx),
    }};
    if (pending.event.params_size != 8 ||
        bpf_probe_read_kernel(&pending.event.timeslice_us, sizeof(pending.event.timeslice_us), (void *)PT_REGS_PARM5(ctx)))
        pending.event.read_error = 1;
    if (bpf_map_update_elem(&rpc_calls, &id, &pending, BPF_NOEXIST)) count(2);
    else count(0);
    return 0;
}
SEC("kprobe/_issueRpcAndWait")
int wait_enter(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_rpc_pending *pending = bpf_map_lookup_elem(&rpc_calls, &id);
    if (pending) pending->event.issue_count++;
    return 0;
}
SEC("kretprobe/_issueRpcAndWait")
int wait_exit(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_rpc_pending *pending = bpf_map_lookup_elem(&rpc_calls, &id);
    if (pending) {
        pending->event.wait_count++;
        pending->event.wait_status = PT_REGS_RC(ctx);
        if (pending->event.wait_status) pending->event.wait_errors++;
    }
    return 0;
}
SEC("kretprobe/rpcRmApiControl_GSP")
int rpc_exit(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_rpc_pending *pending = bpf_map_lookup_elem(&rpc_calls, &id);
    if (!pending) return 0;
    if (pending->nested) { pending->nested--; return 0; }
    pending->event.elapsed_ns = bpf_ktime_get_ns() - pending->event.entered_ns;
    pending->event.return_status = PT_REGS_RC(ctx);
    if (bpf_ringbuf_output(&events, &pending->event, sizeof(pending->event), 0)) count(3);
    else count(1);
    bpf_map_delete_elem(&rpc_calls, &id);
    return 0;
}
char LICENSE[] SEC("license") = "GPL";
