/* SPDX-License-Identifier: GPL-2.0 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "gpreempt_context_smoke_rpc.h"

const volatile __u32 target_pid = 0;
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

/* Called only AFTER the real RPC wait; core RM notrace restrictions remain
 * intact. No driver field is written and no return value is overridden. */
SEC("kprobe/nv_gpu_sched_gsp_control_complete")
int rpc_completed(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    if (target_pid && (id >> 32) != target_pid) return 0;
    struct gp_rpc_event event = { .pid_tgid = id, .completed_ns = bpf_ktime_get_ns() };
    if (bpf_probe_read_kernel(&event.completion, sizeof(event.completion), (void *)PT_REGS_PARM1(ctx))) {
        count(2);
        return 0;
    }
    if (event.completion.command != 0xa06c0103) return 0;
    count(0);
    if (bpf_ringbuf_output(&events, &event, sizeof(event), 0)) count(3);
    else count(1);
    return 0;
}
char LICENSE[] SEC("license") = "GPL";
