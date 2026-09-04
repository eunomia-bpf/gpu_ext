/* SPDX-License-Identifier: GPL-2.0 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "revision_init_trace.h"

const volatile __u32 target_tgid = 0;

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 262144);
} events SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, REVISION_INIT_TRACE_STAT_COUNT);
	__type(key, __u32);
	__type(value, __u64);
} stats SEC(".maps");

static __always_inline void count(__u32 key)
{
	__u64 *value = bpf_map_lookup_elem(&stats, &key);

	if (value)
		__sync_fetch_and_add(value, 1);
}

SEC("kprobe/nv_gpu_sched_init_diagnostic")
int observe_init_diagnostic(struct pt_regs *ctx)
{
	__u64 id = bpf_get_current_pid_tgid();
	struct revision_init_trace_event event = {
		.pid_tgid = id,
		.timestamp_ns = bpf_ktime_get_ns(),
		.kind = REVISION_INIT_EVENT_DIAGNOSTIC,
	};

	if (!target_tgid || (id >> 32) != target_tgid)
		return 0;
	count(REVISION_INIT_DIAGNOSTIC_OBSERVED);
	if (bpf_probe_read_kernel(&event.diagnostic, sizeof(event.diagnostic),
				  (void *)PT_REGS_PARM1(ctx))) {
		count(REVISION_INIT_DIAGNOSTIC_READ_ERROR);
		return 0;
	}
	if (bpf_ringbuf_output(&events, &event, sizeof(event), 0))
		count(REVISION_INIT_DIAGNOSTIC_DROP);
	else
		count(REVISION_INIT_DIAGNOSTIC_EMITTED);
	return 0;
}

SEC("kprobe/nv_gpu_sched_gsp_control_complete")
int observe_gsp_completion(struct pt_regs *ctx)
{
	__u64 id = bpf_get_current_pid_tgid();
	struct revision_init_trace_event event = {
		.pid_tgid = id,
		.timestamp_ns = bpf_ktime_get_ns(),
		.kind = REVISION_INIT_EVENT_GSP,
	};

	if (!target_tgid || (id >> 32) != target_tgid)
		return 0;
	if (bpf_probe_read_kernel(&event.gsp, sizeof(event.gsp),
				  (void *)PT_REGS_PARM1(ctx))) {
		count(REVISION_INIT_GSP_READ_ERROR);
		return 0;
	}
	if (event.gsp.command != REVISION_INIT_GSP_SET_TIMESLICE &&
	    event.gsp.command != REVISION_INIT_GSP_SET_INTERLEAVE)
		return 0;
	count(REVISION_INIT_GSP_OBSERVED);
	if (bpf_ringbuf_output(&events, &event, sizeof(event), 0))
		count(REVISION_INIT_GSP_DROP);
	else
		count(REVISION_INIT_GSP_EMITTED);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
