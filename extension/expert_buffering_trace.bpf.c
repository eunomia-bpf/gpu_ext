/* SPDX-License-Identifier: GPL-2.0 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

enum expert_trace_event_type {
	EXPERT_TRACE_GRAPH = 1,
	EXPERT_TRACE_LAYOUT = 2,
	EXPERT_TRACE_ROUTE = 3,
};

enum expert_trace_stat {
	EXPERT_TRACE_STAT_GRAPH = 0,
	EXPERT_TRACE_STAT_LAYOUT,
	EXPERT_TRACE_STAT_ROUTE,
	EXPERT_TRACE_STAT_DROPPED,
	EXPERT_TRACE_STAT_MAX,
};

struct expert_trace_event {
	u64 timestamp_ns;
	u64 pid_tgid;
	u64 graph_ordinal;
	u64 tensor_base;
	u64 total_bytes;
	u64 per_expert_bytes;
	u32 type;
	u32 n_experts;
	u32 is_bias;
	u32 expert_id;
	char tensor_name[64];
};

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 4 * 1024 * 1024);
} events SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} graph_sequence SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 256);
	__type(key, u64);
	__type(value, u64);
} thread_graph SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, EXPERT_TRACE_STAT_MAX);
	__type(key, u32);
	__type(value, u64);
} stats SEC(".maps");

static __always_inline void count_stat(u32 key)
{
	u64 *value = bpf_map_lookup_elem(&stats, &key);

	if (value)
		(*value)++;
}

static __always_inline struct expert_trace_event *reserve_event(u32 type)
{
	struct expert_trace_event *event;

	event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
	if (!event) {
		count_stat(EXPERT_TRACE_STAT_DROPPED);
		return NULL;
	}

	__builtin_memset(event, 0, sizeof(*event));
	event->timestamp_ns = bpf_ktime_get_ns();
	event->pid_tgid = bpf_get_current_pid_tgid();
	event->type = type;
	return event;
}

SEC("uprobe")
int BPF_UPROBE(trace_graph_begin, void *sched, void *graph)
{
	u32 zero = 0;
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u64 ordinal;
	u64 *sequence;
	struct expert_trace_event *event;

	(void)sched;
	(void)graph;

	sequence = bpf_map_lookup_elem(&graph_sequence, &zero);
	if (!sequence)
		return 0;

	ordinal = __sync_fetch_and_add(sequence, 1) + 1;
	bpf_map_update_elem(&thread_graph, &pid_tgid, &ordinal, BPF_ANY);
	count_stat(EXPERT_TRACE_STAT_GRAPH);

	event = reserve_event(EXPERT_TRACE_GRAPH);
	if (!event)
		return 0;
	event->graph_ordinal = ordinal;
	bpf_ringbuf_submit(event, 0);
	return 0;
}

SEC("uprobe")
int BPF_UPROBE(trace_tensor_layout,
	       const char *name,
	       const void *base,
	       u64 total_bytes,
	       u64 per_expert_bytes,
	       u32 n_experts,
	       u32 is_bias)
{
	struct expert_trace_event *event;

	count_stat(EXPERT_TRACE_STAT_LAYOUT);
	event = reserve_event(EXPERT_TRACE_LAYOUT);
	if (!event)
		return 0;

	event->tensor_base = (u64)base;
	event->total_bytes = total_bytes;
	event->per_expert_bytes = per_expert_bytes;
	event->n_experts = n_experts;
	event->is_bias = is_bias;
	bpf_probe_read_user_str(event->tensor_name, sizeof(event->tensor_name), name);
	bpf_ringbuf_submit(event, 0);
	return 0;
}

SEC("uprobe")
int BPF_UPROBE(trace_expert_route, const void *tensor_base, u32 expert_id)
{
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u64 *ordinal;
	struct expert_trace_event *event;

	count_stat(EXPERT_TRACE_STAT_ROUTE);
	event = reserve_event(EXPERT_TRACE_ROUTE);
	if (!event)
		return 0;

	ordinal = bpf_map_lookup_elem(&thread_graph, &pid_tgid);
	if (ordinal)
		event->graph_ordinal = *ordinal;
	event->tensor_base = (u64)tensor_base;
	event->expert_id = expert_id;
	bpf_ringbuf_submit(event, 0);
	return 0;
}
