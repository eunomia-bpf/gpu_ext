/* Device raw-record path and independently aggregated control. */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502
#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527

struct raw_record {
	u64 sequence;
	u64 block_x;
	u64 block_y;
	u64 block_z;
	u64 thread_x;
	u64 thread_y;
	u64 thread_z;
};

struct aggregate_state {
	u64 callbacks;
	u64 sequence_sum;
	u64 block_x_sum;
	u64 thread_x_sum;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, struct raw_record);
} raw_records SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct aggregate_state);
} aggregate SEC(".maps");

static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kretprobe/raw_map_kernel")
int cuda__capture_return(void)
{
	struct raw_record record = {};
	struct aggregate_state *state;
	u32 key = 0;

	bpf_get_block_idx(&record.block_x, &record.block_y, &record.block_z);
	bpf_get_thread_idx(&record.thread_x, &record.thread_y, &record.thread_z);
	state = bpf_map_lookup_elem(&aggregate, &key);
	if (state) {
		state->callbacks += 1;
		record.sequence = state->callbacks;
		state->sequence_sum += record.sequence;
		state->block_x_sum += record.block_x;
		state->thread_x_sum += record.thread_x;
	}
	bpf_perf_event_output(NULL, &raw_records, 0, &record, sizeof(record));
	return 0;
}

char LICENSE[] SEC("license") = "GPL";

