#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define ONEVALUE_ARRAY_VALUE_U64 u64
#include "onevalue_array_value.h"

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct onevalue_value);
} arena SEC(".maps");

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe(void)
{
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	u64 block_dim_x, block_dim_y, block_dim_z;
	u64 linear_thread, warps_per_block;
	u64 coordinate_x, coordinate_y, coordinate_z;
	u64 timestamp, slot;
	u64 *counter;
	struct onevalue_value *value;
	struct onevalue_event *event;
	u32 key = 0;

	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	bpf_get_block_dim(&block_dim_x, &block_dim_y, &block_dim_z);
	linear_thread = thread_x + thread_y * block_dim_x;
	linear_thread += thread_z * block_dim_x * block_dim_y;
	if ((linear_thread & 31) != 0)
		return 0;
	warps_per_block = (block_dim_x * block_dim_y * block_dim_z + 31) >> 5;
	coordinate_x = block_x * warps_per_block + (linear_thread >> 5);
	coordinate_y = block_y;
	coordinate_z = block_z;
	timestamp = bpf_get_globaltimer();

	value = bpf_map_lookup_elem(&arena, &key);
	if (!value)
		return 0;
	if (coordinate_x >= ONEVALUE_ARRAY_MAX_WARPS || coordinate_y != 0 ||
	    coordinate_z != 0) {
		value->total_out_of_range += 1;
		return 0;
	}
	counter = &value->warp_counters[coordinate_x];
	slot = *counter;
	if (slot >= ONEVALUE_ARRAY_MAX_EVENTS_PER_WARP) {
		value->total_overflow += 1;
		return 0;
	}
	event = &value->events[coordinate_x *
				      ONEVALUE_ARRAY_MAX_EVENTS_PER_WARP +
				slot];
	event->coordinate_x = coordinate_x;
	event->coordinate_y = coordinate_y;
	event->coordinate_z = coordinate_z;
	event->timestamp = timestamp;
	*counter = slot + 1;
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
