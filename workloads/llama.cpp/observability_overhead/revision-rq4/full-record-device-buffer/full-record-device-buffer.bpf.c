#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define FULL_RECORD_VALUE_U64 u64
#include "full_record_device_buffer_value.h"

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_get_grid_dim)(u64 *x, u64 *y, u64 *z) = (void *)508;

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, FRDB_NUM_BANKS);
	__type(key, u32);
	__type(value, struct frdb_value);
} arena SEC(".maps");

SEC("kretprobe/_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli")
int cuda__retprobe(void)
{
	struct frdb_record record;
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	u64 block_dim_x, block_dim_y, block_dim_z;
	u64 grid_x, grid_y, grid_z;
	u64 width, height, thread_id;
	u64 bank, slot, counter;
	struct frdb_value *value;
	u32 key;

	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	bpf_get_block_dim(&block_dim_x, &block_dim_y, &block_dim_z);
	bpf_get_grid_dim(&grid_x, &grid_y, &grid_z);

	record.block_x = block_x;
	record.block_y = block_y;
	record.block_z = block_z;
	record.thread_x = thread_x;
	record.thread_y = thread_y;
	record.thread_z = thread_z;
	record.block_dim_x = block_dim_x;
	record.block_dim_y = block_dim_y;
	record.block_dim_z = block_dim_z;
	record.timestamp = bpf_get_globaltimer();

	/* Existing per-thread linear coordinate (getGlobalThreadId),
	 * computed in u64 before any bank/slot indexing. */
	width = grid_x * block_dim_x;
	height = grid_y * block_dim_y;
	thread_id = (block_z * block_dim_z + thread_z) * width * height
		   + (block_y * block_dim_y + thread_y) * width
		   + (block_x * block_dim_x + thread_x);

	if (thread_id >= FRDB_TOTAL_SLOTS) {
		/* Outside the supported 524288-slot geometry: count it in the
		 * bank-0 sink and do not index the record storage. */
		key = 0;
		value = bpf_map_lookup_elem(&arena, &key);
		if (value)
			value->total_out_of_range += 1;
		return 0;
	}

	bank = thread_id / FRDB_SLOTS_PER_BANK;
	slot = thread_id % FRDB_SLOTS_PER_BANK;
	key = (u32)bank;
	value = bpf_map_lookup_elem(&arena, &key);
	if (!value)
		return 0;
	counter = value->slot_counters[slot];
	if (counter >= FRDB_RECORDS_PER_SLOT) {
		value->total_overflow += 1;
		return 0;
	}
	value->records[slot * FRDB_RECORDS_PER_SLOT + counter] = record;
	value->slot_counters[slot] = counter + 1;
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
