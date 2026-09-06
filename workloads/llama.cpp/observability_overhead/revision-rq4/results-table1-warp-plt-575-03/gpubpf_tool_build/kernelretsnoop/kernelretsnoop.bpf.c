#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502
#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527

struct data {
	u64 coordinate_x, coordinate_y, coordinate_z;
	u64 timestamp;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 256);
	__type(key, u32);
	__type(value, struct data);
} rb SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kretprobe/_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli")
int cuda__retprobe()
{
	struct data data = {};
	u64 block_x = 0, block_y = 0, block_z = 0;
	u64 thread_x = 0, thread_y = 0, thread_z = 0;
	u64 block_dim_x = 0, block_dim_y = 0, block_dim_z = 0;
	u64 linear_thread = 0;
	u64 warps_per_block = 0;

	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	bpf_get_block_dim(&block_dim_x, &block_dim_y, &block_dim_z);
	linear_thread = thread_x + thread_y * block_dim_x;
	linear_thread += thread_z * block_dim_x * block_dim_y;
	if ((linear_thread & 31) != 0)
		return 0;
	warps_per_block = (block_dim_x * block_dim_y * block_dim_z + 31) >> 5;
	data.coordinate_x = block_x * warps_per_block + (linear_thread >> 5);
	data.coordinate_y = block_y;
	data.coordinate_z = block_z;
	data.timestamp = bpf_get_globaltimer();
	return bpf_perf_event_output(NULL, &rb, 0, &data,
				     sizeof(struct data));
}

char LICENSE[] SEC("license") = "GPL";
