#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP 1513
#define UPDATE_MAGIC 0x51a7cafe00000001ULL

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} device_values SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} host_values SEC(".maps");

/* Common device-resident sink for the two lookup programs. */
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} observed_values SEC(".maps");

SEC("kprobe/fig15_map_kernel")
int cuda__noop(void)
{
	return 0;
}

SEC("kprobe/fig15_map_kernel")
int cuda__dev_up(void)
{
	u32 key = 0;
	u64 value = UPDATE_MAGIC;
	return (int)bpf_map_update_elem(&device_values, &key, &value, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__host_up(void)
{
	u32 key = 0;
	u64 value = UPDATE_MAGIC;
	return (int)bpf_map_update_elem(&host_values, &key, &value, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__dev_look(void)
{
	u32 key = 0;
	u64 *value = bpf_map_lookup_elem(&device_values, &key);
	if (!value)
		return -1;
	u64 copy = *value;
	return (int)bpf_map_update_elem(&observed_values, &key, &copy, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__host_look(void)
{
	u32 key = 0;
	u64 *value = bpf_map_lookup_elem(&host_values, &key);
	if (!value)
		return -1;
	u64 copy = *value;
	return (int)bpf_map_update_elem(&observed_values, &key, &copy, BPF_ANY);
}

char LICENSE[] SEC("license") = "GPL";
