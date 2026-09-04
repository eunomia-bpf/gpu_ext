#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP 1513
#define MAP_ENTRIES 32
#define UPDATE_MAGIC 0x51a7000000000000ULL

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, MAP_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} device_values SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, MAP_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} host_values SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAP_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} rpc_values SEC(".maps");

/* Common device-resident sink used only by the three lookup programs. */
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, MAP_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} observed_values SEC(".maps");

static const u64 (*bpf_get_lane_id)(void) = (void *)511;

static __always_inline int lane_key(u32 *key)
{
	u64 lane = bpf_get_lane_id();
	if (lane >= MAP_ENTRIES)
		return -1;
	*key = (u32)lane;
	return 0;
}

static __always_inline u64 update_value(u32 key)
{
	return UPDATE_MAGIC ^ (u64)key;
}

SEC("kprobe/fig15_map_kernel")
int cuda__noop(void)
{
	return 0;
}

SEC("kprobe/fig15_map_kernel")
int cuda__device_update(void)
{
	u32 key;
	if (lane_key(&key))
		return -1;
	u64 value = update_value(key);
	return (int)bpf_map_update_elem(&device_values, &key, &value, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__host_update(void)
{
	u32 key;
	if (lane_key(&key))
		return -1;
	u64 value = update_value(key);
	return (int)bpf_map_update_elem(&host_values, &key, &value, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__rpc_update(void)
{
	u32 key;
	if (lane_key(&key))
		return -1;
	u64 value = update_value(key);
	return (int)bpf_map_update_elem(&rpc_values, &key, &value, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__device_lookup(void)
{
	u32 key;
	if (lane_key(&key))
		return -1;
	u64 *value = bpf_map_lookup_elem(&device_values, &key);
	if (!value)
		return -2;
	u64 copy = *value;
	return (int)bpf_map_update_elem(&observed_values, &key, &copy, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__host_lookup(void)
{
	u32 key;
	if (lane_key(&key))
		return -1;
	u64 *value = bpf_map_lookup_elem(&host_values, &key);
	if (!value)
		return -2;
	u64 copy = *value;
	return (int)bpf_map_update_elem(&observed_values, &key, &copy, BPF_ANY);
}

SEC("kprobe/fig15_map_kernel")
int cuda__rpc_lookup(void)
{
	u32 key;
	if (lane_key(&key))
		return -1;
	u64 *value = bpf_map_lookup_elem(&rpc_values, &key);
	if (!value)
		return -2;
	u64 copy = *value;
	return (int)bpf_map_update_elem(&observed_values, &key, &copy, BPF_ANY);
}

char LICENSE[] SEC("license") = "GPL";
