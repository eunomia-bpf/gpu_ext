#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define WARP_MAP_ENTRIES 64
#define WARP_MAGIC 0x57504d4150000000ULL

static const u64 (*bpf_get_warp_id)(void) = (void *)510;

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, WARP_MAP_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} warp_values SEC(".maps");

SEC("kprobe/fig15_warp_map_kernel")
int cuda__noop(void)
{
	return 0;
}

SEC("kprobe/fig15_warp_map_kernel")
int cuda__shared(void)
{
	u32 key = 0;
	u64 value = WARP_MAGIC;
	return (int)bpf_map_update_elem(&warp_values, &key, &value, BPF_ANY);
}

SEC("kprobe/fig15_warp_map_kernel")
int cuda__warp(void)
{
	u64 warp = bpf_get_warp_id();
	u32 key = (u32)warp;
	if (key >= WARP_MAP_ENTRIES)
		return -1;
	u64 value = WARP_MAGIC ^ warp;
	return (int)bpf_map_update_elem(&warp_values, &key, &value, BPF_ANY);
}

char LICENSE[] SEC("license") = "GPL";
