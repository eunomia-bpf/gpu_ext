/* Device return-event counter, adapted from bpftime's threadhist example. */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#ifdef DEVICE_SMOKE_NEGATIVE_LANE_BRANCH
static u64 (*device_lane_id)(void) = (void *)511;
#endif

struct {
    __uint(type, 1502); /* BPFTIME per-GPU-thread array, not a kernel map. */
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} call_count SEC(".maps");

SEC("kretprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__count_return(void)
{
#ifdef DEVICE_SMOKE_NEGATIVE_LANE_BRANCH
    /* Strict-only negative: never run this object with verification bypassed. */
    if (device_lane_id() != 0)
        return 0;
#endif
    u32 key = 0;
    u64 *count = bpf_map_lookup_elem(&call_count, &key);
    if (count)
        *count += 1;
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
