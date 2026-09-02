/* Device return-event counter, adapted from bpftime's threadhist example. */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

struct {
    __uint(type, 1502); /* BPFTIME per-GPU-thread array, not a kernel map. */
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} call_count SEC(".maps");

SEC("kretprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__count_return(void)
{
    u32 key = 0;
    u64 *count = bpf_map_lookup_elem(&call_count, &key);
    if (count)
        *count += 1;
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
