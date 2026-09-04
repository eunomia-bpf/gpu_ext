#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#ifndef TRAMPOLINE_SCALING_MATRIX_HEADER
#define TRAMPOLINE_SCALING_MATRIX_HEADER "matrix.h"
#endif
#include TRAMPOLINE_SCALING_MATRIX_HEADER

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502

struct {
    __uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} marker_count SEC(".maps");

#ifndef TRAMPOLINE_SCALING_NOOP
struct {
    __uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP);
    __uint(max_entries, SCALE_COUNTER_KEYS);
    __type(key, u32);
    __type(value, u64);
} target_count SEC(".maps");

static const u64 (*gpu_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*gpu_grid_dim)(u64 *x, u64 *y, u64 *z) = (void *)508;
#endif

SEC("kprobe/trampoline_marker_kernel")
int cuda__scale_marker(void)
{
    u32 key = 0;
    u64 *count = bpf_map_lookup_elem(&marker_count, &key);
    if (count)
        *count += 1;
    return 0;
}

SEC("kprobe/trampoline_scale_kernel")
int cuda__scale_target(void)
{
#ifndef TRAMPOLINE_SCALING_NOOP
    u64 grid_x = 0, grid_y = 0, grid_z = 0;
    u64 block_x = 0, block_y = 0, block_z = 0;
    gpu_grid_dim(&grid_x, &grid_y, &grid_z);
    gpu_block_dim(&block_x, &block_y, &block_z);

    if (grid_y != 1 || grid_z != 1 || block_y != 1 || block_z != 1)
        return 0;

    u32 key = SCALE_COUNTER_KEYS;
#define MATCH_GEOMETRY(blocks, threads, counter_key)           \
    if (grid_x == (blocks) && block_x == (threads))              \
        key = (counter_key);
    SCALE_COUNTER_GEOMETRY_LIST(MATCH_GEOMETRY)
#undef MATCH_GEOMETRY
    if (key >= SCALE_COUNTER_KEYS)
        return 0;

    u64 *count = bpf_map_lookup_elem(&target_count, &key);
    if (count)
        *count += 1;
#endif
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
