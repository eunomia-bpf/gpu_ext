// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * CUDA Launch Kernel Trace - Trace CUDA kernel launches using uprobes
 *
 * Traces cuLaunchKernel (CUDA Driver API) and cudaLaunchKernel (CUDA Runtime API)
 * to monitor GPU kernel launches with grid/block dimensions and stream info.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "cuda_launch_trace_event.h"

char LICENSE[] SEC("license") = "GPL";

// Ring buffer for outputting events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Statistics counters
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

#define STAT_CULAUNCHKERNEL 0
#define STAT_CUDALAUNCHKERNEL 1
#define STAT_DROPPED 2

static __always_inline void inc_stat(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

/*
 * Hook: cuLaunchKernel (CUDA Driver API)
 *
 * CUresult cuLaunchKernel(
 *     CUfunction f,
 *     unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
 *     unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
 *     unsigned int sharedMemBytes,
 *     CUstream hStream,
 *     void **kernelParams,
 *     void **extra
 * );
 */
SEC("uprobe/cuLaunchKernel")
int trace_cuLaunchKernel(struct pt_regs *ctx)
{
    struct cuda_launch_event *e;
    __u64 pid_tgid;

    inc_stat(STAT_CULAUNCHKERNEL);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    pid_tgid = bpf_get_current_pid_tgid();
    e->pid = pid_tgid >> 32;
    e->tid = (__u32)pid_tgid;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_CULAUNCHKERNEL;

    // Read function arguments
    // On x86_64: RDI, RSI, RDX, RCX, R8, R9, then stack
    e->function = PT_REGS_PARM1(ctx);
    e->grid_dim_x = (__u32)PT_REGS_PARM2(ctx);
    e->grid_dim_y = (__u32)PT_REGS_PARM3(ctx);
    e->grid_dim_z = (__u32)PT_REGS_PARM4(ctx);
    e->block_dim_x = (__u32)PT_REGS_PARM5(ctx);
    e->block_dim_y = (__u32)PT_REGS_PARM6(ctx);

    // blockDimZ, sharedMemBytes, hStream are on the stack
    // Read from user stack
    void *sp = (void *)PT_REGS_SP(ctx);
    bpf_probe_read_user(&e->block_dim_z, sizeof(e->block_dim_z), sp + 8);
    bpf_probe_read_user(&e->shared_mem_bytes, sizeof(e->shared_mem_bytes), sp + 16);
    bpf_probe_read_user(&e->stream, sizeof(e->stream), sp + 24);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook: cudaLaunchKernel (CUDA Runtime API)
 *
 * cudaError_t cudaLaunchKernel(
 *     const void *func,
 *     dim3 gridDim,
 *     dim3 blockDim,
 *     void **args,
 *     size_t sharedMem,
 *     cudaStream_t stream
 * );
 *
 * dim3 is a struct with {x, y, z} each being unsigned int
 */
SEC("uprobe/cudaLaunchKernel")
int trace_cudaLaunchKernel(struct pt_regs *ctx)
{
    struct cuda_launch_event *e;
    __u64 pid_tgid;
    struct {
        __u32 x;
        __u32 y;
        __u32 z;
    } dim3_grid, dim3_block;

    inc_stat(STAT_CUDALAUNCHKERNEL);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    pid_tgid = bpf_get_current_pid_tgid();
    e->pid = pid_tgid >> 32;
    e->tid = (__u32)pid_tgid;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_CUDALAUNCHKERNEL;

    // Read function arguments
    e->function = PT_REGS_PARM1(ctx);

    // Read dim3 gridDim (passed by value, so it's on stack or in registers depending on ABI)
    // On x86_64 System V ABI, small structs are passed in registers
    // dim3 gridDim is in RSI, RDX (first two uints in RSI, third in RDX lower 32 bits)
    __u64 grid_packed = PT_REGS_PARM2(ctx);
    e->grid_dim_x = (__u32)(grid_packed & 0xFFFFFFFF);
    e->grid_dim_y = (__u32)(grid_packed >> 32);
    e->grid_dim_z = (__u32)PT_REGS_PARM3(ctx);

    // Read dim3 blockDim (in RCX, R8)
    __u64 block_packed = PT_REGS_PARM4(ctx);
    e->block_dim_x = (__u32)(block_packed & 0xFFFFFFFF);
    e->block_dim_y = (__u32)(block_packed >> 32);
    e->block_dim_z = (__u32)PT_REGS_PARM5(ctx);

    // Read sharedMem and stream from stack
    void *sp = (void *)PT_REGS_SP(ctx);
    bpf_probe_read_user(&e->shared_mem_bytes, sizeof(e->shared_mem_bytes), sp + 8);
    bpf_probe_read_user(&e->stream, sizeof(e->stream), sp + 16);

    bpf_ringbuf_submit(e, 0);
    return 0;
}
