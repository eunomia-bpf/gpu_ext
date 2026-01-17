// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "cuda_launch_trace.skel.h"
#include "cuda_launch_trace_event.h"

// Default CUDA library paths to try
static const char *cuda_driver_libs[] = {
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib64/libcuda.so",
    NULL
};

static const char *cuda_runtime_libs[] = {
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/lib/x86_64-linux-gnu/libcudart.so",
    "/usr/lib64/libcudart.so",
    NULL
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t exiting = 0;
static __u64 start_time_ns = 0;

static void print_stats(struct cuda_launch_trace_bpf *skel)
{
    int stats_fd = bpf_map__fd(skel->maps.stats);
    __u32 key;
    __u64 val;
    __u64 stats[3] = {0};

    // Read all stats
    for (key = 0; key < 3; key++) {
        if (bpf_map_lookup_elem(stats_fd, &key, &val) == 0) {
            stats[key] = val;
        }
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "CUDA KERNEL LAUNCH TRACE SUMMARY\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "cuLaunchKernel (Driver)   %8llu\n", stats[0]);
    fprintf(stderr, "cudaLaunchKernel (Runtime)%8llu\n", stats[1]);
    fprintf(stderr, "--------------------------------------------------------------------------------\n");
    fprintf(stderr, "TOTAL                     %8llu\n", stats[0] + stats[1]);
    if (stats[2] > 0) {
        fprintf(stderr, "DROPPED                   %8llu\n", stats[2]);
    }
    fprintf(stderr, "================================================================================\n");
}

static void sig_handler(int sig)
{
    exiting = 1;
}

static const char *hook_type_str(__u32 hook_type)
{
    switch (hook_type) {
    case HOOK_CULAUNCHKERNEL:
        return "cuLaunchKernel";
    case HOOK_CUDALAUNCHKERNEL:
        return "cudaLaunchKernel";
    default:
        return "unknown";
    }
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
    const struct cuda_launch_event *e = data;
    __u64 elapsed_ms;

    if (start_time_ns == 0)
        start_time_ns = e->timestamp_ns;

    elapsed_ms = (e->timestamp_ns - start_time_ns) / 1000000;

    // CSV output format:
    // time_ms,hook_type,pid,tid,comm,function,grid(x,y,z),block(x,y,z),shared_mem,stream
    printf("%llu,%s,%u,%u,%s,0x%llx,(%u,%u,%u),(%u,%u,%u),%u,0x%llx\n",
           (unsigned long long)elapsed_ms,
           hook_type_str(e->hook_type),
           e->pid,
           e->tid,
           e->comm,
           (unsigned long long)e->function,
           e->grid_dim_x, e->grid_dim_y, e->grid_dim_z,
           e->block_dim_x, e->block_dim_y, e->block_dim_z,
           e->shared_mem_bytes,
           (unsigned long long)e->stream);

    return 0;
}

// Find first existing library from a list
static const char *find_library(const char **lib_paths)
{
    for (int i = 0; lib_paths[i] != NULL; i++) {
        if (access(lib_paths[i], F_OK) == 0) {
            return lib_paths[i];
        }
    }
    return NULL;
}

// Attach uprobe to a library and function
static int attach_uprobe(struct bpf_program *prog, const char *lib_path,
                         const char *func_name, struct bpf_link **link)
{
    LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts,
        .func_name = func_name,
        .retprobe = false,
    );

    *link = bpf_program__attach_uprobe_opts(prog, -1 /* any process */,
                                             lib_path, 0 /* offset */,
                                             &uprobe_opts);
    if (!*link) {
        fprintf(stderr, "Failed to attach uprobe to %s in %s: %s\n",
                func_name, lib_path, strerror(errno));
        return -1;
    }

    fprintf(stderr, "Attached uprobe to %s in %s\n", func_name, lib_path);
    return 0;
}

int main(int argc, char **argv)
{
    struct cuda_launch_trace_bpf *skel = NULL;
    struct ring_buffer *rb = NULL;
    struct bpf_link *link_culaunch = NULL;
    struct bpf_link *link_cudalaunch = NULL;
    const char *cuda_driver_lib = NULL;
    const char *cuda_runtime_lib = NULL;
    int err;

    // Parse command-line arguments for custom library paths
    if (argc > 1) {
        cuda_driver_lib = argv[1];
    } else {
        cuda_driver_lib = find_library(cuda_driver_libs);
    }

    if (argc > 2) {
        cuda_runtime_lib = argv[2];
    } else {
        cuda_runtime_lib = find_library(cuda_runtime_libs);
    }

    // Set up libbpf errors and debug info callback
    libbpf_set_print(libbpf_print_fn);

    // Open BPF application
    skel = cuda_launch_trace_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    // Load & verify BPF programs
    err = cuda_launch_trace_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load and verify BPF skeleton: %d\n", err);
        goto cleanup;
    }

    // Attach uprobes manually
    if (cuda_driver_lib) {
        fprintf(stderr, "Using CUDA Driver library: %s\n", cuda_driver_lib);
        err = attach_uprobe(skel->progs.trace_cuLaunchKernel, cuda_driver_lib,
                           "cuLaunchKernel", &link_culaunch);
        if (err && err != -ENOENT) {
            fprintf(stderr, "Warning: Failed to attach cuLaunchKernel uprobe\n");
        }
    } else {
        fprintf(stderr, "Warning: CUDA Driver library (libcuda.so) not found\n");
    }

    if (cuda_runtime_lib) {
        fprintf(stderr, "Using CUDA Runtime library: %s\n", cuda_runtime_lib);
        err = attach_uprobe(skel->progs.trace_cudaLaunchKernel, cuda_runtime_lib,
                           "cudaLaunchKernel", &link_cudalaunch);
        if (err && err != -ENOENT) {
            fprintf(stderr, "Warning: Failed to attach cudaLaunchKernel uprobe\n");
        }
    } else {
        fprintf(stderr, "Warning: CUDA Runtime library (libcudart.so) not found\n");
    }

    if (!link_culaunch && !link_cudalaunch) {
        fprintf(stderr, "Error: No CUDA libraries found or failed to attach uprobes\n");
        fprintf(stderr, "Usage: %s [libcuda.so path] [libcudart.so path]\n", argv[0]);
        err = 1;
        goto cleanup;
    }

    // Set up ring buffer polling
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) {
        err = -1;
        fprintf(stderr, "Failed to create ring buffer\n");
        goto cleanup;
    }

    // Set up signal handler
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Print CSV header
    printf("time_ms,hook_type,pid,tid,comm,function,grid_dim,block_dim,shared_mem,stream\n");

    fprintf(stderr, "Tracing CUDA kernel launches... Press Ctrl-C to stop.\n");

    // Process events
    while (!exiting) {
        err = ring_buffer__poll(rb, 100 /* timeout, ms */);
        // Ctrl-C will cause -EINTR
        if (err == -EINTR) {
            err = 0;
            break;
        }
        if (err < 0) {
            fprintf(stderr, "Error polling ring buffer: %d\n", err);
            break;
        }
    }

    print_stats(skel);

cleanup:
    bpf_link__destroy(link_culaunch);
    bpf_link__destroy(link_cudalaunch);
    ring_buffer__free(rb);
    cuda_launch_trace_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
