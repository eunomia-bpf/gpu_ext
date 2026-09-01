/* SPDX-License-Identifier: GPL-2.0 */
/* Ownership-safe loader for the revision-only host stride + LFU ablation. */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_stride_lfu.skel.h"

#define CONFIG_CONFIDENCE_THRESHOLD 0
#define CONFIG_PREFETCH_PAGES 1
#define CONFIG_MAX_STRIDE 2

struct engagement_stats {
    __u64 page_fault_calls;
    __u64 stride_detections;
    __u64 prefetches_issued;
    __u64 lfu_activations;
    __u64 lfu_accesses;
    __u64 lfu_sampled_updates;
    __u64 lfu_reorder_requests;
    __u64 eviction_prepares;
};

static volatile sig_atomic_t exiting;

static void handle_signal(int signo)
{
    (void)signo;
    exiting = 1;
}

static int libbpf_print_fn(enum libbpf_print_level level,
                           const char *format,
                           va_list args)
{
    if (level == LIBBPF_DEBUG)
        return 0;
    return vfprintf(stderr, format, args);
}

static int update_config(int fd, __u32 key, __u64 value)
{
    if (bpf_map_update_elem(fd, &key, &value, BPF_ANY) == 0)
        return 0;
    fprintf(stderr, "failed to set config key %u: %s\n", key,
            strerror(errno));
    return -errno;
}

static __u32 map_id(const struct bpf_map *map)
{
    struct bpf_map_info info = {};
    __u32 size = sizeof(info);
    int fd = bpf_map__fd(map);

    if (fd < 0 || bpf_obj_get_info_by_fd(fd, &info, &size))
        return 0;
    return info.id;
}

static __u32 prog_id(const struct bpf_program *prog)
{
    struct bpf_prog_info info = {};
    __u32 size = sizeof(info);
    int fd = bpf_program__fd(prog);

    if (fd < 0 || bpf_obj_get_info_by_fd(fd, &info, &size))
        return 0;
    return info.id;
}

static __u32 link_id(const struct bpf_link *link)
{
    struct bpf_link_info info = {};
    __u32 size = sizeof(info);
    int fd = bpf_link__fd(link);

    if (fd < 0 || bpf_obj_get_info_by_fd(fd, &info, &size))
        return 0;
    return info.id;
}

static int emit_stats(struct prefetch_stride_lfu_bpf *skel,
                      const char *event)
{
    struct engagement_stats stats = {};
    struct engagement_stats *per_cpu = NULL;
    __u32 key = 0;
    int fd = bpf_map__fd(skel->maps.engagement);
    int cpu_count = libbpf_num_possible_cpus();
    int cpu;

    if (cpu_count <= 0)
        return cpu_count ? cpu_count : -EINVAL;
    per_cpu = calloc((size_t)cpu_count, sizeof(*per_cpu));
    if (!per_cpu)
        return -ENOMEM;
    if (bpf_map_lookup_elem(fd, &key, per_cpu)) {
        fprintf(stderr, "failed to read engagement map: %s\n",
                strerror(errno));
        free(per_cpu);
        return -errno;
    }
    for (cpu = 0; cpu < cpu_count; ++cpu) {
        stats.page_fault_calls += per_cpu[cpu].page_fault_calls;
        stats.stride_detections += per_cpu[cpu].stride_detections;
        stats.prefetches_issued += per_cpu[cpu].prefetches_issued;
        stats.lfu_activations += per_cpu[cpu].lfu_activations;
        stats.lfu_accesses += per_cpu[cpu].lfu_accesses;
        stats.lfu_sampled_updates += per_cpu[cpu].lfu_sampled_updates;
        stats.lfu_reorder_requests += per_cpu[cpu].lfu_reorder_requests;
        stats.eviction_prepares += per_cpu[cpu].eviction_prepares;
    }
    free(per_cpu);

    printf("{\"event\":\"%s\",\"pid\":%ld,"
           "\"page_fault_calls\":%llu,\"stride_detections\":%llu,"
           "\"prefetches_issued\":%llu,\"lfu_activations\":%llu,"
           "\"lfu_accesses\":%llu,\"lfu_sampled_updates\":%llu,"
           "\"lfu_reorder_requests\":%llu,\"eviction_prepares\":%llu}\n",
           event, (long)getpid(),
           (unsigned long long)stats.page_fault_calls,
           (unsigned long long)stats.stride_detections,
           (unsigned long long)stats.prefetches_issued,
           (unsigned long long)stats.lfu_activations,
           (unsigned long long)stats.lfu_accesses,
           (unsigned long long)stats.lfu_sampled_updates,
           (unsigned long long)stats.lfu_reorder_requests,
           (unsigned long long)stats.eviction_prepares);
    fflush(stdout);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [-t confidence] [-n pages] [-m max_stride]\n",
            program);
}

int main(int argc, char **argv)
{
    struct prefetch_stride_lfu_bpf *skel = NULL;
    struct bpf_link *struct_link = NULL;
    __u64 confidence = 2;
    __u64 prefetch_pages = 2;
    __u64 max_stride = 128;
    int config_fd;
    int opt;
    int err = 0;

    while ((opt = getopt(argc, argv, "t:n:m:h")) != -1) {
        char *end = NULL;
        unsigned long long parsed;

        if (opt == 'h') {
            usage(argv[0]);
            return 0;
        }
        if (opt != 't' && opt != 'n' && opt != 'm') {
            usage(argv[0]);
            return 2;
        }
        errno = 0;
        parsed = strtoull(optarg, &end, 10);
        if (errno || !end || *end != '\0') {
            fprintf(stderr, "invalid numeric option: %s\n", optarg);
            return 2;
        }
        if (opt == 't')
            confidence = parsed;
        else if (opt == 'n')
            prefetch_pages = parsed;
        else
            max_stride = parsed;
    }
    if (optind != argc || prefetch_pages == 0 || max_stride == 0 ||
        confidence > 2147483647ULL || prefetch_pages > 2147483647ULL ||
        max_stride > 2147483647ULL) {
        usage(argv[0]);
        return 2;
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    libbpf_set_print(libbpf_print_fn);

    skel = prefetch_stride_lfu_bpf__open();
    if (!skel) {
        fprintf(stderr, "failed to open BPF skeleton\n");
        return 1;
    }
    err = prefetch_stride_lfu_bpf__load(skel);
    if (err) {
        fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
        goto out;
    }

    config_fd = bpf_map__fd(skel->maps.policy_config);
    err = update_config(config_fd, CONFIG_CONFIDENCE_THRESHOLD, confidence);
    if (!err)
        err = update_config(config_fd, CONFIG_PREFETCH_PAGES, prefetch_pages);
    if (!err)
        err = update_config(config_fd, CONFIG_MAX_STRIDE, max_stride);
    if (err)
        goto out;

    skel->links.prefetch_get_hint_va_block =
        bpf_program__attach(skel->progs.prefetch_get_hint_va_block);
    err = libbpf_get_error(skel->links.prefetch_get_hint_va_block);
    if (err) {
        skel->links.prefetch_get_hint_va_block = NULL;
        fprintf(stderr, "failed to attach va_block kprobe: %s\n",
                strerror(-err));
        goto out;
    }

    struct_link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_stride_lfu);
    err = libbpf_get_error(struct_link);
    if (err) {
        struct_link = NULL;
        fprintf(stderr, "failed to attach struct_ops: %s\n", strerror(-err));
        goto out;
    }

    printf("{\"event\":\"ready\",\"pid\":%ld,"
           "\"struct_link_id\":%u,\"kprobe_link_id\":%u,"
           "\"struct_map_id\":%u,\"engagement_map_id\":%u,"
           "\"config_map_id\":%u,\"program_ids\":{"
           "\"page_prefetch\":%u,\"block_activate\":%u,"
           "\"block_access\":%u,\"evict_prepare\":%u}}\n",
           (long)getpid(), link_id(struct_link),
           link_id(skel->links.prefetch_get_hint_va_block),
           map_id(skel->maps.uvm_ops_stride_lfu), map_id(skel->maps.engagement),
           map_id(skel->maps.policy_config),
           prog_id(skel->progs.gpu_page_prefetch),
           prog_id(skel->progs.gpu_block_activate),
           prog_id(skel->progs.gpu_block_access),
           prog_id(skel->progs.gpu_evict_prepare));

    while (!exiting) {
        sleep(1);
        if (!exiting)
            emit_stats(skel, "engagement");
    }
    emit_stats(skel, "final_engagement");

out:
    /* Destroy only links created and held by this exact process. */
    bpf_link__destroy(struct_link);
    prefetch_stride_lfu_bpf__destroy(skel);
    return err < 0 ? -err : err;
}
