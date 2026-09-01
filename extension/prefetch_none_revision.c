/* SPDX-License-Identifier: GPL-2.0 */
/* Ownership-safe loader for the revision no-prefetch policy. */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_none_revision.skel.h"

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

static __u32 map_id(const struct bpf_map *map)
{
    struct bpf_map_info info = {};
    __u32 size = sizeof(info);
    int fd = bpf_map__fd(map);

    if (fd < 0 || bpf_obj_get_info_by_fd(fd, &info, &size))
        return 0;
    return info.id;
}

static __u32 prog_id(const struct bpf_program *program)
{
    struct bpf_prog_info info = {};
    __u32 size = sizeof(info);
    int fd = bpf_program__fd(program);

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

int main(void)
{
    struct prefetch_none_revision_bpf *skel = NULL;
    struct bpf_link *link = NULL;
    int err = 0;

    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    libbpf_set_print(libbpf_print_fn);

    skel = prefetch_none_revision_bpf__open();
    if (!skel) {
        fprintf(stderr, "failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_none_revision_bpf__load(skel);
    if (err) {
        fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
        goto out;
    }

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_none_revision);
    err = libbpf_get_error(link);
    if (err) {
        link = NULL;
        fprintf(stderr, "failed to attach struct_ops: %s\n", strerror(-err));
        goto out;
    }

    printf("{\"event\":\"ready\",\"pid\":%ld,"
           "\"struct_link_id\":%u,\"struct_map_id\":%u,"
           "\"page_prefetch_program_id\":%u}\n",
           (long)getpid(), link_id(link),
           map_id(skel->maps.uvm_ops_none_revision),
           prog_id(skel->progs.gpu_page_prefetch));

    while (!exiting)
        sleep(1);

    printf("{\"event\":\"detaching\",\"pid\":%ld}\n", (long)getpid());

out:
    bpf_link__destroy(link);
    prefetch_none_revision_bpf__destroy(skel);
    return err < 0 ? -err : err;
}
