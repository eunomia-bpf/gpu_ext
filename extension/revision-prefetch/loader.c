/* SPDX-License-Identifier: GPL-2.0 */
/* Ownership pattern from ../prefetch_none_revision.c; no pinning/detach-all. */
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "fixture.h"
#include "fixture.skel.h"

static volatile sig_atomic_t exiting;
static void stopped(int sig) { (void)sig; exiting = 1; }
static int logger(enum libbpf_print_level level, const char *fmt, va_list args)
{
    return level == LIBBPF_DEBUG ? 0 : vfprintf(stderr, fmt, args);
}
static unsigned int map_id(struct bpf_map *map)
{
    struct bpf_map_info info = {};
    unsigned int size = sizeof(info);
    return bpf_obj_get_info_by_fd(bpf_map__fd(map), &info, &size) ? 0 : info.id;
}
static unsigned int link_id(struct bpf_link *link)
{
    struct bpf_link_info info = {};
    unsigned int size = sizeof(info);
    return !link || bpf_obj_get_info_by_fd(bpf_link__fd(link), &info, &size) ? 0 : info.id;
}

static int final_metrics(struct fixture_bpf *skel, unsigned int action)
{
    int cpus = libbpf_num_possible_cpus(), ok = 1, first = 1;
    unsigned int key = 0;
    unsigned long long frame_key;
    struct prefetch_metrics total = {}, *per_cpu;
    struct bpf_program *program;
    if (cpus <= 0)
        return 1;
    per_cpu = calloc((size_t)cpus, sizeof(*per_cpu));
    if (!per_cpu || bpf_map_lookup_elem(bpf_map__fd(skel->maps.metrics), &key, per_cpu)) {
        free(per_cpu);
        return 1;
    }
    for (int cpu = 0; cpu < cpus; cpu++) {
#define SUM(name) total.name += per_cpu[cpu].name;
        PREFETCH_COUNTERS(SUM)
#undef SUM
    }
    errno = 0;
    int empty_frames = bpf_map_get_next_key(bpf_map__fd(skel->maps.frames), NULL,
                                           &frame_key) < 0 && errno == ENOENT;
    ok = empty_frames && total.mask_enter > 0 && total.wrapper_enter > 0 &&
         total.mask_enter == total.mask_exit &&
         total.wrapper_enter == total.wrapper_exit &&
         total.wrapper_exit == total.decisions_complete &&
         total.wrapper_exit == total.returned_default + total.returned_bypass + total.returned_invalid99 &&
         total.mask_exit == total.empty_masks + total.nonempty_masks;
    ok &= (action == 0 ? total.returned_default : action == 1 ?
           total.returned_bypass : total.returned_invalid99) == total.wrapper_exit;
    if (action == 0)
        ok &= total.policy_calls == 0 && total.setter_ok == 0;
    else
        ok &= total.policy_calls == total.wrapper_exit &&
              total.setter_ok == total.policy_calls;
    if (action == 1)
        ok &= total.bypass_decisions == total.wrapper_exit &&
              total.native_decisions == 0 && total.range_calls == 0 &&
              total.nonempty_masks == 0;
    else
        ok &= total.native_decisions == total.wrapper_exit &&
              total.bypass_decisions == 0 && total.range_calls >= total.wrapper_exit;
    ok &= !(total.map_errors || total.nesting_errors || total.missing_frame ||
            total.identity_errors || total.order_errors || total.read_errors ||
            total.request_errors || total.action_errors || total.traversal_errors ||
            total.iterator_calls || total.mask_bounds_errors);
    printf("{\"event\":\"final_metrics\",\"action\":%u,\"empty_frames\":%s,",
           action, empty_frames ? "true" : "false");
#define PRINT(name) printf("\"" #name "\":%llu,", total.name);
    PREFETCH_COUNTERS(PRINT)
#undef PRINT
    printf("\"programs\":[");
    bpf_object__for_each_program(program, skel->obj) {
        struct bpf_prog_info info = {};
        unsigned int size = sizeof(info);
        if (bpf_obj_get_info_by_fd(bpf_program__fd(program), &info, &size)) {
            ok = 0;
            continue;
        }
        if (info.recursion_misses)
            ok = 0;
        printf("%s{\"name\":\"%s\",\"id\":%u,\"run_count\":%llu,"
               "\"recursion_misses\":%llu}", first ? "" : ",",
               bpf_program__name(program), info.id,
               (unsigned long long)info.run_cnt,
               (unsigned long long)info.recursion_misses);
        first = 0;
    }
    printf("],\"mask_samples\":[");
    first = 1;
    for (int cpu = 0; cpu < cpus; cpu++) {
        struct prefetch_metrics *m = &per_cpu[cpu];
        if (!m->mask_exit)
            continue;
        printf("%s{\"cpu\":%d,\"first\":%llu,\"outer\":%llu,\"bitmap\":[",
               first ? "" : ",", cpu, m->sample_first, m->sample_outer);
        for (int i = 0; i < 8; i++)
            printf("%s%llu", i ? "," : "", m->sample_bitmap[i]);
        printf("]}");
        first = 0;
    }
    printf("],\"valid\":%s}\n", ok ? "true" : "false");
    free(per_cpu);
    return !ok;
}

int main(int argc, char **argv)
{
    struct fixture_bpf *skel = NULL;
    struct bpf_program *program;
    struct bpf_link *links[PREFETCH_OBSERVER_COUNT] = {}, *policy = NULL;
    unsigned int action;
    int count = 0, stats_fd = -1, result = 1, ready = 0;
    if (argc != 2 || (strcmp(argv[1], "native") && strcmp(argv[1], "bypass") &&
                      strcmp(argv[1], "invalid99"))) {
        fprintf(stderr, "usage: %s native|bypass|invalid99\n", argv[0]);
        return 2;
    }
    action = !strcmp(argv[1], "native") ? 0 : !strcmp(argv[1], "bypass") ? 1 : 99;
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, stopped);
    signal(SIGTERM, stopped);
    libbpf_set_print(logger);
    stats_fd = bpf_enable_stats(BPF_STATS_RUN_TIME);
    if (stats_fd < 0)
        goto out;
    skel = fixture_bpf__open();
    if (!skel)
        goto out;
    skel->rodata->action = action;
    if (fixture_bpf__load(skel))
        goto out;
    bpf_object__for_each_program(program, skel->obj) {
        if (bpf_program__type(program) == BPF_PROG_TYPE_STRUCT_OPS)
            continue;
        if (count >= PREFETCH_OBSERVER_COUNT)
            goto out;
        links[count] = bpf_program__attach(program);
        if (libbpf_get_error(links[count])) {
            links[count] = NULL;
            goto out;
        }
        count++;
    }
    if (count != PREFETCH_OBSERVER_COUNT)
        goto out;
    if (action) {
        policy = bpf_map__attach_struct_ops(skel->maps.invalid_prefetch_ops);
        if (libbpf_get_error(policy)) {
            policy = NULL;
            goto out;
        }
    }
    if (!map_id(skel->maps.invalid_prefetch_ops) || (action && !link_id(policy)))
        goto out;
    for (int i = 0; i < count; i++)
        if (!link_id(links[i]))
            goto out;
    printf("{\"event\":\"ready\",\"pid\":%ld,\"mode\":\"%s\",\"action\":%u,"
           "\"struct_map_id\":%u,\"struct_link_id\":%u,\"observer_link_ids\":[",
           (long)getpid(), argv[1], action,
           map_id(skel->maps.invalid_prefetch_ops), link_id(policy));
    for (int i = 0; i < count; i++)
        printf("%s%u", i ? "," : "", link_id(links[i]));
    printf("]}\n");
    ready = 1;
    while (!exiting)
        sleep(1);
out:
    /* The coordinator must stop/reap its target before signaling this loader. */
    bpf_link__destroy(policy);
    for (int i = count - 1; i >= 0; i--)
        bpf_link__destroy(links[i]);
    if (ready)
        result = final_metrics(skel, action);
    else
        fprintf(stderr, "prefetch fixture did not reach complete attach readiness\n");
    fixture_bpf__destroy(skel);
    if (stats_fd >= 0)
        close(stats_fd);
    return result;
}
