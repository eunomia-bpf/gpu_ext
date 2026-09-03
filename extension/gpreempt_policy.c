/* SPDX-License-Identifier: GPL-2.0 */
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "gpreempt_bridge.h"
#include "gpreempt_policy.skel.h"

static volatile sig_atomic_t exiting;
static void stop(int signal_number) { (void)signal_number; exiting = 1; }
static const char *names[GP_STAT_COUNT] = {
    "scope_enter", "scope_leave", "gr_init", "other_engine", "unknown_engine",
    "timeslice_ok", "setter_error", "alloc_captured", "alloc_error", "registered",
    "register_error", "bind_shadow_match", "bind_shadow_mismatch", "destroy",
    "map_error", "scope_error",
};

static int print_stats(struct gpreempt_policy_bpf *skeleton)
{
    unsigned long long failures = 0;
    printf("gpreempt_policy_stats:");
    for (__u32 key = 0; key < GP_STAT_COUNT; ++key) {
        __u64 value = 0;
        if (bpf_map_lookup_elem(bpf_map__fd(skeleton->maps.stats), &key, &value)) return -1;
        printf(" %s=%llu", names[key], (unsigned long long)value);
        if (key == GP_UNKNOWN_ENGINE || key == GP_SETTER_ERROR || key == GP_ALLOC_ERROR ||
            key == GP_REGISTER_ERROR || key == GP_BIND_MISMATCH || key == GP_MAP_ERROR || key == GP_SCOPE_ERROR)
            failures += value;
    }
    putchar('\n');
    fflush(stdout);
    return failures ? -1 : 0;
}

static int number(const char *argument, long minimum, long maximum, long *value)
{
    char *end;
    errno = 0;
    *value = strtol(argument, &end, 10);
    return !errno && end != argument && *end == '\0' && *value >= minimum && *value <= maximum ? 0 : -1;
}

int main(int argc, char **argv)
{
    const char *library = NULL, *directory = NULL;
    long duration = 300, pid = -1;
    char scope_path[PATH_MAX], record_path[PATH_MAX];
    struct gpreempt_policy_bpf *skeleton = NULL;
    struct bpf_link *links[7] = {};
    bool directory_owned = false, scopes_pinned = false, records_pinned = false;
    int result = 1, option;
    static const struct option options[] = {
        {"library", required_argument, NULL, 'l'}, {"pin-dir", required_argument, NULL, 'm'},
        {"duration", required_argument, NULL, 'd'}, {"pid", required_argument, NULL, 'p'},
        {"help", no_argument, NULL, 'h'}, {NULL, 0, NULL, 0},
    };
    while ((option = getopt_long(argc, argv, "l:m:d:p:h", options, NULL)) != -1) {
        switch (option) {
        case 'l': library = optarg; break;
        case 'm': directory = optarg; break;
        case 'd': if (number(optarg, 1, 3600, &duration)) goto usage; break;
        case 'p': if (number(optarg, 1, INT_MAX, &pid)) goto usage; break;
        case 'h': result = 0; goto usage;
        default: goto usage;
        }
    }
    if (!library || library[0] != '/' || !directory ||
        strncmp(directory, "/sys/fs/bpf/", 12) || !directory[12] ||
        strstr(directory, "..") || optind != argc) goto usage;
    if (snprintf(scope_path, sizeof(scope_path), "%s/scopes", directory) >= (int)sizeof(scope_path) ||
        snprintf(record_path, sizeof(record_path), "%s/records", directory) >= (int)sizeof(record_path)) goto usage;
    signal(SIGTERM, stop);
    signal(SIGINT, stop);
    if (mkdir(directory, 0700)) { perror("create fresh BPF pin directory"); goto cleanup; }
    directory_owned = true;
    skeleton = gpreempt_policy_bpf__open_and_load();
    if (!skeleton) { fprintf(stderr, "cannot load GPReempt BPF policy\n"); goto cleanup; }
    if (bpf_map__pin(skeleton->maps.scopes, scope_path)) goto cleanup;
    scopes_pinned = true;
    if (bpf_map__pin(skeleton->maps.records, record_path)) goto cleanup;
    records_pinned = true;
    links[0] = bpf_program__attach_kprobe(skeleton->progs.ioctl_enter, false, "nvidia_unlocked_ioctl");
    links[1] = bpf_program__attach_kprobe(skeleton->progs.ioctl_exit, true, "nvidia_unlocked_ioctl");
    links[2] = bpf_program__attach_tracepoint(skeleton->progs.thread_exit, "sched", "sched_process_exit");
    struct bpf_program *markers[] = {skeleton->progs.scope_enter, skeleton->progs.register_context,
                                     skeleton->progs.scope_leave};
    const char *symbols[] = {"gpreempt_bpf_scope_enter", "gpreempt_bpf_register", "gpreempt_bpf_scope_leave"};
    for (int i = 0; i < 3; ++i) {
        LIBBPF_OPTS(bpf_uprobe_opts, options, .func_name = symbols[i]);
        links[i + 3] = bpf_program__attach_uprobe_opts(markers[i], pid, library, 0, &options);
    }
    links[6] = bpf_map__attach_struct_ops(skeleton->maps.gpreempt_ops);
    for (unsigned int i = 0; i < 7; ++i) {
        long error = libbpf_get_error(links[i]);
        if (!links[i] || error) {
            fprintf(stderr, "GPReempt attachment %u failed: %ld\n", i, error);
            goto cleanup;
        }
    }
    printf("gpreempt_policy_ready: pid=%ld pin_dir=%s scope=single_gpu "
           "lc_timeslice_us=1000000 be_timeslice_us=1 engine_filter=rm_gr_1_to_8\n", pid, directory);
    fflush(stdout);
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    do {
        sleep(1);
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while (!exiting && now.tv_sec - start.tv_sec < duration);
    result = print_stats(skeleton) ? 1 : 0;
cleanup:
    /* Detach policy first, then markers/captures. Only our freshly-created pins
     * are unlinked. No pre-existing pin, program, or module is touched. */
    for (int i = 6; i >= 0; --i)
        if (links[i] && !libbpf_get_error(links[i])) bpf_link__destroy(links[i]);
    if (records_pinned && unlink(record_path)) { perror("unlink own records pin"); result = 1; }
    if (scopes_pinned && unlink(scope_path)) { perror("unlink own scopes pin"); result = 1; }
    gpreempt_policy_bpf__destroy(skeleton);
    if (directory_owned && rmdir(directory)) { perror("remove own pin directory"); result = 1; }
    return result;
usage:
    fprintf(stderr, "Usage: %s --library ABS_LIB --pin-dir /sys/fs/bpf/NEW_DIR "
                    "[--duration 1..3600] [--pid PID]\n", argv[0]);
    return result;
}
