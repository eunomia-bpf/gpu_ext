#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "eviction_freq_pid_decay.skel.h"
#include "cleanup_struct_ops.h"
#include "eviction_common.h"

static __u64 g_priority_pid = 0;
static __u64 g_low_priority_pid = 0;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

static void print_stats(struct eviction_freq_pid_decay_bpf *skel) {
    int pid_stats_fd = bpf_map__fd(skel->maps.pid_chunk_count);
    struct pid_chunk_stats ps;
    __u32 pid;
    __u64 total_current = 0;
    __u64 total_activate = 0;
    __u64 total_used = 0;
    __u64 total_allow = 0;
    __u64 total_deny = 0;

    printf("\n=== Per-PID Statistics ===\n");

    if (g_priority_pid > 0) {
        pid = (__u32)g_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 used_total = ps.policy_allow + ps.policy_deny;
            printf("  High priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    Policy allow (moved): %llu", ps.policy_allow);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_allow / used_total);
            printf("\n");
            printf("    Policy deny (not moved): %llu", ps.policy_deny);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_deny / used_total);
            printf("\n");

            total_current += ps.current_count;
            total_activate += ps.total_activate;
            total_used += ps.total_used;
            total_allow += ps.policy_allow;
            total_deny += ps.policy_deny;
        } else {
            printf("  High priority PID %u: no data\n", pid);
        }
    }

    if (g_low_priority_pid > 0) {
        pid = (__u32)g_low_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 used_total = ps.policy_allow + ps.policy_deny;
            printf("  Low priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    Policy allow (moved): %llu", ps.policy_allow);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_allow / used_total);
            printf("\n");
            printf("    Policy deny (not moved): %llu", ps.policy_deny);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_deny / used_total);
            printf("\n");

            total_current += ps.current_count;
            total_activate += ps.total_activate;
            total_used += ps.total_used;
            total_allow += ps.policy_allow;
            total_deny += ps.policy_deny;
        } else {
            printf("  Low priority PID %u: no data\n", pid);
        }
    }

    printf("\n=== Summary ===\n");
    printf("  Total current chunks: %llu\n", total_current);
    printf("  Total activated: %llu\n", total_activate);
    printf("  Total used calls: %llu\n", total_used);
    __u64 grand_total = total_allow + total_deny;
    printf("  Policy allow (moved): %llu", total_allow);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_allow / grand_total);
    printf("\n");
    printf("  Policy deny (not moved): %llu", total_deny);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_deny / grand_total);
    printf("\n");
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -p PID     Set high priority PID\n");
    printf("  -P N       Set high priority decay (move every N accesses, default 1)\n");
    printf("  -l PID     Set low priority PID\n");
    printf("  -L N       Set low priority decay (move every N accesses, default 10)\n");
    printf("  -d N       Set default decay for other PIDs (default 5)\n");
    printf("  -h         Show this help\n");
    printf("\nFrequency decay eviction policy:\n");
    printf("  - High priority: move_tail every N accesses (N=1 means always)\n");
    printf("  - Low priority: move_tail every N accesses (larger N = less protection)\n");
    printf("\nExample:\n");
    printf("  %s -p 1234 -P 1 -l 5678 -L 10\n", prog);
    printf("  High priority (PID 1234): move every access (always protected)\n");
    printf("  Low priority (PID 5678): move every 10 accesses (decayed protection)\n");
}

int main(int argc, char **argv) {
    struct eviction_freq_pid_decay_bpf *skel;
    struct bpf_link *link;
    int err;
    __u64 priority_pid = 0;
    __u64 priority_param = 1;      /* Default: every access */
    __u64 low_priority_pid = 0;
    __u64 low_priority_param = 10; /* Default: every 10 accesses */
    __u64 default_param = 5;       /* Default: every 5 accesses */
    int opt;

    while ((opt = getopt(argc, argv, "p:P:l:L:d:h")) != -1) {
        switch (opt) {
            case 'p':
                priority_pid = atoi(optarg);
                g_priority_pid = priority_pid;
                break;
            case 'P':
                priority_param = atoll(optarg);
                if (priority_param == 0) priority_param = 1;
                break;
            case 'l':
                low_priority_pid = atoi(optarg);
                g_low_priority_pid = low_priority_pid;
                break;
            case 'L':
                low_priority_param = atoll(optarg);
                if (low_priority_param == 0) low_priority_param = 1;
                break;
            case 'd':
                default_param = atoll(optarg);
                if (default_param == 0) default_param = 1;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);

    cleanup_old_struct_ops();

    skel = eviction_freq_pid_decay_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = eviction_freq_pid_decay_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set configuration */
    int config_fd = bpf_map__fd(skel->maps.config);
    __u32 key;

    key = CONFIG_PRIORITY_PID;
    bpf_map_update_elem(config_fd, &key, &priority_pid, BPF_ANY);

    key = CONFIG_PRIORITY_PARAM;
    bpf_map_update_elem(config_fd, &key, &priority_param, BPF_ANY);

    key = CONFIG_LOW_PRIORITY_PID;
    bpf_map_update_elem(config_fd, &key, &low_priority_pid, BPF_ANY);

    key = CONFIG_LOW_PRIORITY_PARAM;
    bpf_map_update_elem(config_fd, &key, &low_priority_param, BPF_ANY);

    key = CONFIG_DEFAULT_PARAM;
    bpf_map_update_elem(config_fd, &key, &default_param, BPF_ANY);

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_freq_pid_decay);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded frequency decay eviction policy!\n");
    printf("\nConfiguration (decay = move every N accesses):\n");
    printf("  High priority PID: %llu (decay: %llu)\n", priority_pid, priority_param);
    printf("  Low priority PID:  %llu (decay: %llu)\n", low_priority_pid, low_priority_param);
    printf("  Default decay:     %llu\n", default_param);
    printf("\nPress Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(5);
        print_stats(skel);
    }

    printf("\nDetaching struct_ops...\n");
    print_stats(skel);
    bpf_link__destroy(link);

cleanup:
    eviction_freq_pid_decay_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
