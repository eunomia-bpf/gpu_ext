/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Migration-debt eviction policy loader (LMCache/gpubpf prototype).
 *
 * Exposes the warm-phase disk-durable flag through the existing BPF
 * map/control path (debt_config, key DEBT_CONFIG_DISK_DURABLE): set it
 * to 1 when the LMCache warm phase has durably written the KV pool to
 * local NVMe.  Chunks activated while the flag is set are tracked as
 * disk-durable low-reuse candidates.  Turning the flag ON with the
 * 'w' key is retroactive: chunks activated before the warm command are
 * already tracked, so the loader walks the chunk_debt keys, reads each
 * debt_chunk_state, sets disk_durable = 1 on chunks not yet marked,
 * writes the state back, and reports how many chunks were marked.
 * The loader does not know the exact LMCache chunk -> UVM chunk/page
 * identity; that limitation is documented in eviction_debt.bpf.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <poll.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "eviction_debt.skel.h"
#include "cleanup_struct_ops.h"
#include "eviction_common.h"

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
#include "eviction_debt_model.h"

#define PRESSURE_THRESHOLD_DEFAULT 32

static int g_warm;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig)
{
    exiting = true;
}

/*
 * Set the coarse warm-phase disk-durable flag through the debt_config
 * map.  Setting the flag only affects future activations inside the BPF
 * policy; chunks activated before the warm command were recorded with
 * disk_durable == 0.  Marking is therefore retroactive when the flag
 * goes ON: walk the chunk_debt keys (u64 chunk pointers), read each
 * debt_chunk_state, set disk_durable = 1 for chunks not yet marked,
 * update the entry, and report how many chunks were marked.
 */
static void set_warm_flag(struct eviction_debt_bpf *skel, int on)
{
    u32 key = DEBT_CONFIG_DISK_DURABLE;
    u64 val = on ? 1 : 0;
    int fd = bpf_map__fd(skel->maps.debt_config);
    int chunks_fd = bpf_map__fd(skel->maps.chunk_debt);
    u64 prev_chunk = 0, next_chunk = 0, marked = 0;
    struct debt_chunk_state state;

    if (bpf_map_update_elem(fd, &key, &val, BPF_ANY)) {
        fprintf(stderr, "Failed to set warm-phase disk-durable flag: %s\n",
                strerror(errno));
        return;
    }
    g_warm = on;
    printf("warm-phase disk-durable flag: %s\n", on ? "ON" : "off");

    if (!on)
        return;

    while (bpf_map_get_next_key(chunks_fd, &prev_chunk, &next_chunk) == 0) {
        if (bpf_map_lookup_elem(chunks_fd, &next_chunk, &state) == 0 &&
            state.disk_durable == 0) {
            state.disk_durable = 1;
            if (bpf_map_update_elem(chunks_fd, &next_chunk, &state,
                                    BPF_EXIST) == 0)
                marked++;
        }
        prev_chunk = next_chunk;
    }
    printf("  retroactively marked %llu tracked chunks disk-durable\n",
           (unsigned long long)marked);
}

static void print_stats(struct eviction_debt_bpf *skel)
{
    int pressure_fd = bpf_map__fd(skel->maps.debt_pressure);
    int chunks_fd = bpf_map__fd(skel->maps.chunk_debt);
    int pid_fd = bpf_map__fd(skel->maps.pid_chunk_count);
    u32 pkey = 0;
    u64 pressure = 0;
    u64 prev_chunk = 0, next_chunk = 0, tracked = 0;
    u32 next_pid = 0;
    u64 total_allow = 0, total_deny = 0;
    u64 total_activate = 0, total_used = 0;

    printf("\n=== Migration-Debt Statistics ===\n");

    if (bpf_map_lookup_elem(pressure_fd, &pkey, &pressure) == 0)
        printf("  Aggregate debt pressure: %llu\n",
               (unsigned long long)pressure);
    /* chunk_debt keys are u64 chunk pointers. */
    while (bpf_map_get_next_key(chunks_fd, &prev_chunk, &next_chunk) == 0) {
        tracked++;
        prev_chunk = next_chunk;
    }
    printf("  Tracked chunks: %llu\n", (unsigned long long)tracked);
    printf("  Warm-phase disk-durable flag: %s\n", g_warm ? "ON" : "off");

    printf("\n  Per-PID:\n");
    while (bpf_map_get_next_key(pid_fd, &next_pid, &next_pid) == 0) {
        struct pid_chunk_stats ps;

        if (bpf_map_lookup_elem(pid_fd, &next_pid, &ps) != 0)
            continue;
        total_activate += ps.total_activate;
        total_used += ps.total_used;
        total_allow += ps.policy_allow;
        total_deny += ps.policy_deny;
        printf("    PID %u: active=%llu used=%llu saved=%llu evicted=%llu\n",
               next_pid, ps.current_count, ps.total_used,
               ps.policy_allow, ps.policy_deny);
    }

    printf("\n  Totals: activated=%llu used=%llu saved=%llu evicted=%llu\n",
           (unsigned long long)total_activate, (unsigned long long)total_used,
           (unsigned long long)total_allow, (unsigned long long)total_deny);
}

static void usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -w 0|1    Warm-phase disk-durable flag at startup (default 0)\n");
    printf("  -m N      Debt cap: candidate hits before a chunk is low-reuse (default %d)\n",
           DEBT_DEFAULT_MAX);
    printf("  -p N      Aggregate debt pressure threshold that suppresses\n");
    printf("            speculative prefetch; 0 disables the gate (default %d)\n",
           PRESSURE_THRESHOLD_DEFAULT);
    printf("  -h        Show this help\n");
    printf("\nMigration-debt eviction policy:\n");
    printf("  - Eviction candidates accumulate an at-risk debt signal;\n");
    printf("    a later observed reuse clears it and saves the chunk.\n");
    printf("  - High aggregate debt suppresses speculative prefetch.\n");
    printf("  - Disk-durable low-reuse chunks are preferred eviction victims.\n");
    printf("\nKeys while running:\n");
    printf("  w  set warm-phase disk-durable flag ON  (after warm phase;\n");
    printf("     retroactively marks already-tracked chunks)\n");
    printf("  c  set warm-phase disk-durable flag off\n");
    printf("  q  quit\n");
}

int main(int argc, char **argv)
{
    struct eviction_debt_bpf *skel;
    struct bpf_link *link;
    int err;
    u64 warm = 0;
    u64 debt_max = DEBT_DEFAULT_MAX;
    u64 threshold = PRESSURE_THRESHOLD_DEFAULT;
    int opt;

    while ((opt = getopt(argc, argv, "w:m:p:h")) != -1) {
        switch (opt) {
            case 'w':
                warm = atoi(optarg) ? 1 : 0;
                break;
            case 'm':
                debt_max = atoll(optarg);
                break;
            case 'p':
                threshold = atoll(optarg);
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

    skel = eviction_debt_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = eviction_debt_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set configuration through the debt_config map (existing control path). */
    int config_fd = bpf_map__fd(skel->maps.debt_config);
    u32 key;
    u64 val;

    key = DEBT_CONFIG_DISK_DURABLE;
    val = warm;
    bpf_map_update_elem(config_fd, &key, &val, BPF_ANY);

    key = DEBT_CONFIG_DEBT_MAX;
    val = debt_max;
    bpf_map_update_elem(config_fd, &key, &val, BPF_ANY);

    key = DEBT_CONFIG_PRESSURE_THRESHOLD;
    val = threshold;
    bpf_map_update_elem(config_fd, &key, &val, BPF_ANY);

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_debt);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n",
                strerror(-err), err);
        goto cleanup;
    }

    g_warm = (int)warm;
    printf("Successfully loaded migration-debt eviction policy!\n");
    printf("\nConfiguration:\n");
    printf("  Warm-phase disk-durable flag: %s (note: coarse warm-phase\n",
           g_warm ? "ON" : "off");
    printf("  signal; exact LMCache chunk -> UVM page identity unavailable)\n");
    printf("  Debt cap: %llu\n", (unsigned long long)debt_max);
    printf("  Prefetch suppression pressure threshold: %llu\n",
           (unsigned long long)threshold);
    printf("\nPress Ctrl-C to exit...\n");

    while (!exiting) {
        int i;
        for (i = 0; i < 5 && !exiting; i++) {
            struct pollfd pfd = {
                .fd = STDIN_FILENO,
                .events = POLLIN,
            };

            sleep(1);
            if (poll(&pfd, 1, 0) > 0) {
                int c = getchar();
                if (c == 'q' || c == 'Q')
                    exiting = true;
                else if (c == 'w' || c == 'W')
                    set_warm_flag(skel, 1);
                else if (c == 'c' || c == 'C')
                    set_warm_flag(skel, 0);
            }
        }
        if (exiting)
            break;
        print_stats(skel);
    }

    printf("\nDetaching struct_ops...\n");
    print_stats(skel);
    bpf_link__destroy(link);

cleanup:
    eviction_debt_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
