/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Migration-debt eviction policy loader (LMCache/gpubpf prototype).
 *
 * Tracks the single-KV-pool range by attaching a uprobe/uretprobe pair
 * (func_name uvm_kv_malloc) on the workload's allocator shared object
 * (-a path).  The BPF side saves the enter args in a HASH map and, on a
 * successful (non-NULL) return, records {start, end, tgid} in a single
 * ARRAY-map slot, keeping only the largest successful allocation.
 * gpu_block_activate marks a chunk is_kv only when its va_block start
 * lies inside the recorded range for the same tgid; the warm-phase
 * disk-durable flag (debt_config key DEBT_CONFIG_DISK_DURABLE) is
 * sampled at activation only for KV chunks, so UVM memory outside the
 * recorded range is never claimed durable.  Turning the flag ON with
 * the 'w' key is retroactive but KV-scoped: the loader walks the
 * chunk_debt keys, reads each debt_chunk_state, and sets disk_durable=1
 * only on entries already marked is_kv.
 *
 * Documented limitation: only the single largest successful
 * uvm_kv_malloc allocation is tracked.  A KV pool split across several
 * smaller allocations is only partially covered (its largest piece),
 * and a freed pool stays recorded until a larger allocation replaces
 * it.
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

static int attach_uprobe_symbol(struct bpf_program *prog, const char *path,
                                const char *symbol, bool retprobe,
                                struct bpf_link **link_out)
{
    LIBBPF_OPTS(bpf_uprobe_opts, opts,
        .func_name = symbol,
        .retprobe = retprobe,
    );
    struct bpf_link *link;
    int err;

    link = bpf_program__attach_uprobe_opts(prog, -1, path, 0, &opts);
    err = libbpf_get_error(link);
    if (err) {
        errno = -err;
        return err;
    }

    *link_out = link;
    return 0;
}

/*
 * Set the warm-phase disk-durable flag through the debt_config map.
 * The flag is sampled by the BPF policy only for chunks activated
 * inside the recorded single-KV-pool range (is_kv).  Marking is
 * retroactive but KV-scoped when the flag goes ON: walk the chunk_debt
 * keys (u64 chunk pointers), read each debt_chunk_state, set
 * disk_durable = 1 only on is_kv entries not yet marked, update the
 * entry, and report how many chunks were marked.  Tracked chunks
 * outside the KV range are never marked durable.
 */
static void set_warm_flag(struct eviction_debt_bpf *skel, int on)
{
    u32 key = DEBT_CONFIG_DISK_DURABLE;
    u64 val = on ? 1 : 0;
    int fd = bpf_map__fd(skel->maps.debt_config);
    int chunks_fd = bpf_map__fd(skel->maps.chunk_debt);
    u64 prev_chunk = 0, next_chunk = 0, marked = 0, scanned = 0;
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
        if (bpf_map_lookup_elem(chunks_fd, &next_chunk, &state) == 0) {
            scanned++;
            if (state.is_kv && state.disk_durable == 0) {
                state.disk_durable = 1;
                if (bpf_map_update_elem(chunks_fd, &next_chunk, &state,
                                        BPF_EXIST) == 0)
                    marked++;
            }
        }
        prev_chunk = next_chunk;
    }
    printf("  retroactively marked %llu of %llu tracked chunks disk-durable"
           " (KV-range entries only)\n",
           (unsigned long long)marked, (unsigned long long)scanned);
}

static void print_stats(struct eviction_debt_bpf *skel)
{
    int pressure_fd = bpf_map__fd(skel->maps.debt_pressure);
    int chunks_fd = bpf_map__fd(skel->maps.chunk_debt);
    int pid_fd = bpf_map__fd(skel->maps.pid_chunk_count);
    int kv_fd = bpf_map__fd(skel->maps.kv_pool_range);
    u32 pkey = 0;
    u64 pressure = 0;
    u64 prev_chunk = 0, next_chunk = 0, tracked = 0;
    u64 kv_tracked = 0, kv_durable = 0;
    u32 next_pid = 0;
    u64 total_allow = 0, total_deny = 0;
    u64 total_activate = 0, total_used = 0;
    u32 kv_key = 0;
    struct debt_kv_range kv = {0};
    struct debt_chunk_state state;

    printf("\n=== Migration-Debt Statistics ===\n");

    if (bpf_map_lookup_elem(pressure_fd, &pkey, &pressure) == 0)
        printf("  Aggregate debt pressure: %llu\n",
               (unsigned long long)pressure);
    /* chunk_debt keys are u64 chunk pointers. */
    while (bpf_map_get_next_key(chunks_fd, &prev_chunk, &next_chunk) == 0) {
        tracked++;
        if (bpf_map_lookup_elem(chunks_fd, &next_chunk, &state) == 0 &&
            state.is_kv) {
            kv_tracked++;
            if (state.disk_durable)
                kv_durable++;
        }
        prev_chunk = next_chunk;
    }
    printf("  Tracked chunks: %llu (KV-range: %llu, KV disk-durable: %llu)\n",
           (unsigned long long)tracked, (unsigned long long)kv_tracked,
           (unsigned long long)kv_durable);

    bpf_map_lookup_elem(kv_fd, &kv_key, &kv);
    if (kv.start)
        printf("  KV pool range: [0x%llx, 0x%llx) tgid %u (%llu MiB;"
               " largest uvm_kv_malloc)\n",
               (unsigned long long)kv.start, (unsigned long long)kv.end,
               kv.tgid, (unsigned long long)((kv.end - kv.start) >> 20));
    else
        printf("  KV pool range: none captured yet\n");
    printf("  Warm-phase disk-durable flag: %s (sampled for KV-range"
           " chunks only; single-largest-allocation tracking)\n",
           g_warm ? "ON" : "off");

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
    printf("  -a PATH   Required. Allocator shared object providing\n");
    printf("            uvm_kv_malloc\n");
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
    printf("  - KV-range chunks (largest uvm_kv_malloc allocation, same\n");
    printf("    tgid) are the only disk-durable / preferred victims.\n");
    printf("\nKeys while running:\n");
    printf("  w  set warm-phase disk-durable flag ON  (after warm phase;\n");
    printf("     retroactively marks tracked KV-range entries only)\n");
    printf("  c  set warm-phase disk-durable flag off\n");
    printf("  q  quit\n");
}

int main(int argc, char **argv)
{
    struct eviction_debt_bpf *skel;
    struct bpf_link *link = NULL;
    struct bpf_link *link_malloc_enter = NULL;
    struct bpf_link *link_malloc_ret = NULL;
    const char *allocator_path = NULL;
    int err;
    u64 warm = 0;
    u64 debt_max = DEBT_DEFAULT_MAX;
    u64 threshold = PRESSURE_THRESHOLD_DEFAULT;
    int opt;

    while ((opt = getopt(argc, argv, "a:w:m:p:h")) != -1) {
        switch (opt) {
            case 'a':
                allocator_path = optarg;
                break;
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

    if (!allocator_path) {
        fprintf(stderr, "Error: -a PATH is required\n\n");
        usage(argv[0]);
        return 1;
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

    err = attach_uprobe_symbol(skel->progs.uvm_kv_malloc_enter, allocator_path,
                               "uvm_kv_malloc", false, &link_malloc_enter);
    if (err) {
        fprintf(stderr, "Failed to attach uprobe on %s:uvm_kv_malloc: %s (%d)\n",
                allocator_path, strerror(errno), err);
        goto cleanup;
    }
    printf("uprobe attached: %s:uvm_kv_malloc\n", allocator_path);

    err = attach_uprobe_symbol(skel->progs.uvm_kv_malloc_ret, allocator_path,
                               "uvm_kv_malloc", true, &link_malloc_ret);
    if (err) {
        fprintf(stderr, "Failed to attach uretprobe on %s:uvm_kv_malloc: %s (%d)\n",
                allocator_path, strerror(errno), err);
        goto cleanup;
    }
    printf("uretprobe attached: %s:uvm_kv_malloc\n", allocator_path);

    g_warm = (int)warm;
    /* Ready marker: only after struct_ops and both uprobe attachments. */
    printf("Successfully loaded migration-debt eviction policy!\n");
    printf("\nConfiguration:\n");
    printf("  KV allocator: %s (uvm_kv_malloc uprobe/uretprobe)\n",
           allocator_path);
    printf("  Warm-phase disk-durable flag: %s (sampled only for chunks\n",
           g_warm ? "ON" : "off");
    printf("  inside the recorded KV pool range; single-largest-allocation\n");
    printf("  tracking, so a KV pool split across smaller allocations is\n");
    printf("  only partially covered)\n");
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

cleanup:
    if (link_malloc_ret)
        bpf_link__destroy(link_malloc_ret);
    if (link_malloc_enter)
        bpf_link__destroy(link_malloc_enter);
    if (link)
        bpf_link__destroy(link);
    eviction_debt_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
