#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_adaptive_sequential.skel.h"
#include "cleanup_struct_ops.h"
#include "nvml_monitor.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

#define PREFETCH_FORWARD       0
#define PREFETCH_BACKWARD      1
#define PREFETCH_FORWARD_START 2

static volatile bool exiting = false;
static nvmlDevice_t nvml_device = NULL;

/* Configuration */
static struct {
    int fixed_pct;           /* -1 means adaptive mode */
    unsigned int min_pct;    /* min percentage for adaptive mode */
    unsigned int max_pct;    /* max percentage for adaptive mode */
    unsigned long long max_mbps;  /* max PCIe throughput for scaling */
    int invert;              /* invert the adaptive logic */
    unsigned int direction;  /* 0=forward, 1=backward */
    unsigned int num_pages;  /* 0=use percentage, >0=fixed page count */
} config = {
    .fixed_pct = -1,
    .min_pct = 30,
    .max_pct = 100,
    .max_mbps = 20480ULL,    /* 20 GB/s */
    .invert = 0,
    .direction = PREFETCH_FORWARD,  /* Default: forward */
    .num_pages = 0,
};

void handle_signal(int sig) {
    exiting = true;
}

/* Get GPU PCIe throughput in MB/s using NVML */
static unsigned long long get_pcie_throughput_mbps(void) {
    if (!nvml_device) {
        return 0;
    }

    unsigned long long throughput_kbps = nvml_get_pcie_throughput_kbps(nvml_device);
    return throughput_kbps / 1024;
}

/* Calculate prefetch percentage based on PCIe throughput
 *
 * Normal mode (invert=0):
 *   High traffic -> high percentage -> more prefetch
 *   Low traffic  -> low percentage  -> less prefetch
 *
 * Inverted mode (invert=1):
 *   High traffic -> low percentage  -> less prefetch (bandwidth constrained)
 *   Low traffic  -> high percentage -> more prefetch (bandwidth available)
 */
static unsigned int calculate_prefetch_percentage(unsigned long long throughput_mbps) {
    if (throughput_mbps >= config.max_mbps) {
        return config.invert ? config.min_pct : config.max_pct;
    }

    double ratio = (double)throughput_mbps / (double)config.max_mbps; /* 0..1 */

    if (config.invert) {
        ratio = 1.0 - ratio;  /* invert: high traffic -> low ratio */
    }

    unsigned int pct = (unsigned int)(config.min_pct +
                                      (config.max_pct - config.min_pct) * ratio + 0.5);
    if (pct < config.min_pct) pct = config.min_pct;
    if (pct > config.max_pct) pct = config.max_pct;
    return pct;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nPrefetch Adaptive Sequential Policy with Direction Support\n");
    printf("Controls prefetch percentage, direction, and page count.\n");
    printf("\nOptions:\n");
    printf("  -p PCT        Set fixed prefetch percentage (0-100), disables adaptive mode\n");
    printf("  -m MIN        Set minimum percentage for adaptive mode (default: %u)\n", config.min_pct);
    printf("  -M MAX        Set maximum percentage for adaptive mode (default: %u)\n", config.max_pct);
    printf("  -b MBPS       Set max PCIe bandwidth for scaling in MB/s (default: %llu)\n", config.max_mbps);
    printf("  -i            Invert adaptive logic (high traffic -> less prefetch)\n");
    printf("  -d DIR        Prefetch direction: 'forward' (default), 'backward', or 'forward_start'\n");
    printf("  -n NUM        Number of pages to prefetch (0=use percentage, default: 0)\n");
    printf("  -h            Show this help\n");
    printf("\nDirection modes:\n");
    printf("  forward:       Prefetch pages AFTER the faulting page (higher addresses) (default)\n");
    printf("                 For sequential access patterns (low -> high)\n");
    printf("  backward:      Prefetch pages BEFORE the faulting page (lower addresses)\n");
    printf("                 For reverse access patterns (high -> low)\n");
    printf("  forward_start: Prefetch from region start [max_first, max_first+n)\n");
    printf("                 Original sequential prefetch behavior\n");
    printf("\nExamples:\n");
    printf("  %s -p 100              # Fixed 100%% prefetch (like always_max)\n", prog);
    printf("  %s -p 0                # Fixed 0%% prefetch (like none)\n", prog);
    printf("  %s -p 50               # Fixed 50%% prefetch\n", prog);
    printf("  %s                     # Adaptive mode (default)\n", prog);
    printf("  %s -m 20 -M 80         # Adaptive with custom range 20-80%%\n", prog);
    printf("  %s -i                  # Inverted: less prefetch when busy\n", prog);
    printf("  %s -d backward         # Backward prefetch direction\n", prog);
    printf("  %s -d forward -n 32    # Forward from fault, fixed 32 pages\n", prog);
    printf("  %s -d forward_start    # Forward from region start (original)\n", prog);
    printf("  %s -d backward -p 50   # Backward, 50%% prefetch\n", prog);
    printf("\nWithout -p, uses adaptive mode based on PCIe throughput.\n");
    printf("If -n is set to >0, it overrides the percentage-based calculation.\n");
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_sequential_bpf *skel;
    struct bpf_link *link;
    int err;
    int pct_map_fd;
    unsigned int key = 0;
    int opt;

    while ((opt = getopt(argc, argv, "p:m:M:b:id:n:h")) != -1) {
        switch (opt) {
        case 'p':
            config.fixed_pct = atoi(optarg);
            if (config.fixed_pct < 0 || config.fixed_pct > 100) {
                fprintf(stderr, "Error: percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'm':
            config.min_pct = (unsigned int)atoi(optarg);
            if (config.min_pct > 100) {
                fprintf(stderr, "Error: min percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'M':
            config.max_pct = (unsigned int)atoi(optarg);
            if (config.max_pct > 100) {
                fprintf(stderr, "Error: max percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'b':
            config.max_mbps = (unsigned long long)atoll(optarg);
            break;
        case 'i':
            config.invert = 1;
            break;
        case 'd':
            if (strcmp(optarg, "forward") == 0 || strcmp(optarg, "f") == 0) {
                config.direction = PREFETCH_FORWARD;
            } else if (strcmp(optarg, "backward") == 0 || strcmp(optarg, "b") == 0) {
                config.direction = PREFETCH_BACKWARD;
            } else if (strcmp(optarg, "forward_start") == 0 || strcmp(optarg, "fs") == 0) {
                config.direction = PREFETCH_FORWARD_START;
            } else {
                fprintf(stderr, "Invalid direction: %s\n", optarg);
                print_usage(argv[0]);
                return 1;
            }
            break;
        case 'n':
            config.num_pages = (unsigned int)atoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Validate min/max */
    if (config.min_pct > config.max_pct) {
        fprintf(stderr, "Error: min percentage (%u) cannot be greater than max (%u)\n",
                config.min_pct, config.max_pct);
        return 1;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Initialize NVML for adaptive mode */
    if (config.fixed_pct < 0) {
        nvml_device = nvml_init_device();
        if (!nvml_device) {
            fprintf(stderr, "Warning: Failed to initialize NVML, using fixed 100%% mode\n");
            config.fixed_pct = 100;
        }
    }

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = prefetch_adaptive_sequential_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = prefetch_adaptive_sequential_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Get prefetch percentage map FD */
    pct_map_fd = bpf_map__fd(skel->maps.prefetch_pct_map);
    if (pct_map_fd < 0) {
        fprintf(stderr, "Failed to get prefetch_pct_map FD\n");
        err = pct_map_fd;
        goto cleanup;
    }

    /* Set initial percentage */
    unsigned int initial_pct = (config.fixed_pct >= 0) ? (unsigned int)config.fixed_pct : config.max_pct;
    err = bpf_map_update_elem(pct_map_fd, &key, &initial_pct, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set initial percentage: %d\n", err);
        goto cleanup;
    }

    /* Get and set direction map */
    int dir_map_fd = bpf_map__fd(skel->maps.prefetch_direction_map);
    if (dir_map_fd < 0) {
        fprintf(stderr, "Failed to get prefetch_direction_map FD\n");
        err = dir_map_fd;
        goto cleanup;
    }
    err = bpf_map_update_elem(dir_map_fd, &key, &config.direction, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set direction: %d\n", err);
        goto cleanup;
    }

    /* Get and set num_pages map */
    int num_map_fd = bpf_map__fd(skel->maps.prefetch_num_pages_map);
    if (num_map_fd < 0) {
        fprintf(stderr, "Failed to get prefetch_num_pages_map FD\n");
        err = num_map_fd;
        goto cleanup;
    }
    err = bpf_map_update_elem(num_map_fd, &key, &config.num_pages, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set num_pages: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_adaptive_sequential);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF adaptive_sequential policy!\n");

    /* Print direction */
    const char *dir_str;
    switch (config.direction) {
    case PREFETCH_FORWARD:
        dir_str = "FORWARD (prefetch after fault page)";
        break;
    case PREFETCH_BACKWARD:
        dir_str = "BACKWARD (prefetch before fault page)";
        break;
    case PREFETCH_FORWARD_START:
    default:
        dir_str = "FORWARD_START (prefetch from region start)";
        break;
    }
    printf("Direction: %s\n", dir_str);

    /* Print num_pages */
    if (config.num_pages > 0) {
        printf("Num pages: %u (overrides percentage)\n", config.num_pages);
    } else {
        printf("Num pages: 0 (use percentage)\n");
    }

    if (config.fixed_pct >= 0) {
        printf("Mode: Fixed prefetch percentage = %d%%\n", config.fixed_pct);
    } else {
        printf("Mode: Adaptive (based on PCIe throughput)\n");
        printf("  Range: %u%% - %u%%\n", config.min_pct, config.max_pct);
        printf("  Max bandwidth: %llu MB/s\n", config.max_mbps);
        printf("  Invert: %s\n", config.invert ? "yes" : "no");
        printf("Monitoring PCIe traffic and updating percentage every second...\n");
    }
    printf("Monitor tracepipe for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n\n");

    /* Main loop */
    while (!exiting) {
        if (config.fixed_pct < 0) {
            /* Adaptive mode: update percentage based on PCIe throughput */
            unsigned long long throughput = get_pcie_throughput_mbps();
            unsigned int pct = calculate_prefetch_percentage(throughput);

            err = bpf_map_update_elem(pct_map_fd, &key, &pct, BPF_ANY);
            if (err) {
                fprintf(stderr, "Failed to update prefetch percentage: %d\n", err);
            } else {
                printf("[%ld] PCIe: %llu MB/s -> Prefetch: %u%%\n",
                       time(NULL), throughput, pct);
            }
        }
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_adaptive_sequential_bpf__destroy(skel);

    if (nvml_device) {
        nvml_cleanup();
    }

    return err < 0 ? -err : 0;
}
