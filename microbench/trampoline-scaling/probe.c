#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix.h"

static volatile sig_atomic_t stopping;

static void handle_signal(int signal_number)
{
    (void)signal_number;
    stopping = 1;
}

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return -1.0;
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int emit_segments(const char *map_name, int fd, uint32_t key,
                         uint64_t *values, uint32_t thread_count)
{
    if (bpf_map_lookup_elem(fd, &key, values) != 0) {
        fprintf(stderr, "map lookup failed for %s key %u: %s\n",
                map_name, key, strerror(errno));
        return -1;
    }

    uint32_t begin = 0;
    while (begin < thread_count) {
        uint32_t end = begin + 1;
        while (end < thread_count && values[end] == values[begin])
            ++end;
        printf("{\"event\":\"counter_segment\",\"map\":\"%s\","
               "\"key\":%u,\"begin\":%u,\"end\":%u,\"value\":%" PRIu64 "}\n",
               map_name, key, begin, end, values[begin]);
        begin = end;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 5 ||
        (strcmp(argv[2], "noop") != 0 && strcmp(argv[2], "counter") != 0)) {
        fprintf(stderr, "usage: %s OBJECT {noop|counter} GPU_THREADS TIMEOUT_SECONDS\n",
                argv[0]);
        return 2;
    }

    char *end = NULL;
    errno = 0;
    unsigned long parsed_threads = strtoul(argv[3], &end, 10);
    if (errno || !end || *end || parsed_threads != SCALE_MAX_THREADS) {
        fprintf(stderr, "GPU_THREADS must equal %u\n", SCALE_MAX_THREADS);
        return 2;
    }
    end = NULL;
    errno = 0;
    unsigned long parsed_timeout = strtoul(argv[4], &end, 10);
    if (errno || !end || *end || parsed_timeout < 1 || parsed_timeout > 3600) {
        fprintf(stderr, "TIMEOUT_SECONDS must be in [1, 3600]\n");
        return 2;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    struct bpf_object *object = bpf_object__open_file(argv[1], NULL);
    if (!object || libbpf_get_error(object)) {
        fprintf(stderr, "failed to open BPF object %s\n", argv[1]);
        return 3;
    }

    int result = 4;
    struct bpf_link **links = NULL;
    size_t program_count = 0;
    struct bpf_program *program;
    bpf_object__for_each_program(program, object)
        ++program_count;
    struct bpf_program *target_program =
        bpf_object__find_program_by_name(object, "cuda__scale_target");
    struct bpf_program *marker_program =
        bpf_object__find_program_by_name(object, "cuda__scale_marker");
    if (program_count != 2 || !target_program || !marker_program ||
        strcmp(bpf_program__section_name(target_program),
               "kprobe/trampoline_scale_kernel") != 0 ||
        strcmp(bpf_program__section_name(marker_program),
               "kprobe/trampoline_marker_kernel") != 0) {
        fprintf(stderr, "expected two BPF programs, found %zu\n", program_count);
        goto done;
    }
    /* The entry pass resolves the module-wide explicit stub for the first link. */
    struct bpf_program *attach_order[] = {target_program, marker_program};
    links = calloc(program_count, sizeof(*links));
    if (!links)
        goto done;
    if (bpf_object__load(object) != 0) {
        fprintf(stderr, "failed to load BPF object\n");
        goto done;
    }

    size_t link_count = 0;
    for (size_t i = 0; i < program_count; ++i) {
        program = attach_order[i];
        struct bpf_link *link = bpf_program__attach(program);
        if (!link || libbpf_get_error(link)) {
            fprintf(stderr, "failed to attach BPF program %s\n",
                    bpf_program__name(program));
            goto done;
        }
        links[link_count++] = link;
    }

    struct bpf_map *marker = bpf_object__find_map_by_name(object, "marker_count");
    struct bpf_map *target = bpf_object__find_map_by_name(object, "target_count");
    const int counter_mode = strcmp(argv[2], "counter") == 0;
    if (!marker || (!!target != counter_mode)) {
        fprintf(stderr, "BPF map set does not match mode %s\n", argv[2]);
        goto done;
    }

    printf("{\"event\":\"ready\",\"mode\":\"%s\",\"programs\":%zu,"
           "\"gpu_threads\":%u,\"target_map\":%s,"
           "\"attach_order\":[\"%s\",\"%s\"]}\n",
           argv[2], program_count, SCALE_MAX_THREADS,
           target ? "true" : "false",
           bpf_program__name(attach_order[0]),
           bpf_program__name(attach_order[1]));

    const double deadline = monotonic_seconds() + (double)parsed_timeout;
    struct timespec pause_time = {.tv_sec = 0, .tv_nsec = 100000000};
    while (!stopping && monotonic_seconds() < deadline)
        nanosleep(&pause_time, NULL);
    if (!stopping) {
        fprintf(stderr, "loader timed out before runner requested readback\n");
        result = 5;
        goto done;
    }

    uint64_t *values = calloc(SCALE_MAX_THREADS, sizeof(*values));
    if (!values) {
        result = 6;
        goto done;
    }
    if (emit_segments("marker_count", bpf_map__fd(marker), 0, values,
                      SCALE_MAX_THREADS) != 0) {
        free(values);
        result = 7;
        goto done;
    }
    if (target) {
        for (uint32_t key = 0; key < SCALE_COUNTER_KEYS; ++key) {
            if (emit_segments("target_count", bpf_map__fd(target), key,
                              values, SCALE_MAX_THREADS) != 0) {
                free(values);
                result = 7;
                goto done;
            }
        }
    }
    free(values);
    result = 0;

done:
    if (links) {
        for (size_t i = program_count; i > 0; --i) {
            if (links[i - 1])
                bpf_link__destroy(links[i - 1]);
        }
    }
    if (result == 0)
        printf("{\"event\":\"detached\",\"links\":%zu}\n", program_count);
    free(links);
    bpf_object__close(object);
    return result;
}
