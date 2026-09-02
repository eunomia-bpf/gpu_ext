#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 2)
        return 2;
    setvbuf(stdout, NULL, _IONBF, 0);
    struct bpf_object *object = bpf_object__open_file(argv[1], NULL);
    if (!object || libbpf_get_error(object))
        return 3;
    int result = 4;
    struct bpf_link *link = NULL;
    if (bpf_object__load(object))
        goto done;
    struct bpf_program *program = bpf_object__find_program_by_name(object, "cuda__count_return");
    struct bpf_map *map = bpf_object__find_map_by_name(object, "call_count");
    if (!program || !map)
        goto done;
    link = bpf_program__attach(program);
    if (!link || libbpf_get_error(link)) {
        link = NULL;
        goto done;
    }
    puts("{\"event\":\"ready\",\"expected_threads\":4096,\"expected_launches\":8}");
    uint64_t counts[4096] = {0};
    uint32_t key = 0;
    double deadline = seconds() + 90;
    double next_report = 0;
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 100000000};
    while (seconds() < deadline) {
        if (bpf_map_lookup_elem(bpf_map__fd(map), &key, counts))
            goto done;
        uint64_t sum = 0, correct = 0, nonzero = 0, maximum = 0;
        for (unsigned i = 0; i < 4096; ++i) {
            sum += counts[i];
            correct += counts[i] == 8;
            nonzero += counts[i] != 0;
            if (counts[i] > maximum)
                maximum = counts[i];
        }
        if (seconds() >= next_report || maximum > 8) {
            printf("{\"event\":\"counter_snapshot\",\"device_thread_returns\":%llu,\"nonzero_threads\":%llu,\"threads_with_eight_returns\":%llu,\"maximum_returns\":%llu}\n",
                   (unsigned long long)sum, (unsigned long long)nonzero,
                   (unsigned long long)correct, (unsigned long long)maximum);
            next_report = seconds() + 1;
        }
        if (maximum > 8)
            goto done;
        if (correct == 4096) {
            printf("{\"event\":\"engagement\",\"device_thread_returns\":%llu,\"threads_with_eight_returns\":%llu}\n",
                   (unsigned long long)sum, (unsigned long long)correct);
            result = 0;
            break;
        }
        nanosleep(&delay, NULL);
    }
done:
    if (link)
        bpf_link__destroy(link);
    bpf_object__close(object);
    return result;
}
