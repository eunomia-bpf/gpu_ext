/* SPDX-License-Identifier: GPL-2.0 */
/* Owns exactly one diagnostic fentry link and, for BPF cells, one struct_ops link. */
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "live.skel.h"
#include "observer.h"

enum { VERIFIER_LOG_BYTES = 4 * 1024 * 1024 };

static volatile sig_atomic_t exiting;

static void stopped(int signo)
{
    (void)signo;
    exiting = 1;
}

static int libbpf_log(enum libbpf_print_level level, const char *format,
                      va_list args)
{
    if (level == LIBBPF_DEBUG)
        return 0;
    return vfprintf(stderr, format, args);
}

static unsigned int link_id(struct bpf_link *link)
{
    struct bpf_link_info info = {};
    unsigned int length = sizeof(info);

    if (!link || bpf_obj_get_info_by_fd(bpf_link__fd(link), &info, &length))
        return 0;
    return info.id;
}

static unsigned int map_id(struct bpf_map *map)
{
    struct bpf_map_info info = {};
    unsigned int length = sizeof(info);

    if (!map || bpf_obj_get_info_by_fd(bpf_map__fd(map), &info, &length))
        return 0;
    return info.id;
}

static int write_verifier_log(const char *path, char *const *logs,
                              const char *const *names, size_t count,
                              int load_error)
{
    FILE *output = fopen(path, "wx");

    if (!output) {
        fprintf(stderr, "cannot create verifier log %s: %s\n", path,
                strerror(errno));
        return -errno;
    }
    fprintf(output, "load_error=%d\nprogram_count=%zu\n", load_error, count);
    for (size_t index = 0; index < count; ++index) {
        fprintf(output, "\nprogram=%s\n", names[index]);
        if (logs[index][0]) {
            fputs(logs[index], output);
            if (logs[index][strlen(logs[index]) - 1] != '\n')
                fputc('\n', output);
        }
    }
    if (fclose(output))
        return -EIO;
    return 0;
}

static int decision_record(void *context, void *data, size_t size)
{
    const struct stale_state_v1_observer_event *event = data;
    const struct uvm_stale_state_v1_diagnostic *d;
    unsigned int expected_mode = *(unsigned int *)context;
    const char *implementation;
    const char *phase;
    const char *action;
    const char *effect;
    bool valid;

    if (size != sizeof(*event)) {
        fprintf(stderr, "observer returned an unexpected record size: %zu\n", size);
        return -EMSGSIZE;
    }
    d = &event->diagnostic;
    implementation = expected_mode == STALE_STATE_V1_MODE_NATIVE ? "native" : "bpf";
    phase = d->input.snapshot.phase == STALE_STATE_V1_PHASE_DENSE ? "dense" :
            d->input.snapshot.phase == STALE_STATE_V1_PHASE_SPARSE ? "sparse" : "invalid";
    action = d->action == STALE_STATE_V1_ACTION_PREFETCH_MAX ? "prefetch_max" :
             d->action == STALE_STATE_V1_ACTION_DISCARD_PREFETCH ? "discard_prefetch" : "invalid";
    effect = d->action == STALE_STATE_V1_ACTION_PREFETCH_MAX ? "prefetch" :
             d->action == STALE_STATE_V1_ACTION_DISCARD_PREFETCH ? "discard" : "invalid";
    valid = d->diagnostic_phase == STALE_STATE_V1_DIAG_FINISHED &&
            d->mode == expected_mode &&
            d->status == STALE_STATE_V1_STATUS_EFFECT_APPLIED &&
            d->input.abi_version == STALE_STATE_DRIVER_V1_ABI_VERSION &&
            d->input.reserved == 0 && d->reserved == 0 &&
            d->input.snapshot.reserved == 0 &&
            d->input.snapshot.sequence > 0 && d->input.decision_sequence > 0 &&
            d->input.decision_mono_ns >= d->input.snapshot.published_mono_ns &&
            d->decision_age_ns == d->input.decision_mono_ns -
                                  d->input.snapshot.source_mono_ns &&
            d->action_attempted == 1 && d->action_conflict == 0 &&
            d->action_request_calls == 1 &&
            d->region_result == STALE_STATE_V1_TRANSITION_APPLY &&
            d->initial_effect == STALE_STATE_V1_INITIAL_BYPASS &&
            d->callback_return == (long long)d->action &&
            strcmp(phase, "invalid") && strcmp(action, "invalid");
    if (d->action == STALE_STATE_V1_ACTION_PREFETCH_MAX)
        valid = valid && d->requested_first == d->input.max_first &&
                d->requested_outer == d->input.max_outer &&
                d->output_first == d->input.max_first &&
                d->output_outer == d->input.max_outer;
    else if (d->action == STALE_STATE_V1_ACTION_DISCARD_PREFETCH)
        valid = valid && d->requested_first == 0 && d->requested_outer == 0 &&
                d->output_first == 0 && d->output_outer == 0;
    if (!valid) {
        fprintf(stderr,
                "invalid completed diagnostic: sequence=%llu mode=%u status=%u action=%u\n",
                d->input.decision_sequence, d->mode, d->status, d->action);
        return -EPROTO;
    }

    printf("{\"event\":\"policy_decision\",\"implementation\":\"%s\","
           "\"decision_sequence\":%llu,\"snapshot_read_path\":"
           "\"driver_%s_read_only_context\",\"decision_mono_ns\":%llu,"
           "\"snapshot_sequence\":%llu,\"snapshot_phase\":\"%s\","
           "\"decision_age_ns\":%llu,\"action\":\"%s\",\"effect\":\"%s\","
           "\"effect_source\":\"driver_diagnostic\",\"fault_page_index\":%llu,"
           "\"legal_max_first\":%llu,\"legal_max_outer\":%llu,"
           "\"output_first\":%llu,\"output_outer\":%llu,"
           "\"observer_mono_ns\":%llu,\"target_tgid\":%u}\n",
           implementation, d->input.decision_sequence, implementation,
           d->input.decision_mono_ns, d->input.snapshot.sequence, phase,
           d->decision_age_ns, action, effect, d->input.page_index,
           d->input.max_first, d->input.max_outer, d->output_first,
           d->output_outer, event->observed_mono_ns,
           (unsigned int)(event->pid_tgid >> 32));
    return 0;
}

static int read_metrics(struct live_bpf *skel,
                        struct stale_state_v1_observer_metrics *total)
{
    int cpu_count = libbpf_num_possible_cpus();
    struct stale_state_v1_observer_metrics *per_cpu;
    unsigned int key = 0;

    if (cpu_count <= 0)
        return -EINVAL;
    per_cpu = calloc((size_t)cpu_count, sizeof(*per_cpu));
    if (!per_cpu)
        return -ENOMEM;
    if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.observer_metrics), &key,
                            per_cpu)) {
        free(per_cpu);
        return -errno;
    }
    memset(total, 0, sizeof(*total));
    for (int cpu = 0; cpu < cpu_count; ++cpu) {
#define SUM_COUNTER(name) total->name += per_cpu[cpu].name;
        STALE_STATE_V1_OBSERVER_COUNTERS(SUM_COUNTER)
#undef SUM_COUNTER
    }
    free(per_cpu);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s --target-pid PID --implementation native|bpf "
            "--verifier-log PATH\n", program);
}

int main(int argc, char **argv)
{
    struct live_bpf *skel = NULL;
    struct bpf_link *observer = NULL, *policy = NULL;
    struct ring_buffer *ring = NULL;
    struct stale_state_v1_observer_config config = {};
    struct stale_state_v1_observer_metrics total = {};
    struct bpf_program *program;
    char *logs[2] = {};
    const char *names[2] = {};
    const char *verifier_path = NULL;
    const char *implementation = NULL;
    unsigned int key = 0, expected_mode = 0;
    size_t program_count = 0;
    int result = 1, load_error = 0, poll_error = 0;

    for (int index = 1; index < argc; ++index) {
        if (!strcmp(argv[index], "--target-pid") && index + 1 < argc) {
            char *end = NULL;
            unsigned long value = strtoul(argv[++index], &end, 10);
            if (!end || *end || !value || value > 0xffffffffUL) {
                usage(argv[0]);
                return 2;
            }
            config.target_tgid = (unsigned int)value;
        }
        else if (!strcmp(argv[index], "--implementation") && index + 1 < argc)
            implementation = argv[++index];
        else if (!strcmp(argv[index], "--verifier-log") && index + 1 < argc)
            verifier_path = argv[++index];
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!config.target_tgid || !verifier_path || !implementation ||
        (strcmp(implementation, "native") && strcmp(implementation, "bpf"))) {
        usage(argv[0]);
        return 2;
    }
    expected_mode = !strcmp(implementation, "native") ?
                        STALE_STATE_V1_MODE_NATIVE : STALE_STATE_V1_MODE_BPF;
    config.expected_mode = expected_mode;
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, stopped);
    signal(SIGTERM, stopped);
    libbpf_set_print(libbpf_log);

    skel = live_bpf__open();
    if (!skel)
        goto out;
    bpf_object__for_each_program(program, skel->obj) {
        if (program_count >= 2)
            goto out;
        logs[program_count] = calloc(1, VERIFIER_LOG_BYTES);
        if (!logs[program_count])
            goto out;
        names[program_count] = bpf_program__name(program);
        if (bpf_program__set_log_level(program, 1) ||
            bpf_program__set_log_buf(program, logs[program_count],
                                     VERIFIER_LOG_BYTES))
            goto out;
        ++program_count;
    }
    if (program_count != 2)
        goto out;
    load_error = live_bpf__load(skel);
    if (write_verifier_log(verifier_path, logs, names, program_count,
                           load_error))
        goto out;
    if (load_error)
        goto out;
    if (bpf_map_update_elem(bpf_map__fd(skel->maps.observer_config), &key,
                            &config, BPF_ANY))
        goto out;
    observer = bpf_program__attach(skel->progs.stale_state_v1_diagnostic_observer);
    if (libbpf_get_error(observer)) {
        observer = NULL;
        goto out;
    }
    if (expected_mode == STALE_STATE_V1_MODE_BPF) {
        policy = bpf_map__attach_struct_ops(skel->maps.stale_state_v1_ops);
        if (libbpf_get_error(policy)) {
            policy = NULL;
            goto out;
        }
    }
    if (!link_id(observer) || (expected_mode == STALE_STATE_V1_MODE_BPF &&
                               !link_id(policy)))
        goto out;
    ring = ring_buffer__new(bpf_map__fd(skel->maps.observer_events),
                            decision_record, &expected_mode, NULL);
    if (!ring)
        goto out;
    printf("{\"event\":\"ready\",\"pid\":%ld,\"target_pid\":%u,"
           "\"implementation\":\"%s\",\"observer_link_id\":%u,"
           "\"struct_link_id\":%u,\"struct_map_id\":%u}\n",
           (long)getpid(), config.target_tgid, implementation, link_id(observer),
           link_id(policy), map_id(skel->maps.stale_state_v1_ops));
    while (!exiting) {
        poll_error = ring_buffer__poll(ring, 100);
        if (poll_error < 0 && poll_error != -EINTR)
            goto out;
    }
    do {
        poll_error = ring_buffer__poll(ring, 0);
    } while (poll_error > 0);
    if (poll_error < 0 && poll_error != -EINTR)
        goto out;
    if (read_metrics(skel, &total))
        goto out;
    result = total.diagnostic_calls == total.selected_seen + total.finished_seen &&
             total.selected_seen == total.finished_seen &&
             total.finished_seen == total.records_emitted &&
             total.finished_seen > 0 && !total.foreign_tgid &&
             !total.read_errors && !total.ringbuf_drops && !total.phase_errors ? 0 : 1;
    printf("{\"event\":\"observer_final\",\"implementation\":\"%s\","
           "\"observer_link_id\":%u,\"struct_link_id\":%u,"
           "\"diagnostic_calls\":%llu,\"selected_seen\":%llu,"
           "\"finished_seen\":%llu,\"records_emitted\":%llu,"
           "\"foreign_tgid\":%llu,\"read_errors\":%llu,"
           "\"ringbuf_drops\":%llu,\"phase_errors\":%llu,"
           "\"valid\":%s}\n",
           implementation, link_id(observer), link_id(policy),
           total.diagnostic_calls, total.selected_seen, total.finished_seen,
           total.records_emitted, total.foreign_tgid, total.read_errors,
           total.ringbuf_drops, total.phase_errors, result ? "false" : "true");
out:
    ring_buffer__free(ring);
    bpf_link__destroy(policy);
    bpf_link__destroy(observer);
    live_bpf__destroy(skel);
    for (size_t index = 0; index < 2; ++index)
        free(logs[index]);
    return result;
}
