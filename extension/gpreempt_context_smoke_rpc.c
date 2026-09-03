/* SPDX-License-Identifier: GPL-2.0 */
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpreempt_context_smoke_rpc.h"
#include "gpreempt_context_smoke_rpc.skel.h"

static volatile sig_atomic_t exiting;
static unsigned long long received;
static void stop(int number) { (void)number; exiting = 1; }
static int event(void *unused, void *data, size_t size)
{
    (void)unused;
    if (size != sizeof(struct gp_rpc_event)) return -EINVAL;
    const struct gp_rpc_event *e = data;
    const struct gp_gsp_completion *c = &e->completion;
    printf("{\"event\":\"gsp_timeslice_rpc\",\"source\":\"kernel_open_rpc_completion\","
           "\"pid\":%llu,\"tid\":%llu,\"hclient\":%u,\"hobject\":%u,\"command\":%u,"
           "\"params_size\":%u,\"wire_size\":%u,\"timeslice_us\":%llu,\"input_valid\":%u,"
           "\"transport_status\":%u,\"gsp_status\":%u,\"gsp_status_valid\":%u,\"completed_ns\":%llu}\n",
           e->pid_tgid >> 32, e->pid_tgid & 0xffffffffULL, c->hClient, c->hObject, c->command,
           c->input_size, c->wire_size, c->input_value, c->input_valid, c->transport_status,
           c->gsp_status, c->gsp_status_valid, e->completed_ns);
    ++received;
    fflush(stdout);
    return 0;
}
static int positive(const char *arg, unsigned int maximum, unsigned int *out)
{
    char *end;
    errno = 0;
    unsigned long value = strtoul(arg, &end, 10);
    if (errno || end == arg || *end || !value || value > maximum) return -1;
    *out = value;
    return 0;
}
int main(int argc, char **argv)
{
    unsigned int seconds = 120, pid = 0;
    if (argc > 3 || (argc > 1 && positive(argv[1], 3600, &seconds)) ||
        (argc > 2 && positive(argv[2], 0x7fffffff, &pid))) {
        fprintf(stderr, "Usage: %s [seconds 1..3600] [target PID]\n", argv[0]);
        return 1;
    }
    struct gpreempt_context_smoke_rpc_bpf *skeleton = gpreempt_context_smoke_rpc_bpf__open();
    struct ring_buffer *ring = NULL;
    int result = 1;
    if (!skeleton) return 1;
    skeleton->rodata->target_pid = pid;
    if (gpreempt_context_smoke_rpc_bpf__load(skeleton) ||
        gpreempt_context_smoke_rpc_bpf__attach(skeleton)) goto done;
    ring = ring_buffer__new(bpf_map__fd(skeleton->maps.events), event, NULL, NULL);
    if (!ring) goto done;
    signal(SIGTERM, stop); signal(SIGINT, stop);
    printf("gpreempt_rpc_observer_ready: pid=%u seconds=%u source=kernel_open_rpc_completion\n", pid, seconds);
    fflush(stdout);
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    do {
        int rc = ring_buffer__poll(ring, 100);
        if (rc < 0 && rc != -EINTR) goto done;
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while (!exiting && now.tv_sec - start.tv_sec < seconds);
    if (ring_buffer__consume(ring) < 0) goto done;
    __u64 values[4] = {};
    for (__u32 key = 0; key < 4; ++key)
        if (bpf_map_lookup_elem(bpf_map__fd(skeleton->maps.stats), &key, &values[key])) goto done;
    printf("{\"event\":\"gpreempt_rpc_observer_summary\",\"observed\":%llu,\"completed\":%llu,"
           "\"read_errors\":%llu,\"ring_drops\":%llu,\"received\":%llu}\n",
           (unsigned long long)values[0], (unsigned long long)values[1],
           (unsigned long long)values[2], (unsigned long long)values[3], received);
    /* Still correlate entries with queried role handles; zero events is NOT a pass. */
    result = !values[0] || values[0] != values[1] || values[1] != received || values[2] || values[3];
done:
    ring_buffer__free(ring);
    gpreempt_context_smoke_rpc_bpf__destroy(skeleton);
    return result;
}
