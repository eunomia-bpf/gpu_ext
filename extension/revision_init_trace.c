/* SPDX-License-Identifier: GPL-2.0 */
#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "revision_init_trace.h"
#include "revision_init_trace.skel.h"

static volatile sig_atomic_t exiting;
static unsigned long long diagnostic_received;
static unsigned long long gsp_received;

static void stop(int number)
{
	(void)number;
	exiting = 1;
}

static int positive(const char *text, unsigned int maximum, unsigned int *value)
{
	char *end;
	unsigned long parsed;

	errno = 0;
	parsed = strtoul(text, &end, 10);
	if (errno || end == text || *end || !parsed || parsed > maximum)
		return -1;
	*value = parsed;
	return 0;
}

static int print_event(void *unused, void *data, size_t size)
{
	const struct revision_init_trace_event *event = data;
	unsigned long long pid;
	unsigned long long tid;

	(void)unused;
	if (size != sizeof(*event))
		return -EMSGSIZE;
	pid = event->pid_tgid >> 32;
	tid = event->pid_tgid & 0xffffffffULL;
	if (event->kind == REVISION_INIT_EVENT_DIAGNOSTIC) {
		const struct revision_init_diagnostic *d = &event->diagnostic;

		printf("{\"event\":\"scheduler_init_diagnostic\",\"pid\":%llu,\"tid\":%llu,"
		       "\"timestamp_ns\":%llu,\"abi_version\":%u,\"abi_size\":%u,"
		       "\"phase\":%u,\"field\":%u,\"h_client\":%u,\"h_resource\":%u,"
		       "\"gpu_instance\":%u,\"subdevice_instance\":%u,\"group_id\":%u,"
		       "\"runlist_id\":%u,\"engine_type\":%u,\"constructor_epoch\":%u,"
		       "\"default_timeslice\":%llu,\"minimum_timeslice\":%llu,"
		       "\"default_interleave\":%u,\"timeslice_attempted\":%u,"
		       "\"timeslice_conflict\":%u,\"timeslice_request_value\":%llu,"
		       "\"interleave_attempted\":%u,\"interleave_conflict\":%u,"
		       "\"interleave_request_value\":%u,\"timeslice_validation_result\":%u,"
		       "\"interleave_validation_result\":%u,\"effective_timeslice\":%llu,"
		       "\"effective_interleave\":%u,\"timeslice_native_status\":%u,"
		       "\"timeslice_post_value\":%llu,\"interleave_native_status\":%u,"
		       "\"interleave_post_value\":%u,\"constructor_status\":%u,"
		       "\"final_interleave\":%u,\"final_timeslice\":%llu,"
		       "\"final_snapshot_valid\":%u}\n",
		       pid, tid, (unsigned long long)event->timestamp_ns,
		       d->abi_version, d->abi_size, d->phase, d->field,
		       d->h_client, d->h_resource, d->gpu_instance,
		       d->subdevice_instance, d->group_id, d->runlist_id,
		       d->engine_type, d->constructor_epoch,
		       (unsigned long long)d->default_timeslice,
		       (unsigned long long)d->minimum_timeslice,
		       d->default_interleave, d->timeslice_attempted,
		       d->timeslice_conflict,
		       (unsigned long long)d->timeslice_request_value,
		       d->interleave_attempted, d->interleave_conflict,
		       d->interleave_request_value, d->timeslice_validation_result,
		       d->interleave_validation_result,
		       (unsigned long long)d->effective_timeslice,
		       d->effective_interleave, d->timeslice_native_status,
		       (unsigned long long)d->timeslice_post_value,
		       d->interleave_native_status, d->interleave_post_value,
		       d->constructor_status, d->final_interleave,
		       (unsigned long long)d->final_timeslice,
		       d->final_snapshot_valid);
		++diagnostic_received;
	} else if (event->kind == REVISION_INIT_EVENT_GSP) {
		const struct revision_init_gsp_completion *g = &event->gsp;

		printf("{\"event\":\"scheduler_init_gsp_completion\",\"pid\":%llu,"
		       "\"tid\":%llu,\"timestamp_ns\":%llu,\"h_client\":%u,"
		       "\"h_object\":%u,\"command\":%u,\"input_size\":%u,"
		       "\"wire_size\":%u,\"input_value\":%llu,\"input_valid\":%u,"
		       "\"transport_status\":%u,\"gsp_status\":%u,"
		       "\"gsp_status_valid\":%u}\n",
		       pid, tid, (unsigned long long)event->timestamp_ns,
		       g->h_client, g->h_object, g->command, g->input_size,
		       g->wire_size, (unsigned long long)g->input_value,
		       g->input_valid, g->transport_status, g->gsp_status,
		       g->gsp_status_valid);
		++gsp_received;
	} else {
		return -EPROTO;
	}
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	struct revision_init_trace_bpf *skeleton = NULL;
	struct ring_buffer *ring = NULL;
	unsigned int seconds = 120;
	unsigned int target_tgid;
	struct timespec start;
	struct timespec now;
	__u64 stats[REVISION_INIT_TRACE_STAT_COUNT] = {};
	int result = 1;
	int rc;

	if (argc != 2 && argc != 3) {
		fprintf(stderr, "Usage: %s TARGET_TGID [SECONDS 1..3600]\n", argv[0]);
		return 2;
	}
	if (positive(argv[1], 0x7fffffffU, &target_tgid) ||
	    (argc == 3 && positive(argv[2], 3600, &seconds))) {
		fprintf(stderr, "invalid target TGID or duration\n");
		return 2;
	}

	skeleton = revision_init_trace_bpf__open();
	if (!skeleton)
		goto done;
	skeleton->rodata->target_tgid = target_tgid;
	if (revision_init_trace_bpf__load(skeleton) ||
	    revision_init_trace_bpf__attach(skeleton))
		goto done;
	ring = ring_buffer__new(bpf_map__fd(skeleton->maps.events), print_event,
				NULL, NULL);
	if (!ring)
		goto done;
	signal(SIGINT, stop);
	signal(SIGTERM, stop);
	printf("{\"event\":\"scheduler_init_observer_ready\",\"target_tgid\":%u}\n",
	       target_tgid);
	fflush(stdout);
	clock_gettime(CLOCK_MONOTONIC, &start);
	do {
		rc = ring_buffer__poll(ring, 100);
		if (rc < 0 && rc != -EINTR)
			goto done;
		clock_gettime(CLOCK_MONOTONIC, &now);
	} while (!exiting && now.tv_sec - start.tv_sec < seconds);
	if (ring_buffer__consume(ring) < 0)
		goto done;
	for (__u32 key = 0; key < REVISION_INIT_TRACE_STAT_COUNT; ++key) {
		if (bpf_map_lookup_elem(bpf_map__fd(skeleton->maps.stats),
					&key, &stats[key]))
			goto done;
	}
	printf("{\"event\":\"scheduler_init_observer_summary\","
	       "\"diagnostic\":{\"observed\":%llu,\"emitted\":%llu,"
	       "\"read_errors\":%llu,\"ring_drops\":%llu,\"received\":%llu},"
	       "\"gsp\":{\"observed\":%llu,\"emitted\":%llu,"
	       "\"read_errors\":%llu,\"ring_drops\":%llu,\"received\":%llu}}\n",
	       (unsigned long long)stats[REVISION_INIT_DIAGNOSTIC_OBSERVED],
	       (unsigned long long)stats[REVISION_INIT_DIAGNOSTIC_EMITTED],
	       (unsigned long long)stats[REVISION_INIT_DIAGNOSTIC_READ_ERROR],
	       (unsigned long long)stats[REVISION_INIT_DIAGNOSTIC_DROP],
	       diagnostic_received,
	       (unsigned long long)stats[REVISION_INIT_GSP_OBSERVED],
	       (unsigned long long)stats[REVISION_INIT_GSP_EMITTED],
	       (unsigned long long)stats[REVISION_INIT_GSP_READ_ERROR],
	       (unsigned long long)stats[REVISION_INIT_GSP_DROP],
	       gsp_received);
	fflush(stdout);
	result = (stats[REVISION_INIT_DIAGNOSTIC_OBSERVED] !=
		  stats[REVISION_INIT_DIAGNOSTIC_EMITTED] ||
		  stats[REVISION_INIT_DIAGNOSTIC_EMITTED] != diagnostic_received ||
		  stats[REVISION_INIT_DIAGNOSTIC_READ_ERROR] ||
		  stats[REVISION_INIT_DIAGNOSTIC_DROP] ||
		  stats[REVISION_INIT_GSP_OBSERVED] !=
		  stats[REVISION_INIT_GSP_EMITTED] ||
		  stats[REVISION_INIT_GSP_EMITTED] != gsp_received ||
		  stats[REVISION_INIT_GSP_READ_ERROR] ||
		  stats[REVISION_INIT_GSP_DROP]);
done:
	ring_buffer__free(ring);
	revision_init_trace_bpf__destroy(skeleton);
	return result;
}
