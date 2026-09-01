/* SPDX-License-Identifier: MIT */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "expert_buffering_trace.skel.h"

enum expert_trace_event_type {
	EXPERT_TRACE_GRAPH = 1,
	EXPERT_TRACE_LAYOUT = 2,
	EXPERT_TRACE_ROUTE = 3,
};

enum expert_trace_stat {
	EXPERT_TRACE_STAT_GRAPH = 0,
	EXPERT_TRACE_STAT_LAYOUT,
	EXPERT_TRACE_STAT_ROUTE,
	EXPERT_TRACE_STAT_DROPPED,
	EXPERT_TRACE_STAT_MAX,
};

struct expert_trace_event {
	uint64_t timestamp_ns;
	uint64_t pid_tgid;
	uint64_t graph_ordinal;
	uint64_t tensor_base;
	uint64_t total_bytes;
	uint64_t per_expert_bytes;
	uint32_t type;
	uint32_t n_experts;
	uint32_t is_bias;
	uint32_t expert_id;
	char tensor_name[64];
};

static volatile sig_atomic_t exiting;

static void handle_signal(int signo)
{
	(void)signo;
	exiting = 1;
}

static int libbpf_print_fn(enum libbpf_print_level level,
			   const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG)
		return 0;
	return vfprintf(stderr, format, args);
}

static struct bpf_link *attach_symbol(struct bpf_program *program,
				      const char *library,
				      const char *symbol,
				      pid_t pid)
{
	LIBBPF_OPTS(bpf_uprobe_opts, opts,
		.func_name = symbol,
		.retprobe = false,
	);
	struct bpf_link *link;
	int err;

	link = bpf_program__attach_uprobe_opts(program, pid, library, 0, &opts);
	err = libbpf_get_error(link);
	if (err) {
		errno = -err;
		return NULL;
	}
	return link;
}

static void print_json_string(const char *value, size_t size)
{
	size_t i;

	putchar('"');
	for (i = 0; i < size && value[i]; ++i) {
		unsigned char c = (unsigned char)value[i];

		if (c == '"' || c == '\\') {
			putchar('\\');
			putchar(c);
		} else if (c >= 0x20 && c < 0x7f) {
			putchar(c);
		} else {
			printf("\\u%04x", c);
		}
	}
	putchar('"');
}

static int handle_event(void *ctx, void *data, size_t size)
{
	const struct expert_trace_event *event = data;
	uint32_t tgid;
	uint32_t tid;

	(void)ctx;
	if (size < sizeof(*event))
		return 0;

	tgid = event->pid_tgid >> 32;
	tid = (uint32_t)event->pid_tgid;
	if (event->type == EXPERT_TRACE_GRAPH) {
		printf("{\"event\":\"graph\",\"tgid\":%u,\"tid\":%u,"
		       "\"graph\":%llu,\"timestamp_ns\":%llu}\n",
		       tgid, tid,
		       (unsigned long long)event->graph_ordinal,
		       (unsigned long long)event->timestamp_ns);
	} else if (event->type == EXPERT_TRACE_LAYOUT) {
		printf("{\"event\":\"layout\",\"tgid\":%u,\"tid\":%u,"
		       "\"name\":", tgid, tid);
		print_json_string(event->tensor_name, sizeof(event->tensor_name));
		printf(",\"base\":%llu,\"total_bytes\":%llu,"
		       "\"per_expert_bytes\":%llu,\"n_experts\":%u,"
		       "\"is_bias\":%u,\"timestamp_ns\":%llu}\n",
		       (unsigned long long)event->tensor_base,
		       (unsigned long long)event->total_bytes,
		       (unsigned long long)event->per_expert_bytes,
		       event->n_experts, event->is_bias,
		       (unsigned long long)event->timestamp_ns);
	} else if (event->type == EXPERT_TRACE_ROUTE) {
		printf("{\"event\":\"route\",\"tgid\":%u,\"tid\":%u,"
		       "\"graph\":%llu,\"tensor_base\":%llu,"
		       "\"expert_id\":%u,\"timestamp_ns\":%llu}\n",
		       tgid, tid,
		       (unsigned long long)event->graph_ordinal,
		       (unsigned long long)event->tensor_base,
		       event->expert_id,
		       (unsigned long long)event->timestamp_ns);
	}
	fflush(stdout);
	return 0;
}

static int parse_nonnegative(const char *arg, long *value)
{
	char *end = NULL;
	long parsed;

	errno = 0;
	parsed = strtol(arg, &end, 10);
	if (errno || !end || *end != '\0' || parsed < 0)
		return -EINVAL;
	*value = parsed;
	return 0;
}

static uint64_t monotonic_seconds(void)
{
	struct timespec now = {};

	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec;
}

int main(int argc, char **argv)
{
	struct expert_buffering_trace_bpf *skel = NULL;
	struct bpf_link *graph_link = NULL;
	struct bpf_link *layout_link = NULL;
	struct bpf_link *route_link = NULL;
	struct ring_buffer *ring = NULL;
	const char *library;
	long pid_arg = 0;
	long duration = 0;
	pid_t attach_pid;
	uint64_t deadline = 0;
	int err = 0;
	int stats_fd;
	int ncpus;

	if (argc < 2 || argc > 4) {
		fprintf(stderr, "usage: %s LIBGGML_BASE [PID] [SECONDS]\n", argv[0]);
		return 2;
	}
	library = argv[1];
	if (argc >= 3 && parse_nonnegative(argv[2], &pid_arg)) {
		fprintf(stderr, "invalid PID: %s\n", argv[2]);
		return 2;
	}
	if (argc == 4 && parse_nonnegative(argv[3], &duration)) {
		fprintf(stderr, "invalid duration: %s\n", argv[3]);
		return 2;
	}
	if (access(library, R_OK) != 0) {
		fprintf(stderr, "cannot read library %s: %s\n", library, strerror(errno));
		return 1;
	}
	attach_pid = pid_arg == 0 ? -1 : (pid_t)pid_arg;
	if (duration)
		deadline = monotonic_seconds() + (uint64_t)duration;

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	libbpf_set_print(libbpf_print_fn);

	skel = expert_buffering_trace_bpf__open_and_load();
	if (!skel) {
		fprintf(stderr, "failed to open/load expert trace BPF\n");
		return 1;
	}

	graph_link = attach_symbol(skel->progs.trace_graph_begin, library,
				   "ggml_backend_sched_graph_compute_async", attach_pid);
	layout_link = attach_symbol(skel->progs.trace_tensor_layout, library,
				    "gpubpf_expert_tensor_layout", attach_pid);
	route_link = attach_symbol(skel->progs.trace_expert_route, library,
				   "gpubpf_expert_route", attach_pid);
	if (!graph_link || !layout_link || !route_link) {
		fprintf(stderr, "failed to attach one or more expert trace uprobes: %s\n",
			strerror(errno));
		err = 1;
		goto out;
	}

	ring = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event,
				NULL, NULL);
	if (!ring) {
		fprintf(stderr, "failed to create expert trace ring buffer\n");
		err = 1;
		goto out;
	}

	printf("{\"event\":\"ready\",\"library\":");
	print_json_string(library, strlen(library));
	printf(",\"pid\":%ld,\"duration_seconds\":%ld}\n", pid_arg, duration);
	fflush(stdout);

	while (!exiting && (!deadline || monotonic_seconds() < deadline)) {
		err = ring_buffer__poll(ring, 100);
		if (err == -EINTR) {
			err = 0;
			break;
		}
		if (err < 0) {
			fprintf(stderr, "ring buffer poll failed: %s\n", strerror(-err));
			goto out;
		}
	}
	err = 0;

	stats_fd = bpf_map__fd(skel->maps.stats);
	ncpus = libbpf_num_possible_cpus();
	if (stats_fd >= 0 && ncpus > 0) {
		uint64_t *percpu = calloc((size_t)ncpus, sizeof(*percpu));
		uint64_t totals[EXPERT_TRACE_STAT_MAX] = {};
		uint32_t key;

		if (!percpu) {
			err = 1;
			goto out;
		}
		for (key = 0; key < EXPERT_TRACE_STAT_MAX; ++key) {
			int cpu;

			memset(percpu, 0, (size_t)ncpus * sizeof(*percpu));
			if (bpf_map_lookup_elem(stats_fd, &key, percpu) != 0)
				continue;
			for (cpu = 0; cpu < ncpus; ++cpu)
				totals[key] += percpu[cpu];
		}
		printf("{\"event\":\"final\",\"graphs\":%llu,\"layouts\":%llu,"
		       "\"routes\":%llu,\"dropped\":%llu}\n",
		       (unsigned long long)totals[EXPERT_TRACE_STAT_GRAPH],
		       (unsigned long long)totals[EXPERT_TRACE_STAT_LAYOUT],
		       (unsigned long long)totals[EXPERT_TRACE_STAT_ROUTE],
		       (unsigned long long)totals[EXPERT_TRACE_STAT_DROPPED]);
		free(percpu);
	}

out:
	ring_buffer__free(ring);
	bpf_link__destroy(route_link);
	bpf_link__destroy(layout_link);
	bpf_link__destroy(graph_link);
	expert_buffering_trace_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
