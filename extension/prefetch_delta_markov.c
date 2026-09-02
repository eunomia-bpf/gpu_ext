/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
/* Userspace loader for the block-local delta/Markov prefetch policy. */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "loader_identity.h"
#include "prefetch_delta_markov.skel.h"

#define CONFIG_CONFIDENCE_THRESHOLD 0
#define CONFIG_PREFETCH_PAGES 1
#define CONFIG_MAX_DELTA 2

struct delta_markov_stats {
	__u64 context_captures;
	__u64 callbacks;
	__u64 blocks_initialized;
	__u64 deltas_observed;
	__u64 invalid_deltas;
	__u64 transitions_created;
	__u64 transition_matches;
	__u64 transition_decays;
	__u64 transition_replacements;
	__u64 confident_predictions;
	__u64 prefetch_requests;
	__u64 empty_requests;
	__u64 map_errors;
	__u64 request_errors;
};

static volatile sig_atomic_t exiting;

static void handle_signal(int signal_number)
{
	(void)signal_number;
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

static int set_config(int fd, __u32 key, __u64 value)
{
	if (bpf_map_update_elem(fd, &key, &value, BPF_ANY) == 0)
		return 0;
	fprintf(stderr, "failed to set config key %u: %s\n", key,
		strerror(errno));
	return -errno;
}

static int read_metrics(struct prefetch_delta_markov_bpf *skel,
			struct delta_markov_stats *total)
{
	struct delta_markov_stats *per_cpu;
	__u32 key = 0;
	int cpu_count;
	int cpu;

	cpu_count = libbpf_num_possible_cpus();
	if (cpu_count <= 0)
		return cpu_count ? cpu_count : -EINVAL;
	per_cpu = calloc((size_t)cpu_count, sizeof(*per_cpu));
	if (!per_cpu)
		return -ENOMEM;

	memset(total, 0, sizeof(*total));
	if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.metrics), &key,
				per_cpu) != 0) {
		int err = -errno;

		free(per_cpu);
		return err;
	}

#define ADD_FIELD(name) total->name += per_cpu[cpu].name
	for (cpu = 0; cpu < cpu_count; cpu++) {
		ADD_FIELD(context_captures);
		ADD_FIELD(callbacks);
		ADD_FIELD(blocks_initialized);
		ADD_FIELD(deltas_observed);
		ADD_FIELD(invalid_deltas);
		ADD_FIELD(transitions_created);
		ADD_FIELD(transition_matches);
		ADD_FIELD(transition_decays);
		ADD_FIELD(transition_replacements);
		ADD_FIELD(confident_predictions);
		ADD_FIELD(prefetch_requests);
		ADD_FIELD(empty_requests);
		ADD_FIELD(map_errors);
		ADD_FIELD(request_errors);
	}
#undef ADD_FIELD

	free(per_cpu);
	return 0;
}

static void emit_metrics(struct prefetch_delta_markov_bpf *skel,
			 const char *event)
{
	struct delta_markov_stats stats;
	int err = read_metrics(skel, &stats);

	if (err) {
		fprintf(stderr, "failed to read metrics: %s\n", strerror(-err));
		return;
	}

	printf("{\"event\":\"%s\",\"context_captures\":%llu,"
	       "\"callbacks\":%llu,\"blocks_initialized\":%llu,"
	       "\"deltas_observed\":%llu,\"invalid_deltas\":%llu,"
	       "\"transitions_created\":%llu,\"transition_matches\":%llu,"
	       "\"transition_decays\":%llu,\"transition_replacements\":%llu,"
	       "\"confident_predictions\":%llu,\"prefetch_requests\":%llu,"
	       "\"empty_requests\":%llu,\"map_errors\":%llu,"
	       "\"request_errors\":%llu}\n",
	       event,
	       (unsigned long long)stats.context_captures,
	       (unsigned long long)stats.callbacks,
	       (unsigned long long)stats.blocks_initialized,
	       (unsigned long long)stats.deltas_observed,
	       (unsigned long long)stats.invalid_deltas,
	       (unsigned long long)stats.transitions_created,
	       (unsigned long long)stats.transition_matches,
	       (unsigned long long)stats.transition_decays,
	       (unsigned long long)stats.transition_replacements,
	       (unsigned long long)stats.confident_predictions,
	       (unsigned long long)stats.prefetch_requests,
	       (unsigned long long)stats.empty_requests,
	       (unsigned long long)stats.map_errors,
	       (unsigned long long)stats.request_errors);
	fflush(stdout);
}

static void usage(const char *program)
{
	fprintf(stderr,
		"usage: %s [-c confidence] [-n prefetch_pages] [-m max_delta] "
		"[-i metrics_interval_seconds]\n"
		"  -c  matching transitions required (1..64, default 2)\n"
		"  -n  contiguous pages per prediction (1..512, default 2)\n"
		"  -m  largest learned absolute delta (1..4096, default 128)\n"
		"  -i  metrics interval in seconds (1..3600, default 5)\n",
		program);
}

static int parse_u64(const char *text, __u64 *value)
{
	char *end = NULL;
	unsigned long long parsed;

	errno = 0;
	parsed = strtoull(text, &end, 10);
	if (errno || !end || *end != '\0')
		return -EINVAL;
	*value = parsed;
	return 0;
}

int main(int argc, char **argv)
{
	struct prefetch_delta_markov_bpf *skel = NULL;
	struct bpf_link *struct_link = NULL;
	__u64 confidence = 2;
	__u64 prefetch_pages = 2;
	__u64 maximum_delta = 128;
	__u64 metrics_interval = 5;
	__u32 struct_map_id;
	__u32 struct_link_id;
	__u32 kprobe_link_id;
	int config_fd;
	int option;
	int err = 0;

	while ((option = getopt(argc, argv, "c:n:m:i:h")) != -1) {
		__u64 parsed;

		if (option == 'h') {
			usage(argv[0]);
			return 0;
		}
		if ((option != 'c' && option != 'n' && option != 'm' &&
		     option != 'i') || parse_u64(optarg, &parsed) != 0) {
			usage(argv[0]);
			return 2;
		}
		if (option == 'c')
			confidence = parsed;
		else if (option == 'n')
			prefetch_pages = parsed;
		else if (option == 'm')
			maximum_delta = parsed;
		else
			metrics_interval = parsed;
	}

	if (optind != argc || confidence < 1 || confidence > 64 ||
	    prefetch_pages < 1 || prefetch_pages > 512 ||
	    maximum_delta < 1 || maximum_delta > 4096 ||
	    metrics_interval < 1 || metrics_interval > 3600) {
		usage(argv[0]);
		return 2;
	}

	setvbuf(stdout, NULL, _IOLBF, 0);
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	libbpf_set_print(libbpf_print_fn);

	skel = prefetch_delta_markov_bpf__open();
	if (!skel) {
		fprintf(stderr, "failed to open BPF skeleton\n");
		return 1;
	}
	err = prefetch_delta_markov_bpf__load(skel);
	if (err) {
		fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
		goto out;
	}

	config_fd = bpf_map__fd(skel->maps.policy_config);
	err = set_config(config_fd, CONFIG_CONFIDENCE_THRESHOLD, confidence);
	if (!err)
		err = set_config(config_fd, CONFIG_PREFETCH_PAGES,
				 prefetch_pages);
	if (!err)
		err = set_config(config_fd, CONFIG_MAX_DELTA, maximum_delta);
	if (err)
		goto out;

	skel->links.capture_va_block =
		bpf_program__attach(skel->progs.capture_va_block);
	err = libbpf_get_error(skel->links.capture_va_block);
	if (err) {
		skel->links.capture_va_block = NULL;
		fprintf(stderr, "failed to attach VA-block observation kprobe: %s\n",
			strerror(-err));
		goto out;
	}

	struct_link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_delta_markov);
	err = libbpf_get_error(struct_link);
	if (err) {
		struct_link = NULL;
		fprintf(stderr, "failed to attach struct_ops: %s\n",
			strerror(-err));
		goto out;
	}
	err = safe_loader_map_id(skel->maps.uvm_ops_delta_markov,
				 &struct_map_id);
	if (!err)
		err = safe_loader_link_id(struct_link, &struct_link_id);
	if (!err)
		err = safe_loader_link_id(skel->links.capture_va_block,
				  &kprobe_link_id);
	if (err) {
		fprintf(stderr, "failed to resolve owned BPF object IDs: %s\n",
			strerror(-err));
		goto out;
	}

	printf("{\"event\":\"ready\",\"pid\":%ld,"
	       "\"struct_map_id\":%u,\"struct_link_id\":%u,"
	       "\"kprobe_link_id\":%u,"
	       "\"confidence\":%llu,\"prefetch_pages\":%llu,"
	       "\"maximum_delta\":%llu,"
	       "\"metrics_interval_seconds\":%llu}\n",
	       (long)getpid(), struct_map_id, struct_link_id, kprobe_link_id,
	       (unsigned long long)confidence,
	       (unsigned long long)prefetch_pages,
	       (unsigned long long)maximum_delta,
	       (unsigned long long)metrics_interval);

	while (!exiting) {
		sleep((unsigned int)metrics_interval);
		if (!exiting)
			emit_metrics(skel, "metrics");
	}
	emit_metrics(skel, "final_metrics");

out:
	/* Destroy only links owned by this loader process. */
	bpf_link__destroy(struct_link);
	prefetch_delta_markov_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
