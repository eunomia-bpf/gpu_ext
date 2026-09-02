/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
/* Userspace loader for the approximate 2Q / segmented-LRU policy. */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "eviction_2q_approx.skel.h"
#include "loader_identity.h"

#define CONFIG_PROMOTE_AFTER 0
#define CONFIG_MAX_GENERATION_GAP 1

struct twoq_stats {
	__u64 activate_events;
	__u64 access_events;
	__u64 admissions;
	__u64 identity_resets;
	__u64 generation_resets;
	__u64 same_episode_events;
	__u64 probation_head_requests;
	__u64 promotions;
	__u64 protected_tail_requests;
	__u64 reorder_errors;
	__u64 eviction_prepares;
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

static int read_metrics(struct eviction_2q_approx_bpf *skel,
			struct twoq_stats *total)
{
	struct twoq_stats *per_cpu;
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

	for (cpu = 0; cpu < cpu_count; cpu++) {
		total->activate_events += per_cpu[cpu].activate_events;
		total->access_events += per_cpu[cpu].access_events;
		total->admissions += per_cpu[cpu].admissions;
		total->identity_resets += per_cpu[cpu].identity_resets;
		total->generation_resets += per_cpu[cpu].generation_resets;
		total->same_episode_events += per_cpu[cpu].same_episode_events;
		total->probation_head_requests +=
			per_cpu[cpu].probation_head_requests;
		total->promotions += per_cpu[cpu].promotions;
		total->protected_tail_requests +=
			per_cpu[cpu].protected_tail_requests;
		total->reorder_errors += per_cpu[cpu].reorder_errors;
		total->eviction_prepares += per_cpu[cpu].eviction_prepares;
	}

	free(per_cpu);
	return 0;
}

static void emit_metrics(struct eviction_2q_approx_bpf *skel,
			 const char *event)
{
	struct twoq_stats stats;
	int err = read_metrics(skel, &stats);

	if (err) {
		fprintf(stderr, "failed to read metrics: %s\n", strerror(-err));
		return;
	}

	printf("{\"event\":\"%s\",\"activate_events\":%llu,"
	       "\"access_events\":%llu,\"admissions\":%llu,"
	       "\"identity_resets\":%llu,\"generation_resets\":%llu,"
	       "\"same_episode_events\":%llu,"
	       "\"probation_head_requests\":%llu,\"promotions\":%llu,"
	       "\"protected_tail_requests\":%llu,\"reorder_errors\":%llu,"
	       "\"eviction_prepares\":%llu}\n",
	       event,
	       (unsigned long long)stats.activate_events,
	       (unsigned long long)stats.access_events,
	       (unsigned long long)stats.admissions,
	       (unsigned long long)stats.identity_resets,
	       (unsigned long long)stats.generation_resets,
	       (unsigned long long)stats.same_episode_events,
	       (unsigned long long)stats.probation_head_requests,
	       (unsigned long long)stats.promotions,
	       (unsigned long long)stats.protected_tail_requests,
	       (unsigned long long)stats.reorder_errors,
	       (unsigned long long)stats.eviction_prepares);
	fflush(stdout);
}

static void usage(const char *program)
{
	fprintf(stderr,
		"usage: %s [-p promote_after] [-g maximum_generation_gap] "
		"[-i metrics_interval_seconds]\n"
		"  -p  observations before promotion (2..64, default 2)\n"
		"  -g  generation gap treated as a recycled root (1..64, "
		"default 2)\n"
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
	struct eviction_2q_approx_bpf *skel = NULL;
	struct bpf_link *struct_link = NULL;
	__u64 promote_after = 2;
	__u64 maximum_generation_gap = 2;
	__u64 metrics_interval = 5;
	__u32 struct_map_id;
	__u32 struct_link_id;
	int config_fd;
	int option;
	int err = 0;

	while ((option = getopt(argc, argv, "p:g:i:h")) != -1) {
		__u64 parsed;

		if (option == 'h') {
			usage(argv[0]);
			return 0;
		}
		if ((option != 'p' && option != 'g' && option != 'i') ||
		    parse_u64(optarg, &parsed) != 0) {
			usage(argv[0]);
			return 2;
		}
		if (option == 'p')
			promote_after = parsed;
		else if (option == 'g')
			maximum_generation_gap = parsed;
		else
			metrics_interval = parsed;
	}

	if (optind != argc || promote_after < 2 || promote_after > 64 ||
	    maximum_generation_gap < 1 || maximum_generation_gap > 64 ||
	    metrics_interval < 1 || metrics_interval > 3600) {
		usage(argv[0]);
		return 2;
	}

	setvbuf(stdout, NULL, _IOLBF, 0);
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	libbpf_set_print(libbpf_print_fn);

	skel = eviction_2q_approx_bpf__open();
	if (!skel) {
		fprintf(stderr, "failed to open BPF skeleton\n");
		return 1;
	}

	err = eviction_2q_approx_bpf__load(skel);
	if (err) {
		fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
		goto out;
	}

	config_fd = bpf_map__fd(skel->maps.policy_config);
	err = set_config(config_fd, CONFIG_PROMOTE_AFTER, promote_after);
	if (!err)
		err = set_config(config_fd, CONFIG_MAX_GENERATION_GAP,
				 maximum_generation_gap);
	if (err)
		goto out;

	struct_link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_2q_approx);
	err = libbpf_get_error(struct_link);
	if (err) {
		struct_link = NULL;
		fprintf(stderr, "failed to attach struct_ops: %s\n",
			strerror(-err));
		goto out;
	}
	err = safe_loader_map_id(skel->maps.uvm_ops_2q_approx,
				 &struct_map_id);
	if (!err)
		err = safe_loader_link_id(struct_link, &struct_link_id);
	if (err) {
		fprintf(stderr, "failed to resolve owned struct_ops IDs: %s\n",
			strerror(-err));
		goto out;
	}

	printf("{\"event\":\"ready\",\"pid\":%ld,"
	       "\"struct_map_id\":%u,\"struct_link_id\":%u,"
	       "\"promote_after\":%llu,\"maximum_generation_gap\":%llu,"
	       "\"metrics_interval_seconds\":%llu}\n",
	       (long)getpid(), struct_map_id, struct_link_id,
	       (unsigned long long)promote_after,
	       (unsigned long long)maximum_generation_gap,
	       (unsigned long long)metrics_interval);

	while (!exiting) {
		sleep((unsigned int)metrics_interval);
		if (!exiting)
			emit_metrics(skel, "metrics");
	}
	emit_metrics(skel, "final_metrics");

out:
	/* Destroy only the link owned by this loader process. */
	bpf_link__destroy(struct_link);
	eviction_2q_approx_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
