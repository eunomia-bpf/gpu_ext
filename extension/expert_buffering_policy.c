/* SPDX-License-Identifier: MIT */

#include <errno.h>
#include <linux/bpf.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "expert_buffering_policy.skel.h"

#define EXPERT_MAX_LAYOUT_BLOCKS 65536U

enum expert_block_class {
	EXPERT_BLOCK_DEFAULT = 0,
	EXPERT_BLOCK_COLD = 1,
	EXPERT_BLOCK_HOT = 2,
	EXPERT_BLOCK_SHARED = 3,
};

enum expert_policy_mode {
	EXPERT_POLICY_PAGE_LIFO = 1,
	EXPERT_POLICY_HOT_LIFO = 2,
	EXPERT_POLICY_PROTECT = 3,
	EXPERT_POLICY_OBSERVE = 4,
};

enum expert_policy_stat {
	EXPERT_STAT_ACTIVATE = 0,
	EXPERT_STAT_MAPPED,
	EXPERT_STAT_HOT_TAIL,
	EXPERT_STAT_COLD_HEAD,
	EXPERT_STAT_SHARED_TAIL,
	EXPERT_STAT_DEFAULT,
	EXPERT_STAT_SETTER_FAILURE,
	EXPERT_STAT_ACCESS,
	EXPERT_STAT_COLD_NATIVE,
	EXPERT_STAT_HOT_ACCESS_TAIL,
	EXPERT_STAT_SHARED_ACCESS_TAIL,
	EXPERT_STAT_OBSERVE_ACTIVATE,
	EXPERT_STAT_OBSERVE_ACCESS,
	EXPERT_STAT_MAX,
};

struct expert_layout_control {
	uint64_t base;
	uint32_t blocks;
	uint32_t mode;
	uint32_t ready;
	uint32_t reserved;
};

static volatile sig_atomic_t exiting;
static volatile sig_atomic_t snapshot_requested;

static void handle_signal(int signo)
{
	if (signo == SIGUSR1)
		snapshot_requested = 1;
	else
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

static int refuse_existing_struct_ops(void)
{
	uint32_t id = 0;

	for (;;) {
		uint32_t next;
		int fd;
		struct bpf_map_info info = {};
		uint32_t info_len = sizeof(info);

		if (bpf_map_get_next_id(id, &next) != 0) {
			if (errno == ENOENT)
				return 0;
			return -errno;
		}
		id = next;
		fd = bpf_map_get_fd_by_id(id);
		if (fd < 0)
			continue;
		if (bpf_map_get_info_by_fd(fd, &info, &info_len) == 0 &&
		    info.type == BPF_MAP_TYPE_STRUCT_OPS) {
			close(fd);
			fprintf(stderr, "refusing existing struct_ops map id %u\n", id);
			return -EBUSY;
		}
		close(fd);
	}
}

static int parse_mode(const char *arg, uint32_t *mode)
{
	if (!strcmp(arg, "page")) {
		*mode = EXPERT_POLICY_PAGE_LIFO;
		return 0;
	}
	if (!strcmp(arg, "hot")) {
		*mode = EXPERT_POLICY_HOT_LIFO;
		return 0;
	}
	if (!strcmp(arg, "protect")) {
		*mode = EXPERT_POLICY_PROTECT;
		return 0;
	}
	if (!strcmp(arg, "observe")) {
		*mode = EXPERT_POLICY_OBSERVE;
		return 0;
	}
	return -EINVAL;
}

static const char *mode_name(uint32_t mode)
{
	if (mode == EXPERT_POLICY_PAGE_LIFO)
		return "page";
	if (mode == EXPERT_POLICY_HOT_LIFO)
		return "hot";
	if (mode == EXPERT_POLICY_PROTECT)
		return "protect";
	return "observe";
}

static int read_class_table(const char *path,
			    uint8_t classes[EXPERT_MAX_LAYOUT_BLOCKS],
			    struct expert_layout_control *control,
			    uint64_t *hot_bytes,
			    uint32_t *nondefault)
{
	FILE *input;
	char line[256];
	unsigned long long base;
	unsigned long long protected_bytes;
	unsigned int blocks;
	unsigned int registrations;
	unsigned int line_number = 0;
	bool saw_header = false;

	input = fopen(path, "r");
	if (!input) {
		fprintf(stderr, "cannot open class table %s: %s\n", path, strerror(errno));
		return -errno;
	}
	while (fgets(line, sizeof(line), input)) {
		unsigned int index;
		unsigned int class_value;
		char extra;

		line_number++;
		if (line[0] == '#' || line[0] == '\n')
			continue;
		if (!saw_header) {
			if (sscanf(line,
				   "base %llu blocks %u hot_bytes %llu registrations %u %c",
				   &base, &blocks, &protected_bytes, &registrations,
				   &extra) != 4 || blocks == 0 ||
			    blocks > EXPERT_MAX_LAYOUT_BLOCKS) {
				fprintf(stderr, "invalid class-table header at line %u\n",
					line_number);
				fclose(input);
				return -EINVAL;
			}
			(void)registrations;
			control->base = base;
			control->blocks = blocks;
			*hot_bytes = protected_bytes;
			saw_header = true;
			continue;
		}
		if (sscanf(line, "%u %u %c", &index, &class_value, &extra) != 2 ||
		    index >= control->blocks || class_value < EXPERT_BLOCK_COLD ||
		    class_value > EXPERT_BLOCK_SHARED || classes[index] != 0) {
			fprintf(stderr, "invalid class-table entry at line %u\n", line_number);
			fclose(input);
			return -EINVAL;
		}
		classes[index] = (uint8_t)class_value;
		(*nondefault)++;
	}
	if (ferror(input)) {
		fprintf(stderr, "failed reading class table %s\n", path);
		fclose(input);
		return -EIO;
	}
	fclose(input);
	if (!saw_header || *nondefault == 0)
		return -EINVAL;
	return 0;
}

static int map_identity(int fd, uint32_t *id)
{
	struct bpf_map_info info = {};
	uint32_t info_len = sizeof(info);

	if (bpf_map_get_info_by_fd(fd, &info, &info_len) != 0)
		return -errno;
	*id = info.id;
	return 0;
}

static int program_identity(int fd, uint32_t *id)
{
	struct bpf_prog_info info = {};
	uint32_t info_len = sizeof(info);

	if (bpf_prog_get_info_by_fd(fd, &info, &info_len) != 0)
		return -errno;
	*id = info.id;
	return 0;
}

static void print_stats(int stats_fd)
{
	uint64_t totals[EXPERT_STAT_MAX] = {};
	int ncpus = libbpf_num_possible_cpus();
	uint64_t *percpu;
	uint32_t key;

	if (stats_fd < 0 || ncpus <= 0)
		return;
	percpu = calloc((size_t)ncpus, sizeof(*percpu));
	if (!percpu)
		return;
	for (key = 0; key < EXPERT_STAT_MAX; ++key) {
		int cpu;

		memset(percpu, 0, (size_t)ncpus * sizeof(*percpu));
		if (bpf_map_lookup_elem(stats_fd, &key, percpu) != 0)
			continue;
		for (cpu = 0; cpu < ncpus; ++cpu)
			totals[key] += percpu[cpu];
	}
	printf("{\"event\":\"policy_stats\",\"activate\":%llu,"
	       "\"mapped\":%llu,\"hot_tail\":%llu,\"cold_head\":%llu,"
	       "\"shared_tail\":%llu,\"default\":%llu,"
	       "\"setter_failure\":%llu,\"access\":%llu,"
	       "\"cold_native\":%llu,\"hot_access_tail\":%llu,"
	       "\"shared_access_tail\":%llu,\"observe_activate\":%llu,"
	       "\"observe_access\":%llu}\n",
	       (unsigned long long)totals[EXPERT_STAT_ACTIVATE],
	       (unsigned long long)totals[EXPERT_STAT_MAPPED],
	       (unsigned long long)totals[EXPERT_STAT_HOT_TAIL],
	       (unsigned long long)totals[EXPERT_STAT_COLD_HEAD],
	       (unsigned long long)totals[EXPERT_STAT_SHARED_TAIL],
	       (unsigned long long)totals[EXPERT_STAT_DEFAULT],
	       (unsigned long long)totals[EXPERT_STAT_SETTER_FAILURE],
	       (unsigned long long)totals[EXPERT_STAT_ACCESS],
	       (unsigned long long)totals[EXPERT_STAT_COLD_NATIVE],
	       (unsigned long long)totals[EXPERT_STAT_HOT_ACCESS_TAIL],
	       (unsigned long long)totals[EXPERT_STAT_SHARED_ACCESS_TAIL],
	       (unsigned long long)totals[EXPERT_STAT_OBSERVE_ACTIVATE],
	       (unsigned long long)totals[EXPERT_STAT_OBSERVE_ACCESS]);
	fflush(stdout);
	free(percpu);
}

static void print_activation_snapshot(int counts_fd,
				      const uint8_t *classes,
				      uint64_t base,
				      uint32_t blocks,
				      uint32_t ordinal)
{
	uint32_t index;
	uint32_t records = 0;

	printf("{\"event\":\"block_snapshot_begin\",\"ordinal\":%u,"
	       "\"base\":%llu,\"blocks\":%u}\n", ordinal,
	       (unsigned long long)base, blocks);
	for (index = 0; index < blocks; ++index) {
		uint64_t count = 0;

		if (classes[index] != EXPERT_BLOCK_HOT)
			continue;
		if (bpf_map_lookup_elem(counts_fd, &index, &count) != 0)
			continue;
		printf("{\"event\":\"hot_block_activation\",\"ordinal\":%u,"
		       "\"index\":%u,\"class\":%u,\"count\":%llu}\n",
		       ordinal, index, classes[index], (unsigned long long)count);
		records++;
	}
	printf("{\"event\":\"block_snapshot_end\",\"ordinal\":%u,"
	       "\"records\":%u}\n", ordinal, records);
	fflush(stdout);
}

int main(int argc, char **argv)
{
	struct expert_buffering_policy_bpf *skel = NULL;
	struct bpf_link *policy_link = NULL;
	uint8_t *classes = NULL;
	struct expert_layout_control control = {};
	uint64_t hot_bytes = 0;
	uint32_t nondefault = 0;
	uint32_t zero = 0;
	uint32_t map_id = 0;
	uint32_t program_id = 0;
	uint32_t counts_map_id = 0;
	uint32_t snapshot_ordinal = 0;
	uint32_t index;
	int classes_fd;
	int control_fd;
	int stats_fd;
	int counts_fd;
	int err;

	if (argc != 3) {
		fprintf(stderr,
			"usage: %s {page|hot|protect|observe} CLASS_TABLE\n",
			argv[0]);
		return 2;
	}
	if (parse_mode(argv[1], &control.mode)) {
		fprintf(stderr, "invalid mode %s\n", argv[1]);
		return 2;
	}
	classes = calloc(EXPERT_MAX_LAYOUT_BLOCKS, sizeof(*classes));
	if (!classes)
		return 1;
	err = read_class_table(argv[2], classes, &control, &hot_bytes, &nondefault);
	if (err)
		goto out;
	err = refuse_existing_struct_ops();
	if (err)
		goto out;

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	signal(SIGUSR1, handle_signal);
	libbpf_set_print(libbpf_print_fn);
	skel = expert_buffering_policy_bpf__open_and_load();
	if (!skel) {
		fprintf(stderr, "failed to open/load expert policy BPF\n");
		err = -EINVAL;
		goto out;
	}

	classes_fd = bpf_map__fd(skel->maps.block_classes);
	control_fd = bpf_map__fd(skel->maps.layout_control);
	stats_fd = bpf_map__fd(skel->maps.policy_stats);
	counts_fd = bpf_map__fd(skel->maps.activation_counts);
	for (index = 0; index < control.blocks; ++index) {
		if (!classes[index])
			continue;
		if (bpf_map_update_elem(classes_fd, &index, &classes[index], BPF_ANY) != 0) {
			err = -errno;
			fprintf(stderr, "failed to populate block class %u: %s\n",
				index, strerror(errno));
			goto out;
		}
	}
	control.ready = 1;
	if (bpf_map_update_elem(control_fd, &zero, &control, BPF_ANY) != 0) {
		err = -errno;
		fprintf(stderr, "failed to publish layout control: %s\n", strerror(errno));
		goto out;
	}

	policy_link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_expert_buffering);
	err = libbpf_get_error(policy_link);
	if (err) {
		policy_link = NULL;
		fprintf(stderr, "failed to attach expert policy: %s (%d)\n",
			strerror(-err), err);
		goto out;
	}
	map_identity(bpf_map__fd(skel->maps.uvm_ops_expert_buffering), &map_id);
	map_identity(counts_fd, &counts_map_id);
	program_identity(bpf_program__fd(skel->progs.gpu_block_activate), &program_id);
	printf("{\"event\":\"policy_ready\",\"mode\":\"%s\","
	       "\"layout_base\":%llu,\"layout_blocks\":%u,"
	       "\"classified_blocks\":%u,\"hot_bytes\":%llu,"
	       "\"struct_ops_map_id\":%u,\"activate_program_id\":%u,"
	       "\"activation_counts_map_id\":%u,\"pid\":%ld}\n",
	       mode_name(control.mode),
	       (unsigned long long)control.base, control.blocks, nondefault,
	       (unsigned long long)hot_bytes, map_id, program_id,
	       counts_map_id, (long)getpid());
	fflush(stdout);

	while (!exiting) {
		sleep(1);
		if (snapshot_requested) {
			snapshot_requested = 0;
			print_activation_snapshot(counts_fd, classes, control.base,
						  control.blocks,
						  ++snapshot_ordinal);
		}
		print_stats(stats_fd);
	}
	print_stats(stats_fd);
	err = 0;

out:
	bpf_link__destroy(policy_link);
	expert_buffering_policy_bpf__destroy(skel);
	free(classes);
	return err < 0 ? -err : err;
}
