/* SPDX-License-Identifier: GPL-2.0 */
#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <linux/bpf.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "revision_init_records.h"

static volatile sig_atomic_t exiting;

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

static struct bpf_map *find_rodata(struct bpf_object *object)
{
	struct bpf_map *map;

	bpf_object__for_each_map(map, object) {
		const char *name = bpf_map__name(map);
		size_t length = strlen(name);

		if (length >= 7 && !strcmp(name + length - 7, ".rodata"))
			return map;
	}
	return NULL;
}

static int map_id(int fd, uint32_t *id)
{
	struct bpf_map_info info = {};
	uint32_t size = sizeof(info);

	if (bpf_obj_get_info_by_fd(fd, &info, &size))
		return -errno;
	*id = info.id;
	return 0;
}

static int link_id(int fd, uint32_t *id)
{
	struct bpf_link_info info = {};
	uint32_t size = sizeof(info);

	if (bpf_obj_get_info_by_fd(fd, &info, &size))
		return -errno;
	*id = info.id;
	return 0;
}

static int dump_requests(int fd, unsigned int target_tgid,
			 unsigned long long *count)
{
	struct revision_init_key current;
	struct revision_init_key next;
	struct revision_init_key *previous = NULL;

	while (bpf_map_get_next_key(fd, previous, &next) == 0) {
		struct revision_init_record record;

		if (bpf_map_lookup_elem(fd, &next, &record))
			return -errno;
		if ((next.pid_tgid >> 32) != target_tgid)
			return -EPROTO;
		printf("{\"event\":\"scheduler_init_policy_request\","
		       "\"pid\":%llu,\"tid\":%llu,\"timestamp_ns\":%llu,"
		       "\"tsg_id\":%llu,\"runlist_id\":%u,\"engine_type\":%u,"
		       "\"default_timeslice\":%llu,\"default_interleave\":%u,"
		       "\"fixture\":%u,\"complete\":%u,"
		       "\"timeslice_count\":%u,\"timeslice_returns\":[%d,%d,%d],"
		       "\"interleave_count\":%u,\"interleave_returns\":[%d,%d,%d]}\n",
		       (unsigned long long)(next.pid_tgid >> 32),
		       (unsigned long long)(next.pid_tgid & 0xffffffffULL),
		       (unsigned long long)record.timestamp_ns,
		       (unsigned long long)next.tsg_id, next.runlist_id,
		       record.input.engine_type,
		       (unsigned long long)record.input.default_timeslice,
		       record.input.default_interleave, record.fixture, record.complete,
		       record.requests.timeslice_count,
		       record.requests.timeslice[0], record.requests.timeslice[1],
		       record.requests.timeslice[2],
		       record.requests.interleave_count,
		       record.requests.interleave[0], record.requests.interleave[1],
		       record.requests.interleave[2]);
		++*count;
		current = next;
		previous = &current;
	}
	return errno == ENOENT ? 0 : -errno;
}

int main(int argc, char **argv)
{
	struct bpf_object *object = NULL;
	struct bpf_map *rodata;
	struct bpf_map *ops;
	struct bpf_map *requests;
	struct bpf_map *stats_map;
	struct bpf_link *link = NULL;
	unsigned int target_tgid;
	unsigned int seconds = 120;
	uint32_t ops_map_id = 0;
	uint32_t ops_link_id = 0;
	uint64_t stats[INIT_STAT_COUNT] = {};
	unsigned long long records = 0;
	struct timespec start;
	struct timespec now;
	size_t rodata_size = 0;
	void *initial;
	int result = 1;
	long error;

	if (argc != 3 && argc != 4) {
		fprintf(stderr, "Usage: %s FIXTURE.bpf.o TARGET_TGID [SECONDS 1..3600]\n",
			argv[0]);
		return 2;
	}
	if (positive(argv[2], 0x7fffffffU, &target_tgid) ||
	    (argc == 4 && positive(argv[3], 3600, &seconds))) {
		fprintf(stderr, "invalid target TGID or duration\n");
		return 2;
	}

	object = bpf_object__open_file(argv[1], NULL);
	error = libbpf_get_error(object);
	if (error) {
		object = NULL;
		goto done;
	}
	rodata = find_rodata(object);
	if (!rodata)
		goto done;
	initial = bpf_map__initial_value(rodata, &rodata_size);
	/* Each frozen fixture has exactly one four-byte rodata variable. Refuse an
	 * altered object rather than guessing a variable offset. */
	if (!initial || rodata_size != sizeof(target_tgid))
		goto done;
	memcpy(initial, &target_tgid, sizeof(target_tgid));
	if (bpf_object__load(object))
		goto done;
	ops = bpf_object__find_map_by_name(object, "revision_init_ops");
	requests = bpf_object__find_map_by_name(object, "init_requests");
	stats_map = bpf_object__find_map_by_name(object, "init_stats");
	if (!ops || !requests || !stats_map ||
	    bpf_map__type(ops) != BPF_MAP_TYPE_STRUCT_OPS)
		goto done;
	link = bpf_map__attach_struct_ops(ops);
	error = libbpf_get_error(link);
	if (error) {
		link = NULL;
		goto done;
	}
	if (map_id(bpf_map__fd(ops), &ops_map_id) ||
	    link_id(bpf_link__fd(link), &ops_link_id))
		goto done;
	signal(SIGINT, stop);
	signal(SIGTERM, stop);
	printf("{\"event\":\"scheduler_init_loader_ready\",\"target_tgid\":%u,"
	       "\"struct_ops_map_id\":%u,\"struct_ops_link_id\":%u}\n",
	       target_tgid, ops_map_id, ops_link_id);
	fflush(stdout);
	clock_gettime(CLOCK_MONOTONIC, &start);
	do {
		struct timespec pause = { .tv_sec = 0, .tv_nsec = 100000000 };

		nanosleep(&pause, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
	} while (!exiting && now.tv_sec - start.tv_sec < seconds);

	if (dump_requests(bpf_map__fd(requests), target_tgid, &records))
		goto done;
	for (uint32_t key = 0; key < INIT_STAT_COUNT; ++key) {
		if (bpf_map_lookup_elem(bpf_map__fd(stats_map), &key, &stats[key]))
			goto done;
	}
	printf("{\"event\":\"scheduler_init_loader_summary\",\"target_tgid\":%u,"
	       "\"struct_ops_map_id\":%u,\"struct_ops_link_id\":%u,"
	       "\"init_seen\":%llu,\"init_recorded\":%llu,"
	       "\"init_record_error\":%llu,\"request_records\":%llu}\n",
	       target_tgid, ops_map_id, ops_link_id,
	       (unsigned long long)stats[INIT_SEEN],
	       (unsigned long long)stats[INIT_RECORDED],
	       (unsigned long long)stats[INIT_RECORD_ERROR], records);
	fflush(stdout);
	result = (!records || stats[INIT_SEEN] != records ||
		  stats[INIT_RECORDED] != records || stats[INIT_RECORD_ERROR]);
done:
	bpf_link__destroy(link);
	bpf_object__close(object);
	return result;
}
