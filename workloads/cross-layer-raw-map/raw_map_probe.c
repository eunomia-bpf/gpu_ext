#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <bpftime_gpu_ringbuf.h>

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { RING_CAPACITY = 4 };

struct raw_record {
	uint64_t sequence;
	uint64_t block_x;
	uint64_t block_y;
	uint64_t block_z;
	uint64_t thread_x;
	uint64_t thread_y;
	uint64_t thread_z;
};

struct aggregate_state {
	uint64_t callbacks;
	uint64_t sequence_sum;
	uint64_t block_x_sum;
	uint64_t thread_x_sum;
};

_Static_assert(sizeof(struct raw_record) == 56, "raw-record ABI changed");
_Static_assert(sizeof(struct aggregate_state) == 32, "aggregate ABI changed");

typedef int (*poll_gpu_fn)(int, void *,
			   void (*)(const void *, uint64_t, void *));
typedef int (*stats_gpu_fn)(int, struct bpftime_gpu_ringbuf_stats *);

static volatile sig_atomic_t drain_requested;
static volatile sig_atomic_t exiting;

struct callback_state {
	uint64_t records;
	uint64_t malformed;
};

static void on_signal(int signal_number)
{
	if (signal_number == SIGUSR1)
		drain_requested = 1;
	else
		exiting = 1;
}

static double monotonic_seconds(void)
{
	struct timespec value;
	clock_gettime(CLOCK_MONOTONIC, &value);
	return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static int parse_positive(const char *text, uint64_t *value)
{
	char *end = NULL;
	errno = 0;
	unsigned long long parsed = strtoull(text, &end, 10);
	if (errno || !end || *end != '\0' || parsed == 0)
		return -1;
	*value = (uint64_t)parsed;
	return 0;
}

static void raw_callback(const void *data, uint64_t size, void *opaque)
{
	struct callback_state *state = opaque;
	if (!state)
		return;
	if (!data || size != sizeof(struct raw_record)) {
		state->malformed++;
		return;
	}
	const struct raw_record *record = data;
	state->records++;
	printf(
		"{\"event\":\"raw_record\",\"sequence\":%" PRIu64
		",\"block_x\":%" PRIu64 ",\"block_y\":%" PRIu64
		",\"block_z\":%" PRIu64 ",\"thread_x\":%" PRIu64
		",\"thread_y\":%" PRIu64 ",\"thread_z\":%" PRIu64 "}\n",
		record->sequence, record->block_x, record->block_y,
		record->block_z, record->thread_x, record->thread_y,
		record->thread_z);
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr,
			"usage: %s BPF_OBJECT THREAD_SLOTS THREADS_PER_BLOCK LAUNCHES\n",
			argv[0]);
		return 64;
	}
	uint64_t thread_slots = 0, threads_per_block = 0, launches = 0;
	if (parse_positive(argv[2], &thread_slots)
			|| parse_positive(argv[3], &threads_per_block)
			|| parse_positive(argv[4], &launches)
			|| thread_slots % threads_per_block
			|| thread_slots > (1ULL << 20) || launches > 1024) {
		fprintf(stderr, "invalid expected geometry\n");
		return 65;
	}

	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);
	signal(SIGUSR1, on_signal);
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);

	struct bpf_object *object = bpf_object__open_file(argv[1], NULL);
	if (!object || libbpf_get_error(object)) {
		fprintf(stderr, "failed to open BPF object\n");
		return 3;
	}
	int result = 4;
	struct bpf_link *link = NULL;
	struct aggregate_state *states = NULL;
	if (bpf_object__load(object)) {
		fprintf(stderr, "failed to load BPF object\n");
		goto done;
	}
	struct bpf_program *program =
		bpf_object__find_program_by_name(object, "cuda__capture_return");
	struct bpf_map *raw_map =
		bpf_object__find_map_by_name(object, "raw_records");
	struct bpf_map *aggregate_map =
		bpf_object__find_map_by_name(object, "aggregate");
	if (!program || !raw_map || !aggregate_map) {
		fprintf(stderr, "BPF object inventory is incomplete\n");
		goto done;
	}
	link = bpf_program__attach(program);
	if (!link || libbpf_get_error(link)) {
		link = NULL;
		fprintf(stderr, "failed to attach BPF program\n");
		goto done;
	}

	poll_gpu_fn poll_gpu = (poll_gpu_fn)dlsym(
		RTLD_DEFAULT, "bpftime_syscall_server__poll_gpu_ringbuf_map");
	stats_gpu_fn stats_gpu = (stats_gpu_fn)dlsym(
		RTLD_DEFAULT, "bpftime_syscall_server__get_gpu_ringbuf_stats");
	if (!poll_gpu || !stats_gpu) {
		fprintf(stderr, "GPU ring-buffer poll/stats ABI is unavailable\n");
		goto done;
	}

	printf(
		"{\"event\":\"ready\",\"thread_slots\":%" PRIu64
		",\"threads_per_block\":%" PRIu64 ",\"launches\":%" PRIu64
		",\"ring_capacity_per_thread\":%d}\n",
		thread_slots, threads_per_block, launches, RING_CAPACITY);
	double deadline = monotonic_seconds() + 90.0;
	struct timespec delay = {.tv_sec = 0, .tv_nsec = 10000000};
	while (!drain_requested && !exiting && monotonic_seconds() < deadline)
		nanosleep(&delay, NULL);
	if (!drain_requested || exiting) {
		fprintf(stderr, "drain was not requested within the bounded lifetime\n");
		goto done;
	}

	if (thread_slots > SIZE_MAX / sizeof(*states)) {
		fprintf(stderr, "aggregate allocation overflow\n");
		goto done;
	}
	states = calloc((size_t)thread_slots, sizeof(*states));
	if (!states) {
		fprintf(stderr, "aggregate allocation failed\n");
		goto done;
	}

	const uint64_t expected_callbacks = thread_slots * launches;
	struct callback_state callback = {};
	struct bpftime_gpu_ringbuf_stats stats = {};
	deadline = monotonic_seconds() + 10.0;
	for (;;) {
		int polled = poll_gpu(bpf_map__fd(raw_map), &callback, raw_callback);
		if (polled < 0) {
			fprintf(stderr, "GPU ring-buffer poll failed: %d\n", polled);
			goto done;
		}
		if (bpf_map_lookup_elem(bpf_map__fd(aggregate_map),
					&((uint32_t){0}), states)) {
			fprintf(stderr, "aggregate lookup failed\n");
			goto done;
		}
		if (stats_gpu(bpf_map__fd(raw_map), &stats)) {
			fprintf(stderr, "GPU ring-buffer stats query failed\n");
			goto done;
		}
		uint64_t callbacks = 0;
		for (uint64_t slot = 0; slot < thread_slots; ++slot)
			callbacks += states[slot].callbacks;
		const uint64_t drops = stats.oob_drops + stats.full_drops
			+ stats.bad_size_drops + stats.other_drops;
		if (callbacks == expected_callbacks
				&& stats.committed_records + drops == expected_callbacks
				&& stats.pending_records == 0 && stats.dirty_slots == 0)
			break;
		if (monotonic_seconds() >= deadline) {
			fprintf(stderr, "raw/aggregate accounting did not converge\n");
			goto done;
		}
		nanosleep(&delay, NULL);
	}

	uint64_t callbacks = 0, sequence_sum = 0;
	uint64_t block_x_sum = 0, thread_x_sum = 0, slot_mismatches = 0;
	const uint64_t expected_sequence_sum = launches * (launches + 1) / 2;
	for (uint64_t slot = 0; slot < thread_slots; ++slot) {
		const uint64_t expected_block = slot / threads_per_block;
		const uint64_t expected_thread = slot % threads_per_block;
		const struct aggregate_state *state = &states[slot];
		callbacks += state->callbacks;
		sequence_sum += state->sequence_sum;
		block_x_sum += state->block_x_sum;
		thread_x_sum += state->thread_x_sum;
		if (state->callbacks != launches
				|| state->sequence_sum != expected_sequence_sum
				|| state->block_x_sum != launches * expected_block
				|| state->thread_x_sum != launches * expected_thread)
			slot_mismatches++;
	}
	printf(
		"{\"event\":\"aggregate_summary\",\"thread_slots\":%" PRIu64
		",\"checked_slots\":%" PRIu64 ",\"callbacks\":%" PRIu64
		",\"sequence_sum\":%" PRIu64 ",\"block_x_sum\":%" PRIu64
		",\"thread_x_sum\":%" PRIu64 ",\"slot_mismatches\":%" PRIu64 "}\n",
		thread_slots, thread_slots, callbacks, sequence_sum, block_x_sum,
		thread_x_sum, slot_mismatches);
	printf(
		"{\"event\":\"ring_summary\",\"value_size\":%" PRIu64
		",\"entries_per_thread\":%" PRIu64
		",\"allocated_thread_slots\":%" PRIu64
		",\"committed_records\":%" PRIu64
		",\"collected_records\":%" PRIu64
		",\"pending_records\":%" PRIu64
		",\"oob_drops\":%" PRIu64 ",\"full_drops\":%" PRIu64
		",\"bad_size_drops\":%" PRIu64 ",\"other_drops\":%" PRIu64
		",\"dirty_slots\":%" PRIu64 ",\"callback_records\":%" PRIu64
		",\"malformed_records\":%" PRIu64 "}\n",
		stats.value_size, stats.entries_per_thread,
		stats.allocated_thread_slots, stats.committed_records,
		stats.collected_records, stats.pending_records, stats.oob_drops,
		stats.full_drops, stats.bad_size_drops, stats.other_drops,
		stats.dirty_slots, callback.records, callback.malformed);
	if (slot_mismatches || callback.malformed
			|| callback.records != stats.collected_records) {
		fprintf(stderr, "observer validation failed\n");
		goto done;
	}
	result = 0;

done:
	free(states);
	if (link)
		bpf_link__destroy(link);
	bpf_object__close(object);
	return result;
}

