#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAP_ENTRIES 32U
#define LOOKUP_MAGIC UINT64_C(0x10c4000000000000)

static volatile sig_atomic_t stopping;

static const char *const modes[] = {
	"noop",          "device_update", "host_update", "rpc_update",
	"device_lookup", "host_lookup",   "rpc_lookup",
};

static void handle_signal(int signal_number)
{
	(void)signal_number;
	stopping = 1;
}

static double monotonic_seconds(void)
{
	struct timespec now;
	if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
		return -1.0;
	return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int known_mode(const char *mode)
{
	for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i)
		if (strcmp(mode, modes[i]) == 0)
			return 1;
	return 0;
}

static const char *program_name(const char *mode)
{
	if (strcmp(mode, "noop") == 0)
		return "cuda__noop";
	if (strcmp(mode, "device_update") == 0)
		return "cuda__device_update";
	if (strcmp(mode, "host_update") == 0)
		return "cuda__host_update";
	if (strcmp(mode, "rpc_update") == 0)
		return "cuda__rpc_update";
	if (strcmp(mode, "device_lookup") == 0)
		return "cuda__device_lookup";
	if (strcmp(mode, "host_lookup") == 0)
		return "cuda__host_lookup";
	return "cuda__rpc_lookup";
}

static const char *source_map_name(const char *mode)
{
	if (strncmp(mode, "device_", 7) == 0)
		return "device_values";
	if (strncmp(mode, "host_", 5) == 0)
		return "host_values";
	return "rpc_values";
}

static int initialize_lookup_map(struct bpf_object *object, const char *mode)
{
	if (!strstr(mode, "_lookup"))
		return 0;
	struct bpf_map *map = bpf_object__find_map_by_name(
		object, source_map_name(mode));
	if (!map)
		return -1;
	for (uint32_t key = 0; key < MAP_ENTRIES; ++key) {
		uint64_t value = LOOKUP_MAGIC ^ (uint64_t)key;
		if (bpf_map_update_elem(bpf_map__fd(map), &key, &value, BPF_ANY)) {
			fprintf(stderr, "failed to initialize %s key %u: %s\n",
				source_map_name(mode), key, strerror(errno));
			return -1;
		}
	}
	return 0;
}

static int emit_map(struct bpf_object *object, const char *mode)
{
	if (strcmp(mode, "noop") == 0)
		return 0;
	const char *name = strstr(mode, "_lookup") ? "observed_values"
						 : source_map_name(mode);
	struct bpf_map *map = bpf_object__find_map_by_name(object, name);
	if (!map)
		return -1;
	for (uint32_t key = 0; key < MAP_ENTRIES; ++key) {
		uint64_t value = 0;
		if (bpf_map_lookup_elem(bpf_map__fd(map), &key, &value)) {
			fprintf(stderr, "failed to read %s key %u: %s\n", name, key,
				strerror(errno));
			return -1;
		}
		printf("FIG15_MAP\t%s\t%u\t%" PRIu64 "\n", name, key, value);
	}
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 4 || !known_mode(argv[2])) {
		fprintf(stderr, "usage: %s OBJECT MODE TIMEOUT_SECONDS\n", argv[0]);
		return 2;
	}
	char *end = NULL;
	errno = 0;
	unsigned long timeout = strtoul(argv[3], &end, 10);
	if (errno || !end || *end || timeout < 1 || timeout > 3600)
		return 2;

	setvbuf(stdout, NULL, _IONBF, 0);
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	libbpf_set_strict_mode(LIBBPF_STRICT_ALL);

	struct bpf_object *object = bpf_object__open_file(argv[1], NULL);
	const long open_error = object ? libbpf_get_error(object) : -ENOMEM;
	if (!object || open_error) {
		fprintf(stderr, "failed to open BPF object %s: error=%ld (%s)\n",
			argv[1], open_error,
			open_error < 0 ? strerror((int)-open_error) : "unknown");
		return 3;
	}
	int result = 4;
	struct bpf_program *selected = NULL;
	struct bpf_program *program;
	size_t program_count = 0;
	bpf_object__for_each_program(program, object) {
		++program_count;
		const int enabled = strcmp(bpf_program__name(program),
					   program_name(argv[2])) == 0;
		bpf_program__set_autoload(program, enabled);
		if (enabled)
			selected = program;
	}
	if (program_count != 7 || !selected ||
	    strcmp(bpf_program__section_name(selected),
		   "kprobe/fig15_map_kernel") != 0) {
		fprintf(stderr, "unexpected BPF program inventory\n");
		goto done;
	}
	if (bpf_object__load(object)) {
		fprintf(stderr, "failed to load BPF object\n");
		goto done;
	}
	if (initialize_lookup_map(object, argv[2]))
		goto done;
	struct bpf_link *link = bpf_program__attach(selected);
	if (!link || libbpf_get_error(link)) {
		fprintf(stderr, "failed to attach %s\n", program_name(argv[2]));
		goto done;
	}
	printf("FIG15_READY\t%s\t1\n", argv[2]);

	const double deadline = monotonic_seconds() + (double)timeout;
	const struct timespec pause_time = {.tv_sec = 0, .tv_nsec = 100000000};
	while (!stopping && monotonic_seconds() < deadline)
		nanosleep(&pause_time, NULL);
	if (!stopping) {
		fprintf(stderr, "loader timed out\n");
		bpf_link__destroy(link);
		result = 5;
		goto done;
	}
	if (emit_map(object, argv[2])) {
		bpf_link__destroy(link);
		result = 6;
		goto done;
	}
	bpf_link__destroy(link);
	printf("FIG15_DETACHED\t1\n");
	result = 0;

done:
	bpf_object__close(object);
	return result;
}
