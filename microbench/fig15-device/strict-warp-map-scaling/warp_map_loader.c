#define _POSIX_C_SOURCE 200809L
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define WARP_MAP_ENTRIES 64

static volatile sig_atomic_t stopping;

static const char *const modes[] = {
	"noop", "shared_update", "warp_update",
};

static int capture_libbpf_log(enum libbpf_print_level level,
			      const char *format, va_list arguments)
{
	(void)level;
	return vfprintf(stderr, format, arguments);
}

static int prime_bpftime_server(void)
{
	int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "failed to prime syscall server: %s\n",
			strerror(errno));
		return -1;
	}
	if (close(fd) != 0) {
		fprintf(stderr, "failed to close syscall-server prime fd: %s\n",
			strerror(errno));
		return -1;
	}
	printf("FIG15_WARP_SERVER_PRIMED\t1\n");
	return 0;
}

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
	for (size_t index = 0; index < sizeof(modes) / sizeof(modes[0]); ++index)
		if (strcmp(mode, modes[index]) == 0)
			return 1;
	return 0;
}

static const char *program_name(const char *mode)
{
	if (strcmp(mode, "noop") == 0)
		return "cuda__noop";
	if (strcmp(mode, "shared_update") == 0)
		return "cuda__shared";
	return "cuda__warp";
}

static int initialize_map(struct bpf_object *object)
{
	struct bpf_map *map = bpf_object__find_map_by_name(object, "warp_values");
	if (!map) {
		fprintf(stderr, "failed to find warp_values\n");
		return -1;
	}
	const uint64_t zero = 0;
	for (uint32_t key = 0; key < WARP_MAP_ENTRIES; ++key) {
		if (bpf_map_update_elem(bpf_map__fd(map), &key, &zero, BPF_ANY)) {
			fprintf(stderr, "failed to initialize key %u: %s\n", key,
				strerror(errno));
			return -1;
		}
	}
	return 0;
}

static int emit_map(struct bpf_object *object)
{
	struct bpf_map *map = bpf_object__find_map_by_name(object, "warp_values");
	if (!map) {
		fprintf(stderr, "failed to find warp_values\n");
		return -1;
	}
	unsigned nonzero = 0;
	for (uint32_t key = 0; key < WARP_MAP_ENTRIES; ++key) {
		uint64_t value = 0;
		if (bpf_map_lookup_elem(bpf_map__fd(map), &key, &value)) {
			fprintf(stderr, "failed to read key %u: %s\n", key,
				strerror(errno));
			return -1;
		}
		if (value != 0) {
			printf("FIG15_WARP_MAP\t%u\t%" PRIu64 "\n", key, value);
			++nonzero;
		}
	}
	printf("FIG15_WARP_MAP_COUNT\t%u\n", nonzero);
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
	libbpf_set_print(capture_libbpf_log);
	if (prime_bpftime_server())
		return 7;

	errno = 0;
	struct bpf_object *object = bpf_object__open_file(argv[1], NULL);
	const int saved_open_errno = errno;
	const long open_error = object ? libbpf_get_error(object)
				       : -(long)(saved_open_errno ? saved_open_errno : EIO);
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
	if (program_count != 3 || !selected ||
	    strcmp(bpf_program__section_name(selected),
		   "kprobe/fig15_warp_map_kernel") != 0) {
		fprintf(stderr, "unexpected BPF program inventory\n");
		goto done;
	}
	if (bpf_object__load(object)) {
		fprintf(stderr, "failed to load BPF object\n");
		goto done;
	}
	if (initialize_map(object))
		goto done;

	struct bpf_link *link = bpf_program__attach(selected);
	if (!link || libbpf_get_error(link)) {
		fprintf(stderr, "failed to attach %s\n", program_name(argv[2]));
		goto done;
	}
	printf("FIG15_WARP_READY\t%s\t1\n", argv[2]);

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
	if (emit_map(object)) {
		bpf_link__destroy(link);
		result = 6;
		goto done;
	}
	bpf_link__destroy(link);
	printf("FIG15_WARP_DETACHED\t1\n");
	result = 0;

done:
	bpf_object__close(object);
	return result;
}
