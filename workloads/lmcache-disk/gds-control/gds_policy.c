// SPDX-License-Identifier: GPL-2.0
/*
 * Minimal libbpf loader for the GDS gpu_storage_ops struct_ops policy.
 *
 * Opens gds_policy.bpf.o, loads it (struct_ops map gds_ops against the
 * live nvidia_uvm module BTF), attaches it with bpf_map__attach_struct_ops,
 * prints "attached", and holds the link until SIGINT/SIGTERM.
 */

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "libbpf.h"

#define DEFAULT_BPF_PATH	"gds_policy.bpf.o"
#define PROG_NAME		"gds_gpu_storage_decide"
#define OPS_MAP_NAME		"gds_ops"

static volatile sig_atomic_t exiting;

static void on_signal(int sig)
{
	(void)sig;
	exiting = 1;
}

static int fail(const char *what, int err)
{
	fprintf(stderr, "%s: %s\n", what, err ? strerror(err) : "failed");
	return 1;
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : DEFAULT_BPF_PATH;
	struct sigaction sa = { 0 };
	struct bpf_object *obj = NULL;
	struct bpf_program *prog;
	struct bpf_map *ops;
	struct bpf_link *link;
	int err;

	sa.sa_handler = on_signal;
	sigemptyset(&sa.sa_mask);
	if (sigaction(SIGINT, &sa, NULL) < 0 ||
	    sigaction(SIGTERM, &sa, NULL) < 0) {
		perror("sigaction");
		return 1;
	}

	obj = bpf_object__open_file(path, NULL);
	err = libbpf_get_error(obj);
	if (err)
		return fail("bpf_object__open_file", err);

	prog = bpf_object__find_program_by_name(obj, PROG_NAME);
	if (!prog) {
		bpf_object__close(obj);
		return fail("find program " PROG_NAME, 0);
	}

	ops = bpf_object__find_map_by_name(obj, OPS_MAP_NAME);
	if (!ops) {
		bpf_object__close(obj);
		return fail("find map " OPS_MAP_NAME, 0);
	}

	err = bpf_object__load(obj);
	if (err) {
		bpf_object__close(obj);
		return fail("bpf_object__load", -err);
	}

	link = bpf_map__attach_struct_ops(ops);
	if (!link) {
		bpf_object__close(obj);
		return fail("bpf_map__attach_struct_ops", errno);
	}

	printf("attached\n");
	fflush(stdout);

	while (!exiting)
		pause();

	bpf_link__destroy(link);
	bpf_object__close(obj);
	return 0;
}
