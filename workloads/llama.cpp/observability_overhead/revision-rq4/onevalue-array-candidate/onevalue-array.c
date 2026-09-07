// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* One-value GPU_ARRAY map (type 1503) collector for the kernelretsnoop
 * warp-exit workload.
 *
 * The collector owns the skeleton, so the map (and its device buffer) is
 * created in this process at skeleton load time. A CUDA context is therefore
 * initialized here before map creation and kept alive through the final
 * whole-value drain lookup; after completion the host reads the entire single
 * map value with one bpf_map_lookup_elem(key=0) and reports the bulk-drain
 * bytes and wall time separately from prefill throughput.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "onevalue_array_value.h"
#include "./.output/onevalue-array.skel.h"

#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t exiting = 0;

static void sig_handler(int sig)
{
	exiting = true;
}

static CUcontext owner_cuda_context = NULL;

static bool init_owner_cuda_context(void)
{
	if (cuInit(0) != CUDA_SUCCESS) {
		warn("cuInit failed\n");
		return false;
	}
	CUdevice device;
	if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
		warn("cuDeviceGet failed\n");
		return false;
	}
	if (cuCtxCreate_v2(&owner_cuda_context, 0, device) != CUDA_SUCCESS ||
	    owner_cuda_context == NULL) {
		warn("cuCtxCreate failed\n");
		return false;
	}
	return true;
}

int main(void)
{
	/* Start the syscall server before cuInit: CUDA's first fopen otherwise
	 * re-enters the server's own cuInit while the driver is initializing. */
	uint32_t ignored_map_id = 0;
	(void)bpf_map_get_next_id(0, &ignored_map_id);
	if (!init_owner_cuda_context())
		return 1;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	struct onevalue_array_bpf *skel = NULL;
	int err = 0;
	int status = 1;

	skel = onevalue_array_bpf__open();
	if (!skel) {
		warn("Failed to open and load BPF skeleton\n");
		return 1;
	}
	err = onevalue_array_bpf__load(skel);
	if (err) {
		warn("Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}
	err = onevalue_array_bpf__attach(skel);
	if (err) {
		warn("Failed to attach BPF skeleton: %d\n", err);
		goto cleanup;
	}

	printf("One-value map type: %d\n", bpf_map__type(skel->maps.arena));
	printf("One-value map max entries: %u\n",
	       bpf_map__max_entries(skel->maps.arena));
	printf("One-value map value bytes: %u\n",
	       bpf_map__value_size(skel->maps.arena));
	printf("One-value probe attached; waiting for SIGINT\n");
	fflush(stdout);

	while (!exiting)
		usleep(100000);

	uint32_t key = 0;
	void *value = malloc(sizeof(struct onevalue_value));
	if (!value) {
		warn("Failed to allocate whole-value drain buffer\n");
		goto cleanup;
	}

	struct timespec drain_start, drain_end;
	clock_gettime(CLOCK_MONOTONIC, &drain_start);
	err = bpf_map_lookup_elem(bpf_map__fd(skel->maps.arena), &key, value);
	clock_gettime(CLOCK_MONOTONIC, &drain_end);
	if (err) {
		warn("Whole-value drain lookup failed: %d (%s)\n", err,
		     strerror(errno));
		free(value);
		goto cleanup;
	}
	const uint64_t drain_bytes = sizeof(struct onevalue_value);
	const uint64_t drain_time_ns =
		(uint64_t)(drain_end.tv_sec - drain_start.tv_sec) * 1000000000ULL +
		(uint64_t)(drain_end.tv_nsec - drain_start.tv_nsec);

	const struct onevalue_value *v = value;
	uint64_t counter_sum = 0;
	uint64_t active_warps = 0;
	uint64_t extent_x = 0;
	uint64_t nonzero_timestamps = 0;
	uint64_t mismatched_events = 0;
	for (uint64_t w = 0; w < ONEVALUE_ARRAY_MAX_WARPS; w++) {
		counter_sum += v->warp_counters[w];
		if (v->warp_counters[w] != 0)
			active_warps++;
	}
	for (uint64_t w = 0; w < ONEVALUE_ARRAY_MAX_WARPS; w++) {
		const uint64_t count = v->warp_counters[w];
		for (uint64_t k = 0; k < count; k++) {
			const struct onevalue_event *e =
				&v->events[w * ONEVALUE_ARRAY_MAX_EVENTS_PER_WARP +
					   k];
			nonzero_timestamps += e->timestamp != 0;
			if (e->coordinate_x != w || e->coordinate_y != 0 ||
			    e->coordinate_z != 0)
				mismatched_events++;
		}
		if (count > 0 && w + 1 > extent_x)
			extent_x = w + 1;
	}

	printf("One-value drain bytes: %" PRIu64 "\n", drain_bytes);
	printf("One-value drain time ns: %" PRIu64 "\n", drain_time_ns);
	printf("Warp append counters total: %" PRIu64 "\n", counter_sum);
	printf("Active warps: %" PRIu64 "\n", active_warps);
	printf("Coordinate extent x: %" PRIu64 "\n", extent_x);
	printf("Stored events: %" PRIu64 "\n", counter_sum);
	printf("Event coordinate mismatches: %" PRIu64 "\n",
	       mismatched_events);
	printf("Reserved total committed (device): %" PRIu64 "\n",
	       v->total_committed);
	printf("Overflow events: %" PRIu64 "\n", v->total_overflow);
	printf("Out-of-range events: %" PRIu64 "\n", v->total_out_of_range);
	printf("Nonzero timestamps: %" PRIu64 "\n", nonzero_timestamps);
	fflush(stdout);
	status = 0;
	free(value);

cleanup:
	onevalue_array_bpf__destroy(skel);
	if (owner_cuda_context)
		cuCtxDestroy(owner_cuda_context);
	return status;
}
