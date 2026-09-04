// SPDX-License-Identifier: MIT
/*
 * Diagnostic for the public NVIDIA 575 CPU/PTIMER correlation control.
 *
 * This is deliberately independent of bpftime and launchlate.  A passing run
 * admits implementation of the repaired clock path; it is not itself a
 * launch-latency result.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/ioctl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define NV_IOCTL_MAGIC 'F'
#define NV_IOCTL_BASE 200
#define NV_ESC_IOCTL_XFER_CMD (NV_IOCTL_BASE + 11)
#define NV_ESC_RM_FREE 0x29U
#define NV_ESC_RM_CONTROL 0x2aU
#define NV_ESC_RM_ALLOC 0x2bU

#define NV01_ROOT_CLIENT 0x00000041U
#define NV01_DEVICE_0 0x00000080U
#define NV20_SUBDEVICE_0 0x00002080U
#define NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO 0x20800406U
#define NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_ENDPOINTS_V1 \
	0x20800408U
#define NV2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_PLATFORM_API 0x03U
#define NV2080_CTRL_TIMER_GPU_CPU_TIME_MAX_SAMPLES 16U

#define DEFAULT_SAMPLES 200U
#define MAX_SAMPLES 100000U
#define MAX_OUTER_DURATION_NS 10000000ULL
#define TARGET_MEDIAN_BRACKET_NS 1500ULL
#define PTIMER_QUANTIZATION_NS 32ULL

typedef struct {
	uint32_t cmd;
	uint32_t size;
	void *ptr __attribute__((aligned(8)));
} nv_ioctl_xfer_t;

typedef struct {
	uint32_t hRoot;
	uint32_t hObjectParent;
	uint32_t hObjectNew;
	uint32_t hClass;
	void *pAllocParms __attribute__((aligned(8)));
	void *pRightsRequested __attribute__((aligned(8)));
	uint32_t paramsSize;
	uint32_t flags;
	uint32_t status;
} nvos64_parameters;

typedef struct {
	uint32_t hRoot;
	uint32_t hObjectParent;
	uint32_t hObjectOld;
	uint32_t status;
} nvos00_parameters;

typedef struct {
	uint32_t hClient;
	uint32_t hObject;
	uint32_t cmd;
	uint32_t flags;
	void *params __attribute__((aligned(8)));
	uint32_t paramsSize;
	uint32_t status;
} nvos54_parameters;

typedef struct {
	uint32_t deviceId;
	uint32_t hClientShare;
	uint32_t hTargetClient;
	uint32_t hTargetDevice;
	uint32_t flags;
	uint64_t vaSpaceSize __attribute__((aligned(8)));
	uint64_t vaStartInternal __attribute__((aligned(8)));
	uint64_t vaLimitInternal __attribute__((aligned(8)));
	uint32_t vaMode;
} nv0080_alloc_parameters;

typedef struct {
	uint32_t subDeviceId;
} nv2080_alloc_parameters;

typedef struct {
	uint64_t cpuTime __attribute__((aligned(8)));
	uint64_t gpuTime __attribute__((aligned(8)));
} nv2080_timer_gpu_cpu_time_sample;

typedef struct {
	uint8_t cpuClkId;
	uint8_t sampleCount;
	nv2080_timer_gpu_cpu_time_sample
		samples[NV2080_CTRL_TIMER_GPU_CPU_TIME_MAX_SAMPLES]
			__attribute__((aligned(8)));
} nv2080_timer_correlation_parameters;

typedef struct {
	uint64_t cpuBeforeNs __attribute__((aligned(8)));
	uint64_t gpuTimeNs __attribute__((aligned(8)));
	uint64_t cpuAfterNs __attribute__((aligned(8)));
} nv2080_timer_correlation_endpoints_v1_parameters;

_Static_assert(sizeof(nv_ioctl_xfer_t) == 16, "575 xfer ABI size");
_Static_assert(offsetof(nv_ioctl_xfer_t, ptr) == 8, "575 xfer pointer offset");
_Static_assert(sizeof(nvos64_parameters) == 48, "575 NVOS64 ABI size");
_Static_assert(offsetof(nvos64_parameters, pAllocParms) == 16,
	       "575 NVOS64 allocation pointer offset");
_Static_assert(offsetof(nvos64_parameters, status) == 40,
	       "575 NVOS64 status offset");
_Static_assert(sizeof(nvos00_parameters) == 16, "575 NVOS00 ABI size");
_Static_assert(sizeof(nvos54_parameters) == 32, "575 NVOS54 ABI size");
_Static_assert(offsetof(nvos54_parameters, params) == 16,
	       "575 NVOS54 parameter pointer offset");
_Static_assert(sizeof(nv0080_alloc_parameters) == 56,
	       "575 NV0080 allocation ABI size");
_Static_assert(sizeof(nv2080_alloc_parameters) == 4,
	       "575 NV2080 allocation ABI size");
_Static_assert(sizeof(nv2080_timer_gpu_cpu_time_sample) == 16,
	       "575 timer sample ABI size");
_Static_assert(offsetof(nv2080_timer_correlation_parameters, samples) == 8,
	       "575 timer sample-array offset");
_Static_assert(sizeof(nv2080_timer_correlation_parameters) == 264,
	       "575 timer correlation ABI size");
_Static_assert(sizeof(nv2080_timer_correlation_endpoints_v1_parameters) == 24,
	       "575 endpoint-v1 ABI size");
_Static_assert(_Alignof(nv2080_timer_correlation_endpoints_v1_parameters) == 8,
	       "575 endpoint-v1 ABI alignment");
_Static_assert(offsetof(nv2080_timer_correlation_endpoints_v1_parameters,
			cpuBeforeNs) == 0,
	       "575 endpoint-v1 before offset");
_Static_assert(offsetof(nv2080_timer_correlation_endpoints_v1_parameters,
			gpuTimeNs) == 8,
	       "575 endpoint-v1 GPU offset");
_Static_assert(offsetof(nv2080_timer_correlation_endpoints_v1_parameters,
			cpuAfterNs) == 16,
	       "575 endpoint-v1 after offset");

struct rm_handles {
	uint32_t root;
	uint32_t device;
	uint32_t subdevice;
};

struct observation {
	uint64_t before;
	uint64_t after;
	uint64_t midpoint;
	uint64_t gpu;
	uint64_t outer_width;
	uint64_t max_selected_gap;
	uint64_t lower_cpu;
	uint64_t upper_cpu;
	int64_t offset_low;
	int64_t offset_high;
	uint64_t bracket_width;
};

enum control_transport {
	CONTROL_TRANSPORT_XFER = 0,
	CONTROL_TRANSPORT_DIRECT = 1,
};

enum correlation_command {
	CORRELATION_COMMAND_PUBLIC = 0,
	CORRELATION_COMMAND_ENDPOINTS_V1 = 1,
};

static int monotonic_raw_ns(uint64_t *value)
{
	struct timespec now;

	if (!value || clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0)
		return -errno;
	if (now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
	    (uint64_t)now.tv_sec > UINT64_MAX / 1000000000ULL)
		return -ERANGE;
	*value = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
	return 0;
}

static int rm_xfer(int fd, uint32_t command, void *payload, uint32_t size)
{
	nv_ioctl_xfer_t xfer = {
		.cmd = command,
		.size = size,
		.ptr = payload,
	};
	unsigned long request = _IOWR(NV_IOCTL_MAGIC, NV_ESC_IOCTL_XFER_CMD,
				      nv_ioctl_xfer_t);

	if (ioctl(fd, request, &xfer) < 0)
		return -errno;
	return 0;
}

static int rm_control_direct(int fd, nvos54_parameters *control)
{
	unsigned long request = _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL,
				      nvos54_parameters);

	if (ioctl(fd, request, control) < 0)
		return -errno;
	return 0;
}

static int rm_alloc(int fd, uint32_t root, uint32_t parent, uint32_t object_class,
		    void *class_params, uint32_t class_params_size,
		    uint32_t *object, uint32_t *rm_status)
{
	nvos64_parameters alloc = {
		.hRoot = root,
		.hObjectParent = parent,
		.hObjectNew = 0,
		.hClass = object_class,
		.pAllocParms = class_params,
		.pRightsRequested = NULL,
		.paramsSize = class_params_size,
		.flags = 0,
		.status = 0,
	};
	int err;

	if (!object || !rm_status)
		return -EINVAL;
	err = rm_xfer(fd, NV_ESC_RM_ALLOC, &alloc, sizeof(alloc));
	*rm_status = alloc.status;
	*object = alloc.hObjectNew;
	if (err)
		return err;
	if (alloc.status != 0 || alloc.hObjectNew == 0)
		return -EREMOTEIO;
	return 0;
}

static int rm_free_root(int fd, uint32_t root, uint32_t *rm_status)
{
	nvos00_parameters free_args = {
		.hRoot = root,
		.hObjectParent = root,
		.hObjectOld = root,
		.status = 0,
	};
	int err;

	err = rm_xfer(fd, NV_ESC_RM_FREE, &free_args, sizeof(free_args));
	if (rm_status)
		*rm_status = free_args.status;
	if (err)
		return err;
	return free_args.status == 0 ? 0 : -EREMOTEIO;
}

static int rm_open_timer(int fd, struct rm_handles *handles,
			 uint32_t *failed_class, uint32_t *rm_status)
{
	nv0080_alloc_parameters device_params = { .deviceId = 0 };
	nv2080_alloc_parameters subdevice_params = { .subDeviceId = 0 };
	int err;

	memset(handles, 0, sizeof(*handles));
	*failed_class = NV01_ROOT_CLIENT;
	err = rm_alloc(fd, 0, 0, NV01_ROOT_CLIENT, NULL, 0,
		       &handles->root, rm_status);
	if (err)
		return err;

	device_params.hClientShare = handles->root;
	*failed_class = NV01_DEVICE_0;
	err = rm_alloc(fd, handles->root, handles->root, NV01_DEVICE_0,
		       &device_params, sizeof(device_params), &handles->device,
		       rm_status);
	if (err)
		return err;

	*failed_class = NV20_SUBDEVICE_0;
	err = rm_alloc(fd, handles->root, handles->device, NV20_SUBDEVICE_0,
		       &subdevice_params, sizeof(subdevice_params),
		       &handles->subdevice, rm_status);
	return err;
}

static int rm_timer_control(int fd, const struct rm_handles *handles,
			    uint32_t command, void *params,
			    uint32_t params_size,
			    enum control_transport transport,
			    uint32_t *rm_status)
{
	nvos54_parameters control = {
		.hClient = handles->root,
		.hObject = handles->subdevice,
		.cmd = command,
		.flags = 0,
		.params = params,
		.paramsSize = params_size,
		.status = 0,
	};
	int err = transport == CONTROL_TRANSPORT_DIRECT ?
			rm_control_direct(fd, &control) :
			rm_xfer(fd, NV_ESC_RM_CONTROL, &control, sizeof(control));

	*rm_status = control.status;
	if (err)
		return err;
	return control.status == 0 ? 0 : -EREMOTEIO;
}

static int checked_offset(uint64_t gpu, uint64_t cpu, int64_t padding,
			  int64_t *result)
{
	__int128 value = (__int128)gpu - (__int128)cpu + (__int128)padding;

	if (value < INT64_MIN || value > INT64_MAX)
		return -ERANGE;
	*result = (int64_t)value;
	return 0;
}

static int derive_observation(uint64_t before, uint64_t after,
			      uint64_t midpoint, uint64_t gpu,
			      struct observation *out)
{
	uint64_t half_low, half_high, low_candidate, high_candidate;
	uint64_t width;

	if (!out || after < before || midpoint < before || midpoint > after ||
	    midpoint == 0 || gpu == 0)
		return -ERANGE;
	width = after - before;
	if (width >= MAX_OUTER_DURATION_NS)
		return -ETIMEDOUT;

	out->before = before;
	out->after = after;
	out->midpoint = midpoint;
	out->gpu = gpu;
	out->outer_width = width;
	out->max_selected_gap = width / 3;
	half_low = out->max_selected_gap / 2;
	half_high = out->max_selected_gap / 2 +
		    out->max_selected_gap % 2;
	low_candidate = midpoint >= half_low ? midpoint - half_low : 0;
	high_candidate = midpoint <= UINT64_MAX - half_high ?
			 midpoint + half_high : UINT64_MAX;
	out->lower_cpu = low_candidate > before ? low_candidate : before;
	out->upper_cpu = high_candidate < after ? high_candidate : after;
	if (out->lower_cpu > out->upper_cpu ||
	    checked_offset(gpu, out->upper_cpu,
			   -(int64_t)PTIMER_QUANTIZATION_NS,
			   &out->offset_low) ||
	    checked_offset(gpu, out->lower_cpu,
			   (int64_t)PTIMER_QUANTIZATION_NS,
			   &out->offset_high) ||
	    out->offset_high < out->offset_low)
		return -ERANGE;
	out->bracket_width = (uint64_t)out->offset_high -
			     (uint64_t)out->offset_low;
	return 0;
}

static int derive_endpoint_observation(uint64_t before, uint64_t after,
				       uint64_t cpu_before,
				       uint64_t cpu_after, uint64_t gpu,
				       struct observation *out)
{
	uint64_t selected_gap;

	if (!out || after < before || cpu_before < before || cpu_after < cpu_before ||
	    cpu_after > after || cpu_before == 0 || gpu == 0)
		return -ERANGE;
	if (after - before >= MAX_OUTER_DURATION_NS)
		return -ETIMEDOUT;
	selected_gap = cpu_after - cpu_before;
	out->before = before;
	out->after = after;
	out->midpoint = cpu_before + selected_gap / 2;
	out->gpu = gpu;
	out->outer_width = after - before;
	out->max_selected_gap = selected_gap;
	out->lower_cpu = cpu_before;
	out->upper_cpu = cpu_after;
	if (checked_offset(gpu, cpu_after,
			   -(int64_t)PTIMER_QUANTIZATION_NS,
			   &out->offset_low) ||
	    checked_offset(gpu, cpu_before,
			   (int64_t)PTIMER_QUANTIZATION_NS,
			   &out->offset_high) ||
	    out->offset_high < out->offset_low)
		return -ERANGE;
	out->bracket_width = (uint64_t)out->offset_high -
			     (uint64_t)out->offset_low;
	return 0;
}

static int compare_u64(const void *left, const void *right)
{
	uint64_t a = *(const uint64_t *)left;
	uint64_t b = *(const uint64_t *)right;

	return (a > b) - (a < b);
}

static uint64_t median_u64(uint64_t *values, unsigned int count)
{
	qsort(values, count, sizeof(*values), compare_u64);
	if (count % 2)
		return values[count / 2];
	return values[count / 2 - 1] / 2 + values[count / 2] / 2 +
	       ((values[count / 2 - 1] % 2 + values[count / 2] % 2) / 2);
}

static int parse_samples(const char *text, unsigned int *samples)
{
	char *end = NULL;
	unsigned long value;

	errno = 0;
	value = strtoul(text, &end, 10);
	if (errno || !end || *end != '\0' || value == 0 || value > MAX_SAMPLES)
		return -EINVAL;
	*samples = (unsigned int)value;
	return 0;
}

static void usage(const char *program)
{
	fprintf(stderr,
		"Usage: %s [--samples N] [--control-transport xfer|direct] "
		"[--correlation-command public|endpoints-v1] | "
		"--self-test\n",
		program);
}

static int self_test(void)
{
	struct observation observation = { 0 };
	int64_t signed_value = 0;

	if (derive_observation(1000, 1300, 1150, 2000, &observation) ||
	    observation.outer_width != 300 ||
	    observation.max_selected_gap != 100 ||
	    observation.lower_cpu != 1100 || observation.upper_cpu != 1200 ||
	    observation.offset_low != 768 || observation.offset_high != 932 ||
	    observation.bracket_width != 164)
		return 1;
	if (derive_observation(1000, 1303, 1151, 2000, &observation) ||
	    observation.max_selected_gap != 101 ||
	    observation.lower_cpu != 1101 || observation.upper_cpu != 1202 ||
	    observation.offset_low != 766 || observation.offset_high != 931 ||
	    observation.bracket_width != 165)
		return 1;
	if (derive_observation(1000, 1303, 1151, 500, &observation) ||
	    observation.offset_low != -734 || observation.offset_high != -569 ||
	    observation.bracket_width != 165)
		return 1;
	if (derive_observation(1000, 1300, 999, 2000, &observation) != -ERANGE ||
	    derive_observation(1000, 1000 + MAX_OUTER_DURATION_NS, 1100,
			       2000, &observation) != -ETIMEDOUT ||
	    checked_offset(UINT64_MAX, 0, 0, &signed_value) != -ERANGE)
		return 1;
	if (derive_endpoint_observation(900, 1300, 1000, 1100, 2000,
					&observation) ||
	    observation.outer_width != 400 ||
	    observation.max_selected_gap != 100 ||
	    observation.midpoint != 1050 || observation.lower_cpu != 1000 ||
	    observation.upper_cpu != 1100 || observation.offset_low != 868 ||
	    observation.offset_high != 1032 || observation.bracket_width != 164)
		return 1;
	if (derive_endpoint_observation(900, 1300, 1000, 1101, 500,
					&observation) ||
	    observation.max_selected_gap != 101 ||
	    observation.midpoint != 1050 || observation.offset_low != -633 ||
	    observation.offset_high != -468 || observation.bracket_width != 165)
		return 1;
	if (derive_endpoint_observation(1001, 1300, 1000, 1100, 2000,
					&observation) != -ERANGE ||
	    derive_endpoint_observation(900, 1099, 1000, 1100, 2000,
					&observation) != -ERANGE ||
	    derive_endpoint_observation(1300, 900, 1000, 1100, 2000,
					&observation) != -ERANGE ||
	    derive_endpoint_observation(900, 1300, 1100, 1000, 2000,
					&observation) != -ERANGE ||
	    derive_endpoint_observation(0, 1300, 0, 1000, 2000,
					&observation) != -ERANGE ||
	    derive_endpoint_observation(1000,
					1000 + MAX_OUTER_DURATION_NS,
					1100, 1200, 2000,
					&observation) != -ETIMEDOUT)
		return 1;
	printf("rm_ptimer_correlation_sanity self-test: PASS\n");
	return 0;
}

int main(int argc, char **argv)
{
	struct rm_handles handles = { 0 };
	unsigned int requested = DEFAULT_SAMPLES;
	enum control_transport transport = CONTROL_TRANSPORT_XFER;
	enum correlation_command correlation_command = CORRELATION_COMMAND_PUBLIC;
	const char *transport_name = "xfer";
	const char *correlation_command_name = "public";
	unsigned int attempted = 0, accepted = 0, rejected = 0;
	unsigned int cpu_midpoint_regressions = 0, ptimer_regressions = 0;
	uint64_t *widths = NULL, *outer_widths = NULL;
	uint64_t min_width = UINT64_MAX, max_width = 0;
	uint64_t min_outer = UINT64_MAX, max_outer = 0;
	uint64_t median_width = 0, median_outer = 0;
	uint64_t previous_cpu = 0, previous_gpu = 0;
	uint32_t failed_class = 0, rm_status = 0, free_status = 0;
	const char *setup_stage = "host_allocation";
	int have_previous = 0, gate_pass = 0;
	int fd = -1, gpu_fd = -1, err = 0, setup_error = 0, cleanup_error = 0;
	int output_error = 0, exit_code = 1;

	if (argc == 2 && strcmp(argv[1], "--self-test") == 0)
		return self_test();
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
			if (parse_samples(argv[++i], &requested)) {
				usage(argv[0]);
				return 1;
			}
		} else if (strcmp(argv[i], "--control-transport") == 0 &&
			   i + 1 < argc) {
			const char *value = argv[++i];
			if (strcmp(value, "xfer") == 0) {
				transport = CONTROL_TRANSPORT_XFER;
				transport_name = "xfer";
			} else if (strcmp(value, "direct") == 0) {
				transport = CONTROL_TRANSPORT_DIRECT;
				transport_name = "direct";
			} else {
				usage(argv[0]);
				return 1;
			}
		} else if (strcmp(argv[i], "--correlation-command") == 0 &&
			   i + 1 < argc) {
			const char *value = argv[++i];
			if (strcmp(value, "public") == 0) {
				correlation_command = CORRELATION_COMMAND_PUBLIC;
				correlation_command_name = "public";
			} else if (strcmp(value, "endpoints-v1") == 0) {
				correlation_command = CORRELATION_COMMAND_ENDPOINTS_V1;
				correlation_command_name = "endpoints-v1";
			} else {
				usage(argv[0]);
				return 1;
			}
		} else {
			usage(argv[0]);
			return 1;
		}
	}

	widths = calloc(requested, sizeof(*widths));
	outer_widths = calloc(requested, sizeof(*outer_widths));
	if (!widths || !outer_widths) {
		fprintf(stderr, "allocation failed\n");
		setup_error = -ENOMEM;
		goto finalize;
	}

	fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "open /dev/nvidiactl failed: %s\n", strerror(errno));
		setup_stage = "open_nvidiactl";
		setup_error = -errno;
		goto finalize;
	}
	gpu_fd = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
	if (gpu_fd < 0) {
		fprintf(stderr, "open /dev/nvidia0 failed: %s\n", strerror(errno));
		setup_stage = "open_nvidia0";
		setup_error = -errno;
		goto finalize;
	}
	err = rm_open_timer(fd, &handles, &failed_class, &rm_status);
	if (err) {
		fprintf(stderr,
			"RM object allocation failed: class=0x%08" PRIx32
			" syscall_error=%d rm_status=0x%08" PRIx32 "\n",
			failed_class, err, rm_status);
		setup_stage = failed_class == NV01_ROOT_CLIENT ? "rm_alloc_root" :
			      failed_class == NV01_DEVICE_0 ? "rm_alloc_device" :
			      "rm_alloc_subdevice";
		setup_error = err;
		goto finalize;
	}
	setup_stage = "samples";
	fprintf(stderr,
		"RM timer objects ready: root=0x%08" PRIx32
		" device=0x%08" PRIx32 " subdevice=0x%08" PRIx32 "\n",
		handles.root, handles.device, handles.subdevice);

	for (unsigned int i = 0; i < requested; ++i) {
		nv2080_timer_correlation_parameters public_params = {
			.cpuClkId =
				NV2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_PLATFORM_API,
			.sampleCount = 1,
		};
		nv2080_timer_correlation_endpoints_v1_parameters endpoint_params = { 0 };
		struct observation observation = { 0 };
		uint32_t control_command =
			NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO;
		void *control_params = &public_params;
		uint32_t control_params_size = sizeof(public_params);
		uint64_t rm_cpu_before = 0, rm_cpu_after = 0;
		uint64_t rm_cpu_midpoint = 0, rm_gpu = 0;
		uint64_t before = 0, after = 0;
		int before_err, control_err, after_err, derive_err = 0;
		int cpu_regression = 0, gpu_regression = 0;
		int written;

		if (correlation_command == CORRELATION_COMMAND_ENDPOINTS_V1) {
			control_command =
				NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_ENDPOINTS_V1;
			control_params = &endpoint_params;
			control_params_size = sizeof(endpoint_params);
		}
		++attempted;
		rm_status = 0;
		before_err = monotonic_raw_ns(&before);
		control_err = before_err ? -ECANCELED :
			rm_timer_control(fd, &handles, control_command, control_params,
					 control_params_size, transport, &rm_status);
		after_err = monotonic_raw_ns(&after);
		if (correlation_command == CORRELATION_COMMAND_ENDPOINTS_V1) {
			rm_cpu_before = endpoint_params.cpuBeforeNs;
			rm_cpu_after = endpoint_params.cpuAfterNs;
			rm_gpu = endpoint_params.gpuTimeNs;
			if (rm_cpu_after >= rm_cpu_before)
				rm_cpu_midpoint = rm_cpu_before +
					(rm_cpu_after - rm_cpu_before) / 2;
			if (!before_err && !control_err && !after_err)
				derive_err = derive_endpoint_observation(
					before, after, rm_cpu_before, rm_cpu_after,
					rm_gpu, &observation);
		} else {
			rm_cpu_midpoint = public_params.samples[0].cpuTime;
			rm_gpu = public_params.samples[0].gpuTime;
			if (!before_err && !control_err && !after_err)
				derive_err = derive_observation(
					before, after, rm_cpu_midpoint, rm_gpu,
					&observation);
		}

		if (before_err || control_err || after_err || derive_err) {
			++rejected;
			written = printf("{\"record\":\"sample\",\"index\":%u,"
			       "\"control_transport\":\"%s\","
			       "\"correlation_command\":\"%s\","
			       "\"valid\":false,\"before_error\":%d,"
			       "\"control_error\":%d,\"after_error\":%d,"
			       "\"derive_error\":%d,\"rm_status\":%" PRIu32
			       ",\"host_before_ns\":%" PRIu64
			       ",\"host_after_ns\":%" PRIu64
			       ",\"rm_cpu_before_ns\":%" PRIu64
			       ",\"rm_cpu_midpoint_ns\":%" PRIu64
			       ",\"rm_cpu_after_ns\":%" PRIu64
			       ",\"rm_gpu_ptimer_ns\":%" PRIu64 "}\n",
			       i, transport_name, correlation_command_name,
			       before_err, control_err, after_err,
			       derive_err,
			       rm_status, before, after, rm_cpu_before,
			       rm_cpu_midpoint, rm_cpu_after, rm_gpu);
			if (written < 0) {
				output_error = -EIO;
				break;
			}
			continue;
		}

		cpu_regression = have_previous && observation.midpoint < previous_cpu;
		gpu_regression = have_previous && observation.gpu < previous_gpu;
		cpu_midpoint_regressions += cpu_regression;
		ptimer_regressions += gpu_regression;
		previous_cpu = observation.midpoint;
		previous_gpu = observation.gpu;
		have_previous = 1;
		widths[accepted] = observation.bracket_width;
		outer_widths[accepted] = observation.outer_width;
		++accepted;
		if (observation.bracket_width < min_width)
			min_width = observation.bracket_width;
		if (observation.bracket_width > max_width)
			max_width = observation.bracket_width;
		if (observation.outer_width < min_outer)
			min_outer = observation.outer_width;
		if (observation.outer_width > max_outer)
			max_outer = observation.outer_width;
		written = printf("{\"record\":\"sample\",\"index\":%u,"
		       "\"control_transport\":\"%s\","
		       "\"correlation_command\":\"%s\","
		       "\"valid\":true,\"cpu_midpoint_regression\":%s,"
		       "\"ptimer_regression\":%s,"
		       "\"rm_status\":0,\"host_before_ns\":%" PRIu64
		       ",\"host_after_ns\":%" PRIu64
		       ",\"rm_cpu_before_ns\":%" PRIu64
		       ",\"rm_cpu_midpoint_ns\":%" PRIu64
		       ",\"rm_cpu_after_ns\":%" PRIu64
		       ",\"rm_gpu_ptimer_ns\":%" PRIu64
		       ",\"outer_width_ns\":%" PRIu64
		       ",\"max_selected_gap_ns\":%" PRIu64
		       ",\"cpu_lower_ns\":%" PRIu64
		       ",\"cpu_upper_ns\":%" PRIu64
		       ",\"offset_low_ns\":%" PRId64
		       ",\"offset_high_ns\":%" PRId64
		       ",\"bracket_width_ns\":%" PRIu64 "}\n",
		       i, transport_name, correlation_command_name,
		       cpu_regression ? "true" : "false",
		       gpu_regression ? "true" : "false", observation.before,
		       observation.after, rm_cpu_before,
		       observation.midpoint, rm_cpu_after,
		       observation.gpu,
		       observation.outer_width, observation.max_selected_gap,
		       observation.lower_cpu, observation.upper_cpu,
		       observation.offset_low, observation.offset_high,
		       observation.bracket_width);
		if (written < 0) {
			output_error = -EIO;
			break;
		}
	}

	if (accepted > 0) {
		median_width = median_u64(widths, accepted);
		median_outer = median_u64(outer_widths, accepted);
	}

finalize:
	if (fflush(stdout) != 0 && output_error == 0)
		output_error = errno ? -errno : -EIO;
	if (fd >= 0 && handles.root != 0) {
		int free_err = rm_free_root(fd, handles.root, &free_status);
		if (free_err) {
			fprintf(stderr,
				"RM root cleanup failed: syscall_error=%d rm_status=0x%08"
				PRIx32 "\n",
				free_err, free_status);
			cleanup_error = free_err;
		}
	}
	if (fd >= 0)
		close(fd);
	if (gpu_fd >= 0)
		close(gpu_fd);

	if (accepted == 0) {
		min_width = 0;
		min_outer = 0;
	}
	gate_pass = setup_error == 0 && cleanup_error == 0 &&
		    output_error == 0 && attempted == requested &&
		    accepted == requested && rejected == 0 &&
		    cpu_midpoint_regressions == 0 && ptimer_regressions == 0 &&
		    median_width < TARGET_MEDIAN_BRACKET_NS;
	exit_code = setup_error || cleanup_error || output_error ? 1 :
		    gate_pass ? 0 : 2;
	if (printf("{\"record\":\"summary\",\"setup_stage\":\"%s\","
		   "\"control_transport\":\"%s\","
		   "\"correlation_command\":\"%s\","
		   "\"setup_error\":%d,\"cleanup_error\":%d,"
		   "\"cleanup_rm_status\":%" PRIu32
		   ",\"output_error\":%d,\"requested\":%u,"
		   "\"attempted\":%u,\"accepted\":%u,\"rejected\":%u,"
		   "\"cpu_midpoint_regressions\":%u,"
		   "\"ptimer_regressions\":%u,"
		   "\"min_outer_width_ns\":%" PRIu64
		   ",\"median_outer_width_ns\":%" PRIu64
		   ",\"max_outer_width_ns\":%" PRIu64
		   ",\"min_bracket_width_ns\":%" PRIu64
		   ",\"median_bracket_width_ns\":%" PRIu64
		   ",\"max_bracket_width_ns\":%" PRIu64
		   ",\"target_median_bracket_ns\":%" PRIu64
		   ",\"gate_pass\":%s}\n",
		   setup_stage, transport_name, correlation_command_name,
		   setup_error, cleanup_error,
		   free_status,
		   output_error, requested, attempted, accepted, rejected,
		   cpu_midpoint_regressions, ptimer_regressions, min_outer,
		   median_outer, max_outer, min_width, median_width, max_width,
		   (uint64_t)TARGET_MEDIAN_BRACKET_NS,
		   gate_pass ? "true" : "false") < 0 ||
	    fflush(stdout) != 0)
		exit_code = 1;
	free(widths);
	free(outer_widths);
	return exit_code;
}
