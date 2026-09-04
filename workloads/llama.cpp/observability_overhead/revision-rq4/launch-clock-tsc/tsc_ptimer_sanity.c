// SPDX-License-Identifier: MIT
/*
 * Safe, stock-driver admission probe for NVIDIA 575 TSC/PTIMER correlation.
 * It allocates only private RM objects and never changes module or GPU state.
 */

#define _GNU_SOURCE
#include <cpuid.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/ioctl.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

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
#define NV2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_TSC 0x02U
#define RM_MAX_SAMPLES 16U
#define DEFAULT_BATCHES 15U
#define DEFAULT_CPU 23U
#define DEFAULT_PAUSE_MS 1000U
#define PTIMER_ALLOWANCE_NS 32ULL
#define PRECISION_LIMIT_NS 1500ULL
#define RATE_LIMIT_PPB 10000ULL

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
} rm_time_pair;

typedef struct {
	uint8_t cpuClkId;
	uint8_t sampleCount;
	rm_time_pair samples[RM_MAX_SAMPLES] __attribute__((aligned(8)));
} rm_correlation_parameters;

struct rm_handles {
	uint32_t root;
	uint32_t device;
	uint32_t subdevice;
};

struct accepted_sample {
	uint64_t tsc_mid;
	uint64_t tsc_low;
	uint64_t tsc_high;
	uint64_t ptimer_ns;
	uint64_t width_ns;
};

_Static_assert(sizeof(nv_ioctl_xfer_t) == 16, "575 xfer ABI");
_Static_assert(sizeof(nvos64_parameters) == 48, "575 allocation ABI");
_Static_assert(sizeof(nvos00_parameters) == 16, "575 free ABI");
_Static_assert(sizeof(nvos54_parameters) == 32, "575 control ABI");
_Static_assert(sizeof(nv0080_alloc_parameters) == 56, "575 device ABI");
_Static_assert(sizeof(nv2080_alloc_parameters) == 4, "575 subdevice ABI");
_Static_assert(offsetof(rm_correlation_parameters, samples) == 8,
	       "575 sample-array offset");
_Static_assert(sizeof(rm_correlation_parameters) == 264,
	       "575 correlation ABI");

static uint64_t serialized_tsc(unsigned int *aux)
{
	uint64_t value;

	_mm_lfence();
	value = __rdtscp(aux);
	_mm_lfence();
	return value;
}

static int xfer(int fd, uint32_t command, void *payload, uint32_t size)
{
	nv_ioctl_xfer_t args = { .cmd = command, .size = size, .ptr = payload };
	unsigned long request = _IOWR(NV_IOCTL_MAGIC, NV_ESC_IOCTL_XFER_CMD,
				      nv_ioctl_xfer_t);

	return ioctl(fd, request, &args) < 0 ? -errno : 0;
}

static int alloc_object(int fd, uint32_t root, uint32_t parent,
			uint32_t object_class, void *params, uint32_t params_size,
			uint32_t *object, uint32_t *rm_status)
{
	nvos64_parameters args = {
		.hRoot = root,
		.hObjectParent = parent,
		.hClass = object_class,
		.pAllocParms = params,
		.paramsSize = params_size,
	};
	int err = xfer(fd, NV_ESC_RM_ALLOC, &args, sizeof(args));

	*rm_status = args.status;
	*object = args.hObjectNew;
	if (err)
		return err;
	return args.status == 0 && args.hObjectNew != 0 ? 0 : -EREMOTEIO;
}

static int open_timer(int fd, struct rm_handles *handles, uint32_t *rm_status)
{
	nv0080_alloc_parameters device = { .deviceId = 0 };
	nv2080_alloc_parameters subdevice = { .subDeviceId = 0 };
	int err;

	memset(handles, 0, sizeof(*handles));
	err = alloc_object(fd, 0, 0, NV01_ROOT_CLIENT, NULL, 0,
			   &handles->root, rm_status);
	if (err)
		return err;
	device.hClientShare = handles->root;
	err = alloc_object(fd, handles->root, handles->root, NV01_DEVICE_0,
			   &device, sizeof(device), &handles->device, rm_status);
	if (err)
		return err;
	return alloc_object(fd, handles->root, handles->device, NV20_SUBDEVICE_0,
			    &subdevice, sizeof(subdevice), &handles->subdevice,
			    rm_status);
}

static int free_root(int fd, uint32_t root, uint32_t *rm_status)
{
	nvos00_parameters args = {
		.hRoot = root,
		.hObjectParent = root,
		.hObjectOld = root,
	};
	int err = xfer(fd, NV_ESC_RM_FREE, &args, sizeof(args));

	*rm_status = args.status;
	if (err)
		return err;
	return args.status == 0 ? 0 : -EREMOTEIO;
}

static int read_batch(int fd, const struct rm_handles *handles,
		      rm_correlation_parameters *params, uint32_t *rm_status)
{
	nvos54_parameters control = {
		.hClient = handles->root,
		.hObject = handles->subdevice,
		.cmd = NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO,
		.params = params,
		.paramsSize = sizeof(*params),
	};
	unsigned long request = _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL,
				      nvos54_parameters);
	int err = ioctl(fd, request, &control) < 0 ? -errno : 0;

	*rm_status = control.status;
	if (err)
		return err;
	return control.status == 0 ? 0 : -EREMOTEIO;
}

static int tsc_frequency(uint64_t *hz)
{
	unsigned int eax, ebx, ecx, edx;
	__uint128_t value;

	if (!hz || __get_cpuid_max(0, NULL) < 0x15)
		return -ENOTSUP;
	__cpuid_count(0x15, 0, eax, ebx, ecx, edx);
	(void)edx;
	if (!eax || !ebx || !ecx)
		return -ENOTSUP;
	value = (__uint128_t)ecx * ebx;
	if (value % eax != 0 || value / eax > UINT64_MAX)
		return -ERANGE;
	*hz = (uint64_t)(value / eax);
	return *hz ? 0 : -ERANGE;
}

static int cycles_to_ns_ceil(uint64_t cycles, uint64_t hz, uint64_t *ns)
{
	__uint128_t numerator;

	if (!hz || !ns)
		return -EINVAL;
	numerator = (__uint128_t)cycles * 1000000000ULL + hz - 1;
	if (numerator / hz > UINT64_MAX)
		return -ERANGE;
	*ns = (uint64_t)(numerator / hz);
	return 0;
}

static int derive_interior(const rm_time_pair *previous,
			   const rm_time_pair *current,
			   const rm_time_pair *next, uint64_t hz,
			   struct accepted_sample *out)
{
	uint64_t cycles, ns;

	if (!previous || !current || !next || !out || !previous->cpuTime ||
	    !current->cpuTime || !next->cpuTime || !current->gpuTime ||
	    previous->cpuTime >= current->cpuTime ||
	    current->cpuTime >= next->cpuTime)
		return -ERANGE;
	cycles = next->cpuTime - previous->cpuTime;
	if (cycles_to_ns_ceil(cycles, hz, &ns) ||
	    ns > UINT64_MAX - 2 * PTIMER_ALLOWANCE_NS)
		return -ERANGE;
	out->tsc_mid = current->cpuTime;
	out->tsc_low = previous->cpuTime;
	out->tsc_high = next->cpuTime;
	out->ptimer_ns = current->gpuTime;
	out->width_ns = ns + 2 * PTIMER_ALLOWANCE_NS;
	return 0;
}

static int compare_u64(const void *left, const void *right)
{
	uint64_t a = *(const uint64_t *)left;
	uint64_t b = *(const uint64_t *)right;
	return (a > b) - (a < b);
}

static uint64_t median_u64(uint64_t *values, size_t count)
{
	qsort(values, count, sizeof(*values), compare_u64);
	if (count % 2)
		return values[count / 2];
	return values[count / 2 - 1] / 2 + values[count / 2] / 2 +
	       (values[count / 2 - 1] % 2 + values[count / 2] % 2) / 2;
}

static int rate_error_ppb(const struct accepted_sample *first,
			  const struct accepted_sample *last, uint64_t hz,
			  uint64_t *error_ppb)
{
	uint64_t dt, dg;
	__uint128_t predicted_num, observed_num, difference, denominator;

	if (!first || !last || !error_ppb || last->tsc_mid <= first->tsc_mid ||
	    last->ptimer_ns <= first->ptimer_ns)
		return -ERANGE;
	dt = last->tsc_mid - first->tsc_mid;
	dg = last->ptimer_ns - first->ptimer_ns;
	predicted_num = (__uint128_t)dt * 1000000000ULL;
	observed_num = (__uint128_t)dg * hz;
	difference = predicted_num > observed_num ?
		predicted_num - observed_num : observed_num - predicted_num;
	denominator = predicted_num;
	if (!denominator || difference * 1000000000ULL / denominator > UINT64_MAX)
		return -ERANGE;
	*error_ppb = (uint64_t)(difference * 1000000000ULL / denominator);
	return 0;
}

static int pin_cpu(unsigned int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	return sched_setaffinity(0, sizeof(set), &set) == 0 ? 0 : -errno;
}

static int self_test(void)
{
	rm_time_pair pairs[3] = {
		{ .cpuTime = 1000, .gpuTime = 2000 },
		{ .cpuTime = 1100, .gpuTime = 2100 },
		{ .cpuTime = 1300, .gpuTime = 2300 },
	};
	struct accepted_sample sample = {0};
	uint64_t ns = 0, error = 0;
	struct accepted_sample first = { .tsc_mid = 1000, .ptimer_ns = 1000 };
	struct accepted_sample last = { .tsc_mid = 2000, .ptimer_ns = 2000 };

	if (cycles_to_ns_ceil(300, 1000000000ULL, &ns) || ns != 300 ||
	    derive_interior(&pairs[0], &pairs[1], &pairs[2], 1000000000ULL,
			    &sample) || sample.width_ns != 364 ||
	    sample.tsc_low != 1000 || sample.tsc_high != 1300 ||
	    rate_error_ppb(&first, &last, 1000000000ULL, &error) || error != 0)
		return 1;
	pairs[2].cpuTime = 1099;
	if (derive_interior(&pairs[0], &pairs[1], &pairs[2], 1000000000ULL,
			    &sample) != -ERANGE)
		return 1;
	printf("tsc_ptimer_sanity self-test: PASS\n");
	return 0;
}

static void usage(const char *program)
{
	fprintf(stderr,
		"Usage: %s [--batches N] [--cpu N] [--pause-ms N] | --self-test\n",
		program);
}

static int parse_uint(const char *text, unsigned int limit,
		      unsigned int *value)
{
	char *end = NULL;
	unsigned long parsed;

	errno = 0;
	parsed = strtoul(text, &end, 10);
	if (errno || !end || *end || parsed > limit)
		return -EINVAL;
	*value = (unsigned int)parsed;
	return 0;
}

int main(int argc, char **argv)
{
	unsigned int batches = DEFAULT_BATCHES, cpu = DEFAULT_CPU;
	unsigned int pause_ms = DEFAULT_PAUSE_MS;
	struct rm_handles handles = {0};
	struct accepted_sample *accepted_samples = NULL;
	uint64_t *widths = NULL, hz = 0, median_width = 0, max_width = 0;
	uint64_t rate_ppb = UINT64_MAX;
	unsigned int accepted = 0, rejected = 0, regressions = 0;
	unsigned int migration_errors = 0, attempted = 0;
	uint32_t rm_status = 0, cleanup_status = 0;
	int fd = -1, gpu_fd = -1, err = 0, cleanup_err = 0;
	int gate_pass = 0;

	if (argc == 2 && strcmp(argv[1], "--self-test") == 0)
		return self_test();
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--batches") == 0 && i + 1 < argc) {
			if (parse_uint(argv[++i], 10000, &batches) || batches == 0) {
				usage(argv[0]);
				return 1;
			}
		} else if (strcmp(argv[i], "--cpu") == 0 && i + 1 < argc) {
			if (parse_uint(argv[++i], CPU_SETSIZE - 1, &cpu)) {
				usage(argv[0]);
				return 1;
			}
		} else if (strcmp(argv[i], "--pause-ms") == 0 && i + 1 < argc) {
			if (parse_uint(argv[++i], 60000, &pause_ms)) {
				usage(argv[0]);
				return 1;
			}
		} else {
			usage(argv[0]);
			return 1;
		}
	}
	accepted_samples = calloc((size_t)batches * (RM_MAX_SAMPLES - 2),
				  sizeof(*accepted_samples));
	widths = calloc((size_t)batches * (RM_MAX_SAMPLES - 2),
			 sizeof(*widths));
	if (!accepted_samples || !widths || tsc_frequency(&hz) || pin_cpu(cpu)) {
		err = -EINVAL;
		goto cleanup;
	}

	fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
	gpu_fd = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
	if (fd < 0 || gpu_fd < 0) {
		err = -errno;
		goto cleanup;
	}
	err = open_timer(fd, &handles, &rm_status);
	if (err)
		goto cleanup;

	for (unsigned int batch = 0; batch < batches; batch++) {
		rm_correlation_parameters params = {
			.cpuClkId = NV2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_TSC,
			.sampleCount = RM_MAX_SAMPLES,
		};
		unsigned int aux_before = 0, aux_after = 0;
		uint64_t outer_before = serialized_tsc(&aux_before);
		uint64_t outer_after;
		int cpu_before = sched_getcpu(), cpu_after;

		attempted += RM_MAX_SAMPLES - 2;
		err = read_batch(fd, &handles, &params, &rm_status);
		outer_after = serialized_tsc(&aux_after);
		cpu_after = sched_getcpu();
		if (err || cpu_before != (int)cpu || cpu_after != (int)cpu ||
		    aux_before != aux_after || outer_after <= outer_before) {
			rejected += RM_MAX_SAMPLES - 2;
			if (cpu_before != (int)cpu || cpu_after != (int)cpu ||
			    aux_before != aux_after)
				migration_errors++;
			printf("{\"record\":\"batch\",\"batch\":%u,"
			       "\"valid\":false,\"error\":%d,\"rm_status\":%" PRIu32
			       ",\"cpu_before\":%d,\"cpu_after\":%d,"
			       "\"aux_before\":%u,\"aux_after\":%u}\n",
			       batch, err, rm_status, cpu_before, cpu_after,
			       aux_before, aux_after);
			continue;
		}
		for (unsigned int index = 1; index + 1 < RM_MAX_SAMPLES; index++) {
			struct accepted_sample sample = {0};
			int derive_err = derive_interior(&params.samples[index - 1],
							 &params.samples[index],
							 &params.samples[index + 1],
							 hz, &sample);
			if (derive_err || params.samples[index - 1].cpuTime < outer_before ||
			    params.samples[index + 1].cpuTime > outer_after) {
				rejected++;
				printf("{\"record\":\"sample\",\"batch\":%u,"
				       "\"index\":%u,\"valid\":false,"
				       "\"derive_error\":%d}\n",
				       batch, index, derive_err ? derive_err : -ERANGE);
				continue;
			}
			if (accepted &&
			    (sample.tsc_mid <= accepted_samples[accepted - 1].tsc_mid ||
			     sample.ptimer_ns <= accepted_samples[accepted - 1].ptimer_ns))
				regressions++;
			accepted_samples[accepted] = sample;
			widths[accepted] = sample.width_ns;
			if (sample.width_ns > max_width)
				max_width = sample.width_ns;
			accepted++;
			printf("{\"record\":\"sample\",\"batch\":%u,"
			       "\"index\":%u,\"valid\":true,"
			       "\"tsc_mid\":%" PRIu64 ",\"tsc_low\":%" PRIu64
			       ",\"tsc_high\":%" PRIu64 ",\"ptimer_ns\":%" PRIu64
			       ",\"bracket_width_ns\":%" PRIu64 "}\n",
			       batch, index, sample.tsc_mid, sample.tsc_low,
			       sample.tsc_high, sample.ptimer_ns, sample.width_ns);
		}
		if (batch == 0 && batches > 1 && pause_ms) {
			struct timespec delay = {
				.tv_sec = pause_ms / 1000,
				.tv_nsec = (long)(pause_ms % 1000) * 1000000L,
			};
			while (nanosleep(&delay, &delay) != 0 && errno == EINTR)
				;
		}
	}
	if (accepted) {
		median_width = median_u64(widths, accepted);
		if (accepted > 1)
			(void)rate_error_ppb(&accepted_samples[0],
					     &accepted_samples[accepted - 1], hz,
					     &rate_ppb);
	}

cleanup:
	if (fd >= 0 && handles.root)
		cleanup_err = free_root(fd, handles.root, &cleanup_status);
	if (gpu_fd >= 0 && close(gpu_fd) != 0 && !cleanup_err)
		cleanup_err = -errno;
	if (fd >= 0 && close(fd) != 0 && !cleanup_err)
		cleanup_err = -errno;
	gate_pass = !err && !cleanup_err && accepted >= 200 && rejected == 0 &&
		    regressions == 0 && migration_errors == 0 &&
		    median_width <= PRECISION_LIMIT_NS && rate_ppb <= RATE_LIMIT_PPB;
	printf("{\"record\":\"summary\",\"method\":\"stock_rm_tsc_ptimer_v1\","
	       "\"cpu\":%u,\"tsc_hz\":%" PRIu64 ",\"batches\":%u,"
	       "\"attempted\":%u,\"accepted\":%u,\"rejected\":%u,"
	       "\"regressions\":%u,\"migration_errors\":%u,"
	       "\"median_bracket_width_ns\":%" PRIu64
	       ",\"max_bracket_width_ns\":%" PRIu64
	       ",\"rate_error_ppb\":%" PRIu64
	       ",\"precision_limit_ns\":%" PRIu64
	       ",\"rate_limit_ppb\":%" PRIu64
	       ",\"rm_status\":%" PRIu32 ",\"cleanup_status\":%" PRIu32
	       ",\"cleanup_complete\":%s,\"gate_pass\":%s}\n",
	       cpu, hz, batches, attempted, accepted, rejected, regressions,
	       migration_errors, median_width, max_width, rate_ppb,
	       (uint64_t)PRECISION_LIMIT_NS, (uint64_t)RATE_LIMIT_PPB,
	       rm_status, cleanup_status, cleanup_err ? "false" : "true",
	       gate_pass ? "true" : "false");
	free(widths);
	free(accepted_samples);
	return gate_pass ? 0 : 1;
}
