// SPDX-License-Identifier: GPL-2.0
/*
 * Raw userspace probe for the nvidia_uvm UVM_GPU_STORAGE_DECIDE ioctl
 * (raw command 82 = UVM_IOCTL_BASE(82) on Linux).
 *
 * Mirrors the exact 136-byte UVM_GPU_STORAGE_DECIDE_PARAMS layout from
 * kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h and runs
 * explicit request cases, printing the returned action, output priority,
 * defer_ns, batch target, caller tgid, and rmStatus for each.
 */

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef NvU32 NV_STATUS;

/* Raw ioctl command: UVM_GPU_STORAGE_DECIDE = UVM_IOCTL_BASE(82). */
#define UVM_GPU_STORAGE_DECIDE			82

#define UVM_GPU_STORAGE_ABI_VERSION		1u

#define UVM_GPU_STORAGE_OP_READ			0u
#define UVM_GPU_STORAGE_OP_WRITE		1u
#define UVM_GPU_STORAGE_OP_BAD			7u

#define UVM_GPU_STORAGE_REQUEST_FLAG_DEMAND	0x00000001u
#define UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE	0x00000002u
#define UVM_GPU_STORAGE_REQUEST_FLAG_RECOMPUTABLE	0x00000004u
#define UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER	0x00000008u
#define UVM_GPU_STORAGE_REQUEST_FLAG_BAD	0x00000010u

#define UVM_GPU_STORAGE_ACTION_SUBMIT_NOW	0u
#define UVM_GPU_STORAGE_ACTION_DEFER		1u
#define UVM_GPU_STORAGE_ACTION_RECOMPUTE	2u

#define UVM_GPU_STORAGE_MAX_PRIORITY		7u

#define NANOSEC_PER_MSEC			1000000ULL

struct uvm_gpu_storage_decide_params {
	NvU32     abiVersion;
	NvU32     op;
	NvU32     requestFlags;
	NvU32     inputPriority;
	NvU64     requestId;
	NvU64     objectId;
	NvU64     bytes;
	NvU64     tenantId;
	NvU64     callerHint;
	NvU64     deadlineNs;
	NvU64     slackNs;
	NvU64     estimatedTransferNs;
	NvU64     recomputeNs;
	NvU32     queueDepth;
	NvU32     hbmPressurePermille;
	NvU32     action;
	NvU32     outputPriority;
	NvU64     deferNs;
	NvU32     batchTarget;
	NvU64     callerTgid;
	NV_STATUS rmStatus;
};

_Static_assert(sizeof(struct uvm_gpu_storage_decide_params) == 136,
	       "UVM_GPU_STORAGE_DECIDE_PARAMS must be exactly 136 bytes");
_Static_assert(offsetof(struct uvm_gpu_storage_decide_params, requestId) == 16,
	       "requestId offset");
_Static_assert(offsetof(struct uvm_gpu_storage_decide_params, action) == 96,
	       "action offset");
_Static_assert(offsetof(struct uvm_gpu_storage_decide_params, deferNs) == 104,
	       "deferNs offset");
_Static_assert(offsetof(struct uvm_gpu_storage_decide_params, callerTgid) == 120,
	       "callerTgid offset");
_Static_assert(offsetof(struct uvm_gpu_storage_decide_params, rmStatus) == 128,
	       "rmStatus offset");

struct case_desc {
	const char *name;
	NvU32 abiVersion;
	NvU32 op;
	NvU32 requestFlags;
	NvU32 inputPriority;
	NvU64 bytes;
	NvU64 slackNs;
	NvU64 estimatedTransferNs;
	NvU64 recomputeNs;
	NvU32 queueDepth;
	NvU32 hbmPressurePermille;
};

#define FLAG_ALL_DEMAND (UVM_GPU_STORAGE_REQUEST_FLAG_DEMAND | \
			 UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE | \
			 UVM_GPU_STORAGE_REQUEST_FLAG_RECOMPUTABLE | \
			 UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER)

static const struct case_desc cases[] = {
	{
		.name = "zeros_default",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
	},
	{
		.name = "demand_read",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_DEMAND |
				UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 3,
		.bytes = 4 * 1024 * 1024,
		.slackNs = 10 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 1 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 500,
	},
	{
		.name = "spec_read_low_pressure",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 2,
		.bytes = 8 * 1024 * 1024,
		.slackNs = 50 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 20 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 500,
	},
	{
		.name = "recompute_cheaper_fits",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_RECOMPUTABLE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 5,
		.bytes = 2 * 1024 * 1024,
		.slackNs = 10 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 1 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 500,
	},
	{
		.name = "recompute_not_cheaper",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_RECOMPUTABLE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 4,
		.bytes = 2 * 1024 * 1024,
		.slackNs = 10 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 20 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 500,
	},
	{
		.name = "recompute_over_slack",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_RECOMPUTABLE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 4,
		.bytes = 2 * 1024 * 1024,
		.slackNs = 500 * 1000ULL,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 1 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 500,
	},
	{
		.name = "spec_read_defer_900",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 2,
		.bytes = 8 * 1024 * 1024,
		.slackNs = 50 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 20 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 900,
	},
	{
		.name = "spec_read_defer_boundary_800",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 2,
		.bytes = 8 * 1024 * 1024,
		.slackNs = 50 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 20 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 800,
	},
	{
		.name = "spec_read_defer_799",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 2,
		.bytes = 8 * 1024 * 1024,
		.slackNs = 50 * NANOSEC_PER_MSEC,
		.estimatedTransferNs = 5 * NANOSEC_PER_MSEC,
		.recomputeNs = 20 * NANOSEC_PER_MSEC,
		.queueDepth = 8,
		.hbmPressurePermille = 799,
	},
	{
		.name = "write_defer_700",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_WRITE,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 1,
		.bytes = 1 * 1024 * 1024,
		.slackNs = 1234 * 1000ULL,
		.estimatedTransferNs = 500 * 1000ULL,
		.recomputeNs = 0,
		.queueDepth = 32,
		.hbmPressurePermille = 700,
	},
	{
		.name = "write_defer_boundary_600_batch_clamp",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_WRITE,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 1,
		.bytes = 1 * 1024 * 1024,
		.slackNs = 1234 * 1000ULL,
		.estimatedTransferNs = 500 * 1000ULL,
		.recomputeNs = 0,
		.queueDepth = 100,
		.hbmPressurePermille = 600,
	},
	{
		.name = "write_low_pressure",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_WRITE,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 1,
		.bytes = 1 * 1024 * 1024,
		.slackNs = 1234 * 1000ULL,
		.estimatedTransferNs = 500 * 1000ULL,
		.recomputeNs = 0,
		.queueDepth = 64,
		.hbmPressurePermille = 300,
	},
	{
		.name = "write_demand_at_pressure",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_WRITE,
		.requestFlags = FLAG_ALL_DEMAND,
		.inputPriority = 6,
		.bytes = 1 * 1024 * 1024,
		.slackNs = 1234 * 1000ULL,
		.estimatedTransferNs = 500 * 1000ULL,
		.recomputeNs = 0,
		.queueDepth = 64,
		.hbmPressurePermille = 900,
	},
	{
		.name = "bad_abi_2",
		.abiVersion = 2,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_DEMAND,
		.inputPriority = 3,
		.hbmPressurePermille = 500,
	},
	{
		.name = "bad_op_7",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_BAD,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_DEMAND,
		.inputPriority = 3,
		.hbmPressurePermille = 500,
	},
	{
		.name = "bad_flag_0x10",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_BAD,
		.inputPriority = 3,
		.hbmPressurePermille = 500,
	},
	{
		.name = "pressure_overflow_1001",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 2,
		.hbmPressurePermille = 1001,
	},
	{
		.name = "priority_clamp_9",
		.abiVersion = UVM_GPU_STORAGE_ABI_VERSION,
		.op = UVM_GPU_STORAGE_OP_READ,
		.requestFlags = UVM_GPU_STORAGE_REQUEST_FLAG_SPECULATIVE |
				UVM_GPU_STORAGE_REQUEST_FLAG_SAFE_TO_DEFER,
		.inputPriority = 9,
		.hbmPressurePermille = 900,
	},
};

#define CASE_COUNT	(sizeof(cases) / sizeof(cases[0]))
#define OUT_SENTINEL	0xDEADBEEFu

static const char *action_name(NvU32 action)
{
	switch (action) {
	case UVM_GPU_STORAGE_ACTION_SUBMIT_NOW:
		return "SUBMIT_NOW";
	case UVM_GPU_STORAGE_ACTION_DEFER:
		return "DEFER";
	case UVM_GPU_STORAGE_ACTION_RECOMPUTE:
		return "RECOMPUTE";
	default:
		return "?";
	}
}

static void print_case(const struct case_desc *c, size_t idx)
{
	printf("[%zu] %s\n", idx, c->name);
	printf("    in : abi=%u op=%u flags=0x%x prio=%u bytes=%llu "
	       "slack=%llu transfer=%llu recompute=%llu queueDepth=%u "
	       "pressure=%u\n",
	       c->abiVersion, c->op, c->requestFlags, c->inputPriority,
	       (unsigned long long)c->bytes,
	       (unsigned long long)c->slackNs,
	       (unsigned long long)c->estimatedTransferNs,
	       (unsigned long long)c->recomputeNs,
	       c->queueDepth, c->hbmPressurePermille);
}

static void fill_params(struct uvm_gpu_storage_decide_params *p,
			const struct case_desc *c, size_t idx)
{
	memset(p, 0, sizeof(*p));

	p->abiVersion = c->abiVersion;
	p->op = c->op;
	p->requestFlags = c->requestFlags;
	p->inputPriority = c->inputPriority;
	p->requestId = (NvU64)idx + 1;
	p->objectId = 0x1000 + (NvU64)idx;
	p->bytes = c->bytes;
	p->tenantId = 0xA11CEULL;
	p->callerHint = 0;
	p->deadlineNs = c->slackNs + c->estimatedTransferNs;
	p->slackNs = c->slackNs;
	p->estimatedTransferNs = c->estimatedTransferNs;
	p->recomputeNs = c->recomputeNs;
	p->queueDepth = c->queueDepth;
	p->hbmPressurePermille = c->hbmPressurePermille;

	/* Poison the OUT fields so kernel-written values are visible. */
	p->action = OUT_SENTINEL;
	p->outputPriority = OUT_SENTINEL;
	p->deferNs = ((NvU64)OUT_SENTINEL << 32) | OUT_SENTINEL;
	p->batchTarget = OUT_SENTINEL;
	p->callerTgid = ((NvU64)OUT_SENTINEL << 32) | OUT_SENTINEL;
	p->rmStatus = OUT_SENTINEL;
}

static void run_case(int fd, const struct case_desc *c, size_t idx)
{
	struct uvm_gpu_storage_decide_params p;
	NvU32 expected_priority;

	fill_params(&p, c, idx);
	print_case(c, idx);

	errno = 0;
	if (ioctl(fd, (unsigned int)UVM_GPU_STORAGE_DECIDE, &p) != 0) {
		printf("    ioctl: errno=%d (%s)\n",
		       errno, strerror(errno));
		return;
	}

	printf("    out: action=%s(%u) priority=%u deferNs=%llu "
	       "batchTarget=%u callerTgid=%llu "
	       "rmStatus=%u (%s)\n",
	       action_name(p.action), p.action, p.outputPriority,
	       (unsigned long long)p.deferNs, p.batchTarget,
	       (unsigned long long)p.callerTgid,
	       (unsigned int)p.rmStatus,
	       p.rmStatus == 0 ? "NV_OK" : "nonzero");

	/* Baseline expectation while no policy is attached: the driver's
	 * SUBMIT_NOW defaults with the priority clamped to <= 7 and
	 * callerTgid set to our process tgid. */
	expected_priority = c->inputPriority > UVM_GPU_STORAGE_MAX_PRIORITY ?
			    UVM_GPU_STORAGE_MAX_PRIORITY : c->inputPriority;
	printf("    no-policy baseline: action=%s priority=%u deferNs=0 "
	       "batchTarget=1 -> %s\n",
	       action_name(UVM_GPU_STORAGE_ACTION_SUBMIT_NOW),
	       expected_priority,
	       (p.action == UVM_GPU_STORAGE_ACTION_SUBMIT_NOW &&
		p.outputPriority == expected_priority &&
		p.deferNs == 0 &&
		p.batchTarget == 1 &&
		p.callerTgid == (NvU64)getpid() &&
		p.rmStatus == 0) ? "match" : "MISMATCH");
}

int main(int argc, char **argv)
{
	const char *dev = argc > 1 ? argv[1] : "/dev/nvidia-uvm";
	size_t i;
	int fd;

	if (argc > 2) {
		fprintf(stderr, "usage: %s [/dev/nvidia-uvm]\n", argv[0]);
		return 2;
	}

	fd = open(dev, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "open(\"%s\"): %s\n", dev, strerror(errno));
		return 1;
	}

	printf("device=%s fd=%d ioctl_cmd=%d (UVM_IOCTL_BASE(82)) "
	       "params_size=%zu pid=%d\n",
	       dev, fd, UVM_GPU_STORAGE_DECIDE,
	       sizeof(struct uvm_gpu_storage_decide_params), getpid());

	for (i = 0; i < CASE_COUNT; i++)
		run_case(fd, &cases[i], i);

	close(fd);
	return 0;
}
