// SPDX-License-Identifier: GPL-2.0
/*
 * uprobe_preempt_multi.bpf.c - Sleepable uprobe -> multi-target GPU preempt
 *
 * Flow:
 *   1. Optional kprobes auto-capture BE GR TSG handles from the stock NVIDIA
 *      ioctl path and append them into target_tsg[].
 *   2. Sleepable uprobe on cuLaunchKernel preempts every active BE target.
 *   3. A per-CPU cooldown suppresses repeated preempt bursts.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

extern int bpf_nv_gpu_preempt_tsg(u32 hClient, u32 hTsg) __ksym;

#define MAX_BE_TARGETS       16
#define LATENCY_SAMPLES      256
#define NV_ESC_RM_ALLOC      0x2B
#define NV_ESC_IOCTL_XFER_NR 211
#define TSG_CLASS_A06C       0xa06c

#ifndef TASK_COMM_LEN
#define TASK_COMM_LEN 16
#endif

struct tsg_entry {
	__u32 hClient;
	__u32 hTsg;
};

struct pending_alloc {
	__u64 nvos21_ptr;
	__u32 hRoot;
	__u32 engine_type;
	__u32 status_offset;
	__u64 tsg_id;
	__u8 has_engine;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAX_BE_TARGETS);
	__type(key, __u32);
	__type(value, struct tsg_entry);
} target_tsg SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u32);
} be_target_count SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 128);
	__type(key, __u64);
	__type(value, struct pending_alloc);
} pending_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 16);
	__type(key, __u32);
	__type(value, __u64);
} uprobe_stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, LATENCY_SAMPLES);
	__type(key, __u32);
	__type(value, __u64);
} uprobe_latency SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);
} last_preempt_ns SEC(".maps");

enum stat_idx {
	STAT_UPROBE_HIT = 0,
	STAT_PREEMPT_OK = 1,
	STAT_PREEMPT_ERR = 2,
	STAT_LAST_NS = 3,
	STAT_TOTAL_NS = 4,
	STAT_SKIPPED = 5,
	STAT_COOLDOWN_SKIP = 6,
	STAT_TARGETS_HIT = 7,
	STAT_TSG_CAPTURED = 8,
	STAT_TSG_FILTERED = 9,
};

const volatile __u32 enabled = 1;
const volatile __u64 cooldown_ns = 100000ULL;
const volatile char be_comm[TASK_COMM_LEN] = "bench_be";
const volatile char lc_comm[TASK_COMM_LEN] = "bench_lc";

static __always_inline void inc_stat(__u32 idx)
{
	__u64 *val = bpf_map_lookup_elem(&uprobe_stats, &idx);

	if (val)
		__sync_fetch_and_add(val, 1);
}

static __always_inline void add_stat(__u32 idx, __u64 delta)
{
	__u64 *val = bpf_map_lookup_elem(&uprobe_stats, &idx);

	if (val)
		__sync_fetch_and_add(val, delta);
}

static __always_inline void set_stat(__u32 idx, __u64 value)
{
	__u64 *val = bpf_map_lookup_elem(&uprobe_stats, &idx);

	if (val)
		*val = value;
}

static __always_inline int current_comm_matches(const volatile char *expected)
{
	char comm[TASK_COMM_LEN] = {};
	int i;

	if (expected[0] == '\0')
		return 1;

	if (bpf_get_current_comm(comm, sizeof(comm)) != 0)
		return 0;

#pragma unroll
	for (i = 0; i < TASK_COMM_LEN; i++) {
		if (comm[i] != expected[i])
			return 0;
		if (expected[i] == '\0')
			return 1;
	}

	return 1;
}

static __always_inline int current_comm_matches_be(void)
{
	return current_comm_matches(be_comm);
}

static __always_inline int current_comm_matches_lc(void)
{
	return current_comm_matches(lc_comm);
}

static __always_inline void record_latency(__u64 elapsed)
{
	__u32 key;
	__u64 *target_hits;

	set_stat(STAT_LAST_NS, elapsed);
	add_stat(STAT_TOTAL_NS, elapsed);

	key = STAT_TARGETS_HIT;
	target_hits = bpf_map_lookup_elem(&uprobe_stats, &key);
	if (!target_hits || *target_hits == 0)
		return;

	key = ((__u32)(*target_hits - 1)) % LATENCY_SAMPLES;
	bpf_map_update_elem(&uprobe_latency, &key, &elapsed, BPF_ANY);
}

static __always_inline int insert_be_target(__u32 hClient, __u32 hTsg)
{
	__u32 zero = 0;
	__u32 *countp = bpf_map_lookup_elem(&be_target_count, &zero);
	__u32 count;
	int i;

	if (!countp)
		return 0;

	count = *countp;
	if (count > MAX_BE_TARGETS)
		count = MAX_BE_TARGETS;

#pragma unroll
	for (i = 0; i < MAX_BE_TARGETS; i++) {
		__u32 key = i;
		struct tsg_entry *existing;

		if ((__u32)i >= count)
			continue;

		existing = bpf_map_lookup_elem(&target_tsg, &key);
		if (!existing)
			continue;

		if (existing->hClient == hClient && existing->hTsg == hTsg) {
			inc_stat(STAT_TSG_FILTERED);
			return 0;
		}
	}

	if (count >= MAX_BE_TARGETS) {
		inc_stat(STAT_TSG_FILTERED);
		return 0;
	}

	{
		__u32 idx = count;
		struct tsg_entry entry = {
			.hClient = hClient,
			.hTsg = hTsg,
		};

		if (bpf_map_update_elem(&target_tsg, &idx, &entry, BPF_ANY) != 0) {
			inc_stat(STAT_TSG_FILTERED);
			return 0;
		}
	}

	*countp = count + 1;
	inc_stat(STAT_TSG_CAPTURED);
	return 1;
}

SEC("uprobe.s//usr/lib/x86_64-linux-gnu/libcuda.so:cuLaunchKernel")
int preempt_on_launch(struct pt_regs *ctx)
{
	__u32 zero = 0;
	__u32 *countp;
	__u32 count;
	__u64 now = bpf_ktime_get_ns();
	__u64 *lastp;
	__u32 attempted = 0;
	int i;

	inc_stat(STAT_UPROBE_HIT);

	if (!enabled) {
		inc_stat(STAT_SKIPPED);
		return 0;
	}

	if (!current_comm_matches_lc()) {
		inc_stat(STAT_SKIPPED);
		return 0;
	}

	lastp = bpf_map_lookup_elem(&last_preempt_ns, &zero);
	if (lastp && cooldown_ns && *lastp && now - *lastp < cooldown_ns) {
		inc_stat(STAT_COOLDOWN_SKIP);
		return 0;
	}

	countp = bpf_map_lookup_elem(&be_target_count, &zero);
	if (!countp || *countp == 0) {
		inc_stat(STAT_SKIPPED);
		return 0;
	}

	count = *countp;
	if (count > MAX_BE_TARGETS)
		count = MAX_BE_TARGETS;

#pragma unroll
	for (i = 0; i < MAX_BE_TARGETS; i++) {
		__u32 key = i;
		struct tsg_entry *tsg;
		__u64 start;
		__u64 elapsed;
		int ret;

		if ((__u32)i >= count)
			continue;

		tsg = bpf_map_lookup_elem(&target_tsg, &key);
		if (!tsg || tsg->hClient == 0 || tsg->hTsg == 0)
			continue;

		attempted = 1;
		inc_stat(STAT_TARGETS_HIT);

		start = bpf_ktime_get_ns();
		ret = bpf_nv_gpu_preempt_tsg(tsg->hClient, tsg->hTsg);
		elapsed = bpf_ktime_get_ns() - start;
		record_latency(elapsed);

		if (ret == 0) {
			inc_stat(STAT_PREEMPT_OK);
		} else {
			inc_stat(STAT_PREEMPT_ERR);
			bpf_printk("multi uprobe preempt FAIL: ret=%d hClient=0x%x hTsg=0x%x\n",
				   ret, tsg->hClient, tsg->hTsg);
		}
	}

	if (!attempted) {
		inc_stat(STAT_SKIPPED);
		return 0;
	}

	if (lastp)
		*lastp = bpf_ktime_get_ns();

	return 0;
}

SEC("kprobe/nvidia_unlocked_ioctl")
int capture_ioctl_entry(struct pt_regs *ctx)
{
	unsigned int cmd = (unsigned int)PT_REGS_PARM2(ctx);
	unsigned long i_arg = PT_REGS_PARM3(ctx);
	unsigned int nr = cmd & 0xFF;
	__u32 size = (cmd >> 16) & 0x3fff;
	__u64 nvos21_ptr = 0;
	__u32 hClass = 0;
	__u32 hRoot = 0;
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	struct pending_alloc pa = {};

	if (nr == NV_ESC_IOCTL_XFER_NR) {
		__u32 inner_cmd = 0;

		if (size != 16 || bpf_probe_read_user(&inner_cmd, sizeof(inner_cmd), (void *)i_arg) ||
		    bpf_probe_read_user(&size, sizeof(size), (void *)(i_arg + 4)) ||
		    inner_cmd != NV_ESC_RM_ALLOC)
			return 0;

		if (bpf_probe_read_user(&nvos21_ptr, sizeof(nvos21_ptr), (void *)(i_arg + 8)))
			return 0;
	} else if (nr == NV_ESC_RM_ALLOC) {
		nvos21_ptr = i_arg;
	} else {
		return 0;
	}

	if (!nvos21_ptr || (size != 32 && size != 48))
		return 0;

	if (bpf_probe_read_user(&hClass, sizeof(hClass), (void *)(nvos21_ptr + 12)) ||
	    hClass != TSG_CLASS_A06C)
		return 0;

	if (!current_comm_matches_be()) {
		inc_stat(STAT_TSG_FILTERED);
		return 0;
	}

	if (bpf_probe_read_user(&hRoot, sizeof(hRoot), (void *)nvos21_ptr) || !hRoot)
		return 0;

	pa.nvos21_ptr = nvos21_ptr;
	pa.hRoot = hRoot;
	pa.status_offset = size == 32 ? 28 : 40;
	pa.has_engine = 0;

	bpf_map_update_elem(&pending_map, &pid_tgid, &pa, BPF_ANY);
	return 0;
}

SEC("kprobe/nv_gpu_sched_task_init")
int capture_engine_type(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	struct pending_alloc *pa = bpf_map_lookup_elem(&pending_map, &pid_tgid);
	void *init_ctx;

	if (!pa)
		return 0;

	init_ctx = (void *)PT_REGS_PARM1(ctx);
	if (bpf_probe_read_kernel(&pa->tsg_id, sizeof(pa->tsg_id), init_ctx) ||
	    bpf_probe_read_kernel(&pa->engine_type, sizeof(pa->engine_type), init_ctx + 8))
		return 0;
	pa->has_engine = 1;

	return 0;
}

SEC("kretprobe/nvidia_unlocked_ioctl")
int capture_ioctl_exit(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	struct pending_alloc *pa = bpf_map_lookup_elem(&pending_map, &pid_tgid);
	__u64 nvos21_ptr;
	__u32 hRoot;
	__u32 engine_type;
	__u8 has_engine;
	__u32 hObjectNew = 0;
	__u32 status_offset, nvstatus = ~0U;

	if (!pa)
		return 0;

	nvos21_ptr = pa->nvos21_ptr;
	hRoot = pa->hRoot;
	engine_type = pa->engine_type;
	has_engine = pa->has_engine;
	status_offset = pa->status_offset;

	bpf_map_delete_elem(&pending_map, &pid_tgid);

	if (PT_REGS_RC(ctx) != 0 ||
	    bpf_probe_read_user(&nvstatus, sizeof(nvstatus), (void *)(nvos21_ptr + status_offset)) || nvstatus ||
	    bpf_probe_read_user(&hObjectNew, sizeof(hObjectNew), (void *)(nvos21_ptr + 8)) || !hObjectNew)
		return 0;

	if (!has_engine || engine_type < 1 || engine_type > 8) {
		inc_stat(STAT_TSG_FILTERED);
		return 0;
	}

	insert_be_target(hRoot, hObjectNew);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
