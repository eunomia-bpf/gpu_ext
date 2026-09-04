/* SPDX-License-Identifier: GPL-2.0 */
#ifndef REVISION_INIT_RECORDS_H
#define REVISION_INIT_RECORDS_H

#ifdef __VMLINUX_H__
typedef __s32 revision_init_s32;
typedef __u32 revision_init_u32;
typedef __u64 revision_init_u64;
#else
#include <stdint.h>
typedef int32_t revision_init_s32;
typedef uint32_t revision_init_u32;
typedef uint64_t revision_init_u64;
#endif

struct revision_init_returns {
	revision_init_u32 timeslice_count;
	revision_init_u32 interleave_count;
	revision_init_s32 timeslice[3];
	revision_init_s32 interleave[3];
};

struct revision_init_key {
	revision_init_u64 pid_tgid;
	revision_init_u64 tsg_id;
	revision_init_u32 runlist_id;
	revision_init_u32 reserved;
};

struct revision_init_record_input {
	revision_init_u64 tsg_id;
	revision_init_u32 engine_type;
	revision_init_u64 default_timeslice;
	revision_init_u32 default_interleave;
	revision_init_u32 runlist_id;
};

struct revision_init_record {
	struct revision_init_record_input input;
	struct revision_init_returns requests;
	revision_init_u64 timestamp_ns;
	revision_init_u32 fixture;
	revision_init_u32 complete;
};

enum { INIT_SEEN, INIT_RECORDED, INIT_RECORD_ERROR, INIT_STAT_COUNT };

_Static_assert(sizeof(struct revision_init_record_input) == 32,
	       "575 task-init input record ABI");
_Static_assert(sizeof(struct revision_init_key) == 24,
	       "revision init key ABI");
_Static_assert(sizeof(struct revision_init_returns) == 32,
	       "revision init return ABI");
_Static_assert(sizeof(struct revision_init_record) == 80,
	       "revision init record ABI");

#endif /* REVISION_INIT_RECORDS_H */
