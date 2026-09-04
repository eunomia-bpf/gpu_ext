/* SPDX-License-Identifier: GPL-2.0 */
#ifndef REVISION_INIT_TRACE_H
#define REVISION_INIT_TRACE_H

#ifdef __VMLINUX_H__
typedef __u32 revision_u32;
typedef __u64 revision_u64;
#else
#include <stdint.h>
typedef uint32_t revision_u32;
typedef uint64_t revision_u64;
#endif

#define REVISION_INIT_DIAGNOSTIC_ABI_VERSION 1U
#define REVISION_INIT_STATUS_NOT_OBSERVED (~(revision_u32)0)
#define REVISION_INIT_GSP_SET_TIMESLICE 0xa06c0103U
#define REVISION_INIT_GSP_SET_INTERLEAVE 0xa06c0107U

enum revision_init_phase {
	REVISION_INIT_VALIDATED = 1,
	REVISION_INIT_NATIVE_RETURN = 2,
	REVISION_INIT_CONSTRUCTOR_RETURN = 3,
};

enum revision_init_field {
	REVISION_INIT_FIELD_NONE = 0,
	REVISION_INIT_FIELD_TIMESLICE = 1,
	REVISION_INIT_FIELD_INTERLEAVE = 2,
};

/* Address-free mirror of nv_gpu_sched_init_diagnostic_ctx in the 575 ABI. */
struct revision_init_diagnostic {
	revision_u32 abi_version;
	revision_u32 abi_size;
	revision_u32 phase;
	revision_u32 field;

	revision_u32 h_client;
	revision_u32 h_resource;
	revision_u32 gpu_instance;
	revision_u32 subdevice_instance;
	revision_u32 group_id;
	revision_u32 runlist_id;
	revision_u32 engine_type;
	revision_u32 constructor_epoch;

	revision_u64 default_timeslice;
	revision_u64 minimum_timeslice;
	revision_u32 default_interleave;
	revision_u32 timeslice_attempted;
	revision_u32 timeslice_conflict;
	revision_u32 reserved0;
	revision_u64 timeslice_request_value;
	revision_u32 interleave_attempted;
	revision_u32 interleave_conflict;
	revision_u32 interleave_request_value;
	revision_u32 timeslice_validation_result;
	revision_u32 interleave_validation_result;
	revision_u32 reserved1;
	revision_u64 effective_timeslice;
	revision_u32 effective_interleave;
	revision_u32 timeslice_native_status;
	revision_u64 timeslice_post_value;
	revision_u32 interleave_native_status;
	revision_u32 interleave_post_value;
	revision_u32 constructor_status;
	revision_u32 final_interleave;
	revision_u64 final_timeslice;
	revision_u32 final_snapshot_valid;
	revision_u32 reserved2;
};

/* Address-free mirror of nv_gpu_gsp_control_complete_ctx in the 575 ABI. */
struct revision_init_gsp_completion {
	revision_u64 input_value;
	revision_u32 h_client;
	revision_u32 h_object;
	revision_u32 command;
	revision_u32 input_size;
	revision_u32 wire_size;
	revision_u32 input_valid;
	revision_u32 transport_status;
	revision_u32 gsp_status;
	revision_u32 gsp_status_valid;
	revision_u32 reserved;
};

enum revision_init_event_kind {
	REVISION_INIT_EVENT_DIAGNOSTIC = 1,
	REVISION_INIT_EVENT_GSP = 2,
};

struct revision_init_trace_event {
	revision_u64 pid_tgid;
	revision_u64 timestamp_ns;
	revision_u32 kind;
	revision_u32 reserved;
	union {
		struct revision_init_diagnostic diagnostic;
		struct revision_init_gsp_completion gsp;
	};
};

enum revision_init_trace_stat {
	REVISION_INIT_DIAGNOSTIC_OBSERVED = 0,
	REVISION_INIT_DIAGNOSTIC_EMITTED,
	REVISION_INIT_DIAGNOSTIC_READ_ERROR,
	REVISION_INIT_DIAGNOSTIC_DROP,
	REVISION_INIT_GSP_OBSERVED,
	REVISION_INIT_GSP_EMITTED,
	REVISION_INIT_GSP_READ_ERROR,
	REVISION_INIT_GSP_DROP,
	REVISION_INIT_TRACE_STAT_COUNT,
};

#endif /* REVISION_INIT_TRACE_H */
