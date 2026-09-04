/* SPDX-License-Identifier: MIT */
#include <stddef.h>
#include <stdio.h>

#include "revision_init_trace.h"
#include "nv-gpu-rpc-diagnostic.h"
#include "nv-gpu-sched-init-diagnostic.h"

static unsigned int assertions;

#define EXPECT(expression)                                                     \
	do {                                                                     \
		++assertions;                                                      \
		if (!(expression)) {                                               \
			fprintf(stderr, "line %d: %s\n", __LINE__, #expression);    \
			return 1;                                                    \
		}                                                                    \
	} while (0)

#define SAME_DIAGNOSTIC_OFFSET(field)                                          \
	EXPECT(offsetof(struct revision_init_diagnostic, field) ==                \
	       offsetof(struct nv_gpu_sched_init_diagnostic_ctx, field))

static int test_diagnostic_layout(void)
{
	EXPECT(sizeof(struct revision_init_diagnostic) ==
	       sizeof(struct nv_gpu_sched_init_diagnostic_ctx));
	SAME_DIAGNOSTIC_OFFSET(abi_version);
	SAME_DIAGNOSTIC_OFFSET(abi_size);
	SAME_DIAGNOSTIC_OFFSET(phase);
	SAME_DIAGNOSTIC_OFFSET(field);
	SAME_DIAGNOSTIC_OFFSET(h_client);
	SAME_DIAGNOSTIC_OFFSET(h_resource);
	SAME_DIAGNOSTIC_OFFSET(gpu_instance);
	SAME_DIAGNOSTIC_OFFSET(subdevice_instance);
	SAME_DIAGNOSTIC_OFFSET(group_id);
	SAME_DIAGNOSTIC_OFFSET(runlist_id);
	SAME_DIAGNOSTIC_OFFSET(engine_type);
	SAME_DIAGNOSTIC_OFFSET(constructor_epoch);
	SAME_DIAGNOSTIC_OFFSET(default_timeslice);
	SAME_DIAGNOSTIC_OFFSET(minimum_timeslice);
	SAME_DIAGNOSTIC_OFFSET(default_interleave);
	SAME_DIAGNOSTIC_OFFSET(timeslice_attempted);
	SAME_DIAGNOSTIC_OFFSET(timeslice_conflict);
	SAME_DIAGNOSTIC_OFFSET(reserved0);
	SAME_DIAGNOSTIC_OFFSET(timeslice_request_value);
	SAME_DIAGNOSTIC_OFFSET(interleave_attempted);
	SAME_DIAGNOSTIC_OFFSET(interleave_conflict);
	SAME_DIAGNOSTIC_OFFSET(interleave_request_value);
	SAME_DIAGNOSTIC_OFFSET(timeslice_validation_result);
	SAME_DIAGNOSTIC_OFFSET(interleave_validation_result);
	SAME_DIAGNOSTIC_OFFSET(reserved1);
	SAME_DIAGNOSTIC_OFFSET(effective_timeslice);
	SAME_DIAGNOSTIC_OFFSET(effective_interleave);
	SAME_DIAGNOSTIC_OFFSET(timeslice_native_status);
	SAME_DIAGNOSTIC_OFFSET(timeslice_post_value);
	SAME_DIAGNOSTIC_OFFSET(interleave_native_status);
	SAME_DIAGNOSTIC_OFFSET(interleave_post_value);
	SAME_DIAGNOSTIC_OFFSET(constructor_status);
	SAME_DIAGNOSTIC_OFFSET(final_interleave);
	SAME_DIAGNOSTIC_OFFSET(final_timeslice);
	SAME_DIAGNOSTIC_OFFSET(final_snapshot_valid);
	SAME_DIAGNOSTIC_OFFSET(reserved2);
	return 0;
}

#define SAME_GSP_OFFSET(local, driver)                                         \
	EXPECT(offsetof(struct revision_init_gsp_completion, local) ==            \
	       offsetof(struct nv_gpu_gsp_control_complete_ctx, driver))

static int test_gsp_layout(void)
{
	EXPECT(sizeof(struct revision_init_gsp_completion) ==
	       sizeof(struct nv_gpu_gsp_control_complete_ctx));
	SAME_GSP_OFFSET(input_value, input_value);
	SAME_GSP_OFFSET(h_client, hClient);
	SAME_GSP_OFFSET(h_object, hObject);
	SAME_GSP_OFFSET(command, command);
	SAME_GSP_OFFSET(input_size, input_size);
	SAME_GSP_OFFSET(wire_size, wire_size);
	SAME_GSP_OFFSET(input_valid, input_valid);
	SAME_GSP_OFFSET(transport_status, transport_status);
	SAME_GSP_OFFSET(gsp_status, gsp_status);
	SAME_GSP_OFFSET(gsp_status_valid, gsp_status_valid);
	SAME_GSP_OFFSET(reserved, reserved);
	return 0;
}

static int test_constants(void)
{
	EXPECT(REVISION_INIT_DIAGNOSTIC_ABI_VERSION ==
	       NV_GPU_SCHED_INIT_DIAGNOSTIC_ABI_VERSION);
	EXPECT(REVISION_INIT_STATUS_NOT_OBSERVED ==
	       NV_GPU_SCHED_INIT_DIAGNOSTIC_STATUS_NOT_OBSERVED);
	EXPECT((int)REVISION_INIT_VALIDATED ==
	       (int)NV_GPU_SCHED_INIT_DIAGNOSTIC_VALIDATED);
	EXPECT((int)REVISION_INIT_NATIVE_RETURN ==
	       (int)NV_GPU_SCHED_INIT_DIAGNOSTIC_NATIVE_RETURN);
	EXPECT((int)REVISION_INIT_CONSTRUCTOR_RETURN ==
	       (int)NV_GPU_SCHED_INIT_DIAGNOSTIC_CONSTRUCTOR_RETURN);
	EXPECT((int)REVISION_INIT_FIELD_NONE ==
	       (int)NV_GPU_SCHED_INIT_DIAGNOSTIC_FIELD_NONE);
	EXPECT((int)REVISION_INIT_FIELD_TIMESLICE ==
	       (int)NV_GPU_SCHED_INIT_DIAGNOSTIC_FIELD_TIMESLICE);
	EXPECT((int)REVISION_INIT_FIELD_INTERLEAVE ==
	       (int)NV_GPU_SCHED_INIT_DIAGNOSTIC_FIELD_INTERLEAVE);
	EXPECT(sizeof(struct revision_init_trace_event) == 192);
	return 0;
}

int main(void)
{
	if (test_diagnostic_layout() || test_gsp_layout() || test_constants())
		return 1;
	printf("revision_init_trace: 3 cases, %u assertions passed (CPU only)\n",
	       assertions);
	return 0;
}
