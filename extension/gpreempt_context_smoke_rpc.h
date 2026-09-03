/* SPDX-License-Identifier: GPL-2.0 */
#ifndef GPREEMPT_CONTEXT_SMOKE_RPC_H
#define GPREEMPT_CONTEXT_SMOKE_RPC_H
/* Layout checked against the actual 575 header by the CPU-built canary. */
struct gp_gsp_completion {
    unsigned long long input_value;
    unsigned int hClient, hObject, command, input_size, wire_size, input_valid;
    unsigned int transport_status, gsp_status, gsp_status_valid, reserved;
};
struct gp_rpc_event {
    unsigned long long pid_tgid, completed_ns;
    struct gp_gsp_completion completion;
};
#endif
