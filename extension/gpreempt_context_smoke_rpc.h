/* SPDX-License-Identifier: GPL-2.0 */
#ifndef GPREEMPT_CONTEXT_SMOKE_RPC_H
#define GPREEMPT_CONTEXT_SMOKE_RPC_H
struct gp_rpc_event {
    unsigned long long pid_tgid, timeslice_us, entered_ns, elapsed_ns;
    unsigned int hclient, hobject, command, params_size;
    unsigned int issue_count, wait_count, wait_status, return_status, read_error, wait_errors;
};
#endif
