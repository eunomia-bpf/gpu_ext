/* SPDX-License-Identifier: GPL-2.0 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "abi.h"
#define STALE_STATE_575_TYPES_PROVIDED
#include "../stale_state_policy_model.h"

char LICENSE[] SEC("license") = "GPL";

/*
 * The context is driver-owned. The program copies only the immutable input,
 * evaluates the canonical pure model, and submits one action through the
 * trusted setter. It never stores through decision_ctx.
 */
SEC("struct_ops/gpu_stale_state_prefetch_v1")
int BPF_PROG(stale_state_prefetch_v1,
             uvm_stale_state_v1_decision_ctx_t *decision_ctx)
{
    struct uvm_stale_state_v1_input input = {0};
    struct stale_state_575_snapshot snapshot = {0};
    struct stale_state_575_decision decision = {0};
    enum stale_state_575_action action;

    (void)ctx;
    if (bpf_probe_read_kernel(&input, sizeof(input), decision_ctx) != 0)
        return STALE_STATE_575_ACTION_REJECT;
    if (input.abi_version != STALE_STATE_DRIVER_V1_ABI_VERSION ||
        input.reserved != 0)
        return STALE_STATE_575_ACTION_REJECT;

    snapshot.sequence = input.snapshot.sequence;
    snapshot.source_mono_ns = input.snapshot.source_mono_ns;
    snapshot.published_mono_ns = input.snapshot.published_mono_ns;
    snapshot.phase = input.snapshot.phase;
    snapshot.reserved = input.snapshot.reserved;
    action = stale_state_575_choose(&snapshot,
                                    input.decision_mono_ns,
                                    &decision);
    if (action == STALE_STATE_575_ACTION_REJECT)
        return action;

    if (bpf_gpu_stale_state_v1_request(decision_ctx,
                                       (unsigned int)action) != 0)
        return STALE_STATE_575_ACTION_REJECT;
    return action;
}

SEC(".struct_ops")
struct gpu_mem_ops stale_state_v1_ops = {
    .gpu_stale_state_prefetch_v1 = (void *)stale_state_prefetch_v1,
};
