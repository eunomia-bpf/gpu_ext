/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "observer.h"
#define STALE_STATE_575_TYPES_PROVIDED
#include "../stale_state_policy_model.h"

char LICENSE[] SEC("license") = "GPL";

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, unsigned int);
    __type(value, struct stale_state_v1_observer_config);
} observer_config SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, unsigned int);
    __type(value, struct stale_state_v1_observer_metrics);
} observer_metrics SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 64 * 1024 * 1024);
} observer_events SEC(".maps");

static __always_inline struct stale_state_v1_observer_metrics *metrics(void)
{
    unsigned int key = 0;

    return bpf_map_lookup_elem(&observer_metrics, &key);
}

SEC("fentry/uvm_stale_state_v1_diagnostic")
int BPF_PROG(stale_state_v1_diagnostic_observer,
             const struct uvm_stale_state_v1_diagnostic *driver_diagnostic)
{
    struct stale_state_v1_observer_metrics *m = metrics();
    struct stale_state_v1_observer_config *config;
    struct stale_state_v1_observer_event *event;
    struct uvm_stale_state_v1_diagnostic diagnostic = {};
    unsigned long long pid_tgid = bpf_get_current_pid_tgid();
    unsigned int key = 0;

    (void)ctx;
    if (!m)
        return 0;
    m->diagnostic_calls++;
    config = bpf_map_lookup_elem(&observer_config, &key);
    if (!config || !config->target_tgid)
        return 0;
    if (bpf_probe_read_kernel(&diagnostic, sizeof(diagnostic),
                              driver_diagnostic)) {
        m->read_errors++;
        return 0;
    }
    if (diagnostic.owner_tgid != config->target_tgid) {
        m->foreign_tgid++;
        return 0;
    }
    if (diagnostic.diagnostic_phase == STALE_STATE_V1_DIAG_SELECTED) {
        m->selected_seen++;
        return 0;
    }
    if (diagnostic.diagnostic_phase != STALE_STATE_V1_DIAG_FINISHED) {
        m->phase_errors++;
        return 0;
    }
    m->finished_seen++;
    event = bpf_ringbuf_reserve(&observer_events, sizeof(*event), 0);
    if (!event) {
        m->ringbuf_drops++;
        return 0;
    }
    event->observed_mono_ns = bpf_ktime_get_ns();
    event->pid_tgid = pid_tgid;
    __builtin_memcpy(&event->diagnostic, &diagnostic, sizeof(diagnostic));
    bpf_ringbuf_submit(event, 0);
    m->records_emitted++;
    return 0;
}

SEC("struct_ops/gpu_stale_state_prefetch_v1")
int BPF_PROG(stale_state_prefetch_v1,
             uvm_stale_state_v1_decision_ctx_t *decision_ctx)
{
    struct uvm_stale_state_v1_input input = {};
    struct stale_state_575_snapshot snapshot = {};
    struct stale_state_575_decision decision = {};
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
    action = stale_state_575_choose(&snapshot, input.decision_mono_ns, &decision);
    if (action == STALE_STATE_575_ACTION_REJECT)
        return action;
    if (bpf_gpu_stale_state_v1_request(decision_ctx, (unsigned int)action) != 0)
        return STALE_STATE_575_ACTION_REJECT;
    return action;
}

SEC(".struct_ops")
struct gpu_mem_ops stale_state_v1_ops = {
    .gpu_stale_state_prefetch_v1 = (void *)stale_state_prefetch_v1,
};
