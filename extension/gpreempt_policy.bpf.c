/* SPDX-License-Identifier: GPL-2.0 */
#ifndef GP_POLICY_CPU_TEST
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#endif
#include "gpu_sched_set_timeslices.h"
#define GP_BPF_ONLY
#include "gpreempt_bridge.h"

/* Explicit single-GPU experiment scope: task_init's TSG ID has no GPU identity.
 * 575 uses RM_ENGINE_TYPE, NOT the NV_ENGINE_TYPE_* aliases in the old header.
 * RM GR0..GR7 = 1..8; COPY0 starts at 9. Unknown engine 0 is fail-closed. */
struct gp_pending {
    __u64 user_pointer;
    __u64 tsg_id;
    __u32 hclient;
    __u32 status_offset;
    __u32 gr_seen;
    __u32 reserved;
};
struct gp_tsg { struct gp_record record; struct gp_handle_key handles; };

struct {
    __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 64);
    __type(key, __u64); __type(value, struct gp_scope);
} scopes SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 128);
    __type(key, struct gp_handle_key); __type(value, struct gp_record);
} records SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 64);
    __type(key, __u64); __type(value, struct gp_pending);
} pending SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 128);
    __type(key, __u64); __type(value, struct gp_tsg);
} tsgs SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, GP_STAT_COUNT);
    __type(key, __u32); __type(value, __u64);
} stats SEC(".maps");

static __always_inline void count(__u32 key)
{
    __u64 *value = bpf_map_lookup_elem(&stats, &key);
    if (value) __sync_fetch_and_add(value, 1);
}
static __always_inline void error(struct gp_scope *scope, __u32 key)
{
    if (scope) scope->errors++;
    count(key);
}

SEC("uprobe")
int scope_enter(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_scope *old = bpf_map_lookup_elem(&scopes, &id);
    struct gp_scope scope = { .role = (__u32)PT_REGS_PARM1(ctx) };
    if (old) { error(old, GP_SCOPE_ERROR); return 0; }
    if (scope.role > GP_BE) { count(GP_SCOPE_ERROR); return 0; }
    if (bpf_map_update_elem(&scopes, &id, &scope, BPF_NOEXIST)) count(GP_MAP_ERROR);
    else count(GP_SCOPE_ENTER);
    return 0;
}

SEC("uprobe")
int register_context(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_scope *scope = bpf_map_lookup_elem(&scopes, &id);
    struct gp_handle_key key = {
        .hclient = (__u32)PT_REGS_PARM2(ctx), .htsg = (__u32)PT_REGS_PARM3(ctx),
    };
    struct gp_record *record = bpf_map_lookup_elem(&records, &key);
    __u64 context = PT_REGS_PARM1(ctx);
    __u32 role = PT_REGS_PARM4(ctx);
    if (!scope || !record || scope->errors || scope->gr_inits != 1 ||
        scope->registered || record->registered || role > GP_BE ||
        scope->role != role || record->role != role || record->pid_tgid != id || !context) {
        error(scope, GP_REGISTER_ERROR);
        return 0;
    }
    record->cuda_context = context;
    record->registered = 1;
    scope->registered = 1;
    count(GP_REGISTERED);
    return 0;
}

SEC("uprobe")
int scope_leave(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_scope *scope = bpf_map_lookup_elem(&scopes, &id);
    if (!scope || scope->errors || scope->gr_inits != 1 || scope->registered != 1) {
        error(scope, GP_SCOPE_ERROR);
        return 0;
    }
    bpf_map_delete_elem(&scopes, &id);
    bpf_map_delete_elem(&pending, &id);
    count(GP_SCOPE_LEAVE);
    return 0;
}

SEC("kprobe/nvidia_unlocked_ioctl")
int ioctl_enter(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_scope *scope = bpf_map_lookup_elem(&scopes, &id);
    struct gp_pending allocation = {};
    __u32 command = PT_REGS_PARM2(ctx), size = (command >> 16) & 0x3fff;
    __u64 pointer = PT_REGS_PARM3(ctx);
    __u32 number = command & 0xff, hclass = 0;
    if (!scope) return 0;
    bpf_map_delete_elem(&pending, &id);
    if (number == 211) {
        struct { __u32 command, size; __u64 pointer; } transfer = {};
        if (bpf_probe_read_user(&transfer, sizeof(transfer), (void *)pointer)) {
            error(scope, GP_ALLOC_ERROR); return 0;
        }
        if (transfer.command != 0x2b) return 0;
        pointer = transfer.pointer;
        size = transfer.size;
    } else if (number != 0x2b) return 0;
    if (!pointer || (size != 32 && size != 48)) return 0;
    if (bpf_probe_read_user(&hclass, sizeof(hclass), (void *)(pointer + 12))) {
        error(scope, GP_ALLOC_ERROR); return 0;
    }
    if (hclass != 0xa06c) return 0;
    if (bpf_probe_read_user(&allocation.hclient, sizeof(allocation.hclient), (void *)pointer) ||
        !allocation.hclient) { error(scope, GP_ALLOC_ERROR); return 0; }
    allocation.user_pointer = pointer;
    allocation.status_offset = size == 32 ? 28 : 40;
    if (bpf_map_update_elem(&pending, &id, &allocation, BPF_NOEXIST)) error(scope, GP_MAP_ERROR);
    return 0;
}

SEC("struct_ops/gp_task_init")
int BPF_PROG(gp_task_init, struct nv_gpu_task_init_ctx *init)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_scope *scope = bpf_map_lookup_elem(&scopes, &id);
    if (!scope || !init) return 0;
    if (!init->engine_type) { error(scope, GP_UNKNOWN_ENGINE); return 0; }
    if (init->engine_type > 8) { count(GP_OTHER_ENGINE); return 0; }
    struct gp_pending *allocation = bpf_map_lookup_elem(&pending, &id);
    if (!allocation || allocation->gr_seen || scope->role > GP_BE || scope->errors || scope->gr_inits) {
        error(scope, GP_ALLOC_ERROR); return 0;
    }
    struct gp_tsg target = {};
    __u64 tsg_id = init->tsg_id;
    target.record.pid_tgid = id;
    target.record.tsg_id = tsg_id;
    target.record.role = scope->role;
    target.record.engine = init->engine_type;
    target.record.timeslice_us = scope->role == GP_LC ? 1000000 : 1;
    if (bpf_map_update_elem(&tsgs, &tsg_id, &target, BPF_NOEXIST)) {
        error(scope, GP_MAP_ERROR); return 0;
    }
    if (bpf_nv_gpu_set_timeslice(init, target.record.timeslice_us)) {
        bpf_map_delete_elem(&tsgs, &tsg_id);
        error(scope, GP_SETTER_ERROR); return 0;
    }
    allocation->tsg_id = tsg_id;
    allocation->gr_seen = 1;
    scope->gr_inits++;
    count(GP_GR_INIT);
    count(GP_TIMESLICE_OK);
    return 1;
}

SEC("kretprobe/nvidia_unlocked_ioctl")
int ioctl_exit(struct pt_regs *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_pending *source = bpf_map_lookup_elem(&pending, &id);
    if (!source) return 0;
    struct gp_pending allocation = *source;
    bpf_map_delete_elem(&pending, &id);
    if (!allocation.gr_seen) return 0; // Non-GR allocations are not policy targets.
    struct gp_scope *scope = bpf_map_lookup_elem(&scopes, &id);
    struct gp_tsg *target = bpf_map_lookup_elem(&tsgs, &allocation.tsg_id);
    struct gp_handle_key key = { .hclient = allocation.hclient };
    __u32 nvstatus = ~0U;
    if (!scope || !target || PT_REGS_RC(ctx) != 0 ||
        bpf_probe_read_user(&nvstatus, sizeof(nvstatus),
                            (void *)(allocation.user_pointer + allocation.status_offset)) || nvstatus ||
        bpf_probe_read_user(&key.htsg, sizeof(key.htsg), (void *)(allocation.user_pointer + 8)) || !key.htsg) {
        bpf_map_delete_elem(&tsgs, &allocation.tsg_id);
        error(scope, GP_ALLOC_ERROR); return 0;
    }
    target->handles = key;
    if (bpf_map_update_elem(&records, &key, &target->record, BPF_NOEXIST)) {
        error(scope, GP_MAP_ERROR); return 0;
    }
    count(GP_ALLOC_CAPTURED);
    return 0;
}

SEC("struct_ops/gp_bind")
int BPF_PROG(gp_bind, struct nv_gpu_bind_ctx *binding)
{
    if (!binding) return 0;
    __u64 id = binding->tsg_id;
    struct gp_tsg *target = bpf_map_lookup_elem(&tsgs, &id);
    if (target) count(binding->timeslice_us == target->record.timeslice_us ? GP_BIND_MATCH : GP_BIND_MISMATCH);
    return 0; // This observation is host shadow state, NOT proof of hardware execution.
}

SEC("struct_ops/gp_destroy")
int BPF_PROG(gp_destroy, struct nv_gpu_task_destroy_ctx *destroy)
{
    if (!destroy) return 0;
    __u64 id = destroy->tsg_id;
    struct gp_tsg *target = bpf_map_lookup_elem(&tsgs, &id);
    if (target) {
        struct gp_handle_key handles = target->handles;
        bpf_map_delete_elem(&records, &handles);
        bpf_map_delete_elem(&tsgs, &id);
        count(GP_DESTROY);
    }
    return 0;
}

SEC("tracepoint/sched/sched_process_exit")
int thread_exit(void *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct gp_scope *scope = bpf_map_lookup_elem(&scopes, &id);
    if (scope) { count(GP_SCOPE_ERROR); bpf_map_delete_elem(&scopes, &id); }
    bpf_map_delete_elem(&pending, &id);
    return 0;
}

SEC(".struct_ops")
struct nv_gpu_sched_ops gpreempt_ops = {
    .on_task_init = (void *)gp_task_init,
    .on_bind = (void *)gp_bind,
    .on_task_destroy = (void *)gp_destroy,
};
char LICENSE[] SEC("license") = "GPL";
