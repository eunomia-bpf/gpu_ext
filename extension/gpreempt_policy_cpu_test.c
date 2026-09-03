/* SPDX-License-Identifier: GPL-2.0 */
/* Execute the actual policy C functions with bounded maps and a recording
 * setter. This is NOT the kernel verifier, CUDA, GSP, or a concurrency test. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef unsigned int __u32;
typedef unsigned long long __u64;
#define GP_POLICY_CPU_TEST
#define BPF_NO_KFUNC_PROTOTYPES
#define SEC(name)
#define BPF_PROG(name, ...) name(__VA_ARGS__)
#define __uint(name, value) int (*name)[value]
#define __type(name, type) type *name
#define BPF_MAP_TYPE_HASH 1
#define BPF_MAP_TYPE_ARRAY 2
#define BPF_ANY 0
#define BPF_NOEXIST 1
struct pt_regs { __u64 args[4], result; };
#define PT_REGS_PARM1(ctx) ((ctx)->args[0])
#define PT_REGS_PARM2(ctx) ((ctx)->args[1])
#define PT_REGS_PARM3(ctx) ((ctx)->args[2])
#define PT_REGS_PARM4(ctx) ((ctx)->args[3])
#define PT_REGS_RC(ctx) ((ctx)->result)
struct nv_gpu_task_init_ctx;
static __u64 bpf_get_current_pid_tgid(void);
static void *bpf_map_lookup_elem(void *, const void *);
static int bpf_map_update_elem(void *, const void *, const void *, __u64);
static int bpf_map_delete_elem(void *, const void *);
static int bpf_probe_read_user(void *, __u32, const void *);
static int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *, __u64);
#include "gpreempt_policy.bpf.c"

struct slot { int used; unsigned char key[16]; _Alignas(8) unsigned char value[64]; };
struct mock_map { struct slot slots[128]; size_t value_size, key_size; };
static struct mock_map mocks[4];
static __u64 counters[GP_STAT_COUNT], identity, last_timeslice;
static unsigned setter_calls, cases, assertions;
static int setter_failure, read_failure;
static void *map_failure;
#define CHECK(condition) do { ++assertions; assert(condition); } while (0)

static struct mock_map *map_state(void *map)
{
    if (map == &scopes) return &mocks[0];
    if (map == &records) return &mocks[1];
    if (map == &pending) return &mocks[2];
    CHECK(map == &tsgs);
    return &mocks[3];
}
static __u64 bpf_get_current_pid_tgid(void) { return identity; }
static void *bpf_map_lookup_elem(void *map, const void *key)
{
    if (map == &stats) {
        __u32 index = *(const __u32 *)key;
        return index < GP_STAT_COUNT ? &counters[index] : NULL;
    }
    struct mock_map *mock = map_state(map);
    for (unsigned i = 0; i < 128; ++i)
        if (mock->slots[i].used && !memcmp(mock->slots[i].key, key, mock->key_size)) return mock->slots[i].value;
    return NULL;
}
static int bpf_map_update_elem(void *map, const void *key, const void *value, __u64 flags)
{
    if (map == map_failure) return -1;
    struct mock_map *mock = map_state(map);
    void *old = bpf_map_lookup_elem(map, key);
    if (old) {
        if (flags == BPF_NOEXIST) return -1;
        memcpy(old, value, mock->value_size);
        return 0;
    }
    for (unsigned i = 0; i < 128; ++i) {
        if (mock->slots[i].used) continue;
        mock->slots[i].used = 1;
        memcpy(mock->slots[i].key, key, mock->key_size);
        memcpy(mock->slots[i].value, value, mock->value_size);
        return 0;
    }
    return -1;
}
static int bpf_map_delete_elem(void *map, const void *key)
{
    struct mock_map *mock = map_state(map);
    for (unsigned i = 0; i < 128; ++i) {
        if (mock->slots[i].used && !memcmp(mock->slots[i].key, key, mock->key_size)) {
            mock->slots[i].used = 0;
            return 0;
        }
    }
    return -1;
}
static int bpf_probe_read_user(void *to, __u32 size, const void *from)
{
    if (read_failure) return -1;
    memcpy(to, from, size);
    return 0;
}
static int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *input, __u64 value)
{
    CHECK(input->engine_type >= 1 && input->engine_type <= 8);
    ++setter_calls;
    last_timeslice = value;
    return setter_failure;
}
static void reset(void)
{
    ++cases;
    memset(mocks, 0, sizeof(mocks));
    memset(counters, 0, sizeof(counters));
    memset(bind_history, 0, sizeof(bind_history));
    bind_history_count = 0;
    mocks[0].value_size = sizeof(struct gp_scope);
    mocks[1].value_size = sizeof(struct gp_record);
    mocks[2].value_size = sizeof(struct gp_pending);
    mocks[3].value_size = sizeof(struct gp_tsg);
    for (unsigned i = 0; i < 4; ++i) mocks[i].key_size = i == 3 ? sizeof(struct gp_tsg_key) : 8;
    identity = (42ULL << 32) | 101;
    setter_calls = 0;
    setter_failure = read_failure = 0;
    map_failure = NULL;
}
static struct gp_scope *begin(unsigned role)
{
    struct pt_regs marker = { .args = {role} };
    scope_enter(&marker);
    return bpf_map_lookup_elem(&scopes, &identity);
}
static void allocation_enter(__u32 *header, unsigned size, int transfer)
{
    struct { __u32 command, size; __u64 pointer; } xfer = {0x2b, size, (__u64)header};
    struct pt_regs args = { .args = {0, (size << 16) | 0x2b, (__u64)header} };
    if (transfer) { args.args[1] = (16 << 16) | 211; args.args[2] = (__u64)&xfer; }
    ioctl_enter(&args);
}
static void successful_context(unsigned role, unsigned size, int transfer, unsigned engine)
{
    reset();
    __u32 header[12] = {17, 3, 900, 0xa06c};
    struct gp_scope *scope = begin(role);
    CHECK(scope && scope->role == role);
    allocation_enter(header, size, transfer);
    struct nv_gpu_task_init_ctx init = {.tsg_id = 123, .engine_type = 9};
    CHECK(gp_task_init(&init) == 0 && setter_calls == 0 && counters[GP_OTHER_ENGINE] == 1);
    init.engine_type = engine;
    CHECK(gp_task_init(&init) == 1 && setter_calls == 1);
    CHECK(last_timeslice == (role == GP_LC ? 1000000 : 1));
    struct pt_regs exit = {};
    ioctl_exit(&exit);
    struct gp_handle_key key = {17, 900};
    struct gp_record *record = bpf_map_lookup_elem(&records, &key);
    CHECK(record && record->engine == engine && record->role == role && record->pid_tgid == identity);
    struct pt_regs registration = {.args = {0x4321, 17, 900, role}};
    register_context(&registration);
    CHECK(scope->registered == 1 && record->cuda_context == 0x4321 && record->registered == 1);
    CHECK(!scope->errors && counters[GP_ALLOC_CAPTURED] == 1 && counters[GP_REGISTERED] == 1);
    scope_leave(&exit);
    CHECK(!bpf_map_lookup_elem(&scopes, &identity) && !bpf_map_lookup_elem(&pending, &identity));
    struct nv_gpu_bind_ctx binding = {.tsg_id = 123, .timeslice_us = last_timeslice};
    gp_bind(&binding);
    CHECK(counters[GP_BIND_MATCH] == 1 && counters[GP_BIND_MISMATCH] == 0);
    CHECK(bind_history_count == 1 && bind_history[0].expected_us == last_timeslice &&
          bind_history[0].observed_us == last_timeslice && bind_history[0].handle_known);
    ++binding.timeslice_us;
    gp_bind(&binding);
    CHECK(counters[GP_BIND_MISMATCH] == 1);
    struct nv_gpu_task_destroy_ctx destroy = {.tsg_id = 123, .engine_type = engine};
    gp_destroy(&destroy);
    struct gp_tsg_key tsg_key = {.tsg_id = 123};
    CHECK(!bpf_map_lookup_elem(&records, &key) && !bpf_map_lookup_elem(&tsgs, &tsg_key));
    CHECK(counters[GP_DESTROY] == 1);
}

static void cross_runlist_identity(void)
{
    reset();
    for (unsigned role = 0; role < 2; ++role) {
        identity = (42ULL << 32) | (101 + role);
        CHECK(begin(role) != NULL);
        __u32 header[12] = {17, 3, 900 + role, 0xa06c};
        allocation_enter(header, 32, 0);
        struct nv_gpu_task_init_ctx init = {.tsg_id = 123, .runlist_id = 5 + role, .engine_type = 1};
        CHECK(gp_task_init(&init) == 1);
        struct pt_regs exit = {};
        ioctl_exit(&exit);
        struct pt_regs registration = {.args = {0x4321 + role, 17, 900 + role, role}};
        register_context(&registration);
        scope_leave(&exit);
    }
    CHECK(counters[GP_REGISTERED] == 2 && counters[GP_ALLOC_ERROR] == 0);
    struct nv_gpu_bind_ctx binding = {.tsg_id = 123, .runlist_id = 7, .timeslice_us = 3000};
    gp_bind(&binding); // A CE runlist reuses the same grpID: no false shadow mismatch.
    CHECK(bind_history_count == 0 && counters[GP_BIND_MISMATCH] == 0);
    struct nv_gpu_task_destroy_ctx destroy = {.tsg_id = 123, .runlist_id = 7, .engine_type = 9};
    gp_destroy(&destroy); // CE destruction must not delete either GR allocation.
    CHECK(counters[GP_DESTROY] == 0);
    for (unsigned role = 0; role < 2; ++role) {
        struct gp_tsg_key key = {.tsg_id = 123, .runlist_id = 5 + role};
        struct gp_tsg *record = bpf_map_lookup_elem(&tsgs, &key);
        CHECK(record && record->record.role == role);
        binding.runlist_id = 5 + role;
        binding.timeslice_us = role == GP_LC ? 1000000 : 1;
        gp_bind(&binding);
        CHECK(bind_history_count == role + 1 && counters[GP_BIND_MISMATCH] == 0);
        // Wrong engine cannot delete a GR record even when both numeric fields match.
        destroy.runlist_id = 5 + role;
        gp_destroy(&destroy);
        CHECK(bpf_map_lookup_elem(&tsgs, &key) != NULL);
        destroy.engine_type = 1;
        gp_destroy(&destroy);
        CHECK(!bpf_map_lookup_elem(&tsgs, &key));
        struct gp_handle_key handle = {17, 900 + role};
        CHECK(!bpf_map_lookup_elem(&records, &handle));
        destroy.engine_type = 9;
    }
    CHECK(counters[GP_DESTROY] == 2 && counters[GP_BIND_MATCH] == 2);
}

int main(void)
{
    cross_runlist_identity();
    for (unsigned role = 0; role <= 1; ++role)
        for (unsigned size = 32; size <= 48; size += 16)
            for (int transfer = 0; transfer <= 1; ++transfer)
                for (unsigned engine = 1; engine <= 8; ++engine)
                    successful_context(role, size, transfer, engine);
    reset();
    struct nv_gpu_task_init_ctx init = {.tsg_id = 123, .engine_type = 1};
    CHECK(gp_task_init(&init) == 0 && !setter_calls); // Unmarked native context untouched.
    reset();
    CHECK(!begin(2) && counters[GP_SCOPE_ERROR] == 1);
    reset();
    struct gp_scope *scope = begin(0);
    CHECK(begin(1) == scope && scope->role == 0 && scope->errors == 1); // No nested override.
    reset();
    scope = begin(0);
    init.engine_type = 0;
    CHECK(gp_task_init(&init) == 0 && !setter_calls && scope->errors == 1);
    reset();
    scope = begin(0);
    init.engine_type = 1;
    CHECK(gp_task_init(&init) == 0 && !setter_calls && scope->errors == 1); // No ioctl correlation.
    for (unsigned failure = 0; failure < 7; ++failure) {
        reset();
        scope = begin(0);
        __u32 header[12] = {17, 3, 900, 0xa06c};
        allocation_enter(header, 48, 1);
        if (failure == 0) setter_failure = -1;
        if (failure == 1) map_failure = &tsgs;
        int applied = gp_task_init(&init);
        if (failure < 2) { CHECK(!applied && scope->errors == 1); continue; }
        CHECK(applied == 1);
        struct pt_regs exit = {};
        if (failure == 2) exit.result = (__u64)-1;
        if (failure == 3) header[10] = 5; // RM status failure even when ioctl succeeds.
        if (failure == 4) header[2] = 0;
        if (failure == 5) read_failure = 1;
        if (failure == 6) map_failure = &records;
        ioctl_exit(&exit);
        struct gp_handle_key key = {17, 900};
        CHECK(!bpf_map_lookup_elem(&records, &key) && scope->errors == 1);
    }
    reset();
    scope = begin(0);
    __u32 header[12] = {17, 3, 900, 0xa06c};
    allocation_enter(header, 32, 0);
    CHECK(gp_task_init(&init) == 1);
    struct pt_regs exit = {};
    ioctl_exit(&exit);
    ++identity;
    struct pt_regs registration = {.args = {0x4321, 17, 900, 0}};
    register_context(&registration);
    CHECK(!scope->registered && counters[GP_REGISTER_ERROR] == 1); // Another thread cannot claim it.
    --identity;
    registration.args[3] = 1;
    register_context(&registration);
    CHECK(!scope->registered && scope->errors == 1); // Role mismatch.
    thread_exit(NULL);
    CHECK(!bpf_map_lookup_elem(&scopes, &identity));
    printf("{\"test\":\"actual_gpreempt_policy_c_with_mock_helpers\",\"cases\":%u,"
           "\"assertions\":%u,\"gpu_executed\":false,\"verifier_executed\":false}\n", cases, assertions);
    return 0;
}
