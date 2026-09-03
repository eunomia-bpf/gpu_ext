/* SPDX-License-Identifier: GPL-2.0 */
/* Actual process-policy callback with mock maps/helpers; no GPU/verifier. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
typedef unsigned int __u32;
typedef unsigned long long __u64;
typedef unsigned char __u8;
#define GPU_SCHED_CPU_TEST
#define BPF_NO_KFUNC_PROTOTYPES
#define SEC(name)
#define BPF_PROG(name, ...) name(__VA_ARGS__)
#define __uint(name, value) int (*name)[value]
#define __type(name, type) __typeof__(type) *name
#define BPF_MAP_TYPE_HASH 1
#define BPF_MAP_TYPE_ARRAY 2
#define BPF_MAP_TYPE_RINGBUF 27
#define BPF_ANY 0
#define bpf_printk(...) ((void)0)
struct nv_gpu_task_init_ctx;
struct nv_gpu_timeslice_control_ctx;
static void *bpf_map_lookup_elem(void *, const void *);
static int bpf_map_update_elem(void *, const void *, const void *, __u64);
static int bpf_get_current_comm(void *, __u32);
static int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *, __u64);
static int bpf_nv_gpu_set_interleave(struct nv_gpu_task_init_ctx *, __u32);
static int bpf_nv_gpu_override_timeslice(struct nv_gpu_timeslice_control_ctx *, __u64);
#include "gpu_sched_set_timeslices.bpf.c"
static __u64 values[11], configured_timeslice, last_timeslice;
static char current_comm[16];
static unsigned calls, assertions;
static int setter_failure;
#define CHECK(value) do { ++assertions; assert(value); } while (0)
static void *bpf_map_lookup_elem(void *map, const void *key)
{
    if (map == &stats) return &values[*(__u32 *)key];
    if (map == &process_timeslice && !strcmp(key, "bench_lc")) return &configured_timeslice;
    return NULL;
}
static int bpf_map_update_elem(void *m, const void *k, const void *v, __u64 f) { return 0; }
static int bpf_get_current_comm(void *to, __u32 size) { memcpy(to, current_comm, size); return 0; }
static int bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *c, __u64 v) { return 0; }
static int bpf_nv_gpu_set_interleave(struct nv_gpu_task_init_ctx *c, __u32 v) { return 0; }
static int bpf_nv_gpu_override_timeslice(struct nv_gpu_timeslice_control_ctx *c, __u64 v)
{ ++calls; last_timeslice = v; return setter_failure; }
int main(void)
{
    struct nv_gpu_timeslice_control_ctx control = {.requested_timeslice_us = 2048, .phase = 1};
    strcpy(current_comm, "native"); configured_timeslice = 1000000;
    on_timeslice_control(&control); CHECK(calls == 0); // Unmatched process untouched.
    strcpy(current_comm, "bench_lc");
    on_timeslice_control(&control); CHECK(calls == 1 && last_timeslice == 1000000);
    CHECK(control.requested_timeslice_us == 2048 && values[STAT_CONTROL_OVERRIDE] == 1);
    configured_timeslice = 200; control.engine_type = 9;
    on_timeslice_control(&control); CHECK(calls == 2 && last_timeslice == 200); // Existing all-engine policy.
    control.phase = 0; on_timeslice_control(&control); CHECK(calls == 2);
    control.phase = 1; configured_timeslice = 0;
    on_timeslice_control(&control); CHECK(calls == 2);
    configured_timeslice = 1000001; setter_failure = -1;
    on_timeslice_control(&control); CHECK(calls == 3 && values[STAT_SETTER_ERROR] == 1);
    CHECK(values[STAT_CONTROL_OVERRIDE] == 2 && control.requested_timeslice_us == 2048);
    on_timeslice_control(NULL); CHECK(calls == 3);
    printf("process_timeslice_control: %u assertions passed (CPU mock helpers, no GPU)\n", assertions);
}
