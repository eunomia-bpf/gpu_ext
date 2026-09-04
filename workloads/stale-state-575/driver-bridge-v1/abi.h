/* SPDX-License-Identifier: GPL-2.0 */
#ifndef STALE_STATE_575_DRIVER_BRIDGE_V1_ABI_H
#define STALE_STATE_575_DRIVER_BRIDGE_V1_ABI_H

/*
 * Source-side mirror of the version-1 driver ABI. The driver patch remains
 * authoritative; these assertions make layout drift fail during BPF builds.
 */
#define STALE_STATE_DRIVER_V1_ABI_VERSION 1U

struct stale_state_driver_v1_snapshot {
    unsigned long long sequence;
    unsigned long long source_mono_ns;
    unsigned long long published_mono_ns;
    unsigned int phase;
    unsigned int reserved;
};

struct uvm_stale_state_v1_input {
    struct stale_state_driver_v1_snapshot snapshot;
    unsigned long long generation;
    unsigned long long decision_sequence;
    unsigned long long decision_mono_ns;
    unsigned long long page_index;
    unsigned long long max_first;
    unsigned long long max_outer;
    unsigned int abi_version;
    unsigned int reserved;
};

struct stale_state_driver_v1_u32_request {
    unsigned char attempted;
    unsigned char conflict;
    unsigned int value;
};

typedef struct uvm_stale_state_v1_decision_ctx {
    struct uvm_stale_state_v1_input input;
    struct stale_state_driver_v1_u32_request action_request;
    unsigned int request_calls;
    unsigned int request_cookie;
} uvm_stale_state_v1_decision_ctx_t;

typedef unsigned short uvm_page_index_t;
typedef struct uvm_perf_prefetch_bitmap_tree uvm_perf_prefetch_bitmap_tree_t;
typedef struct {
    uvm_page_index_t first;
    uvm_page_index_t outer;
} uvm_va_block_region_t;

typedef struct uvm_bpf_prefetch_decision {
    unsigned char attempted;
    unsigned char conflict;
    unsigned long long first;
    unsigned long long outer;
} uvm_bpf_prefetch_decision_t;

typedef struct uvm_pmm_gpu_struct uvm_pmm_gpu_t;
typedef struct uvm_gpu_chunk_struct uvm_gpu_chunk_t;
typedef struct uvm_bpf_pmm_decision_ctx uvm_bpf_pmm_decision_ctx_t;
struct list_head;

struct gpu_mem_ops {
    int (*gpu_test_trigger)(const char *, int);
    int (*gpu_page_prefetch)(uvm_page_index_t,
                             uvm_perf_prefetch_bitmap_tree_t *,
                             uvm_va_block_region_t *,
                             uvm_bpf_prefetch_decision_t *);
    int (*gpu_page_prefetch_iter)(uvm_perf_prefetch_bitmap_tree_t *,
                                  uvm_va_block_region_t *,
                                  uvm_va_block_region_t *,
                                  unsigned int,
                                  uvm_bpf_prefetch_decision_t *);
    int (*gpu_block_activate)(uvm_pmm_gpu_t *,
                              uvm_gpu_chunk_t *,
                              uvm_bpf_pmm_decision_ctx_t *);
    int (*gpu_block_access)(uvm_pmm_gpu_t *,
                            uvm_gpu_chunk_t *,
                            uvm_bpf_pmm_decision_ctx_t *);
    int (*gpu_evict_prepare)(uvm_pmm_gpu_t *,
                             struct list_head *,
                             struct list_head *);
    int (*gpu_stale_state_prefetch_v1)(uvm_stale_state_v1_decision_ctx_t *);
};

_Static_assert(sizeof(struct stale_state_driver_v1_snapshot) == 32,
               "snapshot ABI size");
_Static_assert(__builtin_offsetof(struct stale_state_driver_v1_snapshot, phase) == 24,
               "snapshot phase offset");
_Static_assert(sizeof(struct uvm_stale_state_v1_input) == 88,
               "read-only input ABI size");
_Static_assert(sizeof(struct stale_state_driver_v1_u32_request) == 8,
               "private request ABI size");
_Static_assert(sizeof(uvm_stale_state_v1_decision_ctx_t) == 104,
               "decision context ABI size");
_Static_assert(__builtin_offsetof(uvm_stale_state_v1_decision_ctx_t,
                                  action_request) == 88,
               "private suffix offset");
_Static_assert(__builtin_offsetof(struct gpu_mem_ops,
                                  gpu_stale_state_prefetch_v1) ==
                   6 * sizeof(void *),
               "append-only struct_ops offset");
_Static_assert(sizeof(struct gpu_mem_ops) == 7 * sizeof(void *),
               "struct_ops ABI size");

#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif

extern int bpf_gpu_stale_state_v1_request(
    uvm_stale_state_v1_decision_ctx_t *decision_ctx,
    unsigned int action) __weak __ksym;

#endif
