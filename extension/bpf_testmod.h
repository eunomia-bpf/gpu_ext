#ifndef _BPF_TESTMOD_H
#define _BPF_TESTMOD_H

/* Note: This header assumes uvm_types.h is included first to provide type definitions */

typedef struct uvm_bpf_prefetch_decision
{
    __u8 attempted;
    __u8 conflict;
    __u64 first;
    __u64 outer;
} uvm_bpf_prefetch_decision_t;

_Static_assert(sizeof(uvm_bpf_prefetch_decision_t) == 24,
               "prefetch decision ABI");
_Static_assert(__builtin_offsetof(uvm_bpf_prefetch_decision_t, first) == 8,
               "prefetch first offset");
_Static_assert(__builtin_offsetof(uvm_bpf_prefetch_decision_t, outer) == 16,
               "prefetch outer offset");

enum nv_gpu_pmm_destination {
	NV_GPU_PMM_DESTINATION_USED = 1,
	NV_GPU_PMM_DESTINATION_UNUSED = 2,
};

enum nv_gpu_pmm_position {
	NV_GPU_PMM_POSITION_HEAD = 1,
	NV_GPU_PMM_POSITION_TAIL = 2,
};

struct uvm_gpu_root_chunk_struct;
typedef struct uvm_gpu_root_chunk_struct uvm_gpu_root_chunk_t;

typedef struct {
	__u64 owner_id;
	__u64 root_id;
	__u64 generation;
	__u32 source;
} nv_gpu_pmm_snapshot_t;

typedef struct {
	__u8 attempted;
	__u8 conflict;
	__u64 destination;
	__u64 position;
} nv_gpu_pmm_request_t;

typedef struct uvm_bpf_pmm_decision_ctx {
	uvm_pmm_gpu_t *pmm;
	uvm_gpu_root_chunk_t *root_chunk;
	nv_gpu_pmm_snapshot_t observed;
	nv_gpu_pmm_request_t request;
} uvm_bpf_pmm_decision_ctx_t;

_Static_assert(sizeof(uvm_bpf_pmm_decision_ctx_t) == 72,
	       "PMM decision ABI");
_Static_assert(__builtin_offsetof(uvm_bpf_pmm_decision_ctx_t, observed) == 16,
	       "PMM observed offset");
_Static_assert(__builtin_offsetof(uvm_bpf_pmm_decision_ctx_t, request) == 48,
	       "PMM request offset");

/* GPU memory policy struct_ops definition */
struct gpu_mem_ops {
	int (*gpu_test_trigger)(const char *, int);
	int (*gpu_page_prefetch)(uvm_page_index_t, uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_bpf_prefetch_decision_t *);
	int (*gpu_page_prefetch_iter)(uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_va_block_region_t *, unsigned int, uvm_bpf_prefetch_decision_t *);

	int (*gpu_block_activate)(uvm_pmm_gpu_t *, uvm_gpu_chunk_t *, uvm_bpf_pmm_decision_ctx_t *);
	int (*gpu_block_access)(uvm_pmm_gpu_t *, uvm_gpu_chunk_t *, uvm_bpf_pmm_decision_ctx_t *);
	int (*gpu_evict_prepare)(uvm_pmm_gpu_t *, struct list_head *, struct list_head *);
};


/* BPF kfuncs */
#ifndef BPF_NO_KFUNC_PROTOTYPES
#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif

/* Prefetch kfuncs */
extern int bpf_gpu_set_prefetch_region(uvm_bpf_prefetch_decision_t *decision_ctx, __u64 first, __u64 outer) __weak __ksym;
extern int bpf_gpu_strstr(const char *str, unsigned int str__sz, const char *substr, unsigned int substr__sz) __weak __ksym;

/* Block eviction policy kfuncs */
extern int bpf_gpu_request_reorder(uvm_bpf_pmm_decision_ctx_t *decision_ctx,
				   __u64 destination,
				   __u64 position) __weak __ksym;

#endif

#endif /* _BPF_TESTMOD_H */
