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

/* GPU memory policy struct_ops definition */
struct gpu_mem_ops {
	int (*gpu_test_trigger)(const char *, int);
	int (*gpu_page_prefetch)(uvm_page_index_t, uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_bpf_prefetch_decision_t *);
	int (*gpu_page_prefetch_iter)(uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_va_block_region_t *, unsigned int, uvm_bpf_prefetch_decision_t *);

	int (*gpu_block_activate)(uvm_pmm_gpu_t *, uvm_gpu_chunk_t *, struct list_head *);
	int (*gpu_block_access)(uvm_pmm_gpu_t *, uvm_gpu_chunk_t *, struct list_head *);
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
extern void bpf_gpu_block_move_head(uvm_gpu_chunk_t *chunk, struct list_head *list) __weak __ksym;
extern void bpf_gpu_block_move_tail(uvm_gpu_chunk_t *chunk, struct list_head *list) __weak __ksym;

#endif

#endif /* _BPF_TESTMOD_H */
