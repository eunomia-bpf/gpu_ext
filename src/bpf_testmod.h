#ifndef _BPF_TESTMOD_H
#define _BPF_TESTMOD_H

#ifndef __KERNEL__
/* Match kernel typedefs exactly for BTF compatibility */
typedef unsigned short __u16;
typedef unsigned int __u32;
typedef unsigned int u32;

/* NVIDIA UVM types - must match kernel exactly */
typedef __u16 NvU16;
typedef NvU16 uvm_page_index_t;
#endif

/* Forward declarations - actual definitions are in uvm_vmlinux.h */
struct uvm_perf_prefetch_bitmap_tree;
struct uvm_va_block_region;

typedef struct uvm_perf_prefetch_bitmap_tree uvm_perf_prefetch_bitmap_tree_t;
typedef struct uvm_va_block_region uvm_va_block_region_t;

/* Shared struct_ops definition between kernel module and BPF program */
struct uvm_gpu_ext {
	int (*uvm_bpf_test_trigger_kfunc)(const char *buf, int len);

	/* Prefetch hooks */
	int (*uvm_prefetch_before_compute)(
		uvm_page_index_t page_index,
		uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
		uvm_va_block_region_t *max_prefetch_region,
		uvm_va_block_region_t *result_region);

	int (*uvm_prefetch_on_tree_iter)(
		uvm_page_index_t page_index,
		uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
		uvm_va_block_region_t *max_prefetch_region,
		uvm_va_block_region_t *current_region,
		unsigned int counter,
		unsigned int subregion_pages);
};

#endif /* _BPF_TESTMOD_H */