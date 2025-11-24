#ifndef __UVM_TYPES_H__
#define __UVM_TYPES_H__

/* Extract only UVM-specific types from nvidia-uvm.ko BTF */

typedef short unsigned int NvU16;
typedef NvU16 uvm_page_index_t;

typedef struct {
	uvm_page_index_t first;
	uvm_page_index_t outer;
} uvm_va_block_region_t;

typedef struct {
	long unsigned int bitmap[8];
} uvm_page_mask_t;

typedef struct uvm_perf_prefetch_bitmap_tree {
	uvm_page_mask_t pages;
	uvm_page_index_t offset;
	NvU16 leaf_count;
	unsigned char level_count;
} uvm_perf_prefetch_bitmap_tree_t;

/* PMM (Physical Memory Manager) types for eviction policy */

/* Forward declarations - opaque types for BPF */
typedef struct uvm_pmm_gpu_struct uvm_pmm_gpu_t;
typedef struct uvm_gpu_chunk_struct uvm_gpu_chunk_t;
typedef struct uvm_va_block_struct uvm_va_block_t;

/* Note: list_head is already defined in vmlinux.h, no need to redefine it */

#endif /* __UVM_TYPES_H__ */
