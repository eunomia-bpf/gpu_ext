/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Test: BPF CO-RE Access to Chunk Attributes in Struct_Ops
 *
 * Validates that we can read chunk->address, chunk->log2_size, etc.
 * directly in struct_ops hooks without needing kfuncs.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Test map: chunk_addr -> access_count */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);    /* chunk physical address */
    __type(value, u64);  /* access count */
} chunk_freq SEC(".maps");

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u64 chunk_addr;
    u64 chunk_size;
    u32 chunk_state;
    u64 va_start = 0;
    u64 va_end = 0;

    if (!chunk)
        return 0;

    // Test 1: Read chunk address using CO-RE
    chunk_addr = BPF_CORE_READ(chunk, address);

    // Test 2: Read bitfield (log2_size) - need to copy struct first
    struct uvm_gpu_chunk_struct chunk_copy;
    if (bpf_probe_read_kernel(&chunk_copy, sizeof(chunk_copy), chunk) == 0) {
        chunk_size = 1ULL << chunk_copy.log2_size;
        chunk_state = chunk_copy.state;
    }

    // Test 3: Read VA block info
    uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
    if (va_block) {
        va_start = BPF_CORE_READ(va_block, start);
        va_end = BPF_CORE_READ(va_block, end);
    }

    // Test 4: Use address as map key
    u64 init = 1;
    bpf_map_update_elem(&chunk_freq, &chunk_addr, &init, BPF_NOEXIST);

    // Print info
    bpf_printk("ACTIVATE: addr=0x%llx size=%llu state=%u VA=[0x%llx, 0x%llx)",
               chunk_addr, chunk_size, chunk_state, va_start, va_end);

    return 0;
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    u64 chunk_addr;
    u64 *freq;

    if (!chunk)
        return 0;

    // Read chunk address
    chunk_addr = BPF_CORE_READ(chunk, address);

    // Increment frequency in map
    freq = bpf_map_lookup_elem(&chunk_freq, &chunk_addr);
    if (freq) {
        __sync_fetch_and_add(freq, 1);
        bpf_printk("USED: addr=0x%llx freq=%llu", chunk_addr, *freq);
    }

    return 0;  // Let kernel do default LRU
}

SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    // Test: Read first chunk from list
    if (!va_block_used)
        return 0;

    struct list_head *first = BPF_CORE_READ(va_block_used, next);
    if (!first || first == va_block_used)
        return 0;

    // Container_of to get chunk
    uvm_gpu_chunk_t *chunk = (uvm_gpu_chunk_t *)((char *)first -
        __builtin_offsetof(struct uvm_gpu_chunk_struct, list));

    u64 chunk_addr = BPF_CORE_READ(chunk, address);
    u64 *freq = bpf_map_lookup_elem(&chunk_freq, &chunk_addr);

    if (freq) {
        bpf_printk("EVICT: addr=0x%llx freq=%llu", chunk_addr, *freq);
        // Clean up map entry
        bpf_map_delete_elem(&chunk_freq, &chunk_addr);
    }

    return 0;
}

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_test_chunk_access = {
    .gpu_test_trigger = (void *)NULL,
    .gpu_page_prefetch = (void *)NULL,
    .gpu_page_prefetch_iter = (void *)NULL,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
