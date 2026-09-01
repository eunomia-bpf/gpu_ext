/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* MRU (Most Recently Used) Eviction Policy
 *
 * 与 LRU 相反：最近访问的 chunk 优先被 evict
 *
 * 适用场景：
 * - 顺序扫描（streaming）：数据用完就不再需要
 * - MoE prefill 阶段：顺序过一遍模型，访问过的不会再访问
 *
 * 实现：
 * - activate: chunk 加到 TAIL（新来的暂时安全）
 * - chunk_used: chunk 移到 HEAD（最近访问的优先 evict）
 * - eviction: 从 HEAD 开始踢（正好是最近访问的）
 *
 * List 结构:
 * HEAD ────────────────────────────────→ TAIL
 * [最近访问] → [...] → [...] → [最久没用]
 *     ↑                              ↑
 *   先 evict                      后 evict
 */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    /* 新 chunk 加到 TAIL，给它一点时间被使用 */
    bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_TAIL);
    bpf_printk("BPF MRU: chunk_activate, moved to tail\n");
    return 1; /* BYPASS */
}

SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_bpf_pmm_decision_ctx_t *decision_ctx)
{
    /* MRU 核心：访问后移到 HEAD，让它优先被 evict */
    bpf_gpu_request_reorder(decision_ctx, NV_GPU_PMM_DESTINATION_USED, NV_GPU_PMM_POSITION_HEAD);
    return 1; /* BYPASS */
}

SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    /* MRU 不需要额外准备，list 已经按 MRU 顺序排列 */
    bpf_printk("BPF MRU: eviction_prepare\n");
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_mru = {
    .gpu_test_trigger = (void *)NULL,
    .gpu_page_prefetch = (void *)NULL,
    .gpu_page_prefetch_iter = (void *)NULL,
    .gpu_block_activate = (void *)gpu_block_activate,
    .gpu_block_access = (void *)gpu_block_access,
    .gpu_evict_prepare = (void *)gpu_evict_prepare,
};
