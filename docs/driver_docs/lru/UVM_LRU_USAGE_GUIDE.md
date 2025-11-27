# NVIDIA UVM LRU 驱逐策略完整使用指南

## 目录
1. [LRU 机制概述](#1-lru-机制概述)
2. [LRU 数据结构详解](#2-lru-数据结构详解)
3. [LRU 工作流程](#3-lru-工作流程)
4. [如何使用 BPF 扩展 LRU](#4-如何使用-bpf-扩展-lru)
5. [实现自定义驱逐策略](#5-实现自定义驱逐策略)
6. [调试和性能分析](#6-调试和性能分析)

---

## 1. LRU 机制概述

### 1.1 什么是 LRU

NVIDIA UVM 使用 **LRU (Least Recently Used)** 策略来管理 GPU 内存的驱逐:
- 当 GPU 内存不足时，驱逐**最久未使用**的内存块
- 粒度: **2MB root chunk** (与 CPU 大页相同)
- 数据结构: Linux 内核双向链表 (`list_head`)

### 1.2 与 Prefetch 的关系

| 机制 | 作用时机 | 目的 |
|------|---------|------|
| **Prefetch** | Page fault 时 | **主动预取**可能访问的页面，减少未来 fault |
| **LRU Eviction** | GPU 内存不足时 | **被动驱逐**最久未使用的页面，腾出空间 |

**协同工作**:
```
用户访问 GPU 内存地址
  ↓
Page Fault
  ↓
├─> Prefetch: 预测并预取周围页面 (uvm_perf_prefetch.c)
└─> 分配 GPU 内存 (uvm_pmm_gpu.c)
      ↓
    内存不足?
      ↓ Yes
    LRU Eviction: 驱逐最久未使用的 2MB chunk
      ↓
    更新 LRU 列表: 新分配的 chunk 移到链表尾部
```

### 1.3 核心设计原则

- ✅ **粗粒度追踪**: 只追踪 2MB root chunk，不追踪单个 4KB 页面
- ✅ **分配时更新**: 在 chunk 分配/unpin 时更新 LRU 位置
- ⚠️ **不追踪访问**: 当前实现不追踪实际的 GPU 访问（见 TODO Line 1487-1488）

---

## 2. LRU 数据结构详解

### 2.1 Root Chunks 结构

**定义位置**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.h:350-362`

```c
struct {
    // Root chunks 数组 (所有 2MB chunks)
    uvm_gpu_root_chunk_t *array;
    size_t count;

    // LRU 列表: 未被 VA block 使用的 chunks
    struct list_head va_block_unused;

    // LRU 列表: 被 VA block 使用的 chunks (主要驱逐来源)
    struct list_head va_block_used;

    // 延迟释放列表
    struct list_head va_block_lazy_free;
    nv_kthread_q_item_t va_block_lazy_free_q_item;
} root_chunks;
```

### 2.2 列表排序规则

**`va_block_used` 列表** (LRU 核心):
```
┌─────────────────────────────────────────────┐
│ HEAD (最久未使用)                            │
├─────────────────────────────────────────────┤
│ Chunk 1: 最早分配/最久未访问                 │
│ Chunk 2: ↓                                   │
│ Chunk 3: ↓                                   │
│ ...                                          │
│ Chunk N: 最近分配/最近访问                   │
├─────────────────────────────────────────────┤
│ TAIL (最近使用)                              │
└─────────────────────────────────────────────┘
```

**更新时机**:
- **`list_move_tail()`**: 在 chunk 分配/unpin 时将其移到尾部
- **`list_del_init()`**: 在 chunk 被 pin 时从列表移除

### 2.3 Chunk 状态机

```c
// uvm_pmm_gpu.h
typedef enum {
    UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED,    // 被 PMA 拥有（未分配）
    UVM_PMM_GPU_CHUNK_STATE_FREE,         // 在 free list 中
    UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED,  // 临时 pin（分配中）
    UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT,     // 已分割成子 chunks
    UVM_PMM_GPU_CHUNK_STATE_ALLOCATED,    // 已分配给 VA block
} uvm_pmm_gpu_chunk_state_t;
```

**状态转换** (与 LRU 相关):
```
PMA_OWNED → TEMP_PINNED → ALLOCATED → (unpin) → va_block_used 列表尾部
                                   ↓
                                (evict) → FREE → PMA_OWNED
```

---

## 3. LRU 工作流程

### 3.1 完整调用链

#### 阶段 1: GPU Page Fault 触发

```
GPU 访问未映射地址
  ↓
Hardware Fault Buffer 记录 fault
  ↓
uvm_parent_gpu_service_replayable_faults()  [uvm_gpu_replayable_faults.c:2906]
  ├─> fetch_fault_buffer_entries()          [读取 fault buffer]
  ├─> preprocess_fault_batch()              [预处理 faults]
  └─> service_fault_batch()                 [处理 faults]
      └─> service_fault_batch_dispatch()
          └─> service_fault_batch_block()
              └─> service_fault_batch_block_locked()
                  └─> uvm_va_block_service_locked()  [uvm_va_block.c:12349]
```

#### 阶段 2: 内存分配与驱逐触发

```c
// uvm_va_block.c:2080-2089
static NV_STATUS block_alloc_gpu_chunk(uvm_va_block_t *va_block,
                                       uvm_gpu_t *gpu,
                                       NvU64 size,
                                       uvm_gpu_chunk_t **out_gpu_chunk,
                                       uvm_va_block_retry_t *retry)
{
    // 第一次尝试: 不驱逐
    status = uvm_pmm_gpu_alloc_user(&gpu->pmm, 1, size,
                                     UVM_PMM_ALLOC_FLAGS_NONE,
                                     &gpu_chunk, &retry->tracker);

    // 如果失败 (NV_ERR_NO_MEMORY), 带驱逐标志重试
    if (status != NV_OK) {
        status = uvm_pmm_gpu_alloc_user(&gpu->pmm, 1, size,
                                         UVM_PMM_ALLOC_FLAGS_EVICT,  // ← 触发 LRU 驱逐
                                         &gpu_chunk, &retry->tracker);
    }

    *out_gpu_chunk = gpu_chunk;
    return status;
}
```

#### 阶段 3: LRU 驱逐选择

```c
// uvm_pmm_gpu.c:1460-1500
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;

    uvm_spin_lock(&pmm->list_lock);

    // 优先级 1: 从 free list 中找 (non-zero 优先)
    chunk = list_first_chunk(find_free_list(pmm,
                                            UVM_PMM_GPU_MEMORY_TYPE_USER,
                                            UVM_CHUNK_SIZE_MAX,
                                            UVM_PMM_LIST_NO_ZERO));
    if (!chunk) {
        chunk = list_first_chunk(find_free_list(pmm,
                                                UVM_PMM_GPU_MEMORY_TYPE_USER,
                                                UVM_CHUNK_SIZE_MAX,
                                                UVM_PMM_LIST_ZERO));
    }

    // 优先级 2: 从 unused 列表中找
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // 优先级 3: 从 used 列表头部找 (LRU - 最久未使用)
    // TODO: Bug 1765193: 未来可能在页面被映射时也更新 LRU
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);  // ← LRU 选择

    if (chunk)
        chunk_start_eviction(pmm, chunk);  // 标记为正在驱逐

    uvm_spin_unlock(&pmm->list_lock);

    if (chunk)
        return root_chunk_from_chunk(pmm, chunk);
    return NULL;
}
```

**关键点**:
- ✅ `list_first_chunk()` 从链表**头部**取 chunk (最久未使用)
- ✅ 三级优先级确保优先使用空闲内存
- ⚠️ TODO 注释表明未来可能改进 (在 map 时也更新 LRU)

#### 阶段 4: LRU 列表更新

```c
// uvm_pmm_gpu.c:627-651
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (chunk_is_root_chunk_pinned(pmm, chunk)) {
            // 如果被 pin，从列表移除
            UVM_ASSERT(root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
                       root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
            list_del_init(&root_chunk->chunk.list);
        }
        else if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            // 关键: 移到 used 列表尾部 (最近使用)
            UVM_ASSERT(root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
                       root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
        }
    }

    // 处理 free chunks
    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_FREE)
        list_move_tail(&chunk->list, find_free_list_chunk(pmm, chunk));
    else if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED)
        list_del_init(&chunk->list);
}
```

**调用时机**:
```c
// uvm_pmm_gpu.c:653-675
static void gpu_unpin_temp(uvm_pmm_gpu_t *pmm,
                           uvm_gpu_chunk_t *chunk,
                           uvm_va_block_t *va_block,
                           bool is_referenced)
{
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    uvm_spin_lock(&pmm->list_lock);

    chunk_unpin(pmm, chunk, UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
    chunk->is_referenced = is_referenced;
    chunk->va_block = va_block;
    chunk_update_lists_locked(pmm, chunk);  // ← 更新 LRU 位置

    uvm_spin_unlock(&pmm->list_lock);
}

// uvm_va_block.c:839
uvm_pmm_gpu_unpin_allocated(&gpu->pmm, gpu_chunk, va_block);  // ← 分配后调用
```

### 3.2 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. GPU Page Fault                                               │
│    uvm_parent_gpu_service_replayable_faults()                   │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 分配 GPU 内存                                                 │
│    uvm_pmm_gpu_alloc_user(..., UVM_PMM_ALLOC_FLAGS_NONE)       │
│    ↓ 失败 (NV_ERR_NO_MEMORY)                                    │
│    uvm_pmm_gpu_alloc_user(..., UVM_PMM_ALLOC_FLAGS_EVICT) ←───┐│
└──────────────────┬──────────────────────────────────────────┐  ││
                   ↓                                          │  ││
┌─────────────────────────────────────────────────────────┐  │  ││
│ 3. LRU 驱逐选择                                          │  │  ││
│    pick_root_chunk_to_evict()                           │  │  ││
│    ├─> 优先级 1: Free list                              │  │  ││
│    ├─> 优先级 2: va_block_unused                        │  │  ││
│    └─> 优先级 3: va_block_used HEAD (LRU)  ←───────────┼──┘  ││
│                                                          │     ││
│    evict_root_chunk()                                   │     ││
│    └─> 迁移页面到 CPU/System Memory                     │     ││
└──────────────────┬──────────────────────────────────────┘     ││
                   ↓                                            ││
┌─────────────────────────────────────────────────────────────┐ ││
│ 4. 分配成功，更新 LRU                                        │ ││
│    uvm_pmm_gpu_unpin_allocated()                            │ ││
│    └─> chunk_update_lists_locked()                          │ ││
│        └─> list_move_tail(..., va_block_used)  ────────────┼─┘│
│                                                              │  │
│    结果: 新 chunk 在链表尾部 (最近使用)                      │  │
└──────────────────────────────────────────────────────────────┘  │
                                                                   │
    下次驱逐时，这个 chunk 最不容易被选中 ───────────────────────┘
```

---

## 4. 如何使用 BPF 扩展 LRU

### 4.1 当前限制

**问题**: 当前 UVM 代码中 LRU **没有 BPF hook 点**
- Prefetch 有 BPF struct_ops (`uvm_prefetch_before_compute`, `uvm_prefetch_on_tree_iter`)
- LRU 驱逐策略**硬编码**在内核中

### 4.2 潜在的 BPF 扩展点

如果要添加 BPF 支持，可以在以下位置插入 hook:

#### Hook 点 1: 驱逐选择 (`pick_root_chunk_to_evict`)

```c
// uvm_pmm_gpu.c:1460
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;
    enum uvm_bpf_action action;

    // 🔥 新增 BPF hook: before_pick_evict_chunk
    action = uvm_bpf_call_before_pick_evict_chunk(pmm, &chunk);

    if (action == UVM_BPF_ACTION_BYPASS) {
        // BPF 直接选择了 chunk
        return root_chunk_from_chunk(pmm, chunk);
    }

    // 原有 LRU 逻辑
    uvm_spin_lock(&pmm->list_lock);
    chunk = list_first_chunk(&pmm->root_chunks.va_block_used);
    ...
}
```

#### Hook 点 2: LRU 更新 (`chunk_update_lists_locked`)

```c
// uvm_pmm_gpu.c:627
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);
    enum uvm_bpf_action action;

    // 🔥 新增 BPF hook: on_chunk_update
    action = uvm_bpf_call_on_chunk_update(pmm, chunk);

    if (action == UVM_BPF_ACTION_BYPASS) {
        // BPF 接管列表更新逻辑
        return;
    }

    // 原有 LRU 逻辑 (list_move_tail)
    if (uvm_gpu_chunk_is_user(chunk)) {
        if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
            root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
        }
    }
}
```

### 4.3 BPF Struct Ops 设计方案

#### 内核侧结构定义

```c
// uvm_bpf_struct_ops.h
struct uvm_eviction_ext {
    /* Eviction selection hook
     * Return: pointer to selected chunk, or NULL to use default LRU
     */
    uvm_gpu_chunk_t *(*pick_evict_chunk)(
        uvm_pmm_gpu_t *pmm,
        struct list_head *va_block_used);

    /* LRU update hook
     * Return: 0 = use default behavior, 1 = BPF handled
     */
    int (*on_chunk_allocated)(
        uvm_pmm_gpu_t *pmm,
        uvm_gpu_chunk_t *chunk,
        uvm_va_block_t *va_block);
};
```

#### BPF 侧示例: FIFO 策略

```c
// evict_fifo.bpf.c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"

/* FIFO 策略: 只在首次分配时加入列表，不更新位置 */
SEC("struct_ops/on_chunk_allocated")
int BPF_PROG(on_chunk_allocated,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_va_block_t *va_block)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    /* 通过 kfunc 检查 chunk 是否已在列表中 */
    bool in_list = bpf_uvm_chunk_in_list(&root_chunk->chunk.list);

    if (!in_list) {
        /* 首次分配: 加到尾部 */
        bpf_uvm_list_add_tail(&root_chunk->chunk.list,
                              &pmm->root_chunks.va_block_used);
    }
    /* 否则: 不更新位置 (FIFO 行为) */

    return 1; /* UVM_BPF_ACTION_BYPASS - BPF 已处理 */
}

SEC(".struct_ops")
struct uvm_eviction_ext uvm_evict_ops_fifo = {
    .on_chunk_allocated = (void *)on_chunk_allocated,
};
```

#### BPF 侧示例: Clock 策略

```c
// evict_clock.bpf.c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "uvm_types.h"

/* BPF map: 为每个 root chunk 维护一个 reference bit */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u64);   /* chunk address */
    __type(value, u8);  /* reference bit (0 or 1) */
} chunk_ref_bits SEC(".maps");

/* Clock 策略: 选择第一个 reference bit = 0 的 chunk */
SEC("struct_ops/pick_evict_chunk")
uvm_gpu_chunk_t *BPF_PROG(pick_evict_chunk,
                           uvm_pmm_gpu_t *pmm,
                           struct list_head *va_block_used)
{
    uvm_gpu_chunk_t *chunk;

    /* 遍历 used 列表 (通过 kfunc helper) */
    bpf_for_each_list_entry(chunk, va_block_used, list) {
        u64 addr = BPF_CORE_READ(chunk, address);
        u8 *ref_bit = bpf_map_lookup_elem(&chunk_ref_bits, &addr);

        if (ref_bit && *ref_bit == 0) {
            /* 找到 reference bit = 0 的 chunk */
            return chunk;
        }

        /* 清除 reference bit (second chance) */
        if (ref_bit) {
            u8 zero = 0;
            bpf_map_update_elem(&chunk_ref_bits, &addr, &zero, BPF_ANY);
        }
    }

    /* 所有 chunk 都有 reference bit，返回第一个 */
    return list_first_entry(va_block_used, uvm_gpu_chunk_t, list);
}

/* 分配时设置 reference bit = 1 */
SEC("struct_ops/on_chunk_allocated")
int BPF_PROG(on_chunk_allocated,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             uvm_va_block_t *va_block)
{
    u64 addr = BPF_CORE_READ(chunk, address);
    u8 ref_bit = 1;
    bpf_map_update_elem(&chunk_ref_bits, &addr, &ref_bit, BPF_ANY);

    return 0; /* 使用默认 LRU 列表更新 */
}

SEC(".struct_ops")
struct uvm_eviction_ext uvm_evict_ops_clock = {
    .pick_evict_chunk = (void *)pick_evict_chunk,
    .on_chunk_allocated = (void *)on_chunk_allocated,
};
```

### 4.4 需要的 Kfuncs

为了支持上述 BPF 程序，需要以下 kfuncs:

```c
/* 检查 chunk 是否在列表中 */
__bpf_kfunc bool bpf_uvm_chunk_in_list(struct list_head *list);

/* 列表操作 kfuncs */
__bpf_kfunc void bpf_uvm_list_add_tail(struct list_head *new, struct list_head *head);
__bpf_kfunc void bpf_uvm_list_move_tail(struct list_head *list, struct list_head *head);
__bpf_kfunc void bpf_uvm_list_del_init(struct list_head *entry);

/* 列表遍历 helper (类似 bpf_for_each_map_elem) */
__bpf_kfunc long bpf_for_each_list_entry(uvm_gpu_chunk_t *chunk,
                                         struct list_head *head,
                                         void *callback_fn,
                                         void *callback_ctx);

/* 获取 root chunk 信息 */
__bpf_kfunc uvm_gpu_root_chunk_t *bpf_uvm_root_chunk_from_chunk(
    uvm_pmm_gpu_t *pmm,
    uvm_gpu_chunk_t *chunk);
```

---

## 5. 实现自定义驱逐策略

### 5.1 方法 1: 修改内核代码 (无 BPF)

如果不使用 BPF，可以直接修改内核代码实现自定义策略。

#### 实现 FIFO (First-In-First-Out)

**修改文件**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c`

**修改点 1**: `chunk_update_lists_locked()` (Line 627)

```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (chunk_is_root_chunk_pinned(pmm, chunk)) {
            list_del_init(&root_chunk->chunk.list);
        }
        else if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            // FIFO 修改: 只在首次加入列表时操作
            if (list_empty(&root_chunk->chunk.list)) {
                list_add_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
            }
            // 否则不更新位置 (保持 FIFO 顺序)
        }
    }

    // 其余逻辑不变
    ...
}
```

**效果**:
- ✅ LRU → FIFO: 最早分配的先驱逐
- ✅ 只需修改 1 个函数
- ⚠️ 不考虑访问模式，可能驱逐热数据

#### 实现访问频率驱逐 (LFU - Least Frequently Used)

**修改点 1**: 添加访问计数字段

```c
// uvm_pmm_gpu.h
typedef struct {
    uvm_gpu_chunk_t chunk;
    uvm_tracker_t tracker;

    /* 新增: 访问计数 */
    atomic64_t access_count;
} uvm_gpu_root_chunk_t;
```

**修改点 2**: 在 chunk 使用时增加计数

```c
// chunk_update_lists_locked()
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
            root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            /* 增加访问计数 */
            atomic64_inc(&root_chunk->access_count);

            /* 仍然移到尾部 (保持列表有序) */
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
        }
    }
}
```

**修改点 3**: 驱逐选择时找访问最少的 chunk

```c
// pick_root_chunk_to_evict()
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;
    uvm_gpu_root_chunk_t *min_chunk = NULL;
    u64 min_count = ULLONG_MAX;

    uvm_spin_lock(&pmm->list_lock);

    /* 遍历 used 列表，找访问计数最小的 */
    list_for_each_entry(chunk, &pmm->root_chunks.va_block_used, list) {
        uvm_gpu_root_chunk_t *root = root_chunk_from_chunk(pmm, chunk);
        u64 count = atomic64_read(&root->access_count);

        if (count < min_count) {
            min_count = count;
            min_chunk = root;
        }
    }

    if (min_chunk)
        chunk_start_eviction(pmm, &min_chunk->chunk);

    uvm_spin_unlock(&pmm->list_lock);

    return min_chunk;
}
```

**效果**:
- ✅ LRU → LFU: 访问最少的先驱逐
- ⚠️ 需要遍历整个列表 (O(n)，可能影响性能)
- ⚠️ 需要原子操作维护计数

### 5.2 方法 2: 通过模块参数切换策略

创建一个可配置的驱逐策略框架:

```c
// uvm_pmm_gpu.c
typedef enum {
    UVM_EVICT_POLICY_LRU,
    UVM_EVICT_POLICY_FIFO,
    UVM_EVICT_POLICY_LFU,
    UVM_EVICT_POLICY_CLOCK,
} uvm_evict_policy_t;

static uvm_evict_policy_t g_evict_policy = UVM_EVICT_POLICY_LRU;
module_param_named(evict_policy, g_evict_policy, int, S_IRUGO);
MODULE_PARM_DESC(evict_policy, "Eviction policy: 0=LRU, 1=FIFO, 2=LFU, 3=Clock");

// 在 pick_root_chunk_to_evict() 中根据策略选择
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    switch (g_evict_policy) {
    case UVM_EVICT_POLICY_LRU:
        return pick_root_chunk_lru(pmm);
    case UVM_EVICT_POLICY_FIFO:
        return pick_root_chunk_fifo(pmm);
    case UVM_EVICT_POLICY_LFU:
        return pick_root_chunk_lfu(pmm);
    case UVM_EVICT_POLICY_CLOCK:
        return pick_root_chunk_clock(pmm);
    default:
        return pick_root_chunk_lru(pmm);
    }
}
```

**加载内核模块时指定策略**:
```bash
sudo modprobe nvidia-uvm evict_policy=1  # 使用 FIFO
```

---

## 6. 调试和性能分析

### 6.1 添加 Tracepoint

在关键路径插入 printk 或 tracepoint:

```c
// uvm_pmm_gpu.c:1490
if (!chunk)
    chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

if (chunk) {
    /* 添加调试输出 */
    pr_info("UVM Evict: Selected chunk at PA 0x%llx, state=%s\n",
            chunk->address,
            uvm_pmm_gpu_chunk_state_string(chunk->state));

    chunk_start_eviction(pmm, chunk);
}
```

### 6.2 统计信息收集

添加驱逐统计计数:

```c
// uvm_pmm_gpu.h
struct {
    atomic64_t eviction_count;          // 总驱逐次数
    atomic64_t eviction_from_lru;       // 从 LRU 列表驱逐
    atomic64_t eviction_from_unused;    // 从 unused 列表驱逐
    atomic64_t eviction_from_free;      // 从 free 列表驱逐
} stats;

// 在 pick_root_chunk_to_evict() 中更新
if (chunk) {
    atomic64_inc(&pmm->stats.eviction_count);
    if (/* from lru */)
        atomic64_inc(&pmm->stats.eviction_from_lru);
}
```

### 6.3 通过 /proc 暴露统计

```c
// uvm_pmm_gpu.c
static int eviction_stats_show(struct seq_file *s, void *data)
{
    uvm_pmm_gpu_t *pmm = s->private;

    seq_printf(s, "Total evictions: %llu\n",
               atomic64_read(&pmm->stats.eviction_count));
    seq_printf(s, "From LRU list: %llu\n",
               atomic64_read(&pmm->stats.eviction_from_lru));
    seq_printf(s, "From unused list: %llu\n",
               atomic64_read(&pmm->stats.eviction_from_unused));
    seq_printf(s, "From free list: %llu\n",
               atomic64_read(&pmm->stats.eviction_from_free));

    return 0;
}

// 创建 /proc/driver/nvidia-uvm/eviction_stats
proc_create_single("eviction_stats", 0, uvm_proc_dir, eviction_stats_show);
```

查看统计:
```bash
cat /proc/driver/nvidia-uvm/eviction_stats
```

### 6.4 使用 eBPF 监控 (如果有 BPF 支持)

```bash
# 监控驱逐事件
sudo bpftrace -e '
kprobe:pick_root_chunk_to_evict {
    printf("Eviction triggered\n");
}

kretprobe:pick_root_chunk_to_evict {
    if (retval != 0) {
        printf("Evicted chunk at 0x%lx\n", retval);
    }
}
'
```

---

## 7. 总结

### 7.1 核心要点

| 特性 | 说明 |
|------|------|
| **粒度** | 2MB root chunk (与 CPU 大页相同) |
| **数据结构** | 双向链表 (`list_head`) |
| **更新时机** | 分配/unpin 时 (不追踪实际访问) |
| **驱逐优先级** | Free → Unused → LRU (从头部选) |
| **列表排序** | HEAD = 最久未使用，TAIL = 最近使用 |

### 7.2 限制和注意事项

1. **不追踪实际访问** (TODO Line 1487-1488)
   - 只在分配时更新 LRU
   - 密集访问场景下退化为 "最早分配先驱逐"

2. **驱逐条件**
   - Chunk 不能处于 `TEMP_PINNED` 或正在驱逐状态
   - 子 chunks 被 pin 会阻止整个 root chunk 驱逐

3. **当前无 BPF 支持**
   - 需要修改内核代码实现自定义策略
   - 或等待社区添加 BPF struct_ops 支持

### 7.3 推荐实践

#### 如果要实现自定义驱逐策略:

**选项 1: 修改内核 (最简单)**
- 修改 `pick_root_chunk_to_evict()` - 改变选择逻辑
- 修改 `chunk_update_lists_locked()` - 改变更新策略

**选项 2: 添加 BPF 支持 (最灵活)**
- 在 `pick_root_chunk_to_evict()` 前添加 BPF hook
- 在 `chunk_update_lists_locked()` 中添加 BPF hook
- 实现 kfuncs 供 BPF 程序使用

**选项 3: 模块参数切换 (折中方案)**
- 预先实现多种策略
- 通过模块参数在加载时选择

### 7.4 与 Prefetch 的协同使用

| 场景 | Prefetch 策略 | LRU 策略 | 效果 |
|------|--------------|---------|------|
| **顺序访问** | Always Max | LRU | 最大化预取，最久未访问先驱逐 |
| **随机访问** | None | FIFO/Clock | 禁用预取，公平驱逐 |
| **热数据集** | Adaptive | LFU | 动态预取，保护热数据 |
| **Thrashing** | Thrashing-aware | Conservative | 避免预取抖动页，避免驱逐 |

---

**文档版本**: v1.0
**创建时间**: 2025-11-23
**作者**: UVM BPF Extension Project
**参考代码**:
- 内核侧: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c`, `uvm_pmm_gpu.h`
- 相关文档: `UVM_LRU_POLICY.md`, `UVM_PREFETCH_POLICY_ANALYSIS.md`
