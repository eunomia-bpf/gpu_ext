# NVIDIA UVM LRU 最小修改 BPF 扩展方案

## 目录
1. [现有架构分析](#1-现有架构分析)
2. [最小修改方案设计](#2-最小修改方案设计)
3. [实现步骤](#3-实现步骤)
4. [BPF Policy 示例](#4-bpf-policy-示例)

---

## 1. 现有架构分析

### 1.1 Prefetch 的 BPF 架构（已实现）

**文件**: `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c`

```c
struct uvm_gpu_ext {
    // Prefetch hooks
    int (*uvm_prefetch_before_compute)(...);
    int (*uvm_prefetch_on_tree_iter)(...);
};

// 全局 ops 指针
static struct uvm_gpu_ext __rcu *uvm_ops;
```

**调用位置**: `kernel-open/nvidia-uvm/uvm_perf_prefetch.c:103-171`

```c
static uvm_va_block_region_t compute_prefetch_region(...)
{
    enum uvm_bpf_action action;

    // 调用 BPF hook
    action = uvm_bpf_call_before_compute_prefetch(...);

    if (action == UVM_BPF_ACTION_BYPASS) {
        // BPF 完全接管
    }
    else {
        // 内核默认逻辑
    }
}
```

### 1.2 LRU 的现有实现（需要扩展）

**文件**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:1460-1500`

**核心函数**: `pick_root_chunk_to_evict()`

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;

    uvm_spin_lock(&pmm->list_lock);

    // 1. 优先从 free lists 选择（非零优先）
    chunk = list_first_chunk(find_free_list(pmm, ..., UVM_PMM_LIST_NO_ZERO));

    if (!chunk)
        chunk = list_first_chunk(find_free_list(pmm, ..., UVM_PMM_LIST_ZERO));

    // 2. 从 unused list 选择（较少使用）
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // 3. 从 used list 选择（LRU - 最久未使用）
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

    // 4. 检查是否可驱逐
    if (chunk) {
        chunk_start_eviction(pmm, chunk);
        uvm_spin_unlock(&pmm->list_lock);
        return root_chunk_from_chunk(pmm, chunk);
    }

    uvm_spin_unlock(&pmm->list_lock);
    return NULL;
}
```

**LRU 更新**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:642`

```c
// unpin_allocated() 调用 -> chunk 被使用后移到尾部
list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
```

**关键数据结构**:
```c
struct {
    struct list_head va_block_used;    // 使用中的 chunks (LRU 排序)
    struct list_head va_block_unused;  // 未使用的 chunks
} root_chunks;
```

---

## 2. 最小修改方案设计 ⭐

### 2.1 设计原则

**复用 Prefetch 架构，最小化代码修改**：

1. ✅ 复用现有的 `struct uvm_gpu_ext`
2. ✅ 复用现有的 `uvm_ops` 全局指针
3. ✅ 复用现有的 kfunc 注册机制
4. ✅ 采用"只调整位置"模型（最安全）

### 2.2 核心修改

**修改 1: 在 `uvm_bpf_struct_ops.c` 扩展 struct_ops**

```c
struct uvm_gpu_ext {
    // 现有 prefetch hooks
    int (*uvm_prefetch_before_compute)(...);
    int (*uvm_prefetch_on_tree_iter)(...);

    // ===== 新增 LRU hooks ===== (最小化设计)

    /**
     * @uvm_lru_on_access - chunk 被访问时调整链表位置
     *
     * @pmm: PMM GPU 指针
     * @chunk_addr: Chunk 的 GPU 物理地址（作为唯一标识）
     *
     * 返回值: 0 = 成功
     *
     * BPF 可调用 kfunc 移动 chunk 在链表中的位置
     */
    int (*uvm_lru_on_access)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);

    /**
     * @uvm_lru_prepare_eviction - 准备驱逐，BPF 将 victim 移到链表头部
     *
     * @pmm: PMM GPU 指针
     *
     * 返回值: 0 = 成功, < 0 = 失败
     *
     * BPF 通过遍历链表并移动 victim 到头部来"建议"驱逐哪个 chunk
     * 内核随后直接取 list_first_entry() 作为 victim
     */
    int (*uvm_lru_prepare_eviction)(uvm_pmm_gpu_t *pmm);
};
```

**修改 2: 添加 LRU kfuncs（复用现有 kfunc 框架）**

```c
/* ============ 链表遍历 kfuncs ============ */

__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_first(struct list_head *head);

__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_next(uvm_gpu_chunk_t *chunk, struct list_head *head);

/* ============ 链表修改 kfuncs ============ */

__bpf_kfunc int
bpf_uvm_list_move_tail(uvm_gpu_chunk_t *chunk, struct list_head *head);

__bpf_kfunc int
bpf_uvm_list_move_head(uvm_gpu_chunk_t *chunk, struct list_head *head);

__bpf_kfunc int
bpf_uvm_list_move_before(uvm_gpu_chunk_t *chunk, uvm_gpu_chunk_t *next_chunk);

/* ============ Chunk 属性访问 kfuncs ============ */

__bpf_kfunc u64
bpf_uvm_chunk_get_address(uvm_gpu_chunk_t *chunk);

__bpf_kfunc bool
bpf_uvm_chunk_is_pinned(uvm_gpu_chunk_t *chunk);
```

**修改 3: 在 `pick_root_chunk_to_evict()` 调用 BPF hook**

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;
    struct uvm_gpu_ext *ops;
    int ret = 0;

    uvm_spin_lock(&pmm->list_lock);

    // ===== 新增：调用 BPF hook =====
    rcu_read_lock();
    ops = rcu_dereference(uvm_ops);
    if (ops && ops->uvm_lru_prepare_eviction) {
        // BPF 会把 victim 移到链表头部
        ret = ops->uvm_lru_prepare_eviction(pmm);
    }
    rcu_read_unlock();

    if (ret < 0) {
        uvm_spin_unlock(&pmm->list_lock);
        return NULL;
    }

    // 原有逻辑：优先从 free lists 选择
    chunk = list_first_chunk(find_free_list(pmm, ..., UVM_PMM_LIST_NO_ZERO));

    if (!chunk)
        chunk = list_first_chunk(find_free_list(pmm, ..., UVM_PMM_LIST_ZERO));

    // 从 unused list 选择
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // 从 used list 选择（BPF 已调整过顺序）
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

    // 检查是否可驱逐
    if (chunk) {
        chunk_start_eviction(pmm, chunk);
        uvm_spin_unlock(&pmm->list_lock);
        return root_chunk_from_chunk(pmm, chunk);
    }

    uvm_spin_unlock(&pmm->list_lock);
    return NULL;
}
```

**修改 4: 在 chunk 访问时调用 BPF hook**

```c
// 文件: kernel-open/nvidia-uvm/uvm_pmm_gpu.c
// 函数: uvm_pmm_gpu_unpin_allocated() 或相关函数

void uvm_pmm_gpu_unpin_allocated(...)
{
    // 原有逻辑
    list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);

    // ===== 新增：调用 BPF hook =====
    struct uvm_gpu_ext *ops;
    rcu_read_lock();
    ops = rcu_dereference(uvm_ops);
    if (ops && ops->uvm_lru_on_access) {
        u64 chunk_addr = root_chunk->chunk.address.address;
        ops->uvm_lru_on_access(pmm, chunk_addr);
    }
    rcu_read_unlock();
}
```

---

## 3. 实现步骤

### 步骤 1: 扩展 `uvm_bpf_struct_ops.c`

```bash
vi kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c
```

**添加内容**：

1. 在 `struct uvm_gpu_ext` 添加 2 个 LRU hooks
2. 添加 CFI stub 函数（默认返回 0）
3. 添加 7 个 kfunc 实现
4. 更新 `BTF_KFUNCS_START` 注册 kfuncs

### 步骤 2: 修改 `uvm_pmm_gpu.c`

```bash
vi kernel-open/nvidia-uvm/uvm_pmm_gpu.c
```

**修改内容**：

1. 在 `pick_root_chunk_to_evict()` 添加 BPF hook 调用（~5 行）
2. 在 `uvm_pmm_gpu_unpin_allocated()` 添加 BPF hook 调用（~5 行）
3. Include `uvm_bpf_struct_ops.h`

### 步骤 3: 暴露必要的符号

```bash
vi kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h
```

**添加声明**：
```c
extern struct uvm_gpu_ext __rcu *uvm_ops;  // 复用现有
```

### 步骤 4: 编译内核模块

```bash
cd kernel-open
make modules
```

---

## 4. BPF Policy 示例

### 4.1 Policy 1: LRU（默认行为）

**文件**: `bpf-progs/lru_default.bpf.c`

```c
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* LRU: 访问时移到尾部，驱逐时头部已是 LRU */

SEC("struct_ops/uvm_lru_on_access")
int BPF_PROG(lru_on_access, uvm_pmm_gpu_t *pmm, u64 chunk_addr)
{
    // 找到这个 chunk
    uvm_gpu_chunk_t *chunk = bpf_uvm_list_first(&pmm->root_chunks.va_block_used);

    #pragma unroll
    for (int i = 0; i < 100 && chunk; i++) {
        if (bpf_uvm_chunk_get_address(chunk) == chunk_addr) {
            // 移到尾部 = Most Recently Used
            bpf_uvm_list_move_tail(chunk, &pmm->root_chunks.va_block_used);
            return 0;
        }
        chunk = bpf_uvm_list_next(chunk, &pmm->root_chunks.va_block_used);
    }

    return 0;
}

SEC("struct_ops/uvm_lru_prepare_eviction")
int BPF_PROG(lru_prepare_eviction, uvm_pmm_gpu_t *pmm)
{
    // LRU: 什么都不做，链表头部已经是 Least Recently Used
    return 0;
}

SEC(".struct_ops")
struct uvm_gpu_ext lru_default = {
    .uvm_lru_on_access = (void *)lru_on_access,
    .uvm_lru_prepare_eviction = (void *)lru_prepare_eviction,
};
```

### 4.2 Policy 2: MRU（扫描场景）

**文件**: `bpf-progs/lru_mru.bpf.c`

```c
/* MRU: 访问时移到尾部，驱逐时从尾部开始找 */

SEC("struct_ops/uvm_lru_on_access")
int BPF_PROG(mru_on_access, uvm_pmm_gpu_t *pmm, u64 chunk_addr)
{
    // 同 LRU
    uvm_gpu_chunk_t *chunk = bpf_uvm_list_first(&pmm->root_chunks.va_block_used);

    #pragma unroll
    for (int i = 0; i < 100 && chunk; i++) {
        if (bpf_uvm_chunk_get_address(chunk) == chunk_addr) {
            bpf_uvm_list_move_tail(chunk, &pmm->root_chunks.va_block_used);
            return 0;
        }
        chunk = bpf_uvm_list_next(chunk, &pmm->root_chunks.va_block_used);
    }

    return 0;
}

SEC("struct_ops/uvm_lru_prepare_eviction")
int BPF_PROG(mru_prepare_eviction, uvm_pmm_gpu_t *pmm)
{
    // MRU: 驱逐最近使用的（尾部 = Most Recently Used）
    // 遍历找到最后一个可驱逐的 chunk，移到头部
    uvm_gpu_chunk_t *chunk = bpf_uvm_list_first(&pmm->root_chunks.va_block_used);
    uvm_gpu_chunk_t *victim = NULL;

    #pragma unroll
    for (int i = 0; i < 100 && chunk; i++) {
        if (!bpf_uvm_chunk_is_pinned(chunk)) {
            victim = chunk;  // 记录最后一个可驱逐的
        }
        chunk = bpf_uvm_list_next(chunk, &pmm->root_chunks.va_block_used);
    }

    if (victim) {
        bpf_uvm_list_move_head(victim, &pmm->root_chunks.va_block_used);
    }

    return 0;
}

SEC(".struct_ops")
struct uvm_gpu_ext lru_mru = {
    .uvm_lru_on_access = (void *)mru_on_access,
    .uvm_lru_prepare_eviction = (void *)mru_prepare_eviction,
};
```

### 4.3 Policy 3: LFU（频率分段，O(1) 驱逐）

**文件**: `bpf-progs/lru_lfu.bpf.c`

```c
/* LFU: 频率分段排序，头部永远是最低频率 */

struct lfu_metadata {
    u32 freq;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);    // chunk 地址
    __type(value, struct lfu_metadata);
    __uint(max_entries, 10000);
} chunk_freq SEC(".maps");

SEC("struct_ops/uvm_lru_on_access")
int BPF_PROG(lfu_on_access, uvm_pmm_gpu_t *pmm, u64 chunk_addr)
{
    struct lfu_metadata *meta = bpf_map_lookup_elem(&chunk_freq, &chunk_addr);

    if (!meta) {
        struct lfu_metadata new_meta = {.freq = 1};
        bpf_map_update_elem(&chunk_freq, &chunk_addr, &new_meta, BPF_ANY);
        return 0;
    }

    u32 new_freq = ++meta->freq;

    // 每 4 次访问才移动一次
    if (new_freq % 4 != 0)
        return 0;

    // 找到这个 chunk，插入到对应频率段
    uvm_gpu_chunk_t *chunk = bpf_uvm_list_first(&pmm->root_chunks.va_block_used);

    #pragma unroll
    for (int i = 0; i < 100 && chunk; i++) {
        if (bpf_uvm_chunk_get_address(chunk) == chunk_addr) {
            // 从当前位置向后找插入点
            uvm_gpu_chunk_t *pos = bpf_uvm_list_next(chunk, &pmm->root_chunks.va_block_used);

            #pragma unroll
            for (int j = 0; j < 8 && pos; j++) {
                u64 pos_addr = bpf_uvm_chunk_get_address(pos);
                struct lfu_metadata *pos_meta = bpf_map_lookup_elem(&chunk_freq, &pos_addr);

                if (pos_meta && pos_meta->freq >= new_freq) {
                    bpf_uvm_list_move_before(chunk, pos);
                    return 0;
                }
                pos = bpf_uvm_list_next(pos, &pmm->root_chunks.va_block_used);
            }

            // 没找到更高频的，移到尾部
            bpf_uvm_list_move_tail(chunk, &pmm->root_chunks.va_block_used);
            return 0;
        }
        chunk = bpf_uvm_list_next(chunk, &pmm->root_chunks.va_block_used);
    }

    return 0;
}

SEC("struct_ops/uvm_lru_prepare_eviction")
int BPF_PROG(lfu_prepare_eviction, uvm_pmm_gpu_t *pmm)
{
    // LFU: 头部已经是最低频率，什么都不做
    return 0;
}

SEC(".struct_ops")
struct uvm_gpu_ext lru_lfu = {
    .uvm_lru_on_access = (void *)lfu_on_access,
    .uvm_lru_prepare_eviction = (void *)lfu_prepare_eviction,
};
```

---

## 5. 方案优势

### 5.1 最小化修改

| 修改文件 | 修改内容 | 行数 |
|---------|---------|------|
| `uvm_bpf_struct_ops.c` | 添加 2 个 hooks + 7 个 kfuncs | ~150 行 |
| `uvm_bpf_struct_ops.h` | 导出符号 | ~5 行 |
| `uvm_pmm_gpu.c` | 2 处调用 BPF hook | ~10 行 |
| **总计** | | **~165 行** |

### 5.2 复用现有基础设施

- ✅ 复用 `struct uvm_gpu_ext`（无需新结构）
- ✅ 复用 `uvm_ops` 全局指针（无需新变量）
- ✅ 复用 kfunc 注册机制（无需新注册函数）
- ✅ 复用 BTF 和 verifier 逻辑

### 5.3 安全性

- ✅ "只调整位置"模型：BPF 不持有 chunk 指针
- ✅ 内核最终决策：`list_first_entry()` 取 victim
- ✅ 类型安全：kfunc 强类型检查
- ✅ 边界安全：有界遍历（最多 100 次迭代）

### 5.4 性能

- ✅ O(1) 驱逐（LRU、FIFO、LFU 分段）
- ✅ 无需创建新链表（复用现有 2 个链表）
- ✅ 最小开销（仅在 access 和 evict 时调用 BPF）

---

## 6. 总结

**核心设计原则** ⭐⭐⭐：

1. **最小侵入性**：只在 2 个关键点添加 BPF hook（~10 行修改）
2. **复用 Prefetch 架构**：无需重新设计，直接扩展现有 `struct_ops`
3. **只调整位置模型**：BPF 只负责排序，内核负责决策
4. **支持所有算法**：LRU、MRU、FIFO、LFU、S3-FIFO 全部可实现

**与现有 Prefetch BPF 的对称性**：

| 特性 | Prefetch BPF | LRU BPF |
|------|-------------|---------|
| **struct_ops** | `uvm_gpu_ext` | 复用同一个 |
| **全局 ops** | `uvm_ops` | 复用同一个 |
| **kfunc 数量** | 2 个 | 7 个 |
| **hook 数量** | 2 个 | 2 个 |
| **调用点** | `compute_prefetch_region()` | `pick_root_chunk_to_evict()` |
| **修改行数** | ~200 行 | ~165 行 |

**这个方案可以在不破坏现有架构的情况下，用最少的代码添加完整的 LRU 可扩展性！**
