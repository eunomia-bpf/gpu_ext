# NVIDIA UVM LRU 链表使用完整分析

## 概览

根据对 `uvm_pmm_gpu.c` 的完整分析，有 **2 个 LRU 链表** 和 **5 个关键使用场景**。

---

## 1. 数据结构定义

**文件**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:3435-3436`

```c
INIT_LIST_HEAD(&pmm->root_chunks.va_block_used);    // 使用中的 chunks (LRU 排序)
INIT_LIST_HEAD(&pmm->root_chunks.va_block_unused);  // 未使用的 chunks
```

**关键区别**：
- `va_block_used`: chunk 正在被 VA block 使用（有关联的虚拟地址）
- `va_block_unused`: chunk 已分配但未被 VA block 使用

---

## 2. 所有使用场景（5 个关键点）

### 场景 1: Chunk Unpin (分配完成后) - **LRU 更新的核心** ⭐⭐⭐

**位置**: `uvm_pmm_gpu.c:642` (在 `chunk_update_lists_locked()`)

```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (chunk_is_root_chunk_pinned(pmm, chunk)) {
            // Pinned chunk: 从链表中移除（不参与 LRU）
            list_del_init(&root_chunk->chunk.list);
        }
        else if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            // **核心 LRU 操作**：移到尾部 = Most Recently Used
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
        }
    }
}
```

**调用路径**:
```
uvm_pmm_gpu_unpin_allocated(chunk, va_block)  [Line 677]
  └─> gpu_unpin_temp(chunk, va_block, false)  [Line 653]
      └─> chunk_update_lists_locked(chunk)    [Line 672]
          └─> list_move_tail(&pmm->root_chunks.va_block_used)  [Line 642] ⭐
```

**触发时机**：
1. **首次分配后**: chunk 从 TEMP_PINNED → ALLOCATED
2. **每次访问后**: VA block unpin chunk（即 chunk 被访问）

**关键洞察**：
> 这是 **唯一更新 LRU 顺序的地方**！
> 每次 chunk 被 unpin（访问/使用）时，都会移到 `va_block_used` 尾部。

---

### 场景 2: 驱逐选择 - **从 LRU 链表头部取 victim** ⭐⭐⭐

**位置**: `uvm_pmm_gpu.c:1485-1490` (在 `pick_root_chunk_to_evict()`)

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;

    uvm_spin_lock(&pmm->list_lock);

    // 1. 优先从 free lists 选择
    chunk = list_first_chunk(find_free_list(pmm, ..., UVM_PMM_LIST_NO_ZERO));
    if (!chunk)
        chunk = list_first_chunk(find_free_list(pmm, ..., UVM_PMM_LIST_ZERO));

    // 2. 从 unused list 选择（较少使用）
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);  // [Line 1485]

    // 3. 从 used list 选择（LRU - 最久未使用）⭐
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);    // [Line 1490]

    if (chunk)
        chunk_start_eviction(pmm, chunk);

    uvm_spin_unlock(&pmm->list_lock);

    if (chunk)
        return root_chunk_from_chunk(pmm, chunk);
    return NULL;
}
```

**驱逐优先级**（从高到低）：
1. Free list (non-zero)
2. Free list (zero)
3. **`va_block_unused` (未使用的 chunks)**
4. **`va_block_used` (LRU - 头部 = 最久未使用)** ⭐

**关键洞察**：
> 链表头部 (`list_first_chunk`) = 最久未被 `unpin` 的 chunk = LRU

---

### 场景 3: 标记 Chunk 为 Used/Unused ⭐ (重要的语义区分)

**位置**: `uvm_pmm_gpu.c:1450-1457`

```c
void uvm_pmm_gpu_mark_root_chunk_used(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_used);
}

void uvm_pmm_gpu_mark_root_chunk_unused(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_unused);
}
```

**实现**：
```c
static void root_chunk_update_eviction_list(uvm_pmm_gpu_t *pmm,
                                            uvm_gpu_chunk_t *chunk,
                                            struct list_head *list)
{
    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == UVM_CHUNK_SIZE_MAX);
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
        !chunk_is_in_eviction(pmm, chunk)) {
        // 移到目标链表的尾部
        list_move_tail(&chunk->list, list);
    }

    uvm_spin_unlock(&pmm->list_lock);
}
```

**调用位置**: `kernel-open/nvidia-uvm/uvm_va_block.c`

**`mark_used` 调用场景** (Line 3565):
```c
static void block_mark_memory_used(uvm_va_block_t *block, uvm_processor_id_t id)
{
    // 当 VA block 有页面驻留在 GPU 时调用
    if (!uvm_va_block_is_hmm(block) &&
        uvm_va_block_size(block) == UVM_CHUNK_SIZE_MAX &&
        uvm_parent_gpu_supports_eviction(gpu->parent)) {
        // 标记 chunk 为 "正在被使用"
        uvm_pmm_gpu_mark_root_chunk_used(&gpu->pmm, gpu_state->chunks[0]);
    }
}
```

**`mark_unused` 调用场景** (Line 3601):
```c
static void block_clear_resident_processor(uvm_va_block_t *block, uvm_processor_id_t id)
{
    // 当 VA block 的所有页面都不再驻留在 GPU 时调用
    if (!uvm_va_block_is_hmm(block) &&
        uvm_va_block_size(block) == UVM_CHUNK_SIZE_MAX &&
        uvm_parent_gpu_supports_eviction(gpu->parent)) {
        if (gpu_state && gpu_state->chunks[0])
            // 标记 chunk 为 "不再被使用"
            uvm_pmm_gpu_mark_root_chunk_unused(&gpu->pmm, gpu_state->chunks[0]);
    }
}
```

**关键语义区别**：

| 状态 | 链表 | 含义 | 触发时机 |
|------|------|------|---------|
| **Used** | `va_block_used` | VA block 有页面驻留在此 chunk | 页面迁移到 GPU 后 |
| **Unused** | `va_block_unused` | Chunk 已分配但无页面驻留 | 页面全部迁移走后 |

**驱逐优先级**：
1. Free chunks (最高优先级)
2. **Unused chunks** (已分配但无数据，易驱逐)
3. **Used chunks** (有数据，需要迁移，LRU 排序)

**这与 `chunk_update_lists_locked()` 的区别**：

| 函数 | 调用时机 | 作用 |
|------|---------|------|
| `chunk_update_lists_locked()` | **每次 chunk 被访问/unpin** | 更新 LRU 顺序（移到尾部） |
| `mark_root_chunk_used/unused()` | **VA block 驻留状态改变** | 在 used/unused 链表间移动 |

**示例流程**：

```
1. Chunk A 分配
   └─> chunk_update_lists_locked()
       └─> list_move_tail(va_block_used)  ← A 在 used 链表

2. VA block 在 chunk A 上有页面驻留
   └─> block_mark_memory_used()
       └─> mark_root_chunk_used()
           └─> list_move_tail(va_block_used)  ← 仍在 used，移到尾部

3. 页面全部从 chunk A 迁移走
   └─> block_clear_resident_processor()
       └─> mark_root_chunk_unused()
           └─> list_move_tail(va_block_unused)  ← 移到 unused 链表！

4. 驱逐时
   └─> pick_root_chunk_to_evict()
       └─> 优先选择 va_block_unused 的 chunk A（因为无数据需要迁移）
```

---

### 场景 4: 初始化

**位置**: `uvm_pmm_gpu.c:3435-3436`

```c
INIT_LIST_HEAD(&pmm->root_chunks.va_block_used);
INIT_LIST_HEAD(&pmm->root_chunks.va_block_unused);
```

**时机**: PMM 初始化时

---

### 场景 5: Pinned Chunk 处理

**位置**: `uvm_pmm_gpu.c:634-637`

```c
if (chunk_is_root_chunk_pinned(pmm, chunk)) {
    // Pinned chunk 不参与 LRU，从链表中移除
    list_del_init(&root_chunk->chunk.list);
}
```

**关键**：Pinned chunk 不会被驱逐，因此从 LRU 链表中移除。

---

## 3. LRU 工作流程总结

### 正常 LRU 流程

```
1. Chunk 分配
   └─> pmm_gpu_alloc() → chunk 状态 = TEMP_PINNED

2. Chunk 关联到 VA block (首次使用)
   └─> uvm_pmm_gpu_unpin_allocated(chunk, va_block)
       └─> chunk_update_lists_locked()
           └─> list_move_tail(&pmm->root_chunks.va_block_used)  ⭐ 移到尾部

3. Chunk 被访问（后续使用）
   └─> uvm_pmm_gpu_unpin_allocated(chunk, va_block)  (重复调用)
       └─> list_move_tail(&pmm->root_chunks.va_block_used)  ⭐ 再次移到尾部

4. 内存不足，需要驱逐
   └─> pick_root_chunk_to_evict()
       └─> list_first_chunk(&pmm->root_chunks.va_block_used)  ⭐ 从头部取 = LRU
```

### 链表状态演示

```
时间 T0 (初始化):
va_block_used: [空]

时间 T1 (chunk A 分配):
va_block_used: [A]  ← 尾部

时间 T2 (chunk B 分配):
va_block_used: [A] → [B]  ← 尾部

时间 T3 (chunk C 分配):
va_block_used: [A] → [B] → [C]  ← 尾部

时间 T4 (chunk A 被访问):
va_block_used: [B] → [C] → [A]  ← 尾部 (A 移到尾部)

时间 T5 (需要驱逐):
victim = list_first_chunk() = B  ← 头部 = Least Recently Used
```

---

## 4. BPF Hook 插入点分析

### 关键洞察

**有 2 个地方更新链表**：
1. `chunk_update_lists_locked()` - 每次 chunk 被访问时更新 LRU 顺序
2. `mark_root_chunk_used/unused()` - VA block 驻留状态改变时切换链表

**但我们只需要 hook 第 1 个**！原因见下文。

### 推荐的 Hook 插入点

#### Hook 1: `on_access` (唯一必需的 hook) ⭐⭐⭐

**位置**: `chunk_update_lists_locked()` 之后

```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    // 原有逻辑
    if (uvm_gpu_chunk_is_user(chunk)) {
        if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
            root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);

            // ===== 添加 BPF hook =====
            struct uvm_gpu_ext *ops;
            rcu_read_lock();
            ops = rcu_dereference(uvm_ops);
            if (ops && ops->uvm_lru_on_access) {
                u64 chunk_addr = root_chunk->chunk.address.address;
                ops->uvm_lru_on_access(pmm, chunk_addr);
            }
            rcu_read_unlock();
        }
    }
}
```

**为什么这里足够？**

1. ✅ **涵盖首次分配**: `unpin_allocated` 第一次调用 → chunk 加入 LRU
2. ✅ **涵盖所有访问**: 每次 unpin → chunk 移到尾部
3. ✅ **无需 `on_alloc` hook**: 因为首次分配也会调用这里！

#### Hook 2: `prepare_eviction` (驱逐前准备)

**位置**: `pick_root_chunk_to_evict()` 开始处

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;
    struct uvm_gpu_ext *ops;
    int ret = 0;

    uvm_spin_lock(&pmm->list_lock);

    // ===== 添加 BPF hook =====
    rcu_read_lock();
    ops = rcu_dereference(uvm_ops);
    if (ops && ops->uvm_lru_prepare_eviction) {
        ret = ops->uvm_lru_prepare_eviction(pmm);
    }
    rcu_read_unlock();

    if (ret < 0) {
        uvm_spin_unlock(&pmm->list_lock);
        return NULL;
    }

    // 原有驱逐逻辑...
}
```

---

## 5. 需要 Hook `mark_used/unused` 吗？

### 问题分析

`mark_root_chunk_used/unused` 也会调用 `list_move_tail`，是否需要额外的 hook？

**答案：不需要！** ⭐

### 原因分析

#### 1. Used/Unused 切换不影响 LRU 语义

```
场景：Chunk A 从 used → unused

之前: va_block_used:   [A] → [B] → [C]
                        ↑
                       LRU

之后: va_block_used:   [B] → [C]
      va_block_unused: [A]
                        ↑
                      优先驱逐（无数据）
```

**关键**：
- `va_block_unused` 的 chunk **优先被驱逐**（在 `pick_root_chunk_to_evict` 中优先级更高）
- 移到 `unused` 本身就是一个"优先级提升"操作
- **BPF 不需要参与这个决策**

#### 2. BPF 只需要关心同一链表内的 LRU 顺序

**BPF 的职责**：
- ✅ 在 `va_block_used` 链表内调整 LRU 顺序
- ❌ 不需要管理 used/unused 切换（这是内核的驱逐策略）

**内核的职责**：
- ✅ 根据驻留状态决定 chunk 在 used/unused 链表
- ✅ 驱逐时优先选择 unused 链表（性能优化）

#### 3. 如果 BPF hook 了 `mark_used/unused` 会怎样？

假设我们添加 hook：
```c
void uvm_pmm_gpu_mark_root_chunk_used(...)
{
    root_chunk_update_eviction_list(..., va_block_used);

    // 如果添加 hook
    if (ops && ops->uvm_lru_on_mark_used) {
        ops->uvm_lru_on_mark_used(pmm, chunk_addr);
    }
}
```

**问题**：
- ❌ BPF 无法改变 used/unused 语义（这是内核根据驻留状态决定的）
- ❌ 增加复杂度但没有收益
- ❌ `chunk_update_lists_locked` 的 hook 已经覆盖了 LRU 更新

#### 4. 实际需求分析

**LRU 算法需要什么？**
- 只需要在**访问时**更新顺序 → `chunk_update_lists_locked` 已覆盖 ✅

**MRU 算法需要什么？**
- 只需要在**驱逐时**选择 MRU → `prepare_eviction` hook 已覆盖 ✅

**LFU 算法需要什么？**
- 只需要在**访问时**更新频率并调整位置 → `chunk_update_lists_locked` 已覆盖 ✅

**FIFO 算法需要什么？**
- 什么都不需要（保持分配顺序）→ 空 hook 即可 ✅

**S3-FIFO 算法需要什么？**
- 访问时标记，驱逐时队列管理 → 现有 2 个 hooks 已覆盖 ✅

### 结论

> ❌ **不需要 hook `mark_root_chunk_used/unused`**
>
> 理由：
> 1. Used/unused 切换是**驱逐策略**而非 LRU 策略
> 2. BPF 只需关心**同一链表内的 LRU 顺序**
> 3. 所有常见算法用现有 2 个 hooks 已足够
> 4. 添加额外 hook 会增加复杂度但无实际收益

---

## 6. 最终结论

### 需要几个 Hook？

**答案：只需要 2 个 hooks！** ⭐⭐⭐

```c
struct uvm_gpu_ext {
    // Prefetch hooks
    int (*uvm_prefetch_before_compute)(...);
    int (*uvm_prefetch_on_tree_iter)(...);

    // LRU hooks (最简方案)
    int (*uvm_lru_on_access)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_prepare_eviction)(uvm_pmm_gpu_t *pmm);
};
```

**不需要 `on_alloc` hook** 的原因：
> `chunk_update_lists_locked()` 在首次分配后就会被调用（通过 `unpin_allocated`），
> 因此 `on_access` hook 可以同时处理分配和访问！

### 各算法的实现

| 算法 | `on_access` | `prepare_eviction` | 说明 |
|------|-----------|-------------------|------|
| **LRU** | 什么都不做 | 什么都不做 | 内核已实现 |
| **MRU** | 什么都不做 | 遍历移尾部到头部 | 只需改驱逐逻辑 |
| **FIFO** | 什么都不做 | 什么都不做 | 类似 LRU |
| **LFU** | freq++, 插入频率段 | 什么都不做 | 头部已是最低频 |
| **S3-FIFO** | 标记访问 | 队列管理 + 移动 | 需要 map |

**关键发现**：
- FIFO 不需要特殊处理！因为 `unpin_allocated` 天然就是分配顺序
- 不需要区分"首次分配"和"后续访问"（对于 FIFO 来说都一样）

---

## 6. 代码修改总结

### 只需修改 2 个位置

1. **`chunk_update_lists_locked()`** - 添加 `on_access` hook (~5 行)
2. **`pick_root_chunk_to_evict()`** - 添加 `prepare_eviction` hook (~10 行)

**总计：~15 行内核代码修改** ⭐

这比之前的方案更简单！
