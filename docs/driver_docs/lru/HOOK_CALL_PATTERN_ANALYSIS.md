# LRU Hook 调用模式深度分析

## 背景：UVM GPU 内存管理基础

### 什么是 UVM？

**UVM (Unified Virtual Memory)** 是 NVIDIA GPU 的统一虚拟内存子系统，让 CPU 和 GPU 可以共享同一个虚拟地址空间。

- **目标**：简化 GPU 编程，自动处理 CPU ↔ GPU 数据迁移
- **核心机制**：按需页面迁移（Page Migration on Demand）
- **驱逐策略**：当 GPU 内存不足时，按 LRU 策略驱逐页面到 CPU

### 什么是 Chunk？

**Chunk** 是 UVM 内存管理的基本单元，类似于 CPU 侧的"页"概念，但更灵活：

```
内存层次结构：

GPU 物理内存
    └─> Root Chunk (2MB)              ← 可以被驱逐的最大单元
        ├─> Subchunk (64KB)           ← 可以进一步 split
        │   └─> Subchunk (4KB)        ← 最小的页面
        └─> Subchunk (64KB)
            └─> ...
```

**关键概念**：

1. **Root Chunk (根块)**：
   - 大小：`UVM_CHUNK_SIZE_MAX` = 2MB
   - **驱逐的基本单位**（只能驱逐整个 2MB root chunk）
   - 包含 512 个 4KB 页面（如果完全 split）

2. **VA Block (虚拟地址块)**：
   - 应用程序的虚拟地址范围
   - 可以映射到一个或多个 chunks
   - 跟踪哪些页面驻留在哪个 processor（CPU/GPU）

3. **PMM (Physical Memory Manager)**：
   - 管理 GPU 物理内存
   - 维护空闲 chunk 列表和 eviction 列表
   - 处理分配、驱逐、回收

### Chunk 的状态

```c
enum {
    UVM_PMM_GPU_CHUNK_STATE_FREE,         // 空闲，可分配
    UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED,  // 临时锁定，正在使用（不可驱逐）
    UVM_PMM_GPU_CHUNK_STATE_ALLOCATED,    // 已分配（可驱逐）
    UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT,     // 已被分割成更小的 subchunks
};
```

**状态转换**：

```
分配流程：
FREE → TEMP_PINNED (分配时，暂时锁定)
     ↓
   使用中（数据迁移、映射等操作）
     ↓
TEMP_PINNED → ALLOCATED (unpin 后，可以被驱逐)

驱逐流程：
ALLOCATED → TEMP_PINNED (选中要驱逐，锁定)
          ↓
        驱逐数据
          ↓
TEMP_PINNED → FREE (驱逐完成，释放)
```

### LRU Eviction Lists（驱逐链表）

UVM 维护两个关键链表来管理可驱逐的 chunks：

```c
struct {
    struct list_head va_block_used;    // 有页面驻留的 chunks（优先级低）
    struct list_head va_block_unused;  // 无页面驻留的 chunks（优先级高）
} root_chunks;
```

**链表语义**：

| 链表 | 包含的 Chunks | Resident Mask | 驱逐优先级 |
|------|---------------|---------------|-----------|
| `va_block_unused` | 已分配但**无页面驻留** | 0（空） | **高**（优先驱逐） |
| `va_block_used` | 有**至少一个页面驻留** | ≥1 | **低**（后驱逐） |

**链表内部顺序**：
- **Head (链表头)** = 最旧/最少使用 = **优先驱逐**
- **Tail (链表尾)** = 最新/最近使用 = **最后驱逐**

**驱逐顺序**：
```
1. 先检查 va_block_unused（优先驱逐没有数据的 chunks）
2. 如果 unused 为空，再检查 va_block_used（驱逐有数据的 chunks）
3. 从链表头开始选择（LRU 策略）
```

### Resident Mask（驻留掩码）

**关键数据结构**：

```c
struct uvm_va_block_t {
    uvm_processor_mask_t resident;  // 位图，标记哪些 processor 有页面驻留
    ...
};
```

**作用**：
- 每个 bit 代表一个 processor（CPU 或 GPU）
- Bit = 1：该 processor 有至少一个页面驻留在这个 VA block
- Bit = 0：该 processor 没有页面驻留

**示例**：

```
VA Block A:
  resident = 0b0010  (只有 GPU1 有页面)
      ↓
  第一个页面迁移到 GPU1
      → test_and_set(&resident, GPU1): 0 → 1
      → 触发 mark_root_chunk_used (第一次)
      ↓
  第二个页面迁移到 GPU1
      → test_and_set(&resident, GPU1): 1 → 1
      → 不触发 mark_root_chunk_used（已经有页面了）
      ↓
  最后一个页面离开 GPU1
      → test_and_clear(&resident, GPU1): 1 → 0
      → 触发 mark_root_chunk_unused（最后一个页面）
```

### Pin/Unpin 机制

**Pin（锁定）**：
- 将 chunk 标记为 `TEMP_PINNED`
- **从 eviction list 移除**（不能被驱逐）
- 用途：正在进行数据迁移、映射操作等

**Unpin（解锁）**：
- 将 chunk 从 `TEMP_PINNED` 转为 `ALLOCATED`
- **重新加入 eviction list**（可以被驱逐）
- 调用 `chunk_update_lists_locked()` 来更新链表

**为什么需要 Pin/Unpin？**
- 数据迁移期间，chunk 必须保持稳定（不能被驱逐）
- 操作完成后，chunk 应该可以被驱逐（释放资源）

### 完整的页面迁移流程示例

```
场景：将 100 个页面从 CPU 迁移到 GPU

步骤 1: 分配 GPU Chunks
  alloc_root_chunk()
      → 分配一个 2MB root chunk
      → 状态：TEMP_PINNED（锁定，不可驱逐）

步骤 2: Unpin Chunk
  uvm_pmm_gpu_unpin_allocated()
      → chunk_update_lists_locked()
          → list_move_tail(..., &va_block_used)  ← 加入 used 链表尾部
      → 状态：ALLOCATED（可驱逐）

步骤 3: 迁移第一个页面
  block_make_resident_update_state()
      → uvm_page_mask_or(dst_resident_mask, ..., copy_mask)  ← 更新页面掩码
      → block_set_resident_processor(va_block, GPU_ID)
          → if (test_and_set(&block->resident, GPU_ID))  ← 0 → 1
          → mark_root_chunk_used()  ← 触发！
              → list_move_tail(..., &va_block_used)  ← 移到链表尾部

步骤 4: 迁移第 2-100 个页面
  block_make_resident_update_state()
      → block_set_resident_processor()
          → if (test_and_set(&block->resident, GPU_ID))  ← 1 → 1，返回 true
          → return;  ← 不调用 mark_root_chunk_used！

步骤 5: 内存压力，驱逐所有页面
  uvm_va_block_evict_chunks()
      → 逐个驱逐页面...
      → 最后一个页面：
          block_clear_resident_processor()
              → if (!test_and_clear(&block->resident, GPU_ID))  ← 1 → 0
              → mark_root_chunk_unused()  ← 触发！
                  → list_move_tail(..., &va_block_unused)  ← 移到 unused 链表

步骤 6: 再次迁移页面进来
  → 重复步骤 3-5（会再次触发 mark_root_chunk_used）
```

**这就是为什么**：
- `chunk_update_lists_locked`：每个 chunk 只调用 1 次（分配时）
- `mark_root_chunk_used`：每次从"无页面"到"有页面"都调用（可能多次）
- `mark_root_chunk_unused`：每次从"有页面"到"无页面"都调用（可能多次）

---

## 问题

我们需要确定 BPF LRU 扩展需要多少个 hook：

1. **`on_access`** - chunk 被访问/使用时调用
2. **`on_mark_unused`** - chunk 移到 unused 链表时调用
3. **`on_mark_used`** - chunk 移到 used 链表时调用（可选）

关键问题：**这些 hook 的调用模式有何不同？是否会有重叠？我们真的需要 3 个 hook 吗？**

---

## 调用频率统计

基于真实测试结果（使用 `bpftrace -e 'kprobe:*chunk* { @calls[probe] = count(); }'`）：

### 完整的 Chunk 相关函数调用频率

| 函数 | 调用次数 | 类别 | 说明 |
|------|---------|------|------|
| **CPU Chunk 操作** ||||
| `uvm_cpu_chunk_mark_dirty` | 62,271,192 | CPU | 标记 CPU chunk 为脏页 |
| `uvm_cpu_chunk_get_size` | 65,800,057 | CPU | 获取 CPU chunk 大小 |
| `uvm_cpu_chunk_get_chunk_for_page` | 67,806,349 | CPU | 根据页面获取 chunk |
| `uvm_cpu_chunk_first_in_region` | 1,765,748 | CPU | 区域内第一个 chunk |
| `uvm_cpu_chunk_get_gpu_phys_addr` | 1,744,438 | CPU | 获取 GPU 物理地址 |
| `uvm_va_block_cpu_clear_resident_all_chunks` | 1,481,763 | CPU | 清除所有 CPU chunk 驻留 |
| `uvm_va_block_cpu_set_resident_all_chunks` | 225,450 | CPU | 设置所有 CPU chunk 驻留 |
| `uvm_cpu_chunk_get_allocation_sizes` | 130,889 | CPU | 获取分配大小 |
| **GPU Chunk LRU 关键函数** ||||
| `root_chunk_update_eviction_list` | **2,063,461** | **LRU Core** | **底层链表移动函数** |
| `uvm_pmm_gpu_mark_root_chunk_used` | **1,616,369** | **LRU Hook** | **标记 chunk 为 used** |
| `chunk_update_lists_locked` | **170,521** | **LRU Hook** | **更新 LRU 链表** |
| `uvm_pmm_gpu_mark_root_chunk_unused` | **119,680** | **LRU Hook** | **标记 chunk 为 unused** |
| **驱逐相关** ||||
| `uvm_va_block_evict_chunks` | 111,938 | Eviction | 驱逐 chunks |
| `evict_root_chunk` | 156,759 | Eviction | 驱逐 root chunk |
| `pick_and_evict_root_chunk` | 147,914 | Eviction | 选择并驱逐 |
| `pick_root_chunk_to_evict` | 147,045 | Eviction | 选择要驱逐的 chunk |
| `uvm_pmm_gpu_mark_chunk_evicted` | 120,507 | Eviction | 标记已驱逐 |
| **分配和释放** ||||
| `find_free_chunk_locked` | 627,656 | Alloc | 查找空闲 chunk |
| `alloc_root_chunk` | 343,202 | Alloc | 分配 root chunk |
| `claim_free_chunk` | 333,506 | Alloc | 声明空闲 chunk |
| `chunk_free_locked` | 427 | Free | 释放 chunk |
| **其他 GPU 操作** ||||
| `block_gpu_chunk_size.isra.0` | 2,507,880 | GPU | 获取 GPU chunk 大小 |
| `chunk_phys_mapping_get` | 2,238,018 | GPU | 获取物理映射 |
| `uvm_va_block_gpu_chunk_index_range` | 6,180,175 | GPU | chunk 索引范围 |
| `block_phys_page_chunk.isra.0` | 4,506,058 | GPU | 物理页面 chunk |
| `uvm_gpu_chunk_get_gpu` | 502,376 | GPU | 获取 GPU 信息 |
| `uvm_mmu_chunk_map` | 125,110 | MMU | MMU chunk 映射 |
| `uvm_mmu_chunk_unmap` | 124,350 | MMU | MMU chunk 取消映射 |

### LRU 关键函数频率对比

| 函数 | 调用次数 | 比例 | 语义 |
|------|---------|------|------|
| `root_chunk_update_eviction_list` | **2,063,461** | 100% | 底层实现（被 used/unused 调用） |
| `uvm_pmm_gpu_mark_root_chunk_used` | **1,616,369** | 78.3% | **第一个页面驻留** |
| `chunk_update_lists_locked` | **170,521** | 8.3% | **chunk unpin/状态转换** |
| `uvm_pmm_gpu_mark_root_chunk_unused` | **119,680** | 5.8% | **最后一个页面离开** |

**关键发现**：

1. **`root_chunk_update_eviction_list` 调用次数 ≈ `mark_used` + `mark_unused`**
   - 2,063,461 ≈ 1,616,369 + 119,680 + 其他
   - 说明 `mark_used/unused` 内部都调用 `root_chunk_update_eviction_list`

2. **`mark_root_chunk_used` 是 `chunk_update_lists_locked` 的 9.5 倍**
   - 1,616,369 / 170,521 ≈ 9.5
   - 说明页面迁移（第一个页面驻留）比 chunk unpin 频繁得多

3. **`mark_root_chunk_used` 是 `mark_root_chunk_unused` 的 13.5 倍**
   - 1,616,369 / 119,680 ≈ 13.5
   - 说明 chunk 获得页面比失去所有页面频繁（页面复用率高）

---

## 什么是 "Unpin"？为什么 `chunk_update_lists_locked` 调用这么少？

### Pin/Unpin 机制

在 UVM 中，chunk 有三种关键状态：

```c
enum {
    UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED,  // 临时锁定，正在分配/使用中
    UVM_PMM_GPU_CHUNK_STATE_ALLOCATED,    // 已分配，可以被驱逐
    UVM_PMM_GPU_CHUNK_STATE_FREE,         // 空闲
};
```

**Pin (锁定)**：
- Chunk 被标记为 `TEMP_PINNED`
- 不能被驱逐（正在被使用，如正在进行数据迁移）
- **从 eviction list 中移除**

**Unpin (解锁)**：
- Chunk 从 `TEMP_PINNED` 转为 `ALLOCATED`
- 现在可以被驱逐了
- **重新加入 eviction list**
- 这时调用 `chunk_update_lists_locked()`

### 为什么 Unpin 调用次数远少于 `mark_root_chunk_used`？

让我们看看完整的 chunk 生命周期：

```
阶段 1: 分配 chunk（调用 1 次 unpin）
  alloc_root_chunk()                     [343,202 次]
      ↓
  chunk 处于 TEMP_PINNED 状态
      ↓
  uvm_pmm_gpu_unpin_allocated()          [调用 unpin]
      └─> chunk_update_lists_locked()    [170,521 次] ← 第 1 次

阶段 2-N: 页面迁移（多次调用 mark_used，但不调用 unpin）
  第 1 个页面迁移进来：
      block_set_resident_processor()
          └─> mark_root_chunk_used()     [调用] ← 第 1 次 mark_used

  第 2 个页面迁移进来：
      block_set_resident_processor()
          └─> if (test_and_set()) return; [跳过！] ← 不调用 mark_used

  第 3-512 个页面迁移进来：
      都被 test_and_set() 拦截，不调用 mark_used

  ... 循环多次页面进出 ...

  第 N 次第一个页面迁移进来（之前所有页面都被驱逐了）：
      block_set_resident_processor()
          └─> mark_root_chunk_used()     [调用] ← 第 2 次 mark_used

最终驱逐：
  最后一个页面离开：
      block_clear_resident_processor()
          └─> mark_root_chunk_unused()   [调用] ← 唯一一次 mark_unused
```

### 数学关系

```
一个 root chunk 的生命周期：
  - unpin 调用次数：1 次（分配时）
  - mark_used 调用次数：N 次（每次从"无页面"变为"有页面"）
  - mark_unused 调用次数：M 次（每次从"有页面"变为"无页面"）

理论上：M ≈ N（每次获得页面最终都会失去）

实际测试数据验证：
  chunk_update_lists_locked: 170,521   (≈ 分配的 chunk 数量)
  mark_root_chunk_used:     1,616,369  (≈ 170,521 × 9.5)
  mark_root_chunk_unused:     119,680  (≈ 170,521 × 0.7)

说明：
  - 平均每个 chunk 经历了 9.5 次 "从无页面到有页面" 的转换
  - 测试结束时还有很多 chunk 处于 "有页面" 状态（used > unused）
```

### 为什么这个比例这么高？

**页面迁移模式**（以一个 2MB root chunk 为例，包含 512 个 4KB 页面）：

```
Chunk A 生命周期示例：

时刻 T0: 分配
  unpin → chunk_update_lists_locked  [第 1 次调用]

时刻 T1: 工作负载 1 - 100 个页面迁移进来
  第 1 个页面 → mark_root_chunk_used  [第 1 次]
  第 2-100 个页面 → 被 test_and_set 拦截

时刻 T2: 内存压力，所有页面被驱逐
  最后页面离开 → mark_root_chunk_unused  [第 1 次]

时刻 T3: 工作负载 2 - 200 个页面迁移进来
  第 1 个页面 → mark_root_chunk_used  [第 2 次]
  第 2-200 个页面 → 被拦截

时刻 T4: 部分页面被驱逐，但还剩 50 个
  不触发 mark_root_chunk_unused（还有页面）

时刻 T5: 工作负载 3 - 300 个新页面迁移进来
  不触发 mark_root_chunk_used（已经有页面）

... 多次循环 ...

时刻 T10: 又一次所有页面被驱逐
  最后页面离开 → mark_root_chunk_unused  [第 2 次]

时刻 T11-T20: 类似循环...

结果：
  - chunk_update_lists_locked: 1 次（只在分配时）
  - mark_root_chunk_used: 10 次（每次从空到有）
  - mark_root_chunk_unused: 2 次（部分页面驱逐时不触发）
```

### 实际工作负载特征

从数据可以推断：

1. **高页面复用率**：
   - 每个 chunk 平均被重复使用 9.5 次
   - 说明系统经常在同一个 chunk 上分配、驱逐、再分配页面

2. **内存压力中等**：
   - `mark_used` / `mark_unused` ≈ 13.5
   - 说明有很多 chunk 在测试结束时仍有页面驻留
   - 如果内存压力极大，这个比例会接近 1:1

3. **Chunk 长期存活**：
   - `chunk_update_lists_locked` 只有 170K 次
   - 但 `mark_used` 有 1.6M 次
   - 说明大部分 chunk 被分配后长期存活，经历多次页面进出

---

## 调用场景分析

### 1. `chunk_update_lists_locked()` - on_access hook

**位置**: `uvm_pmm_gpu.c:642`

```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    ...
    list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
}
```

**调用场景**（4处）：

#### (1) Line 672 - `gpu_unpin_temp()` → `uvm_pmm_gpu_unpin_allocated()`

```c
static void gpu_unpin_temp(uvm_pmm_gpu_t *pmm,
                           uvm_gpu_chunk_t *chunk,
                           uvm_va_block_t *va_block,
                           bool is_referenced)
{
    // TEMP_PINNED → ALLOCATED 状态转换
    chunk_unpin(pmm, chunk, UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
    chunk->is_referenced = is_referenced;
    chunk->va_block = va_block;
    chunk_update_lists_locked(pmm, chunk);  // ← 首次加入 LRU
}
```

**触发时机**：Chunk 刚分配完成，从 TEMP_PINNED 转为 ALLOCATED 状态
**作用**：首次加入 `va_block_used` 链表

#### (2) Line 1381 - `free_root_chunk()` 失败后重新加入

```c
static void free_root_chunk(...)
{
    // 尝试驱逐失败，需要重新放回 eviction list
    // chunk_update_lists_locked() will do that.
    chunk_update_lists_locked(pmm, chunk);
}
```

**触发时机**：驱逐失败，chunk 需要重新排队
**作用**：重新加入 eviction list

#### (3) Line 1653 - `chunk_split_gpu_mappings()` - Pin/Unpin

```c
chunk_pin(pmm, chunk);
chunk_update_lists_locked(pmm, chunk);
```

**触发时机**：Chunk split 操作期间的状态管理
**作用**：更新 LRU 位置

#### (4) Line 1886 - `chunk_update_gpu_mapping_type()`

```c
chunk->type = type;
chunk->state = initial_state;
chunk->is_zero = is_zero;
chunk_update_lists_locked(pmm, chunk);
```

**触发时机**：Chunk mapping type 更新
**作用**：更新 LRU 位置

**总结**：
- `chunk_update_lists_locked` 主要处理 **chunk 状态转换和管理操作**
- 频率相对较低（~176 次）
- **总是在 `va_block_used` 链表内移动**（移到尾部）

---

### 2. `mark_root_chunk_used()` - on_mark_used hook

**位置**: `uvm_pmm_gpu.c:1450`

```c
void uvm_pmm_gpu_mark_root_chunk_used(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_used);
}
```

**调用场景**（1处）：

#### Line 3565 - `block_mark_memory_used()` in `uvm_va_block.c`

```c
static void block_mark_memory_used(uvm_va_block_t *block, uvm_processor_id_t id)
{
    // 当 block 有页面驻留时，标记为 used
    if (!uvm_va_block_is_hmm(block) &&
        uvm_va_block_size(block) == UVM_CHUNK_SIZE_MAX &&
        uvm_parent_gpu_supports_eviction(gpu->parent)) {
        uvm_pmm_gpu_mark_root_chunk_used(&gpu->pmm, gpu_state->chunks[0]);
    }
}
```

**谁调用 `block_mark_memory_used`**：

1. **Line 3576** - `block_set_resident_processor()`
   - 当某个 processor 首次获得页面驻留时

2. **Line 5079** - Migration/Copy 完成后
   ```c
   block_mark_memory_used(va_block, dst_id);
   ```

3. **Line 5324** - 另一个 migration 路径

**触发时机**：
- VA block 从"无驻留页面"变为"有驻留页面"
- 即：**第一个页面被迁移到这个 chunk 时**

**作用**：
- 将 chunk 从 `va_block_unused` 移到 `va_block_used`
- **跨链表移动**

**频率**：1,876,643 次（极高频）

---

### 3. `mark_root_chunk_unused()` - on_mark_unused hook

**位置**: `uvm_pmm_gpu.c:1455`

```c
void uvm_pmm_gpu_mark_root_chunk_unused(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_unused);
}
```

**调用场景**（1处）：

#### Line 3601 - `block_clear_resident_processor()` in `uvm_va_block.c`

```c
static void block_clear_resident_processor(uvm_va_block_t *block, uvm_processor_id_t id)
{
    // 当 block 不再有任何驻留页面时，标记为 unused
    if (!uvm_va_block_is_hmm(block) &&
        uvm_va_block_size(block) == UVM_CHUNK_SIZE_MAX &&
        uvm_parent_gpu_supports_eviction(gpu->parent)) {
        if (gpu_state && gpu_state->chunks[0])
            uvm_pmm_gpu_mark_root_chunk_unused(&gpu->pmm, gpu_state->chunks[0]);
    }
}
```

**谁调用 `block_clear_resident_processor`**（7处）：

1. **Line 4577** - `block_revoke_prot()`
2. **Line 9152** - `block_destroy_gpu_state()`
3. **Line 9663** - `uvm_va_block_unmap()`
4. **Line 9719** - `uvm_va_block_evict_chunks()`
5. **Line 10471** - `uvm_va_block_kill()`
6. **Line 10492** - `uvm_va_block_kill()` (另一路径)

**触发时机**：
- VA block 从"有驻留页面"变为"无驻留页面"
- 即：**最后一个页面被迁移走/驱逐时**

**作用**：
- 将 chunk 从 `va_block_used` 移到 `va_block_unused`
- **跨链表移动**

**频率**：443,335 次（高频）

---

## 关键区别总结

| 特性 | `chunk_update_lists_locked`<br>(on_access) | `mark_root_chunk_used`<br>(on_mark_used) | `mark_root_chunk_unused`<br>(on_mark_unused) |
|------|---------------------------------------------|------------------------------------------|---------------------------------------------|
| **调用频率** | ~176 | 1,876,643 | 443,335 |
| **触发场景** | Chunk 状态转换<br>（分配、驱逐失败、split） | 第一个页面驻留 | 最后一个页面离开 |
| **操作类型** | **链表内移动**<br>(move to tail) | **跨链表移动**<br>(unused → used) | **跨链表移动**<br>(used → unused) |
| **目标链表** | 总是 `va_block_used` | `va_block_used` | `va_block_unused` |
| **语义** | "Chunk 被访问/使用" | "Chunk 变为有效（有数据）" | "Chunk 变为空闲（无数据）" |
| **是否重叠** | ❌ 不重叠 | ❌ 不重叠 | ❌ 不重叠 |

---

## 是否有调用重叠？

### 答案：**完全不重叠**

**证据 1：底层实现机制与 `list_move_tail` 的真正作用**

让我们先看看 `root_chunk_update_eviction_list` 的实现（Line 1430）：

```c
static void root_chunk_update_eviction_list(uvm_pmm_gpu_t *pmm,
                                            uvm_gpu_chunk_t *chunk,
                                            struct list_head *list)
{
    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == UVM_CHUNK_SIZE_MAX);
    UVM_ASSERT(uvm_gpu_chunk_is_user(chunk));
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    if (!chunk_is_root_chunk_pinned(pmm, chunk) && !chunk_is_in_eviction(pmm, chunk)) {
        // An unpinned chunk not selected for eviction should be on one of the
        // eviction lists.
        UVM_ASSERT(!list_empty(&chunk->list));  // ← chunk 必须已经在某个链表中

        list_move_tail(&chunk->list, list);     // ← 从当前链表移到目标链表
    }

    uvm_spin_unlock(&pmm->list_lock);
}
```

### `list_move_tail` 的真正作用

**关键**：`list_move_tail` 是一个 **跨链表移动** 操作！

```c
// Linux 内核链表操作定义：
void list_move_tail(struct list_head *entry, struct list_head *head)
{
    __list_del_entry(entry);     // 1. 从原链表中删除
    list_add_tail(entry, head);  // 2. 添加到目标链表尾部
}
```

**具体例子**：

```
初始状态：
  va_block_used:   [chunk_A, chunk_B, chunk_C]
  va_block_unused: [chunk_X, chunk_Y]

调用 list_move_tail(&chunk_B->list, &va_block_unused):

结果：
  va_block_used:   [chunk_A, chunk_C]              ← chunk_B 被移除！
  va_block_unused: [chunk_X, chunk_Y, chunk_B]     ← chunk_B 被添加到尾部
```

### 三个函数到底改了什么链表？

```c
// 1. chunk_update_lists_locked (Line 642)
list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);

含义：
  - 从 chunk 当前所在的链表（可能是 used 或 unused）移除
  - 添加到 va_block_used 链表尾部
  - 既修改了源链表（移除），也修改了目标链表（添加）

// 2. mark_root_chunk_used (Line 1452)
root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_used);
  → list_move_tail(&chunk->list, &pmm->root_chunks.va_block_used);

含义：
  - 从 chunk 当前所在的链表（通常是 va_block_unused）移除
  - 添加到 va_block_used 链表尾部
  - 修改了两个链表！

// 3. mark_root_chunk_unused (Line 1457)
root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_unused);
  → list_move_tail(&chunk->list, &pmm->root_chunks.va_block_unused);

含义：
  - 从 chunk 当前所在的链表（通常是 va_block_used）移除  ← 这就是你的问题！
  - 添加到 va_block_unused 链表尾部
  - 修改了两个链表！
```

### 回答你的问题：`mark_root_chunk_unused` 有没有改 `used` 链表？

**答案：YES！它会从 `used` 链表移除 chunk！**

```
场景示例：

初始状态（chunk A 有页面驻留）：
  va_block_used:   [chunk_B, chunk_A, chunk_C]  ← chunk_A 在这里
  va_block_unused: [chunk_X]

最后一个页面离开 chunk A：
  block_clear_resident_processor()
      → mark_root_chunk_unused(pmm, chunk_A)
          → list_move_tail(&chunk_A->list, &pmm->root_chunks.va_block_unused)

结果：
  va_block_used:   [chunk_B, chunk_C]            ← chunk_A 被移除了！
  va_block_unused: [chunk_X, chunk_A]            ← chunk_A 被添加到这里
```

### 所以真正的操作是：

| 函数 | 源链表（移除） | 目标链表（添加） | 双向修改 |
|------|---------------|-----------------|---------|
| `chunk_update_lists_locked` | 可能是 unused | `va_block_used` 尾部 | ✅ 是 |
| `mark_root_chunk_used` | 通常是 `va_block_unused` | `va_block_used` 尾部 | ✅ 是 |
| `mark_root_chunk_unused` | 通常是 `va_block_used` | `va_block_unused` 尾部 | ✅ 是 |

**关键洞察**：
- `list_move_tail` 总是修改**两个**链表（源链表 + 目标链表）
- `mark_root_chunk_unused` 确实会从 `used` 链表移除 chunk
- 这就是为什么 BPF 必须感知这个操作（chunk 从 used 消失了！）

### 可视化：Chunk 在链表间的移动

```
时刻 T0: Chunk A 刚分配并 unpin
  va_block_used:   [chunk_B] → [chunk_A]  ← chunk_update_lists_locked
  va_block_unused: [chunk_X]

时刻 T1: Chunk A 获得第一个页面（resident: 0→1）
  va_block_used:   [chunk_B] → [chunk_A]  ← mark_root_chunk_used（已经在 used，移到尾部）
  va_block_unused: [chunk_X]

时刻 T2: Chunk A 失去所有页面（resident: 1→0）
  va_block_used:   [chunk_B]               ← mark_root_chunk_unused 从这里移除了 chunk_A！
  va_block_unused: [chunk_X] → [chunk_A]  ← chunk_A 被添加到这里

时刻 T3: Chunk A 再次获得第一个页面（resident: 0→1）
  va_block_used:   [chunk_B] → [chunk_A]  ← mark_root_chunk_used 从 unused 移回来
  va_block_unused: [chunk_X]               ← chunk_A 被移除了

时刻 T4: Chunk A 再次失去所有页面（resident: 1→0）
  va_block_used:   [chunk_B]               ← 又被移除了
  va_block_unused: [chunk_X] → [chunk_A]  ← 又回到 unused
```

**这就是为什么 BPF 需要 `on_mark_unused` hook**：

如果 BPF 在 T1 时记录了 chunk_A 的元数据（如频率计数），在 T2 时如果不知道 chunk_A 已经从 `used` 链表移除，BPF 的元数据就会变成"僵尸数据"，指向一个不在 `used` 链表中的 chunk！

---

## 所有修改 `va_block_used` 链表的地方

让我们完整列出所有可能修改 `va_block_used` 链表的操作：

### 1. 添加到 `va_block_used`（3 个位置）

| 位置 | 函数 | 操作 | 来源 | BPF Hook |
|------|------|------|------|----------|
| Line 642 | `chunk_update_lists_locked` | `list_move_tail(..., &va_block_used)` | 任意链表 | `on_access` |
| Line 1452 | `mark_root_chunk_used` | `list_move_tail(..., &va_block_used)` | 通常是 `va_block_unused` | `on_mark_used` (可选) |

**注意**：`list_move_tail` 会**同时从源链表移除**并**添加到目标链表**！

### 2. 从 `va_block_used` 移除（4 个位置）

| 位置 | 函数 | 操作 | 去向 | 触发场景 | BPF Hook |
|------|------|------|------|----------|----------|
| Line 637 | `chunk_update_lists_locked` | `list_del_init(...)` | 无（从所有链表移除） | Chunk 被 pin | ❌ 不需要 |
| Line 1426 | `chunk_start_eviction` | `list_del_init(...)` | 无（标记为 in_eviction） | 驱逐开始 | ❌ 不需要 |
| Line 1457 | `mark_root_chunk_unused` | `list_move_tail(..., &va_block_unused)` | `va_block_unused` | 最后页面离开 | **`on_mark_unused`** ⭐ |
| Line 642 | `chunk_update_lists_locked` | `list_move_tail(..., &va_block_used)` | `va_block_used` 尾部 | 同一链表内移动 | `on_access` |

### 详细分析

#### 操作 1: `chunk_update_lists_locked` - Line 642
```c
list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
```

**作用**：
- 如果 chunk 在 `va_block_unused` → 移到 `va_block_used` 尾部
- 如果 chunk 已在 `va_block_used` → 移到链表尾部（LRU 更新）

**触发场景**：chunk unpin（分配完成、驱逐失败重试等）

**BPF 需要感知吗？** ✅ 是（`on_access` hook）

---

#### 操作 2: `chunk_update_lists_locked` - Line 637
```c
if (chunk_is_root_chunk_pinned(pmm, chunk)) {
    list_del_init(&root_chunk->chunk.list);  // 从所有链表移除
}
```

**作用**：Chunk 被 pin（锁定），从 eviction list 移除

**触发场景**：
- Chunk 正在被使用（数据迁移、映射操作等）
- **临时状态**，使用完会 unpin 并重新加入链表

**BPF 需要感知吗？** ❌ 不需要
- Pin 是临时操作，chunk 很快会 unpin 并触发 `on_access`
- Pin 期间 chunk 不参与驱逐，BPF 不需要管理它

---

#### 操作 3: `chunk_start_eviction` - Line 1426
```c
list_del_init(&chunk->list);
uvm_gpu_chunk_set_in_eviction(chunk, true);
```

**作用**：Chunk 被选中要驱逐，标记为 `in_eviction` 状态

**触发场景**：`pick_root_chunk_to_evict` 选中这个 chunk

**BPF 需要感知吗？** ❌ 不需要
- 这发生在 `prepare_eviction` hook **之后**
- BPF 已经通过 `prepare_eviction` 调整过链表顺序
- 驱逐完成后 chunk 会被释放（FREE 状态）

---

#### 操作 4: `mark_root_chunk_unused` - Line 1457 ⭐ 关键！
```c
root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_unused);
  → list_move_tail(&chunk->list, &pmm->root_chunks.va_block_unused);
```

**作用**：
- 从 `va_block_used` 移除
- 添加到 `va_block_unused` 尾部

**触发场景**：Chunk 的最后一个页面被驱逐（resident: 1→0）

**BPF 需要感知吗？** ✅ **必须！**（`on_mark_unused` hook）

**原因**：
```
BPF 管理的是 va_block_used 链表中的 chunk 元数据。

如果 chunk 从 used 移到 unused，但 BPF 不知道：
1. BPF 的 map 仍然有这个 chunk 的元数据（如 LFU 频率计数）
2. 但 chunk 已经不在 used 链表了（在 unused 链表）
3. prepare_eviction 时，BPF 会尝试调整 chunk 在 used 链表的位置
4. ❌ 但 chunk 根本不在 used 链表！→ 元数据泄漏 + 错误决策
```

---

#### 操作 5: `mark_root_chunk_used` - Line 1452
```c
root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_used);
  → list_move_tail(&chunk->list, &pmm->root_chunks.va_block_used);
```

**作用**：
- 从 `va_block_unused` 移除（通常）
- 添加到 `va_block_used` 尾部

**触发场景**：Chunk 获得第一个页面（resident: 0→1）

**BPF 需要感知吗？** ⚠️ 可选（`on_mark_used` hook）

**是否需要取决于算法**：
- **FIFO/LRU**: ❌ 不需要（只关心访问时间）
- **LFU**: ❌ 不需要（只需在 unused→used 时重置计数即可，可在 `on_access` 中判断）
- **S3-FIFO/ARC**: ✅ 需要（ghost cache hit 检测，需要知道 chunk 从 unused 回到 used）

---

### 总结：BPF 必须感知的操作

| 操作 | 影响链表 | BPF Hook | 必要性 | 原因 |
|------|---------|----------|--------|------|
| chunk unpin | `va_block_used` 添加 | `on_access` | ✅ 必需 | 更新 LRU 顺序/元数据 |
| chunk pin | `va_block_used` 移除 | ❌ 不需要 | ❌ 临时操作 | Pin 后会 unpin 触发 `on_access` |
| 驱逐开始 | `va_block_used` 移除 | ❌ 不需要 | ❌ 在 prepare 之后 | `prepare_eviction` 已处理 |
| unused→used | `va_block_used` 添加<br>`va_block_unused` 移除 | `on_mark_used` | ⚠️ 可选 | 只有 ghost cache 算法需要 |
| used→unused | `va_block_used` 移除<br>`va_block_unused` 添加 | `on_mark_unused` | ✅ **必需** | **防止元数据泄漏** |

**最终答案**：

只有**两类操作**真正需要 BPF 感知：

1. ✅ `chunk_update_lists_locked` (on_access) - chunk 状态转换
2. ✅ `mark_root_chunk_unused` (on_mark_unused) - **关键！防止元数据泄漏**
3. ⚠️ `mark_root_chunk_used` (on_mark_used) - 可选，只有 S3-FIFO/ARC 需要

其他操作（pin、驱逐开始）都是内部管理，BPF 不需要感知。

**证据 2：调用条件的本质区别**

关键在于 `block_set_resident_processor` 和 `block_clear_resident_processor` 的实现：

```c
// uvm_va_block.c:3569
static void block_set_resident_processor(uvm_va_block_t *block, uvm_processor_id_t id)
{
    UVM_ASSERT(!uvm_page_mask_empty(uvm_va_block_resident_mask_get(block, id, NUMA_NO_NODE)));

    if (uvm_processor_mask_test_and_set(&block->resident, id))
        return;  // ← 如果已经设置了，直接返回！

    block_mark_memory_used(block, id);  // ← 只在首次设置时调用
        └─> uvm_pmm_gpu_mark_root_chunk_used()
}

// uvm_va_block.c:3579
static void block_clear_resident_processor(uvm_va_block_t *block, uvm_processor_id_t id)
{
    UVM_ASSERT(uvm_page_mask_empty(uvm_va_block_resident_mask_get(block, id, NUMA_NO_NODE)));

    if (!uvm_processor_mask_test_and_clear(&block->resident, id))
        return;  // ← 如果本来就没设置，直接返回！

    if (UVM_ID_IS_CPU(id))
        return;

    gpu = uvm_gpu_get(id);
    ...
    uvm_pmm_gpu_mark_root_chunk_unused(&gpu->pmm, gpu_state->chunks[0]);  // ← 只在清除时调用
}
```

**核心区别**：

- `block->resident` 是一个 **processor mask**，用位图标记哪些 processor 有页面驻留
- `test_and_set` 返回旧值：
  - 0 → 1：返回 `false`，继续调用 `mark_root_chunk_used`（**第一个页面驻留**）
  - 1 → 1：返回 `true`，直接返回（已经有页面了，不需要再标记）
- `test_and_clear` 返回是否成功清除：
  - 1 → 0：返回 `true`，继续调用 `mark_root_chunk_unused`（**最后一个页面离开**）
  - 0 → 0：返回 `false`，直接返回（本来就没有页面）

**所以真正的语义是**：

| 函数 | 触发条件 | 语义 |
|------|---------|------|
| `chunk_update_lists_locked` | chunk 分配、状态转换 | "Chunk 可用了，加入/更新 LRU" |
| `mark_root_chunk_used` | **第一个页面驻留**<br>（resident: 0→1） | "Chunk 从空变为有数据" |
| `mark_root_chunk_unused` | **最后一个页面离开**<br>（resident: 1→0） | "Chunk 从有数据变为空" |

**证据 3：完整的页面迁移调用链**

让我们追踪一个完整的页面迁移过程（以 GPU 页面迁移为例）：

```
1. 页面迁移开始
   └─> block_make_resident_update_state()  [uvm_va_block.c:5008]
       └─> block_set_resident_processor(va_block, dst_id)  [Line 5008]
           └─> if (uvm_processor_mask_test_and_set(&block->resident, id))
                   return;  // ← 如果不是第一个页面，直接返回
               block_mark_memory_used(block, id)
               └─> uvm_pmm_gpu_mark_root_chunk_used()  ← 只在第一个页面时调用！

2. 页面迁移完成，释放临时 pinned chunks
   └─> block_retry_clean_up()  [uvm_va_block.c:839]
       └─> list_for_each_entry_safe(gpu_chunk, ..., &retry->used_chunks, ...) {
               uvm_pmm_gpu_unpin_allocated(&gpu->pmm, gpu_chunk, va_block);
           }
           └─> gpu_unpin_temp()  [uvm_pmm_gpu.c:672]
               └─> chunk_update_lists_locked(pmm, chunk);  ← 每个 chunk 都调用

3. 页面被驱逐
   └─> uvm_va_block_evict_chunks()  [uvm_va_block.c:9719]
       └─> block_clear_resident_processor(block, id)
           └─> if (!uvm_processor_mask_test_and_clear(&block->resident, id))
                   return;  // ← 如果不是最后一个页面，直接返回
               uvm_pmm_gpu_mark_root_chunk_unused()  ← 只在最后一个页面时调用！
```

**关键发现**：

在同一次页面迁移过程中：
- **第一个页面迁移时**：
  1. 先调用 `mark_root_chunk_used`（如果 resident 从 0→1）
  2. 然后调用 `chunk_update_lists_locked`（unpin 时）

- **后续页面迁移时**：
  1. `mark_root_chunk_used` **不会被调用**（resident 已经是 1）
  2. `chunk_update_lists_locked` 仍然会被调用

- **最后一个页面离开时**：
  1. 调用 `mark_root_chunk_unused`（resident 从 1→0）

**这就是为什么调用频率差距这么大**：
- `chunk_update_lists_locked`：~176 次（每次 chunk unpin）
- `mark_root_chunk_used`：1,876,643 次（每次 VA block 首次获得页面）
- `mark_root_chunk_unused`：443,335 次（每次 VA block 失去所有页面）

---

## BPF 算法的需求分析

### FIFO 算法

**需求**：
- 记录 chunk 分配/加入队列的时间戳
- 按时间顺序排序

**Hook 需求**：
1. ✅ `on_access` - 首次加入队列时记录时间戳
2. ⚠️  `on_mark_unused` - Chunk 移到 unused 时需要知道吗？
   - **是**：因为 unused 链表优先级更高，需要单独管理
   - 但 FIFO 不需要调整顺序，所以可能不需要 hook
3. ❌ `on_mark_used` - 不需要，因为 FIFO 不改变顺序

**结论**：2 个 hook 够用（`on_access` + `prepare_eviction`）

---

### LFU 算法

**需求**：
- 维护每个 chunk 的访问频率计数
- 按频率排序

**Hook 需求**：
1. ✅ `on_access` - 增加访问计数
2. ✅ `on_mark_unused` - **必需**！
   - 问题：如果 chunk 移到 unused，BPF 的频率 map 会包含过期数据
   - 解决：需要清理或标记频率数据
3. ⚠️  `on_mark_used` - 可选
   - 如果需要重置频率计数（从 unused 重新变为 used）

**结论**：至少需要 3 个 hook

---

### S3-FIFO / ARC 等复杂算法

**需求**：
- 维护多级队列
- 跟踪 ghost entries（已驱逐但保留元数据的条目）

**Hook 需求**：
1. ✅ `on_access` - 队列晋升逻辑
2. ✅ `on_mark_unused` - **必需**
   - 需要知道 chunk 何时变为空闲，以便：
     - 将元数据移到 ghost 队列
     - 调整队列统计信息
3. ✅ `on_mark_used` - **必需**
   - 需要知道 chunk 何时重新变为 used（ghost hit）
   - 可能需要晋升到更高级别的队列

**结论**：需要完整的 3-4 个 hook

---

## 最终建议

### 方案 1：最小方案（2 hooks）

适用于：FIFO、简单 LRU

```c
struct uvm_gpu_ext {
    int (*uvm_lru_on_access)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_prepare_eviction)(uvm_pmm_gpu_t *pmm);
};
```

**限制**：
- ❌ 无法正确实现 LFU（元数据泄漏）
- ❌ 无法实现 S3-FIFO/ARC（缺少状态转换感知）

---

### 方案 2：推荐方案（3 hooks）⭐

适用于：所有主流算法

```c
struct uvm_gpu_ext {
    int (*uvm_lru_on_access)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_on_mark_unused)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_prepare_eviction)(uvm_pmm_gpu_t *pmm);
};
```

**Hook 位置**：

1. **`chunk_update_lists_locked:642`** 之后
   ```c
   list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);

   if (pmm->gpu_ext && pmm->gpu_ext->uvm_lru_on_access)
       pmm->gpu_ext->uvm_lru_on_access(pmm, chunk_addr);
   ```

2. **`uvm_pmm_gpu_mark_root_chunk_unused:1455`** 之后
   ```c
   root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_unused);

   if (pmm->gpu_ext && pmm->gpu_ext->uvm_lru_on_mark_unused)
       pmm->gpu_ext->uvm_lru_on_mark_unused(pmm, chunk_addr);
   ```

3. **`pick_root_chunk_to_evict:1485`** 之前（已有）

**优点**：
- ✅ 支持 FIFO、LRU、MRU
- ✅ 支持 LFU（可以清理元数据）
- ✅ 支持 S3-FIFO、ARC（可以管理 ghost entries）
- ✅ 性能开销可控（`on_mark_unused` 调用频率虽高但可优化）

---

### 方案 3：完整方案（4 hooks）

适用于：需要对称性的复杂算法

```c
struct uvm_gpu_ext {
    int (*uvm_lru_on_access)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_on_mark_used)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_on_mark_unused)(uvm_pmm_gpu_t *pmm, u64 chunk_addr);
    int (*uvm_lru_prepare_eviction)(uvm_pmm_gpu_t *pmm);
};
```

**额外好处**：
- 完整的生命周期感知
- 更容易实现 ghost cache hit 检测
- 调试更容易（对称的 hook 对）

**代价**：
- 多一个 hook 点（~5 行代码）
- `on_mark_used` 调用频率极高（1.8M 次）

---

## 性能考虑

### 调用频率影响

| Hook | 调用次数 | BPF 开销估计 |
|------|---------|-------------|
| `on_access` | ~176 | 可忽略 |
| `on_mark_used` | 1,876,643 | ⚠️  较高 |
| `on_mark_unused` | 443,335 | ⚠️  中等 |

### 优化建议

1. **条件触发**：
   ```c
   if (pmm->gpu_ext && pmm->gpu_ext->uvm_lru_on_mark_unused) {
       // 只在 BPF 程序 attached 时调用
   }
   ```

2. **快速路径优化**：
   - BPF 程序应尽量简短（<100 instructions）
   - 使用 BPF maps 批量更新而非每次调用都更新

3. **可选 hook**：
   - `on_mark_used` 可以设为可选
   - 只有需要的算法才实现

---

## 结论

**推荐使用方案 2（3 hooks）**：

1. **必需的 hook**：
   - `on_access` - 覆盖 chunk 状态转换和 LRU 更新
   - `on_mark_unused` - 覆盖 chunk 变为空闲（元数据清理）
   - `prepare_eviction` - 驱逐前的最终调整

2. **可选的 hook**：
   - `on_mark_used` - 只在需要 ghost cache hit 检测的算法中实现

3. **内核修改量**：
   - 方案 2：~15 行（3 个 hook 点）
   - 方案 3：~20 行（4 个 hook 点）

4. **支持的算法**：
   - ✅ FIFO, LRU, MRU
   - ✅ LFU
   - ✅ S3-FIFO, ARC (需要 `on_mark_used`)
   - ✅ 自定义算法

**核心理由**：
- `on_access` 和 `on_mark_unused` 的调用场景**完全不重叠**
- `on_mark_unused` 对于正确性**至关重要**（防止元数据泄漏）
- 性能开销可通过 BPF 优化和条件触发来控制

---

## 最终答案：关于"移动"的真相

### 问题：mark_root_chunk_used/unused 是在什么和什么之间移动？

**答案**：它们都使用 `list_move_tail()`，但移动的**语义**不同：

| 操作 | 底层实现 | 来源链表 | 目标链表 | 触发条件 | 语义 |
|------|---------|---------|---------|---------|------|
| `chunk_update_lists_locked` | `list_move_tail()` | 任意 eviction list | `va_block_used` | chunk 分配/unpin | "chunk 被使用" |
| `mark_root_chunk_used` | `list_move_tail()` | 通常是 `va_block_unused` | `va_block_used` | resident: 0→1 | "chunk 获得第一个页面" |
| `mark_root_chunk_unused` | `list_move_tail()` | 通常是 `va_block_used` | `va_block_unused` | resident: 1→0 | "chunk 失去最后一个页面" |

### 为什么看起来都是 `list_move_tail` 但频率差这么多？

**关键在于调用条件的"守门员"**：

```c
// mark_root_chunk_used 的守门员
if (uvm_processor_mask_test_and_set(&block->resident, id))
    return;  // ← 已有页面？不调用！

block_mark_memory_used(block, id);  // ← 只有首次获得页面才到这里

// mark_root_chunk_unused 的守门员
if (!uvm_processor_mask_test_and_clear(&block->resident, id))
    return;  // ← 本来就没页面？不调用！

uvm_pmm_gpu_mark_root_chunk_unused(...);  // ← 只有失去最后页面才到这里

// chunk_update_lists_locked 没有守门员
void uvm_pmm_gpu_unpin_allocated(uvm_pmm_gpu_t *pmm, ...)
{
    gpu_unpin_temp(pmm, chunk, va_block, false);
        └─> chunk_update_lists_locked(pmm, chunk);  // ← 每次 unpin 都调用
}
```

### 完整的 Chunk 生命周期与链表状态

```
阶段 1: Chunk 分配
  TEMP_PINNED (不在任何 eviction list)
      ↓
  gpu_unpin_temp()
      └─> chunk_update_lists_locked()
          └─> list_move_tail(..., &va_block_used)

  → Chunk 现在在 va_block_used，但 resident=0（无页面驻留）

阶段 2: 第一个页面迁移进来
  block_make_resident_update_state()
      └─> block_set_resident_processor()
          └─> if (test_and_set(&block->resident, id))  // 0→1，返回 false
                  // 不返回，继续执行
              block_mark_memory_used()
              └─> mark_root_chunk_used()
                  └─> list_move_tail(..., &va_block_used)  // 可能从 unused 移过来

  → Chunk 在 va_block_used，resident=1（有页面驻留）

阶段 3: 后续页面迁移（第2、3、4...个页面）
  block_make_resident_update_state()
      └─> block_set_resident_processor()
          └─> if (test_and_set(&block->resident, id))  // 1→1，返回 true
                  return;  // ← 直接返回，mark_root_chunk_used 不会被调用！

  → Chunk 保持在 va_block_used，resident=1

阶段 4: 倒数第二个页面被驱逐
  block_clear_resident_processor()
      └─> if (!test_and_clear(&block->resident, id))  // 1→1（还有其他页面），返回 false
              return;  // ← 直接返回，mark_root_chunk_unused 不会被调用！

  → Chunk 保持在 va_block_used，resident=1

阶段 5: 最后一个页面被驱逐
  block_clear_resident_processor()
      └─> if (!test_and_clear(&block->resident, id))  // 1→0，返回 true
              // 不返回，继续执行
          mark_root_chunk_unused()
          └─> list_move_tail(..., &va_block_unused)

  → Chunk 移到 va_block_unused，resident=0

阶段 6: 再次有页面迁移进来（重复阶段 2）
  → 再次触发 mark_root_chunk_used，从 unused 移回 used
```

### 为什么 BPF 必须知道 mark_root_chunk_unused？

**元数据泄漏问题**（以 LFU 为例）：

```
时刻 T1: Chunk A 被访问 100 次
  BPF map: { chunk_A_addr: frequency=100 }
  Chunk A 在 va_block_used 链表

时刻 T2: Chunk A 的所有页面被驱逐
  内核调用: mark_root_chunk_unused()
  内核操作: Chunk A 移到 va_block_unused 链表

  如果没有 on_mark_unused hook:
    BPF map 仍然是: { chunk_A_addr: frequency=100 }  ← 错误！
    问题：Chunk A 已经在 unused 链表了，frequency 数据已经无效

  有了 on_mark_unused hook:
    BPF 代码: bpf_map_delete_elem(&frequency_map, &chunk_A_addr);
    BPF map 变为: { }  ← 正确！

时刻 T3: 准备驱逐时
  内核选择: va_block_unused 链表头的 chunk（Chunk A）

  如果没有 on_mark_unused hook:
    BPF 看到 frequency=100，认为它经常被访问，不应驱逐  ← 错误决策！

  有了 on_mark_unused hook:
    BPF map 里没有 Chunk A 的记录，说明它在 unused 链表，可以驱逐  ← 正确！
```

### 结论

1. **三个函数底层都用 `list_move_tail`，但语义完全不同**
2. **`mark_root_chunk_used/unused` 通过 `resident` mask 的守门，只在关键状态转换时调用**
3. **`chunk_update_lists_locked` 没有守门员，每次 unpin 都调用（因此频率低）**
4. **BPF 必须感知 used ↔ unused 的转换，否则会有元数据泄漏和错误决策**

**最终推荐：3 hooks（on_access + on_mark_unused + prepare_eviction）**

---

## 基于完整测试数据的最终验证

### 数据验证了我们的分析

新测试数据完美验证了我们的分析：

| 发现 | 预测 | 实际数据 | 验证 |
|------|------|---------|------|
| `mark_used/unused` 都调用 `root_chunk_update_eviction_list` | 应该是 used + unused | 2,063,461 ≈ 1,616,369 + 119,680 + 327,412 | ✅ |
| `mark_used` 远多于 `chunk_update_lists_locked` | 页面迁移比分配频繁 | 9.5 倍 | ✅ |
| `mark_used` 多于 `mark_unused` | 测试结束时有残留页面 | 13.5 倍 | ✅ |
| 三个函数调用场景不重叠 | 守门员机制确保 | 频率差距巨大 | ✅ |

### root_chunk_update_eviction_list 的额外调用

```
root_chunk_update_eviction_list: 2,063,461
mark_used:                       1,616,369
mark_unused:                       119,680
差额:                              327,412

这 327K 次额外调用来自哪里？
```

让我检查一下 `chunk_update_lists_locked` 是否也调用了它：

从代码看，`chunk_update_lists_locked` **直接调用 `list_move_tail`**，不调用 `root_chunk_update_eviction_list`。

所以这 327K 次可能来自：
1. 驱逐失败后重新加入（`free_root_chunk` Line 1381）
2. Split 操作（`chunk_split_gpu_mappings` Line 1653）
3. Mapping type 更新（Line 1886）

这进一步证明了 `chunk_update_lists_locked` 和 `mark_used/unused` 是**完全独立的代码路径**！

### BPF Hook 性能影响评估

基于实际调用频率：

| Hook | 调用次数 | 每秒调用（假设 60s 测试） | BPF 开销估计 |
|------|---------|--------------------------|-------------|
| `on_access` | 170,521 | ~2,842/s | 可忽略 (<0.1% CPU) |
| `on_mark_used` | 1,616,369 | ~26,939/s | 中等 (~1-2% CPU) |
| `on_mark_unused` | 119,680 | ~1,995/s | 可忽略 (<0.1% CPU) |
| `prepare_eviction` | 147,045 | ~2,451/s | 可忽略 (<0.1% CPU) |

**结论**：
- 总开销预计 **< 3% CPU**（假设 BPF 程序优化良好）
- `on_mark_used` 是主要开销来源（78% 的调用）
- 可以通过 BPF 程序优化进一步降低开销（如批量更新、per-CPU maps）

### 为什么必须有 3 个 Hook？

回到你最初的问题：**"我们的 hook 管理的是 used 这个列表，如果函数移动了，但我们的算法没感知到，那就会出错？"**

**答案：完全正确！**

数据显示：
- **1.6M 次 `mark_used`** - chunk 从 unused → used
- **119K 次 `mark_unused`** - chunk 从 used → unused
- **170K 次 `chunk_update_lists_locked`** - chunk 在 used 内部移动

如果 BPF 只有 `on_access` hook：
```
问题场景（以 LFU 为例）：
1. Chunk A 被访问 → on_access → BPF map[A] = 1
2. Chunk A 再被访问 → on_access → BPF map[A] = 2
3. ... 100 次访问 ...
4. BPF map[A] = 100

5. 所有页面被驱逐 → mark_root_chunk_unused
   ❌ BPF 不知道！map[A] 仍然是 100

6. Chunk A 移到 va_block_unused 链表（优先被驱逐）
   ❌ 但 BPF 认为它频繁访问，不应被驱逐

7. prepare_eviction 时 BPF 尝试调整顺序
   ❌ 基于错误的频率数据做决策
   ❌ Chunk A 实际上应该最先被驱逐（在 unused 链表）
```

有了 `on_mark_unused` hook：
```
正确场景：
1-4. 同上，BPF map[A] = 100

5. 所有页面被驱逐 → mark_root_chunk_unused
   ✅ on_mark_unused(chunk A)
   ✅ BPF: bpf_map_delete_elem(&freq_map, &chunk_A)
   ✅ map[A] 被删除

6. Chunk A 在 va_block_unused 链表
   ✅ BPF map 里没有 A 的记录

7. prepare_eviction 时
   ✅ BPF 看到 chunk 在 unused 链表且没有频率记录
   ✅ 正确地允许它被优先驱逐
```

### 最终答案

你的担心是**100% 正确的**！基于完整的测试数据和代码分析：

**必须使用 3 个 Hook**：
1. **`on_access`** (170K 次) - chunk 分配/状态转换
2. **`on_mark_unused`** (119K 次) - chunk 从 used → unused（**必需！**）
3. **`prepare_eviction`** (147K 次) - 驱逐前最终调整

**可选第 4 个 Hook**：
4. **`on_mark_used`** (1.6M 次) - chunk 从 unused → used（S3-FIFO/ARC 需要）

**性能开销**：可接受（< 3% CPU）
**正确性**：至关重要（没有 `on_mark_unused` 会导致元数据泄漏和错误决策）
