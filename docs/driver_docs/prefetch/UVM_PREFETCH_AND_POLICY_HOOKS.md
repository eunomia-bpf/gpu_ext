# NVIDIA UVM 预取机制详解与策略替换指南

## 目录
1. [Prefetch 机制详解](#1-prefetch-机制详解)
2. [Eviction Policy 替换方案](#2-eviction-policy-替换方案)
3. [实现 FIFO 示例](#3-实现-fifo-示例)
4. [最小侵入修改点](#4-最小侵入修改点)

---

## 1. Prefetch 机制详解

### 1.1 核心数据结构

#### Bitmap Tree (uvm_perf_prefetch.h:41-50)

```c
typedef struct {
    uvm_page_mask_t pages;       // 每个 bit 表示一个 page 是否存在
    uvm_page_index_t offset;     // 偏移量（用于对齐 big page）
    NvU16 leaf_count;            // 叶子节点数量（页数）
    NvU8 level_count;            // 树的层数
} uvm_perf_prefetch_bitmap_tree_t;
```

**树结构**:
- **满二叉树**: 层数 = `log2(roundup_pow_of_two(leaf_count)) + 1`
- **叶子节点**: 每个叶子对应一个 4KB 页面
- **内部节点**: 每个节点维护子树的"页面存在计数"

#### Prefetch Hint (uvm_perf_prefetch.h:31-36)

```c
typedef struct {
    uvm_page_mask_t prefetch_pages_mask;  // 建议预取的页面掩码
    uvm_processor_id_t residency;         // 预取目标处理器
} uvm_perf_prefetch_hint_t;
```

### 1.2 算法流程

#### 完整调用链

```
uvm_va_block_get_prefetch_hint()                    [uvm_va_block.c:11828]
  └─> uvm_perf_prefetch_get_hint_va_block()         [uvm_perf_prefetch.c:447]
      ├─> uvm_perf_prefetch_prenotify_fault_migrations() [line 327]
      │   ├─> init_bitmap_tree_from_region()        [line 222] ← 初始化树
      │   │   └─> level_count = ilog2(roundup_pow_of_two(leaf_count)) + 1
      │   ├─> update_bitmap_tree_from_va_block()    [line 240] ← 更新树
      │   │   └─> grow_fault_granularity()          [line 164]
      │   │       └─> grow_fault_granularity_if_no_thrashing() [line 148]
      │   └─> compute_prefetch_mask()               [line 299]
      │       └─> compute_prefetch_region()         [line 102] ← 核心算法
      │           └─> traverse tree with 51% threshold
      └─> check min_faults threshold                [line 477]
```

#### 核心算法: compute_prefetch_region() (Line 102-146)

**输入**:
- `page_index`: 发生 fault 的页面索引
- `bitmap_tree`: 当前 VA block 的 bitmap tree
- `max_prefetch_region`: 允许预取的最大区域

**算法步骤**:

```c
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    NvU16 counter;
    uvm_perf_prefetch_bitmap_tree_iter_t iter;
    uvm_va_block_region_t prefetch_region = uvm_va_block_region(0, 0);

    // 从叶子节点向上遍历
    uvm_perf_prefetch_bitmap_tree_traverse_counters(
        counter, bitmap_tree,
        page_index - max_prefetch_region.first + bitmap_tree->offset,
        &iter)
    {
        uvm_va_block_region_t subregion =
            uvm_perf_prefetch_bitmap_tree_iter_get_range(bitmap_tree, &iter);
        NvU16 subregion_pages = uvm_va_block_region_num_pages(subregion);

        // 🔑 关键: 阈值判断 (默认 51%)
        // counter = 子区域中已存在的页数
        // 如果 occupancy > threshold，选择这个子区域
        if (counter * 100 > subregion_pages * g_uvm_perf_prefetch_threshold)
            prefetch_region = subregion;
    }

    // 裁剪到实际可用范围
    return clamp(prefetch_region, max_prefetch_region);
}
```

**逻辑解释**:
1. **自底向上遍历**: 从 fault page 对应的叶子节点开始，向上遍历到根节点
2. **计算 occupancy**: 每层计算 `counter` (已存在页数) / `subregion_pages` (总页数)
3. **阈值判断**: 如果 occupancy > 51%，记录这个子区域
4. **选择最大子区域**: 因为从下往上遍历，最后记录的是**满足阈值的最大子区域**

**示例**:
```
假设 2MB block (512 pages), 发生 fault 的页在 index 128
                Root [256 pages exist / 512 total] → 50% ✗
                /                              \
           L1 [200/256] → 78% ✓          L1 [56/256] → 21% ✗
          /           \
   L2 [150/128] ✓  L2 [50/128] ✗

向上遍历结果:
- 叶子层 (L3): 1/1 = 100% ✓ → prefetch_region = [128, 129)
- L2左子树: 150/128 > 51% ✓ → prefetch_region = [0, 128)
- L1左子树: 200/256 > 51% ✓ → prefetch_region = [0, 256)
- Root: 256/512 = 50% ✗ → 不更新

最终预取: [0, 256) 即左半个 2MB block
```

### 1.3 Thrashing 检测集成

#### 与 Prefetch 的交互点

**Point 1**: 排除 thrashing 页面 (Line 377-381)
```c
// 不计算 thrashing 页面的预取区域
if (thrashing_pages)
    uvm_page_mask_andnot(&scratch_page_mask, faulted_pages, thrashing_pages);
else
    uvm_page_mask_copy(&scratch_page_mask, faulted_pages);

compute_prefetch_mask(faulted_region, max_prefetch_region,
                      bitmap_tree, &scratch_page_mask, prefetch_pages);
```

**Point 2**: 去除已标记的 thrashing 页面 (Line 406-408)
```c
// 避免预取正在 thrashing 的页面
if (thrashing_pages)
    uvm_page_mask_andnot(prefetch_pages, prefetch_pages, thrashing_pages);
```

**Point 3**: 增大非 thrashing 区域的预取粒度 (Line 148-162)
```c
static void grow_fault_granularity_if_no_thrashing(
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t region,
    uvm_page_index_t first,
    const uvm_page_mask_t *faulted_pages,
    const uvm_page_mask_t *thrashing_pages)
{
    // 如果这个区域有 fault 且没有 thrashing
    if (!uvm_page_mask_region_empty(faulted_pages, region) &&
        (!thrashing_pages || uvm_page_mask_region_empty(thrashing_pages, region))) {
        // 标记整个区域的页面为存在，增加 occupancy
        uvm_page_mask_region_fill(&bitmap_tree->pages, region);
    }
}
```

### 1.4 特殊优化

#### First-touch 全填充 (Line 361-366)

```c
// 如果是首次访问且目标是 preferred location，直接填充整个区域
if (uvm_processor_mask_empty(&va_block->resident) &&
    uvm_id_equal(new_residency, policy->preferred_location)) {
    uvm_page_mask_region_fill(prefetch_pages, max_prefetch_region);
}
```

**场景**: 应用首次访问一个 managed memory 区域，且访问的是 preferred location (如 GPU)
**策略**: 直接预取整个 VA block (最多 2MB)，避免后续大量 page faults

#### Big Page 对齐 (Line 271-285)

```c
// 调整 bitmap tree 以适应 big page 边界
if (big_pages_region.first - max_prefetch_region.first > 0) {
    bitmap_tree->offset = big_page_size / PAGE_SIZE -
                          (big_pages_region.first - max_prefetch_region.first);
    bitmap_tree->leaf_count = uvm_va_block_region_num_pages(max_prefetch_region) +
                              bitmap_tree->offset;

    // 左移 page mask 以对齐
    uvm_page_mask_shift_left(&bitmap_tree->pages, &bitmap_tree->pages, bitmap_tree->offset);

    bitmap_tree->level_count = ilog2(roundup_pow_of_two(bitmap_tree->leaf_count)) + 1;
}
```

**目的**: 确保预取区域对齐到 big page (64KB/2MB) 边界，提高 TLB 效率

---

## 2. Eviction Policy 替换方案

### 2.1 当前 LRU 实现分析

#### 数据结构 (uvm_pmm_gpu.h:355)

```c
struct {
    struct list_head va_block_used;    // LRU 列表: 头部=最久未用，尾部=最近使用
    struct list_head va_block_unused;  // 未使用的 chunk 列表
    struct list_head va_block_lazy_free; // 延迟释放列表
} root_chunks;
```

#### 关键函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `pick_root_chunk_to_evict()` | uvm_pmm_gpu.c:1460 | 选择要驱逐的 chunk |
| `chunk_update_lists_locked()` | uvm_pmm_gpu.c:627 | 更新 LRU 位置 |
| `uvm_pmm_gpu_unpin_allocated()` | uvm_pmm_gpu.c:677 | 分配后调用，触发 LRU 更新 |

### 2.2 策略替换的三个层次

#### Level 1: 仅修改选择逻辑 (最小侵入)

**修改点**: `pick_root_chunk_to_evict()` 函数
**位置**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:1460-1500`

**原有逻辑**:
```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;
    uvm_spin_lock(&pmm->list_lock);

    // 优先级1: Free list
    chunk = list_first_chunk(find_free_list(pmm, ...));

    // 优先级2: Unused list
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // 优先级3: LRU (从头部取最久未用)
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

    if (chunk)
        chunk_start_eviction(pmm, chunk);

    uvm_spin_unlock(&pmm->list_lock);
    return chunk ? root_chunk_from_chunk(pmm, chunk) : NULL;
}
```

**FIFO 修改** (只改第3优先级):
```c
// 优先级3: FIFO (从头部取最早分配)
// LRU: list_first_chunk() 取头部 = 最久未访问
// FIFO: list_first_chunk() 取头部 = 最早分配
// → 数据结构不变，只需修改更新策略！
if (!chunk)
    chunk = list_first_chunk(&pmm->root_chunks.va_block_used);
```

**关键**: LRU 和 FIFO 在当前实现下**选择逻辑完全相同**，区别在于**何时更新链表位置**

#### Level 2: 修改更新策略 (中等侵入)

**修改点**: `chunk_update_lists_locked()` 函数
**位置**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:627-651`

**LRU 更新策略** (当前实现):
```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
            root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            // 每次分配后移到尾部 (Most Recently Used)
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
        }
    }
}
```

**FIFO 更新策略** (修改):
```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
            root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            // FIFO: 不移动位置！保持分配顺序
            // 只在首次分配时加到链表尾部
            if (list_empty(&root_chunk->chunk.list)) {
                list_add_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
            }
            // 否则不更新位置
        }
    }
}
```

#### Level 3: 添加新的数据结构 (高侵入)

**场景**: 实现需要额外元数据的策略（如 LFU, Clock, LIRS 等）

**方法**:
1. 在 `uvm_gpu_root_chunk_t` 中添加字段 (uvm_pmm_gpu.h)
2. 在 `uvm_pmm_gpu_t` 中添加辅助数据结构
3. 修改 `pick_root_chunk_to_evict()` 使用新数据结构

**示例: Clock 算法**

在 `uvm_pmm_gpu.h` 添加:
```c
struct uvm_gpu_root_chunk_struct {
    uvm_gpu_chunk_t chunk;
    uvm_tracker_t tracker;

    // 新增: Clock 算法的 reference bit
    bool referenced;
};

struct uvm_pmm_gpu_struct {
    // ...
    struct {
        struct list_head va_block_used;
        // 新增: Clock 指针
        struct list_head *clock_hand;
    } root_chunks;
};
```

在 `uvm_pmm_gpu.c` 实现:
```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict_clock(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;
    struct list_head *pos = pmm->root_chunks.clock_hand;

    if (!pos)
        pos = pmm->root_chunks.va_block_used.next;

    // Clock 扫描
    while (true) {
        if (pos == &pmm->root_chunks.va_block_used) {
            pos = pos->next;  // 跳过链表头
            continue;
        }

        chunk = list_entry(pos, uvm_gpu_chunk_t, list);
        uvm_gpu_root_chunk_t *root = root_chunk_from_chunk(pmm, chunk);

        if (root->referenced) {
            root->referenced = false;  // 清除 reference bit
            pos = pos->next;
        } else {
            pmm->root_chunks.clock_hand = pos->next;
            chunk_start_eviction(pmm, chunk);
            return root;
        }
    }
}
```

---

## 3. 实现 FIFO 示例

### 3.1 最小修改方案 (推荐)

**文件**: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c`

#### 步骤1: 添加条件编译开关

```c
// 在文件开头添加
#define UVM_EVICTION_POLICY_FIFO 1  // 0=LRU, 1=FIFO

#if UVM_EVICTION_POLICY_FIFO
#define UVM_EVICTION_POLICY_NAME "FIFO"
#else
#define UVM_EVICTION_POLICY_NAME "LRU"
#endif
```

#### 步骤2: 修改 chunk_update_lists_locked()

在 **Line 627** 附近:

```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (uvm_gpu_chunk_is_user(chunk)) {
        if (chunk_is_root_chunk_pinned(pmm, chunk)) {
            UVM_ASSERT(root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
                       root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
            list_del_init(&root_chunk->chunk.list);
        }
        else if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            UVM_ASSERT(root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
                       root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);

#if UVM_EVICTION_POLICY_FIFO
            // FIFO: 只在首次加入时添加到链表，之后不移动
            if (list_empty(&root_chunk->chunk.list)) {
                list_add_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
            }
            // 否则保持原位置不变
#else
            // LRU: 每次访问都移到尾部
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
#endif
        }
    }

    // TODO: Bug 1757148: Improve fragmentation of split chunks
    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_FREE)
        list_move_tail(&chunk->list, find_free_list_chunk(pmm, chunk));
    else if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED)
        list_del_init(&chunk->list);
}
```

#### 步骤3: 添加日志 (可选)

在 PMM 初始化函数中添加:

```c
// 在 uvm_pmm_gpu_init() 函数中 (Line ~3400)
NV_STATUS uvm_pmm_gpu_init(uvm_pmm_gpu_t *pmm, uvm_gpu_t *gpu)
{
    // ... 原有代码 ...

    UVM_INFO_PRINT("PMM GPU initialized with %s eviction policy\n",
                   UVM_EVICTION_POLICY_NAME);

    return NV_OK;
}
```

### 3.2 验证和测试

#### 编译

```bash
cd /home/yunwei37/open-gpu-kernel-modules
make modules
```

#### 加载模块

```bash
sudo rmmod nvidia_uvm
sudo insmod kernel-open/nvidia-uvm/nvidia-uvm.ko
dmesg | tail -20  # 查看是否有 "FIFO eviction policy" 日志
```

#### 测试程序

```c
// test_eviction.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    size_t size = 8ULL * 1024 * 1024 * 1024;  // 8GB (超过 GPU 内存)
    char *data;

    cudaMallocManaged(&data, size);

    // 顺序访问，观察驱逐顺序
    for (size_t i = 0; i < size; i += 4096) {
        data[i] = i % 256;
    }

    cudaDeviceSynchronize();
    cudaFree(data);
    return 0;
}
```

使用 `nvidia-smi` 或 UVM events 监控驱逐行为:
- **FIFO**: 驱逐顺序与分配顺序一致
- **LRU**: 驱逐顺序与访问顺序相关

---

## 4. 最小侵入修改点总结

### 4.1 Prefetch Policy Hook Points

| Hook Point | 文件 | 行号 | 功能 | 侵入性 |
|-----------|------|------|------|-------|
| **compute_prefetch_region()** | uvm_perf_prefetch.c | 102 | 预取区域计算核心 | 🟢 低 |
| **g_uvm_perf_prefetch_threshold** | uvm_perf_prefetch.c | 64 | 阈值参数 | 🟢 低 |
| **uvm_perf_prefetch_get_hint_va_block()** | uvm_perf_prefetch.c | 447 | 顶层预取接口 | 🟡 中 |
| **init_bitmap_tree_from_region()** | uvm_perf_prefetch.c | 222 | 树初始化 | 🟡 中 |

### 4.2 Eviction Policy Hook Points

| Hook Point | 文件 | 行号 | 功能 | 侵入性 |
|-----------|------|------|------|-------|
| **pick_root_chunk_to_evict()** | uvm_pmm_gpu.c | 1460 | 选择驱逐目标 | 🟢 低 |
| **chunk_update_lists_locked()** | uvm_pmm_gpu.c | 627 | 更新 LRU/FIFO 列表 | 🟢 低 |
| **uvm_pmm_gpu_unpin_allocated()** | uvm_pmm_gpu.c | 677 | 分配后回调 | 🟡 中 |
| **root_chunks 数据结构** | uvm_pmm_gpu.h | 355 | 添加新元数据 | 🔴 高 |

### 4.3 推荐修改方案

#### 场景1: 替换预取算法 (如改用固定窗口预取)

**修改点**: `compute_prefetch_region()` (uvm_perf_prefetch.c:102)

```c
static uvm_va_block_region_t compute_prefetch_region_fixed_window(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    // 简单的固定窗口: fault page ± 32 pages
    #define PREFETCH_WINDOW 32

    uvm_page_index_t start = (page_index > PREFETCH_WINDOW) ?
                             (page_index - PREFETCH_WINDOW) : 0;
    uvm_page_index_t end = min(page_index + PREFETCH_WINDOW,
                               max_prefetch_region.outer);

    return uvm_va_block_region(start, end);
}
```

**侵入性**: 🟢 低 (只改一个函数)

#### 场景2: 实现 FIFO 驱逐

**修改点**: `chunk_update_lists_locked()` (uvm_pmm_gpu.c:642)

```c
// 注释掉:
// list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);

// 改为:
if (list_empty(&root_chunk->chunk.list)) {
    list_add_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
}
```

**侵入性**: 🟢 低 (修改 1 行代码)

#### 场景3: 实现访问频率驱逐 (LFU)

**修改点**:
1. 添加字段到 `uvm_gpu_root_chunk_t` (uvm_pmm_gpu.h)
2. 修改 `pick_root_chunk_to_evict()` (uvm_pmm_gpu.c:1460)
3. 在 `chunk_update_lists_locked()` 中更新计数

**侵入性**: 🔴 高 (需要修改数据结构和多个函数)

---

## 5. 调试和性能分析

### 5.1 添加 Tracepoint

在 `pick_root_chunk_to_evict()` 添加:

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;

    uvm_spin_lock(&pmm->list_lock);

    chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

    if (chunk) {
        // 添加 trace
        printk(KERN_INFO "UVM: Evicting chunk at PA 0x%llx\n",
               chunk->address);
        chunk_start_eviction(pmm, chunk);
    }

    uvm_spin_unlock(&pmm->list_lock);
    return chunk ? root_chunk_from_chunk(pmm, chunk) : NULL;
}
```

### 5.2 性能计数器

添加到 `uvm_pmm_gpu_t`:

```c
struct {
    atomic64_t eviction_count;
    atomic64_t prefetch_count;
    atomic64_t hit_count;
} stats;
```

在关键路径更新:

```c
// 驱逐时
atomic64_inc(&pmm->stats.eviction_count);

// 预取时
atomic64_add(uvm_page_mask_weight(prefetch_pages), &pmm->stats.prefetch_count);

// 命中时 (页面已在 GPU)
atomic64_inc(&pmm->stats.hit_count);
```

通过 `/proc` 或 `/sys` 导出统计信息。

---

## 6. 参考资料

### 相关代码文件

| 文件 | 功能 |
|------|------|
| `uvm_perf_prefetch.c/h` | 预取策略实现 |
| `uvm_pmm_gpu.c/h` | GPU 物理内存管理和驱逐 |
| `uvm_va_block.c` | VA block 管理和迁移 |
| `uvm_perf_thrashing.c` | Thrashing 检测 |
| `uvm_gpu_replayable_faults.c` | Page fault 处理 |

### 关键宏和辅助函数

```c
// 遍历 bitmap tree
#define uvm_perf_prefetch_bitmap_tree_traverse_counters(counter, tree, page, iter)

// 链表操作
list_first_entry()     // 获取第一个元素
list_add_tail()        // 添加到尾部
list_move_tail()       // 移动到尾部
list_del_init()        // 删除并初始化

// Page mask 操作
uvm_page_mask_region_fill()    // 填充区域
uvm_page_mask_andnot()         // 与非操作
uvm_page_mask_weight()         // 计算 set bits 数量
```

---

**文档版本**: v1.0
**最后更新**: 2025-11-16
**适用代码**: kernel-open/nvidia-uvm (branch: uvm-print-test)
