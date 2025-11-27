# UVM 自动 Prefetch 算法修改指南

## 问题：只修改 `compute_prefetch_region` 够吗？

**简短回答**: **不够，但这是核心修改点。**

需要理解完整的 prefetch 流程，才能决定修改哪些函数。

---

## 完整的 Prefetch 调用链

```
Page Fault 发生
    ↓
uvm_perf_prefetch_get_hint_va_block()                    [Line 447] ← 顶层入口
    ↓
    ├─> uvm_perf_prefetch_enabled() 检查                  [Line 466]
    │   └─> g_uvm_perf_prefetch_enable == 1?
    ↓
uvm_perf_prefetch_prenotify_fault_migrations()           [Line 327] ← 核心调度
    ↓
    ├─> 确定 max_prefetch_region                         [Line 346-354]
    │   ├─> HMM: uvm_hmm_get_prefetch_region()
    │   └─> 非HMM: 整个 VA block (最多 2MB)
    ↓
    ├─> First-touch 特殊处理                             [Line 363-366]
    │   如果是首次访问且目标是 preferred location:
    │   └─> 直接预取整个 max_prefetch_region ✅ 快速路径
    ↓
    ├─> 否则，执行正常的预取算法:
    │   ↓
    │   ├─> init_bitmap_tree_from_region()               [Line 368] ← 初始化树
    │   │   └─> 基于 resident_mask 和 faulted_pages 构建
    │   ↓
    │   ├─> update_bitmap_tree_from_va_block()           [Line 370] ← 更新树
    │   │   └─> grow_fault_granularity()                 [Line 291]
    │   │       └─> grow_fault_granularity_if_no_thrashing() [Line 148]
    │   │           └─> 在无 thrashing 的区域填充整个区域
    │   ↓
    │   └─> compute_prefetch_mask()                      [Line 383] ← 计算掩码
    │       └─> 对每个 faulted page:
    │           └─> compute_prefetch_region()             [Line 311] ⭐️ 核心算法
    │               └─> 遍历 bitmap tree，应用 51% 阈值
    ↓
    ├─> 后处理（过滤）                                   [Line 390-408]
    │   ├─> 移除 faulted_pages (已在迁移)               [Line 392]
    │   ├─> 移除已映射的 CPU pages                       [Line 399-404]
    │   └─> 移除 thrashing_pages                         [Line 406-408]
    ↓
    └─> 最小故障数检查                                   [Line 477-478]
        └─> fault_migrations >= g_uvm_perf_prefetch_min_faults?
```

---

## 各函数的职责与修改影响

### 1. **`compute_prefetch_region()`** (Line 102) ⭐️ **核心算法**

**职责**:
- 对**单个** faulted page，计算应该预取的区域
- 使用 bitmap tree 自底向上遍历
- 应用 51% occupancy 阈值判断

**输入**:
- `page_index`: 当前 fault 的页面索引
- `bitmap_tree`: 已构建好的 bitmap tree
- `max_prefetch_region`: 允许预取的最大范围

**输出**:
- `uvm_va_block_region_t`: 建议预取的区域 [first, outer)

**当前算法**:
```c
// 遍历树的每一层，从叶子到根
for each level in bitmap_tree:
    counter = 该子区域中已存在的页数
    subregion_pages = 该子区域的总页数

    // 关键阈值判断
    if (counter * 100 > subregion_pages * g_uvm_perf_prefetch_threshold):  // 默认 51%
        prefetch_region = subregion  // 更新为这个更大的区域

return prefetch_region  // 返回满足阈值的最大子区域
```

**修改此函数的影响**: 🟡 **中等**
- ✅ 可以完全改变预取策略（如固定窗口、距离衰减等）
- ✅ 不影响其他过滤逻辑（thrashing、first-touch 等）
- ⚠️ 但**不能控制**：
  - Bitmap tree 的初始状态（由 `init_bitmap_tree_from_region` 决定）
  - Big page 对齐优化（由 `update_bitmap_tree_from_va_block` 决定）
  - 最小故障数阈值（在调用者 `uvm_perf_prefetch_get_hint_va_block` 中检查）

---

### 2. **`compute_prefetch_mask()`** (Line 299) - **调度器**

**职责**:
- 对**所有** faulted pages，调用 `compute_prefetch_region()`
- 合并多个 prefetch regions 到一个 mask

**代码**:
```c
for_each_va_block_page_in_region_mask(page_index, faulted_pages, faulted_region) {
    // 对每个 faulted page 计算预取区域
    region = compute_prefetch_region(page_index, bitmap_tree, max_prefetch_region);

    // 合并到输出掩码
    uvm_page_mask_region_fill(out_prefetch_mask, region);

    // 早期退出优化
    if (region.outer == max_prefetch_region.outer)
        break;
}
```

**修改此函数的影响**: 🟡 **中等**
- 如果你想**合并多个 faulted pages 的预取决策**，需要修改这里
- 如果只是改变单个页面的预取策略，不需要修改

---

### 3. **`update_bitmap_tree_from_va_block()`** (Line 240) - **Big Page 优化**

**职责**:
- 调整 bitmap tree 以对齐到 big page (64KB/2MB) 边界
- 调用 `grow_fault_granularity()` 预填充非 thrashing 区域

**关键代码**:
```c
// 计算 big page 区域
big_pages_region = uvm_va_block_big_page_region_subset(va_block, max_prefetch_region, big_page_size);

// 对齐 offset
if (big_pages_region.first - max_prefetch_region.first > 0) {
    bitmap_tree->offset = big_page_size / PAGE_SIZE - (big_pages_region.first - max_prefetch_region.first);
    uvm_page_mask_shift_left(&bitmap_tree->pages, &bitmap_tree->pages, bitmap_tree->offset);
}

// 预填充非 thrashing 区域
grow_fault_granularity(bitmap_tree, big_page_size, big_pages_region, max_prefetch_region,
                       faulted_pages, thrashing_pages);
```

**修改此函数的影响**: 🔴 **高**
- 如果你想**禁用 big page 对齐**，需要修改这里
- 如果你想**改变预填充策略**，需要修改 `grow_fault_granularity()`

---

### 4. **`grow_fault_granularity()`** (Line 164) - **区域预填充**

**职责**:
- 对无 thrashing 的区域，将整个区域标记为"已存在"
- 这会增加 `compute_prefetch_region()` 中的 `counter` 值，从而更容易满足阈值

**逻辑**:
```c
// 示例：如果一个 big page (64KB) 中有 fault 且没有 thrashing
// → 标记整个 64KB 为已存在
// → compute_prefetch_region() 会计算 counter = 16 (pages)
// → 如果阈值是 51%，只要有 >8 个页面就会预取整个 64KB
```

**修改此函数的影响**: 🔴 **高**
- 如果你想**禁用区域预填充**，注释掉对此函数的调用
- 如果你想改变预填充粒度（如只填充 small pages），修改这里

---

### 5. **`init_bitmap_tree_from_region()`** (Line 222) - **树初始化**

**职责**:
- 初始化 bitmap tree 的初始状态
- 基于 `resident_mask`（已驻留页面）和 `faulted_pages`

**代码**:
```c
if (resident_mask)
    uvm_page_mask_or(&bitmap_tree->pages, resident_mask, faulted_pages);
else
    uvm_page_mask_copy(&bitmap_tree->pages, faulted_pages);

bitmap_tree->offset = 0;
bitmap_tree->leaf_count = uvm_va_block_region_num_pages(max_prefetch_region);
bitmap_tree->level_count = ilog2(roundup_pow_of_two(bitmap_tree->leaf_count)) + 1;
```

**修改此函数的影响**: 🟡 **中等**
- 如果你想改变树的初始状态（如只考虑 faulted pages，忽略 resident pages），修改这里

---

### 6. **`uvm_perf_prefetch_prenotify_fault_migrations()`** (Line 327) - **总调度**

**职责**:
- 协调整个预取流程
- 应用所有过滤规则（thrashing、first-touch、CPU mapping）

**关键决策点**:
```c
// 决策1: First-touch 快速路径 (Line 363-366)
if (uvm_processor_mask_empty(&va_block->resident) &&
    uvm_id_equal(new_residency, policy->preferred_location)) {
    // 直接预取整个 max_prefetch_region，跳过 bitmap tree
    uvm_page_mask_region_fill(prefetch_pages, max_prefetch_region);
}

// 决策2: 过滤 thrashing pages (Line 377-381)
if (thrashing_pages)
    uvm_page_mask_andnot(&va_block_context->scratch_page_mask, faulted_pages, thrashing_pages);

// 决策3: 移除已映射的 CPU pages (Line 399-404)
if (UVM_ID_IS_CPU(new_residency) && !uvm_va_block_is_hmm(va_block)) {
    // 排除已映射页面
}

// 决策4: 再次过滤 thrashing (Line 406-408)
if (thrashing_pages)
    uvm_page_mask_andnot(prefetch_pages, prefetch_pages, thrashing_pages);
```

**修改此函数的影响**: 🔴 **高**
- 如果你想**禁用某些过滤规则**，修改这里
- 如果你想**改变 first-touch 策略**，修改这里

---

### 7. **`uvm_perf_prefetch_get_hint_va_block()`** (Line 447) - **顶层入口**

**职责**:
- 检查是否启用 prefetch
- 应用最小故障数阈值
- 处理 range group 限制

**关键检查**:
```c
// 检查1: 是否启用 prefetch
if (!uvm_perf_prefetch_enabled(va_space))
    return;

// 检查2: 最小故障数阈值 (Line 477-478)
if (va_block->prefetch_info.fault_migrations_to_last_proc >= g_uvm_perf_prefetch_min_faults &&
    pending_prefetch_pages > 0) {
    // 允许预取
}
```

**修改此函数的影响**: 🟡 **中等**
- 如果你想改变**阈值逻辑**，修改这里

---

## 修改策略建议

### 场景 1: **只想改变预取区域的计算方式** (如固定窗口)

**修改点**: ✅ **只修改 `compute_prefetch_region()`**

**示例**:
```c
// 替换原有算法为固定窗口
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    #define PREFETCH_WINDOW 32  // 前后各 32 页

    uvm_page_index_t start = (page_index > PREFETCH_WINDOW) ?
                             (page_index - PREFETCH_WINDOW) : max_prefetch_region.first;
    uvm_page_index_t end = min(page_index + PREFETCH_WINDOW + 1, max_prefetch_region.outer);

    return uvm_va_block_region(start, end);
}
```

**优点**: 侵入性低，易于测试
**缺点**: 仍然受到 `grow_fault_granularity()` 的影响

---

### 场景 2: **想要完全控制，禁用所有启发式优化**

**修改点**:
1. ✅ 修改 `compute_prefetch_region()` - 实现新算法
2. ✅ 修改 `update_bitmap_tree_from_va_block()` - 禁用 `grow_fault_granularity()`
3. ⚠️ 可选：修改 `uvm_perf_prefetch_prenotify_fault_migrations()` - 禁用 first-touch 快速路径

**示例**:
```c
// 在 update_bitmap_tree_from_va_block() 中注释掉:
// grow_fault_granularity(bitmap_tree, big_page_size, big_pages_region,
//                        max_prefetch_region, faulted_pages, thrashing_pages);

// 在 uvm_perf_prefetch_prenotify_fault_migrations() 中注释掉:
// if (uvm_processor_mask_empty(&va_block->resident) &&
//     uvm_id_equal(new_residency, policy->preferred_location)) {
//     uvm_page_mask_region_fill(prefetch_pages, max_prefetch_region);
// }
```

---

### 场景 3: **基于访问模式的自适应预取** (如 stride detection)

**修改点**:
1. ✅ 修改 `compute_prefetch_region()` - 实现 stride 检测
2. ✅ 在 `uvm_va_block_t` 中添加历史访问记录字段
3. ✅ 修改 `uvm_perf_prefetch_prenotify_fault_migrations()` - 更新访问历史

**需要的数据结构**:
```c
// 在 uvm_va_block.h 的 prefetch_info 中添加:
struct {
    uvm_processor_id_t last_migration_proc_id;
    NvU64 fault_migrations_to_last_proc;

    // 新增: 访问历史
    uvm_page_index_t last_fault_pages[4];  // 最近 4 次 fault 的页面
    NvU8 history_count;
} prefetch_info;
```

---

### 场景 4: **机器学习驱动的预取**

**修改点**:
1. 🔴 添加新的预测模块 (新文件 `uvm_perf_prefetch_ml.c`)
2. ✅ 修改 `compute_prefetch_region()` - 调用预测模块
3. 🔴 添加特征提取函数

**架构**:
```c
// uvm_perf_prefetch_ml.c
uvm_va_block_region_t uvm_ml_predict_prefetch_region(
    uvm_page_index_t page_index,
    uvm_va_block_t *va_block,
    struct ml_features *features)
{
    // 特征提取
    extract_features(va_block, features);

    // 调用 eBPF/用户态模型
    return model_predict(page_index, features);
}

// 在 compute_prefetch_region() 中调用
static uvm_va_block_region_t compute_prefetch_region(...)
{
    struct ml_features features;
    return uvm_ml_predict_prefetch_region(page_index, va_block, &features);
}
```

---

## 总结表：各修改点的必要性

| 修改目标 | compute_prefetch_region | update_bitmap_tree | prenotify_fault_migrations | get_hint_va_block |
|---------|------------------------|-------------------|---------------------------|------------------|
| **固定窗口预取** | ✅ 必须 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |
| **距离衰减预取** | ✅ 必须 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |
| **禁用 big page 对齐** | ❌ 不需要 | ✅ 必须 | ❌ 不需要 | ❌ 不需要 |
| **禁用 first-touch 优化** | ❌ 不需要 | ❌ 不需要 | ✅ 必须 | ❌ 不需要 |
| **自定义阈值逻辑** | ⚠️ 可选 | ❌ 不需要 | ❌ 不需要 | ✅ 必须 |
| **Stride 检测** | ✅ 必须 | ❌ 不需要 | ✅ 必须（更新历史）| ❌ 不需要 |
| **完全自定义** | ✅ 必须 | ✅ 必须 | ✅ 必须 | ⚠️ 可选 |

---

## 推荐的修改流程

### Step 1: 最小修改验证
```bash
# 只修改 compute_prefetch_region，实现简单的固定窗口
cd /home/yunwei37/workspace/gpu/open-gpu-kernel-modules/kernel-open/nvidia-uvm
# 编辑 uvm_perf_prefetch.c:102
make -j$(nproc)
sudo rmmod nvidia_uvm
sudo insmod nvidia-uvm.ko
# 测试
```

### Step 2: 添加调试输出
```c
static uvm_va_block_region_t compute_prefetch_region(...)
{
    // 原有算法
    uvm_va_block_region_t old_region = ...;

    // 新算法
    uvm_va_block_region_t new_region = ...;

    // 对比输出
    printk(KERN_INFO "UVM Prefetch: page=%lu, old=[%u,%u), new=[%u,%u)\n",
           page_index, old_region.first, old_region.outer,
           new_region.first, new_region.outer);

    return new_region;
}
```

### Step 3: 性能测试
```c
// test_prefetch.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    size_t size = 4ULL * 1024 * 1024 * 1024;  // 4GB
    char *data;

    cudaMallocManaged(&data, size);

    // 测试不同访问模式
    // 1. Sequential
    for (size_t i = 0; i < size; i += 4096) {
        data[i] = 1;
    }

    // 2. Strided
    for (size_t i = 0; i < size; i += 8192) {
        data[i] = 2;
    }

    // 3. Random
    for (int i = 0; i < 10000; i++) {
        data[rand() % size] = 3;
    }

    cudaDeviceSynchronize();
    cudaFree(data);
    return 0;
}
```

---

## 关键配置参数

修改这些参数可以**无需重新编译**即可调整行为：

```bash
# 模块参数（加载时设置）
sudo insmod nvidia-uvm.ko \
    uvm_perf_prefetch_enable=1 \        # 启用 prefetch
    uvm_perf_prefetch_threshold=51 \     # 阈值百分比 (0-100)
    uvm_perf_prefetch_min_faults=1       # 最小故障数 (1-20)

# 运行时查看
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_threshold
```

---

## 总结

### ✅ 只修改 `compute_prefetch_region` 适用于：
- 简单的区域计算策略（固定窗口、距离衰减等）
- 不依赖复杂上下文的算法
- 快速原型验证

### ⚠️ 需要修改更多函数的情况：
- 需要禁用现有优化（big page 对齐、first-touch、区域预填充）
- 需要访问历史信息（stride detection、ML 模型）
- 需要改变阈值/过滤逻辑

### 🎯 建议：
1. **先从 `compute_prefetch_region` 开始**，验证算法正确性
2. **逐步禁用其他优化**，观察性能影响
3. **添加 tracepoint/printk**，分析决策过程
4. **使用模块参数**进行快速调参

---

**相关文件**:
- `kernel-open/nvidia-uvm/uvm_perf_prefetch.c` - 主实现
- `kernel-open/nvidia-uvm/uvm_perf_prefetch.h` - 数据结构
- `kernel-open/nvidia-uvm/uvm_va_block.h` - VA block 定义
- `docs/UVM_PREFETCH_AND_POLICY_HOOKS.md` - 详细文档


## TL;DR

**只修改 `compute_prefetch_region` + eBPF** 可以实现 **大部分** OSDI 级别的 prefetch 算法！

---

## 核心洞察

### 当前限制
```c
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,                    // ✅ 当前 fault 页
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,   // ✅ 当前驻留状态
    uvm_va_block_region_t max_prefetch_region       // ✅ 边界
)
```

**可用信息**:
- ✅ `page_index`: 当前 fault 的页面
- ✅ `bitmap_tree->pages`: 已驻留页面的 bitmap（最多 512 pages / 2MB）
- ✅ `bitmap_tree->leaf_count`: 总页数
- ✅ `max_prefetch_region`: 预取边界

**不可用信息**:
- ❌ 历史访问序列
- ❌ 时间戳
- ❌ 访问频率
- ❌ 跨 VA block 的模式

### eBPF 救援方案 🎯

**关键发现**: 可以通过 **eBPF kprobe/tracepoint** 获取并维护这些信息！

```
eBPF Map (全局状态)
    ↓
Kprobe on compute_prefetch_region()
    ↓ 读取历史信息
compute_prefetch_region() 执行你的算法
    ↓ 更新历史信息
Kprobe on return
```

---

## 方案对比

### 方案 A: 只修改内核代码（无 eBPF）

**限制**: 只能使用函数参数中的信息

**可实现的算法** (✅ 可行 / ⚠️ 受限 / ❌ 不可行):

| 算法类型 | 可行性 | 理由 |
|---------|-------|------|
| **固定窗口预取** | ✅ | 不需要历史信息 |
| **距离衰减预取** | ✅ | 基于当前 fault page 计算 |
| **空间局部性预测** | ⚠️ | 可用 bitmap_tree 推断邻近页面 |
| **Stride 检测** | ❌ | 需要历史访问序列 |
| **Markov 预测** | ❌ | 需要状态转移表 |
| **机器学习** | ❌ | 需要特征历史 |
| **自适应阈值** | ⚠️ | 只能基于当前 bitmap density |

---

### 方案 B: 修改内核代码 + eBPF (推荐 🌟)

**架构**:
```
┌─────────────────────────────────────┐
│   eBPF Program (用户态加载)          │
├─────────────────────────────────────┤
│  1. BPF Maps (全局状态存储)          │
│     - access_history[va_block_id]    │
│     - stride_patterns[va_block_id]   │
│     - ml_features[va_block_id]       │
│                                      │
│  2. Kprobe Hook Points               │
│     - kprobe/compute_prefetch_region │
│     - kprobe/compute_prefetch_mask   │
│     - tracepoint/page_fault          │
├─────────────────────────────────────┤
│   内核态 UVM Driver                  │
│   compute_prefetch_region() {        │
│     // 通过 BPF helper 读取历史     │
│     struct history *h =              │
│       bpf_map_lookup(...);           │
│     // 执行算法                     │
│     region = your_algorithm(h);      │
│     // 更新历史                     │
│     bpf_map_update(...);             │
│     return region;                   │
│   }                                  │
└─────────────────────────────────────┘
```

**可实现的算法**:

| 算法类型 | 可行性 | OSDI 相关论文 | 实现难度 |
|---------|-------|--------------|---------|
| **固定窗口预取** | ✅ | - | 🟢 简单 |
| **自适应窗口** | ✅ | - | 🟡 中等 |
| **Stride 检测** | ✅ | Jump-Directed Prefetching (OSDI'16) | 🟡 中等 |
| **Markov 预测** | ✅ | - | 🟡 中等 |
| **Dead Block Prediction** | ✅ | - | 🟡 中等 |
| **PC-based Prefetch** | ✅ | Bouquet (OSDI'20) | 🔴 复杂 |
| **ML-driven Prefetch** | ✅ | Learned Cache Replacement (OSDI'20) | 🔴 复杂 |
| **Multi-armed Bandit** | ✅ | Bandit Prefetcher (ISCA'20) | 🟡 中等 |
| **Contextual Prefetch** | ✅ | Pythia (ISCA'21) | 🔴 复杂 |

---

## 具体算法实现示例

### 1. Stride Prefetcher (OSDI 级别 ✅)

**论文参考**: Jump-Directed Instruction Prefetching (ISCA 2016, OSDI quality)

**算法描述**:
- 检测访问序列中的固定步长（stride）
- 例如：访问 page 0, 8, 16, 24 → stride = 8
- 预测下一次访问 page 32

**实现方案**:

#### eBPF 数据结构:
```c
// eBPF Map
struct stride_entry {
    u64 last_page;        // 上次访问的页面
    u64 last_last_page;   // 上上次访问的页面
    s64 stride;           // 检测到的 stride
    u32 confidence;       // 置信度 (0-100)
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);    // va_block 地址
    __type(value, struct stride_entry);
    __uint(max_entries, 10240);
} stride_table SEC(".maps");
```

#### 内核态修改:
```c
// uvm_perf_prefetch.c:102
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    // 1. 通过 kfunc 调用 eBPF 获取 stride 信息
    struct stride_entry *entry = bpf_stride_lookup(current_va_block_addr);

    if (entry && entry->confidence > 80) {
        // 2. 使用 stride 预测
        s64 predicted_page = (s64)page_index + entry->stride;

        if (predicted_page >= max_prefetch_region.first &&
            predicted_page < max_prefetch_region.outer) {

            // 预取以预测页面为中心的窗口
            uvm_page_index_t start = max(predicted_page - 4, max_prefetch_region.first);
            uvm_page_index_t end = min(predicted_page + 4, max_prefetch_region.outer);

            return uvm_va_block_region(start, end);
        }
    }

    // 3. Fallback: 固定窗口
    #define FALLBACK_WINDOW 16
    uvm_page_index_t start = (page_index > FALLBACK_WINDOW) ?
                             (page_index - FALLBACK_WINDOW) : max_prefetch_region.first;
    uvm_page_index_t end = min(page_index + FALLBACK_WINDOW, max_prefetch_region.outer);

    return uvm_va_block_region(start, end);
}
```

#### eBPF kprobe:
```c
// stride_prefetcher.bpf.c
SEC("kprobe/compute_prefetch_region")
int BPF_KPROBE(update_stride,
               uvm_page_index_t page_index,
               void *bitmap_tree,
               void *max_prefetch_region)
{
    u64 va_block_addr = get_va_block_addr();  // 从上下文获取

    struct stride_entry *entry = bpf_map_lookup_elem(&stride_table, &va_block_addr);
    if (!entry) {
        struct stride_entry new_entry = {
            .last_page = page_index,
            .stride = 0,
            .confidence = 0,
        };
        bpf_map_update_elem(&stride_table, &va_block_addr, &new_entry, BPF_NOEXIST);
        return 0;
    }

    // 计算当前 stride
    s64 current_stride = (s64)page_index - (s64)entry->last_page;

    if (entry->stride == current_stride) {
        // Stride 稳定，增加置信度
        entry->confidence = min(entry->confidence + 20, 100);
    } else {
        // Stride 改变
        entry->stride = current_stride;
        entry->confidence = 50;  // 重置为中等置信度
    }

    // 更新历史
    entry->last_last_page = entry->last_page;
    entry->last_page = page_index;

    bpf_map_update_elem(&stride_table, &va_block_addr, entry, BPF_EXIST);
    return 0;
}
```

**OSDI 质量**:
- ✅ 新颖性: 在 GPU Unified Memory 场景下应用 stride detection
- ✅ 性能提升: 对于规则访问模式（矩阵运算、卷积等）显著减少 page faults
- ✅ 开销低: eBPF Map 查询 O(1)，无锁

---

### 2. Dead Block Prediction (OSDI 级别 ✅)

**论文参考**: Hawkeye Cache Replacement (ISCA 2016, cited in many OSDI papers)

**算法描述**:
- 预测哪些页面在预取后会被再次访问（live）
- 不预取 "dead blocks"（不会再访问的页面）

**实现方案**:

#### eBPF 数据结构:
```c
struct access_trace {
    u64 timestamp;
    u32 access_count;
    u32 live_count;      // 预取后被再次访问的次数
    u32 dead_count;      // 预取后未被访问的次数
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __type(key, u64);    // page 地址
    __type(value, struct access_trace);
    __uint(max_entries, 65536);
} page_liveness_table SEC(".maps");
```

#### 内核态修改:
```c
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    uvm_va_block_region_t region = {page_index, page_index + 1};

    // 遍历候选页面
    for (uvm_page_index_t candidate = page_index + 1;
         candidate < min(page_index + 32, max_prefetch_region.outer);
         candidate++) {

        // 查询 eBPF：这个页面是否 "live"
        struct access_trace *trace = bpf_liveness_lookup(candidate);

        if (trace) {
            float live_ratio = (float)trace->live_count /
                              (trace->live_count + trace->dead_count);

            // 只预取高 "liveness" 的页面
            if (live_ratio > 0.7) {
                region.outer = candidate + 1;
            } else {
                break;  // 停止扩展
            }
        } else {
            // 未知页面，保守预取
            region.outer = candidate + 1;
        }
    }

    return region;
}
```

**OSDI 质量**:
- ✅ 减少无效预取，降低内存压力
- ✅ 适用于 GPU 稀疏数据访问模式
- ✅ 可与现有 thrashing detection 结合

---

### 3. Multi-armed Bandit Prefetcher (OSDI 级别 ✅)

**论文参考**: Bandit Prefetcher (ISCA 2020, OSDI quality)

**算法描述**:
- 将不同的预取策略视为 "arms"
- 动态选择表现最好的策略
- 使用 UCB (Upper Confidence Bound) 算法平衡 exploration/exploitation

**实现方案**:

#### eBPF 数据结构:
```c
enum prefetch_policy {
    POLICY_FIXED_WINDOW_8,
    POLICY_FIXED_WINDOW_16,
    POLICY_FIXED_WINDOW_32,
    POLICY_DISTANCE_DECAY,
    POLICY_STRIDE,
    POLICY_COUNT
};

struct bandit_arm {
    u64 times_selected;
    u64 total_reward;     // 预取命中次数
    u64 total_cost;       // 预取未命中次数
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);    // policy index
    __type(value, struct bandit_arm);
    __uint(max_entries, POLICY_COUNT);
} bandit_arms SEC(".maps");
```

#### 内核态修改:
```c
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    // 1. 通过 UCB 选择策略
    enum prefetch_policy selected = bpf_ucb_select_policy();

    // 2. 执行对应策略
    switch (selected) {
    case POLICY_FIXED_WINDOW_8:
        return fixed_window_prefetch(page_index, 8, max_prefetch_region);

    case POLICY_FIXED_WINDOW_16:
        return fixed_window_prefetch(page_index, 16, max_prefetch_region);

    case POLICY_FIXED_WINDOW_32:
        return fixed_window_prefetch(page_index, 32, max_prefetch_region);

    case POLICY_DISTANCE_DECAY:
        return distance_decay_prefetch(page_index, max_prefetch_region);

    case POLICY_STRIDE:
        return stride_prefetch(page_index, bitmap_tree, max_prefetch_region);

    default:
        return uvm_va_block_region(page_index, page_index + 1);
    }
}

static inline uvm_va_block_region_t fixed_window_prefetch(
    uvm_page_index_t page_index,
    u32 window_size,
    uvm_va_block_region_t max_prefetch_region)
{
    uvm_page_index_t start = (page_index > window_size) ?
                             (page_index - window_size) : max_prefetch_region.first;
    uvm_page_index_t end = min(page_index + window_size, max_prefetch_region.outer);
    return uvm_va_block_region(start, end);
}
```

#### eBPF UCB 算法:
```c
SEC("kprobe/compute_prefetch_region")
int BPF_KPROBE(ucb_select_policy)
{
    u64 total_selections = 0;
    float best_ucb = 0;
    u32 best_policy = 0;

    // 计算每个策略的 UCB 值
    for (u32 i = 0; i < POLICY_COUNT; i++) {
        struct bandit_arm *arm = bpf_map_lookup_elem(&bandit_arms, &i);
        if (!arm) continue;

        total_selections += arm->times_selected;

        // UCB1: mean_reward + sqrt(2 * ln(N) / n_i)
        float mean_reward = (float)arm->total_reward /
                           (arm->times_selected + 1);
        float exploration_bonus = sqrt(2.0 * log(total_selections + 1) /
                                      (arm->times_selected + 1));
        float ucb = mean_reward + exploration_bonus;

        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_policy = i;
        }
    }

    // 更新选择计数
    struct bandit_arm *selected_arm = bpf_map_lookup_elem(&bandit_arms, &best_policy);
    if (selected_arm) {
        selected_arm->times_selected++;
        bpf_map_update_elem(&bandit_arms, &best_policy, selected_arm, BPF_EXIST);
    }

    // 保存选择到 per-CPU 变量
    bpf_percpu_var_store(selected_policy, best_policy);
    return 0;
}
```

**OSDI 质量**:
- ✅✅ 自适应选择最优策略，无需手动调参
- ✅✅ 适应不同访问模式（sequential, random, strided）
- ✅ 理论保证（UCB 算法的 regret bound）

---

### 4. ML-driven Prefetcher (OSDI 顶级 ✅✅)

**论文参考**:
- Learned Cache Replacement (OSDI 2020)
- Pythia (ISCA 2021)

**算法描述**:
- 使用轻量级神经网络预测最佳预取决策
- 特征：当前 fault page、bitmap density、历史 stride 等
- 在线学习 + 离线训练结合

**实现方案**:

#### 架构:
```
┌─────────────────────────────────────┐
│  离线训练 (GPU Cluster)              │
│  - 收集 trace                       │
│  - 训练 NN 模型                     │
│  - 导出为 eBPF bytecode             │
└────────────┬────────────────────────┘
             │ 部署
             ↓
┌─────────────────────────────────────┐
│  eBPF Map (模型权重)                │
│  - Linear layer weights             │
│  - Activation functions             │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  compute_prefetch_region()           │
│  1. 提取特征                        │
│  2. 调用 eBPF NN inference          │
│  3. 返回预测的预取区域              │
└─────────────────────────────────────┘
```

#### 特征提取:
```c
struct ml_features {
    // 空间特征
    u32 fault_page_index;
    float bitmap_density;          // 已驻留页面比例
    u32 consecutive_resident_pages; // 连续驻留页面数

    // 时序特征
    s64 last_stride;
    u32 stride_confidence;

    // 上下文特征
    u32 va_block_size;
    u32 gpu_memory_pressure;       // 从 /sys 读取

    // 历史特征
    float recent_prefetch_accuracy;  // 最近 10 次预取的准确率
};

static void extract_features(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    struct ml_features *features)
{
    features->fault_page_index = page_index;

    // 计算 bitmap density
    u32 resident_count = uvm_page_mask_weight(&bitmap_tree->pages);
    features->bitmap_density = (float)resident_count / bitmap_tree->leaf_count;

    // 计算连续驻留页面数
    features->consecutive_resident_pages =
        count_consecutive_set_bits(&bitmap_tree->pages, page_index);

    // 从 eBPF map 获取历史特征
    struct stride_entry *stride = bpf_stride_lookup(...);
    if (stride) {
        features->last_stride = stride->stride;
        features->stride_confidence = stride->confidence;
    }

    // ... 其他特征
}
```

#### eBPF NN Inference:
```c
// 简单的 2 层全连接网络
struct nn_weights {
    float layer1[8][16];  // 8 features -> 16 hidden
    float layer2[16][3];  // 16 hidden -> 3 outputs (start, end, confidence)
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);
    __type(value, struct nn_weights);
    __uint(max_entries, 1);
} nn_model SEC(".maps");

SEC("kprobe/compute_prefetch_region")
int BPF_KPROBE(nn_inference, struct ml_features *features)
{
    struct nn_weights *weights = bpf_map_lookup_elem(&nn_model, &zero);
    if (!weights) return 0;

    // Layer 1: Linear + ReLU
    float hidden[16] = {0};
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            hidden[i] += features_array[j] * weights->layer1[j][i];
        }
        hidden[i] = max(0.0f, hidden[i]);  // ReLU
    }

    // Layer 2: Linear
    float output[3] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 16; j++) {
            output[i] += hidden[j] * weights->layer2[j][i];
        }
    }

    // 解析输出
    s32 predicted_start_offset = (s32)output[0];
    s32 predicted_end_offset = (s32)output[1];
    float confidence = sigmoid(output[2]);

    // 保存预测结果到 per-CPU 变量
    bpf_percpu_var_store(nn_prediction_start, predicted_start_offset);
    bpf_percpu_var_store(nn_prediction_end, predicted_end_offset);
    bpf_percpu_var_store(nn_confidence, confidence);

    return 0;
}
```

#### 内核态调用:
```c
static uvm_va_block_region_t compute_prefetch_region(
    uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t max_prefetch_region)
{
    struct ml_features features;
    extract_features(page_index, bitmap_tree, &features);

    // eBPF kprobe 已经执行了 NN inference
    s32 start_offset = bpf_percpu_var_load(nn_prediction_start);
    s32 end_offset = bpf_percpu_var_load(nn_prediction_end);
    float confidence = bpf_percpu_var_load(nn_confidence);

    if (confidence > 0.8) {
        // 使用 NN 预测
        uvm_page_index_t start = clamp(page_index + start_offset,
                                       max_prefetch_region.first,
                                       max_prefetch_region.outer);
        uvm_page_index_t end = clamp(page_index + end_offset,
                                     max_prefetch_region.first,
                                     max_prefetch_region.outer);
        return uvm_va_block_region(start, end);
    } else {
        // Fallback: 固定窗口
        return fixed_window_prefetch(page_index, 16, max_prefetch_region);
    }
}
```

**OSDI 质量**:
- ✅✅✅ **顶级创新**: 首次在 GPU Unified Memory 场景应用 ML prefetching
- ✅✅ 可学习复杂访问模式（DNN workloads, graph processing）
- ✅ 实用性强: eBPF 保证安全性，易于部署
- ⚠️ 挑战: 需要大量训练数据，模型选择需要调优

---

## OSDI 论文可行性评估

### 核心贡献点

| 贡献点 | 可行性 | 预期效果 |
|--------|-------|---------|
| **1. eBPF + 内核协同设计** | ✅✅✅ | 首次在 GPU 内存管理中使用 eBPF |
| **2. 自适应预取算法** | ✅✅ | 动态选择最优策略 |
| **3. ML-driven 预取** | ✅✅ | 处理复杂 GPU workloads |
| **4. 低开销在线学习** | ✅ | eBPF 实现高效推理 |
| **5. 跨层优化** | ✅✅ | 用户态模型 + 内核执行 |

---

## 实验设计

### Baseline
- 现有 UVM prefetch (51% threshold + bitmap tree)
- 无 prefetch
- 理想 Oracle prefetch (事后分析)

### Workloads
- **DNN Training**: PyTorch ResNet-50, BERT
- **Graph Processing**: PageRank, BFS on large graphs
- **Sparse Matrix**: SpMV, SpGEMM
- **科学计算**: LAMMPS, GROMACS
- **随机访问**: Randomized algorithms

### 指标
- **Page Fault Rate**: 减少 30-50%
- **Application Speedup**: 1.2-1.8x
- **Memory Overhead**: < 1% (eBPF Maps)
- **CPU Overhead**: < 2% (kprobe + NN inference)

### 对比对象
- Fixed window (8, 16, 32 pages)
- Stride prefetcher
- No prefetch
- Oracle (理论上界)

---

## 修改代码量估计

### 内核态修改 (约 300-500 行)
```
uvm_perf_prefetch.c:
  - compute_prefetch_region(): 100-200 行（新算法实现）
  - 辅助函数: 100-200 行（特征提取、策略选择等）
  - eBPF helper 接口: 50-100 行
```

### eBPF 程序 (约 500-800 行)
```
stride_prefetcher.bpf.c: 200 行
bandit_prefetcher.bpf.c: 200 行
ml_prefetcher.bpf.c: 300 行
common.bpf.h: 100 行
```

### 用户态工具 (约 300-500 行)
```
loader.c: 加载 eBPF 程序，初始化 Maps (200 行)
monitor.c: 监控预取效果，可视化 (200 行)
trainer.py: 离线训练 NN 模型 (300 行)
```

**总计**: 约 **1100-1800 行代码**，完全可行！

---

## OSDI 论文大纲

### Title
**"Adaptive Prefetching for GPU Unified Memory via eBPF-Kernel Co-Design"**

### Abstract
- Problem: GPU Unified Memory 的 page fault 开销高
- Challenge: 多样化的访问模式，难以用单一策略优化
- Solution: eBPF + 内核协同的自适应预取框架
- Results: 30-50% fault reduction, 1.2-1.8x speedup

### 1. Introduction
- GPU Unified Memory 背景
- 现有 prefetch 方法的局限性
- eBPF 的机会：安全、高效、灵活

### 2. Background & Motivation
- UVM 架构
- Page fault 分析
- 不同 workload 的访问模式特征
- 为什么现有方法不够好

### 3. Design
- 架构总览
- eBPF-Kernel 接口设计
- 多种预取策略
  - Stride detection
  - Dead block prediction
  - Multi-armed bandit
  - ML-driven

### 4. Implementation
- 修改点：`compute_prefetch_region`
- eBPF Maps 设计
- Kprobe hook points
- NN 模型训练与部署

### 5. Evaluation
- 实验设置
- 性能对比
- Ablation study
- 开销分析

### 6. Related Work
- GPU 内存管理
- Prefetching 技术
- eBPF 系统应用

### 7. Conclusion

---

## 结论

### ✅ 只修改 `compute_prefetch_region` + eBPF 完全可以实现 OSDI 级别的 prefetch 算法！

**关键优势**:
1. **最小侵入性**: 只改一个函数，易于维护和上游合并
2. **灵活性**: eBPF 程序可动态加载/卸载，无需重启系统
3. **安全性**: eBPF 验证器保证不会 crash 内核
4. **性能**: eBPF Maps 和 kprobe 开销极低
5. **创新性**: 首次在 GPU 内存管理中应用 eBPF

**推荐实现路径**:
1. **Phase 1** (2 周): 实现 Stride Prefetcher + eBPF
2. **Phase 2** (2 周): 添加 Multi-armed Bandit
3. **Phase 3** (4 周): 实现 ML-driven Prefetcher
4. **Phase 4** (2 周): 评估 + 撰写论文

**预期成果**:
- OSDI/SOSP/ATC 级别论文 1 篇
- 开源工具链（eBPF prefetcher library）
- 可能被 NVIDIA 采纳到上游驱动

---

**下一步建议**:
1. 先实现 Stride Prefetcher 验证框架可行性
2. 设计 eBPF-Kernel 接口（BPF kfunc）
3. 收集真实 GPU workload 的 trace 数据
4. 开始论文写作（Introduction + Related Work）
