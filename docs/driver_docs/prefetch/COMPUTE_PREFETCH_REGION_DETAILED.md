# `compute_prefetch_region` 函数完全解析

基于实际代码的详细分析。

---

## 一、核心数据结构

### 1. `uvm_va_block_region_t`

```c
// uvm_va_block_types.h:70-75
typedef struct {
    uvm_page_index_t first;  // 起始页索引（包含）
    uvm_page_index_t outer;  // 结束页索引（不包含）
} uvm_va_block_region_t;
```

**表示左闭右开区间 `[first, outer)`**

- 页数 = `outer - first`
- 空区域: `first == outer`
- 单页区域: `outer == first + 1`

**例子**:
```c
region = {.first = 10, .outer = 20}
// 表示页面 [10, 11, 12, ..., 19]
// 共 20-10=10 个页面
```

### 2. `uvm_perf_prefetch_bitmap_tree_t`

```c
// uvm_perf_prefetch.h:41-50
typedef struct {
    uvm_page_mask_t pages;       // Bitmap（最多 512 bits，对应 2MB/4KB 的页数）
    uvm_page_index_t offset;     // Big page 对齐偏移量
    NvU16 leaf_count;            // 叶子节点数（实际页数 + offset）
    NvU8 level_count;            // 树的层数
} uvm_perf_prefetch_bitmap_tree_t;
```

#### 字段详解

**`pages` (bitmap)**:
- 每个 bit 表示一个页面是否"存在"（已驻留或将被访问）
- Bit 0 对应 tree 内部页索引 0，Bit 1 对应索引 1，依此类推

**`offset`**:
- **作用**: 将 tree 对齐到 big page 边界
- **计算公式** (Line 276):
  ```c
  offset = big_page_size / PAGE_SIZE - (big_pages_region.first - max_prefetch_region.first)
  ```
- **含义**: 在 bitmap 前面填充 `offset` 个虚拟页面

**`leaf_count`**:
- = `max_prefetch_region 页数` + `offset`
- 用于构建完整二叉树

**`level_count`**:
- = `log2(roundup_pow_of_two(leaf_count)) + 1`
- 树的总层数

---

## 二、`offset` 计算详解

### 为什么需要 offset？

**目的**: 让预取区域对齐到 big page (64KB/2MB) 边界，提高 TLB 效率。

### 计算公式

```c
// Line 276
offset = big_page_size / PAGE_SIZE - (big_pages_region.first - max_prefetch_region.first)
```

其中：
- `big_page_size / PAGE_SIZE` = 一个 big page 的页数（例如 64KB / 4KB = 16）
- `big_pages_region.first` = max_prefetch_region 内第一个对齐的 big page 边界
- `max_prefetch_region.first` = 允许预取的起始页

### 图解

```
VA block 绝对坐标:    0 ... 10 11 12 13 14 15 16 17 18 ... 100
                           ↑                 ↑
                   max_prefetch_region.first  big_pages_region.first
                          (10)                      (16, 64KB 对齐)

big_page_size = 64KB = 16 pages
big_pages_region.first - max_prefetch_region.first = 16 - 10 = 6

offset = 16 - 6 = 10
```

### 为什么是 `16 - 6 = 10`？

因为我们要在 bitmap 前面**填充 10 个虚拟页面**，使得：
- `bitmap_tree->pages[10]` 对应 VA block 页面 10
- `bitmap_tree->pages[16]` 对应 VA block 页面 16（big page 边界）

**Bitmap tree 布局**:
```
Tree 内部索引:        [0 1 2 3 4 5 6 7 8 9] 10 11 12 13 14 15 16 17 ...
                      ↑←────offset=10────→↑                ↑
                      (虚拟 padding)      max_first=10    big page 边界

VA block 绝对坐标:   (不对应实际页)     10 11 12 13 14 15 16 17 ...
```

**验证**:
- Tree 索引 10 → VA block 页 10 ✅
- Tree 索引 16 → VA block 页 16 ✅（big page 边界）

### 实际例子

```
Scenario:
  max_prefetch_region = [10, 100)
  big_page_size = 64KB = 16 pages
  big_pages_region.first = 16 (第一个对齐边界)

计算:
  offset = 16 - (16 - 10) = 10

结果:
  bitmap_tree->pages[0-9]:   padding（不对应实际页面）
  bitmap_tree->pages[10]:    对应 VA block 页 10
  bitmap_tree->pages[16]:    对应 VA block 页 16（big page 边界）
  ...
  bitmap_tree->pages[99]:    对应 VA block 页 99
```

---

## 三、`compute_prefetch_region` 完整流程

### 输入输出坐标系

**输入**:
- `page_index`: VA block 绝对坐标（例如：50）
- `max_prefetch_region`: VA block 绝对坐标（例如：[10, 100)）

**输出**:
- 返回值: VA block 绝对坐标（例如：[40, 60)）

**内部处理**: Bitmap tree 内部坐标（包含 offset）

---

### Step 1: 坐标转换（Line 112）

```c
uvm_perf_prefetch_bitmap_tree_traverse_counters(
    counter,
    bitmap_tree,
    page_index - max_prefetch_region.first + bitmap_tree->offset,  // ← 坐标转换
    &iter
)
```

**转换公式**:
```c
tree_page_index = page_index - max_prefetch_region.first + bitmap_tree->offset
```

**逐步分解**:

1. **`page_index - max_prefetch_region.first`**:
   - 将 VA block 绝对坐标转为相对于 max_prefetch_region 的相对坐标
   - 例如: `page_index=50`, `max_first=10` → `50-10=40`

2. **`+ bitmap_tree->offset`**:
   - 加上 padding，得到 tree 内部坐标
   - 例如: `相对坐标=40`, `offset=10` → `40+10=50`

**完整示例**:
```
假设:
  page_index = 50 (VA block 绝对坐标)
  max_prefetch_region = [10, 100)
  offset = 10

计算:
  相对坐标 = 50 - 10 = 40
  tree 坐标 = 40 + 10 = 50

验证:
  bitmap_tree->pages[50] 对应 VA block 页 50 ✅
```

**坐标对应关系**:
```
VA block 绝对坐标:  0 ... 10 11 ... 50 51 ... 100
                         ↑        ↑
                    max_first  page_index

相对坐标（-10）:         0  1 ... 40 41 ...  90

Tree 内部坐标（+10）: [0...9] 10 11...50 51...100
                      ↑pad↑                ↑
                                    tree_page_index=50
```

---

### Step 2: 遍历树，应用阈值判断（Line 110-120）

```c
uvm_perf_prefetch_bitmap_tree_traverse_counters(counter, bitmap_tree, tree_page_index, &iter) {
    // 获取当前层的子区域（tree 内部坐标）
    uvm_va_block_region_t subregion = uvm_perf_prefetch_bitmap_tree_iter_get_range(bitmap_tree, &iter);

    // 子区域页数
    NvU16 subregion_pages = uvm_va_block_region_num_pages(subregion);  // = subregion.outer - subregion.first

    UVM_ASSERT(counter <= subregion_pages);

    // ⭐ 核心判断：已驻留页数比例 > 51%
    if (counter * 100 > subregion_pages * g_uvm_perf_prefetch_threshold) {
        prefetch_region = subregion;  // 贪心地选择更大的区域
    }
}
```

**遍历顺序**: 从叶子到根（从小到大）

**遍历宏展开** (Line 108-113):
```c
#define uvm_perf_prefetch_bitmap_tree_traverse_counters(counter, tree, page, iter) \
    for (uvm_perf_prefetch_bitmap_tree_iter_init((tree), (page), (iter)),          \
         (counter) = uvm_perf_prefetch_bitmap_tree_iter_get_count((tree), (iter)); \
         (iter)->level_idx >= 0;                                                   \
         (counter) = --(iter)->level_idx < 0 ? 0 :                                 \
                     uvm_perf_prefetch_bitmap_tree_iter_get_count((tree), (iter)))
```

**行为**:
- 初始化 `iter` 指向 `tree_page_index` 对应的叶子节点
- 每次迭代，`level_idx--`，向上移动一层
- `counter` = 当前子区域中已驻留的页数

**具体例子**:

```
假设:
  tree_page_index = 50
  bitmap_tree 状态:
              Root [256/512] 50%
             /               \
      L1 [256/256] 100%    L1 [0/256] 0%
      /          \
  L2 [128/128] L2 [128/128]
  ...
  Leaf[50] = 1 (该页已驻留)

遍历过程:
┌─────┬────────┬──────┬───────────────┬─────────┬─────────────┬────────────┐
│ 层  │ level  │counter│ subregion     │subregion│ 占比        │ 更新?      │
│     │ _idx   │       │(tree 内部坐标)│ _pages  │             │            │
├─────┼────────┼──────┼───────────────┼─────────┼─────────────┼────────────┤
│Leaf │   0    │  1   │ [50, 51)      │    1    │ 1/1=100% >51│ ✅ [50,51) │
│L2   │   1    │ 128  │ [10, 138)     │  128    │128/128=100% │ ✅ [10,138)│
│L1   │   2    │ 256  │ [10, 266)     │  256    │256/256=100% │ ✅ [10,266)│
│Root │   3    │ 256  │ [10, 522)     │  512    │256/512=50%  │ ❌ 保持    │
└─────┴────────┴──────┴───────────────┴─────────┴─────────────┴────────────┘

最终: prefetch_region = [10, 266) (tree 内部坐标)
```

---

### Step 3: 坐标反向转换 + 边界裁剪（Line 123-143）

```c
if (prefetch_region.outer) {  // 如果找到了有效区域
```

**为什么检查 `outer`？**
- 初始值 `prefetch_region = {0, 0}`
- 如果所有层都不满足阈值，`prefetch_region` 仍为 `{0, 0}`
- `outer == 0` 表示空区域，跳过转换直接返回

#### 3.1 转换 `prefetch_region.first`

```c
// Step 1: 加上 max_prefetch_region.first
prefetch_region.first += max_prefetch_region.first;
```

**目的**: 开始反向转换

```
tree 内部: 10
+ max_first (10) = 20
```

```c
// Step 2: 检查是否在 padding 区域
if (prefetch_region.first < bitmap_tree->offset) {
    prefetch_region.first = bitmap_tree->offset;
}
```

**边界情况处理**:
- 如果计算结果 < offset，说明在 padding 区域内
- 设为 offset（padding 的边界）

**实际很少发生**，因为算法通常不会返回 padding 区域。

```c
else {
    // Step 3: 减去 offset，得到 VA block 绝对坐标
    prefetch_region.first -= bitmap_tree->offset;
```

**关键转换**!

```
tree 内部: 10
+ max_first (10) = 20
- offset (10) = 10  ← VA block 绝对坐标
```

**验证推导**:
```
Tree 内部坐标 tree_index
对应的 VA block 绝对坐标 = ?

已知:
  bitmap_tree->pages[offset] 对应 max_prefetch_region.first
  bitmap_tree->pages[tree_index] 对应 VA_abs

推导:
  VA_abs = max_first + (tree_index - offset)
         = tree_index + max_first - offset

代码实现:
  VA_abs = tree_index + max_first  (Step 1)
  VA_abs = VA_abs - offset          (Step 3)

✅ 公式正确！
```

```c
    // Step 4: 裁剪到 max_prefetch_region 边界内
    if (prefetch_region.first < max_prefetch_region.first)
        prefetch_region.first = max_prefetch_region.first;
}
```

**为什么需要裁剪？**

因为 tree 扩展到了 max_prefetch_region 之外（padding 区域）。

**例子**:
```
max_prefetch_region = [10, 100)
offset = 10
→ tree 实际覆盖 [0, 100)（包含 padding）

如果算法返回 tree 内部坐标 5:
  5 + 10 - 10 = 5 < 10
  → 裁剪为 10
```

#### 3.2 转换 `prefetch_region.outer`（完全相同）

```c
prefetch_region.outer += max_prefetch_region.first;

if (prefetch_region.outer < bitmap_tree->offset) {
    prefetch_region.outer = bitmap_tree->offset;
}
else {
    prefetch_region.outer -= bitmap_tree->offset;

    // 裁剪到上界
    if (prefetch_region.outer > max_prefetch_region.outer)
        prefetch_region.outer = max_prefetch_region.outer;
}
```

**唯一区别**: 裁剪检查上界而不是下界

---

## 四、完整例子

### 例子 1: 密集访问，大范围预取

**设置**:
```
VA block: 512 个页面（0-511）
max_prefetch_region = [10, 100)
big_page_size = 64KB = 16 pages
big_pages_region = [16, 96)  (64KB 对齐的区域)

计算 offset:
  offset = 16 - (16 - 10) = 16 - 6 = 10

已驻留页面: 10-99（全部 90 个页面都在 GPU 上）
Fault page: 50
```

**Bitmap tree 状态**:
```
bitmap_tree->pages[10-99] = 1（已驻留）
bitmap_tree->pages[其他] = 0

Tree 结构:
              Root [90/108] 83%
             /               \
      L1 [90/90] 100%     L1 [0/18] 0%
      /          \
  L2 [45/45]  L2 [45/45]
  ...
```

**Step 1: 坐标转换**
```
tree_page_index = 50 - 10 + 10 = 50
```

**Step 2: 遍历**
```
Leaf: [50, 51), counter=1, 1/1=100% > 51% ✅
  → prefetch_region = [50, 51) (tree 坐标)

L2: [10, 55), counter=45, 45/45=100% > 51% ✅
  → prefetch_region = [10, 55)

L1: [10, 100), counter=90, 90/90=100% > 51% ✅
  → prefetch_region = [10, 100)

Root: [10, 118), counter=90, 90/108=83% > 51% ✅
  → prefetch_region = [10, 118)
```

**Step 3: 反向转换**
```
first: 10 + 10 - 10 = 10
outer: 118 + 10 - 10 = 118

裁剪:
  first: 10 >= 10 ✅
  outer: 118 > 100 → 裁剪为 100

最终: prefetch_region = [10, 100)
```

**结果**: 预取整个 max_prefetch_region（90 个页面）

---

### 例子 2: 稀疏访问，小范围预取

**设置**:
```
max_prefetch_region = [10, 100)
offset = 10
已驻留页面: 只有 48-52（共 5 个页面）
Fault page: 50
```

**Bitmap tree 状态**:
```
bitmap_tree->pages[58-62] = 1 (tree 坐标 48+10 到 52+10)
其他 = 0

Tree 结构:
              Root [5/108] 4.6%
             /               \
      L1 [5/90] 5.5%      L1 [0/18] 0%
      ...
```

**Step 2: 遍历**
```
Leaf: [60, 61), counter=1, 1/1=100% > 51% ✅
  → prefetch_region = [60, 61) (tree 坐标)

更高层: counter=5, subregion_pages > 10
  5/10+ = <50% < 51% ❌
  → 不更新

最终: prefetch_region = [60, 61) (tree 坐标)
```

**Step 3: 反向转换**
```
first: 60 + 10 - 10 = 60
outer: 61 + 10 - 10 = 61

最终: prefetch_region = [60, 61)
```

**结果**: 只预取 1 个页面（fault page 本身）

---

### 例子 3: 边界裁剪

**设置**:
```
max_prefetch_region = [10, 50)
offset = 10
Tree 返回: [5, 60) (tree 坐标)
```

**反向转换**:
```
first: 5 + 10 - 10 = 5
outer: 60 + 10 - 10 = 60

裁剪:
  first: 5 < 10 → 裁剪为 10
  outer: 60 > 50 → 裁剪为 50

最终: prefetch_region = [10, 50)
```

---

## 五、总结

### 坐标转换公式

**VA block 绝对坐标 → Tree 内部坐标**:
```c
tree_index = VA_abs - max_first + offset
```

**Tree 内部坐标 → VA block 绝对坐标**:
```c
VA_abs = tree_index + max_first - offset
```

### 算法核心

1. **输入**: Fault page (VA block 绝对坐标)
2. **转换**: → Tree 内部坐标
3. **遍历**: 从叶子到根，贪心选择满足 51% 阈值的最大子区域
4. **反向转换**: Tree 内部坐标 → VA block 绝对坐标
5. **裁剪**: 确保在 max_prefetch_region 范围内

### offset 的作用

- 将 bitmap tree 对齐到 big page 边界
- 在 bitmap 前面添加 padding
- 确保 big page 边界的页面能被完整预取

### 关键特性

- ✅ 自适应: 密集访问 → 大范围预取，稀疏访问 → 小范围预取
- ✅ Big page 友好: 通过 offset 对齐
- ✅ 时间复杂度: O(log N)，N = 页数
- ❌ 局限性: 只考虑空间局部性，不考虑时序模式
