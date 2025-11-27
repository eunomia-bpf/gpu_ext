# UVM BPF Struct_Ops 实现与 CacheBPF 能力对比分析

## 执行摘要

基于我们的追踪数据和代码分析,当前的 UVM BPF struct_ops 实现**具备基础功能**,但与 CacheBPF 论文中的完整能力相比,存在**关键功能缺失**,限制了高级驱逐策略的实现。

### 核心发现

| 能力 | CacheBPF 标准 | UVM 当前实现 | 状态 | 影响 |
|------|--------------|-------------|------|------|
| **链表遍历** | ✅ 完整 API | ❌ 无 API | **严重缺失** | 无法实现 LFU/LRU-K |
| **Chunk 属性读取** | ✅ 完整 | ❌ 无 API | **严重缺失** | 无法基于历史做决策 |
| **链表重排序** | ✅ 支持 | ⚠️ 部分支持 | **不足** | 只能头/尾移动 |
| **BPF Map 支持** | ✅ 支持 | ✅ 支持 | **完整** | 可存储元数据 |
| **Hook 覆盖** | ✅ 完整 | ⚠️ Depopulate失效 | **有缺陷** | 无法追踪完整生命周期 |

---

## 1. 当前实现概览

### 1.1 已有的 Hook 接口

从 `/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/lru_fifo.bpf.c`:

```c
SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)

SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)

SEC("struct_ops/uvm_pmm_chunk_depopulate")
int BPF_PROG(uvm_pmm_chunk_depopulate,  // ❌ 实际从不调用!
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
```

### 1.2 已有的 Kfunc

从 `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c:165-190`:

```c
__bpf_kfunc void bpf_uvm_pmm_chunk_move_head(
    uvm_gpu_chunk_t *chunk,
    struct list_head *list);

__bpf_kfunc void bpf_uvm_pmm_chunk_move_tail(
    uvm_gpu_chunk_t *chunk,
    struct list_head *list);
```

**仅此2个 kfunc!**

### 1.3 实际运行数据

从 `BPF_HOOKS_ACTUAL_BEHAVIOR_ANALYSIS.md`:

```
Hook                    Calls/sec    占比      状态
----------------------------------------------------
ACTIVATE                  6,679      7.9%      ✅ 工作
POPULATE                 70,667     84.1%      ✅ 工作
DEPOPULATE                    0      0.0%      ❌ 失效
EVICTION_PREPARE          6,678      7.9%      ✅ 工作
```

**关键**: Depopulate hook 100% 被 `chunk_is_in_eviction()` 条件拦截。

---

## 2. CacheBPF 标准能力需求

根据 `UVM_LIST_HELPERS.md` 第6节,参考 CacheBPF 论文需要的完整接口:

### 2.1 链表遍历 API (只读)

```c
// ❌ 缺失
__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_first(struct list_head *head);

// ❌ 缺失
__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_next(uvm_gpu_chunk_t *chunk, struct list_head *head);

// ❌ 缺失
__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_last(struct list_head *head);

// ❌ 缺失
__bpf_kfunc bool
bpf_uvm_list_empty(struct list_head *head);
```

**影响**: 无法遍历链表 → **无法实现 LFU, LRU-K, ARC** 等需要扫描的算法。

### 2.2 链表修改 API (精确位置插入)

```c
// ✅ 已有
__bpf_kfunc void
bpf_uvm_pmm_chunk_move_head(uvm_gpu_chunk_t *chunk, struct list_head *head);

// ✅ 已有
__bpf_kfunc void
bpf_uvm_pmm_chunk_move_tail(uvm_gpu_chunk_t *chunk, struct list_head *head);

// ❌ 缺失 - 关键!
__bpf_kfunc int
bpf_uvm_list_move_before(uvm_gpu_chunk_t *chunk, uvm_gpu_chunk_t *next_chunk);

// ❌ 缺失
__bpf_kfunc int
bpf_uvm_list_move_after(uvm_gpu_chunk_t *chunk, uvm_gpu_chunk_t *prev_chunk);
```

**影响**: 只能移动到头/尾 → **无法维护有序链表** (LFU分段, 频率排序)。

### 2.3 Chunk 属性访问 API

```c
// ❌ 全部缺失
__bpf_kfunc u64
bpf_uvm_chunk_get_address(uvm_gpu_chunk_t *chunk);

__bpf_kfunc u64
bpf_uvm_chunk_get_size(uvm_gpu_chunk_t *chunk);

__bpf_kfunc u32
bpf_uvm_chunk_get_state(uvm_gpu_chunk_t *chunk);

__bpf_kfunc u64
bpf_uvm_chunk_get_va_start(uvm_gpu_chunk_t *chunk);

__bpf_kfunc u64
bpf_uvm_chunk_get_va_end(uvm_gpu_chunk_t *chunk);
```

**影响**: 无法读取 chunk 信息 → **无法作为 BPF map key** → **无法存储访问历史/频率**。

---

## 3. 能实现 vs 不能实现的算法

### 3.1 ✅ 当前能实现的算法

#### FIFO (First-In-First-Out)

```c
// 当前实现: lru_fifo.bpf.c
SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate, ...) {
    bpf_uvm_pmm_chunk_move_head(chunk, list);  // ✅ 可用
    return 0;
}
```

**原理**: 新 chunk 放头部 → 头部最老 → 先进先出
**需求**: 只需 `move_head`
**状态**: ✅ **完全可行**

#### LRU (Least Recently Used)

```c
SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate, ...) {
    // 默认行为就是 LRU:
    // 内核已经移动到 tail (MRU 位置)
    return 0;  // 不做任何修改
}
```

**原理**: 新访问的放尾部 → 头部最久未访问
**需求**: 无需修改 (默认行为)
**状态**: ✅ **完全可行**

#### MRU (Most Recently Used)

```c
SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate, ...) {
    bpf_uvm_pmm_chunk_move_head(chunk, list);  // ✅ 可用
    return 0;
}
```

**原理**: 最近访问的反而先驱逐
**需求**: 移动到头部
**状态**: ✅ **完全可行**

### 3.2 ❌ 当前**不能**实现的算法

#### LFU (Least Frequently Used)

**需要**:
1. 遍历链表 (`list_first`, `list_next`) ❌ 缺失
2. 读取 chunk 地址作为 map key ❌ 缺失
3. 精确位置插入 (`move_before`) ❌ 缺失

**伪代码**:
```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);     // chunk address ❌ 无法获取
    __type(value, u32);   // access frequency
} freq_map SEC(".maps");

SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate, ...) {
    u64 addr = bpf_uvm_chunk_get_address(chunk);  // ❌ kfunc不存在
    u32 *freq = bpf_map_lookup_elem(&freq_map, &addr);
    if (freq) (*freq)++;
    else { u32 init = 1; bpf_map_update_elem(&freq_map, &addr, &init, BPF_ANY); }
    return 0;
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare, ...) {
    // 需要遍历链表,根据频率重新排序
    uvm_gpu_chunk_t *chunk = bpf_uvm_list_first(used_list);  // ❌ kfunc不存在
    for (int i = 0; i < 100 && chunk; i++) {
        u64 addr = bpf_uvm_chunk_get_address(chunk);  // ❌ kfunc不存在
        u32 *freq = bpf_map_lookup_elem(&freq_map, &addr);
        // 根据频率插入到对应位置
        // bpf_uvm_list_move_before(chunk, target);  // ❌ kfunc不存在
        chunk = bpf_uvm_list_next(chunk, used_list);  // ❌ kfunc不存在
    }
    return 0;
}
```

**状态**: ❌ **无法实现** - 缺失3个关键能力

#### LRU-K

**需要**:
1. 存储最近 K 次访问时间戳 ❌ 无 chunk 地址读取
2. 遍历链表计算 K-distance ❌ 无遍历 API
3. 按 K-distance 重新排序 ❌ 无精确插入

**状态**: ❌ **无法实现**

#### ARC (Adaptive Replacement Cache)

**需要**:
1. 维护多个链表 (T1, T2, B1, B2)
2. 读取 chunk 地址跟踪状态 ❌ 缺失
3. 动态调整链表大小 (需要遍历+计数) ❌ 缺失

**状态**: ❌ **无法实现**

#### 2Q (Two-Queue)

**需要**:
1. 维护 FIFO + LRU 两个队列
2. 遍历队列判断提升条件 ❌ 缺失
3. 跨队列移动 chunks (需要地址追踪) ❌ 缺失

**状态**: ❌ **无法实现**

---

## 4. 实际场景分析: Chunk 重用与算法限制

### 4.1 我们的关键发现

从 `CHUNK_VA_BLOCK_MAPPING_ANALYSIS.md`:

```
Chunk 0xffffcfd7df464c38 生命周期:

t=0ms:    分配给 VA Block 1 (虚拟地址 0x7870xxx...)
          6次 POPULATE

t=1417ms: 分配给 VA Block 2 (虚拟地址 0x7871xxx...)
          1次 ACTIVATE + 11次 POPULATE

t=3295ms: 分配给 VA Block 3 (虚拟地址 0x7872xxx...)
          1次 ACTIVATE + 12次 POPULATE
```

**问题**: **同一个物理 chunk 在不同时间被不同虚拟地址使用**

### 4.2 对 LFU 算法的致命影响

**场景**:
```c
// 时刻 T0: chunk 被 VA Block 1 使用
// BPF 在 freq_map 中记录: addr=0xffffcfd7df464c38, freq=6

// 时刻 T1: chunk 被 evict,然后分配给 VA Block 2
// ❌ 问题: 我们如何知道这还是"同一个"物理 chunk?
//    - chunk->va_block 指针已经改变
//    - 物理地址不变,但我们无法读取!
//    - BPF 无法维护跨 VA block 的访问历史
```

**根本问题**: 没有 `bpf_uvm_chunk_get_address()` → **无法追踪同一 chunk 的完整历史**。

### 4.3 Depopulate Hook 失效的影响

从 `BPF_HOOKS_ACTUAL_BEHAVIOR_ANALYSIS.md`:

```
DEPOPULATE hook: 0 calls (100% 被拦截)
原因: chunk_is_in_eviction() == true
```

**问题**: 我们**无法知道 chunk 何时从 VA block 解绑**。

**对算法的影响**:
- LFU: 无法清理旧的 `freq_map` 条目 → 内存泄漏
- LRU-K: 无法重置历史时间戳 → 过期数据影响决策
- ARC: 无法将 chunk 从 T1 移动到 B1

---

## 5. 缺失功能的优先级分级

### 5.1 P0 - 必须立即实现 (阻塞所有高级算法)

#### 1. Chunk 地址读取

```c
__bpf_kfunc u64
bpf_uvm_chunk_get_address(uvm_gpu_chunk_t *chunk);
```

**重要性**: ⭐⭐⭐⭐⭐ (5/5)
**难度**: ⭐ (1/5) - 简单读取结构体字段

**理由**:
- 所有基于历史的算法都需要这个作为 map key
- 没有这个,BPF 无法存储任何 per-chunk 状态
- **是实现 LFU/LRU-K/ARC 的先决条件**

**实现**:
```c
__bpf_kfunc u64
bpf_uvm_chunk_get_address(uvm_gpu_chunk_t *chunk)
{
    return chunk ? chunk->address : 0;
}
```

#### 2. 链表遍历 (First + Next)

```c
__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_first(struct list_head *head);

__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_next(uvm_gpu_chunk_t *chunk, struct list_head *head);
```

**重要性**: ⭐⭐⭐⭐⭐ (5/5)
**难度**: ⭐⭐ (2/5) - 需要边界检查

**理由**:
- `eviction_prepare` hook 需要遍历链表重排序
- 所有需要"全局视图"的算法都依赖这个

**实现** (参考 `UVM_LIST_HELPERS.md:641-655`):
```c
__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_first(struct list_head *head)
{
    if (!head || list_empty(head))
        return NULL;
    return list_first_entry(head, uvm_gpu_chunk_t, list);
}

__bpf_kfunc uvm_gpu_chunk_t *
bpf_uvm_list_next(uvm_gpu_chunk_t *chunk, struct list_head *head)
{
    if (!chunk || !head)
        return NULL;

    struct list_head *next = chunk->list.next;
    // 检查是否到达链表尾部 (head)
    if (next == head)
        return NULL;

    return list_entry(next, uvm_gpu_chunk_t, list);
}
```

### 5.2 P1 - 高优先级 (解锁 LFU 类算法)

#### 3. 精确位置插入

```c
__bpf_kfunc int
bpf_uvm_list_move_before(uvm_gpu_chunk_t *chunk, uvm_gpu_chunk_t *next_chunk);
```

**重要性**: ⭐⭐⭐⭐ (4/5)
**难度**: ⭐⭐ (2/5)

**理由**:
- LFU 需要按频率排序链表
- 没有这个只能用头/尾,无法分段

**实现** (参考 `UVM_LIST_HELPERS.md:736`):
```c
__bpf_kfunc int
bpf_uvm_list_move_before(uvm_gpu_chunk_t *chunk, uvm_gpu_chunk_t *next_chunk)
{
    if (!chunk || !next_chunk)
        return -EINVAL;

    // 基于 list_move(new, prev) = 插入到 prev->next 之前
    list_move_tail(&chunk->list, &next_chunk->list);
    return 0;
}
```

#### 4. VA Block 信息读取

```c
__bpf_kfunc u64
bpf_uvm_chunk_get_va_start(uvm_gpu_chunk_t *chunk);

__bpf_kfunc u64
bpf_uvm_chunk_get_va_end(uvm_gpu_chunk_t *chunk);
```

**重要性**: ⭐⭐⭐⭐ (4/5)
**难度**: ⭐⭐ (2/5) - 需要解引用 `chunk->va_block`

**理由**:
- 了解 chunk 的虚拟地址可以实现"热度"预测
- 例如: 相邻虚拟地址可能有相关性

**实现**:
```c
__bpf_kfunc u64
bpf_uvm_chunk_get_va_start(uvm_gpu_chunk_t *chunk)
{
    if (!chunk || !chunk->va_block)
        return 0;
    return chunk->va_block->start;
}
```

### 5.3 P2 - 中优先级 (增强能力)

#### 5. Chunk 大小/状态读取

```c
__bpf_kfunc u64 bpf_uvm_chunk_get_size(uvm_gpu_chunk_t *chunk);
__bpf_kfunc u32 bpf_uvm_chunk_get_state(uvm_gpu_chunk_t *chunk);
```

**重要性**: ⭐⭐⭐ (3/5)
**理由**: 可以根据 chunk 大小优先驱逐小 chunks

#### 6. 链表计数/检查

```c
__bpf_kfunc bool bpf_uvm_list_empty(struct list_head *head);
__bpf_kfunc u64 bpf_uvm_list_count(struct list_head *head, u32 max);
```

**重要性**: ⭐⭐⭐ (3/5)
**理由**: ARC 需要动态调整链表大小

### 5.4 P3 - 低优先级 (Nice-to-Have)

#### 7. Last/Prev 遍历

```c
__bpf_kfunc uvm_gpu_chunk_t *bpf_uvm_list_last(struct list_head *head);
__bpf_kfunc uvm_gpu_chunk_t *bpf_uvm_list_prev(...);
```

**重要性**: ⭐⭐ (2/5)
**理由**: 可以用 First+Next 反向迭代替代

---

## 6. Depopulate Hook 修复建议

### 6.1 问题根源

```c
// uvm_pmm_gpu.c:1445
if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
    !chunk_is_in_eviction(pmm, chunk)) {  // ← 这里 100% 拦截
    if (bpf_hook)
        bpf_hook(pmm, chunk, list);
}
```

### 6.2 修复方案 A: 移除条件检查

```c
// 总是调用 BPF hook
if (bpf_hook)
    bpf_hook(pmm, chunk, list);

// 然后才做条件检查
if (!chunk_is_root_chunk_pinned(pmm, chunk) &&
    !chunk_is_in_eviction(pmm, chunk)) {
    list_move_tail(&chunk->list, list);
}
```

**优点**: BPF 可以观察所有 depopulate 事件
**缺点**: BPF 看到的 chunk 可能不在链表中

### 6.3 修复方案 B: 新增 Hook

```c
// 在 eviction 过程中的 depopulate
void uvm_bpf_call_pmm_chunk_depopulate_in_eviction(
    uvm_pmm_gpu_t *pmm,
    uvm_gpu_chunk_t *chunk,
    struct list_head *list);
```

**优点**: 明确区分两种 depopulate 场景
**缺点**: 增加 API 复杂度

### 6.4 推荐: 方案 A

理由: BPF 程序可以通过 `chunk->in_eviction` 位自己判断,API 保持简单。

---

## 7. 实现路线图

### Phase 1 (1-2周): 核心遍历能力

**目标**: 解锁基础的链表扫描算法

**交付**:
1. ✅ `bpf_uvm_chunk_get_address()` - 读取物理地址
2. ✅ `bpf_uvm_list_first()` - 获取首个 chunk
3. ✅ `bpf_uvm_list_next()` - 遍历链表
4. ✅ 修复 Depopulate hook 调用

**可实现算法**: Clock, Second Chance (基于链表扫描)

### Phase 2 (2-3周): 频率追踪能力

**目标**: 实现 LFU 类算法

**交付**:
1. ✅ `bpf_uvm_list_move_before()` - 精确位置插入
2. ✅ `bpf_uvm_chunk_get_size()` - 读取 chunk 大小
3. ✅ `bpf_uvm_chunk_get_state()` - 读取状态

**可实现算法**: LFU, LFU-Aging, LIRS (部分)

### Phase 3 (3-4周): VA 映射感知

**目标**: 利用虚拟地址信息优化

**交付**:
1. ✅ `bpf_uvm_chunk_get_va_start/end()` - 读取虚拟地址
2. ✅ `bpf_uvm_list_count()` - 链表计数
3. ✅ `bpf_uvm_list_empty()` - 空链表检查

**可实现算法**: ARC, CAR, CART

---

## 8. 总结与建议

### 8.1 当前状态评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **API 完整性** | 2/10 | 只有 2 个 kfunc,缺失 90% 功能 |
| **算法覆盖** | 3/10 | 只能实现 FIFO/LRU/MRU 三种 |
| **生产可用性** | 4/10 | Depopulate hook 失效 |
| **扩展性** | 7/10 | struct_ops 架构良好,易扩展 |

**总体评分**: **4/10 (不及格)** - 需要大幅补充功能

### 8.2 关键行动项

**立即 (1周内)**:
1. 实现 `bpf_uvm_chunk_get_address()` ← **解锁所有高级算法**
2. 实现 `bpf_uvm_list_first/next()` ← **解锁链表遍历**
3. 修复 Depopulate hook 调用逻辑

**短期 (1个月内)**:
4. 实现 `bpf_uvm_list_move_before/after()` ← **解锁 LFU**
5. 补充 chunk 属性读取 API (size, state, va_block)

**中期 (3个月内)**:
6. 优化 BPF verifier 支持更复杂的遍历模式
7. 添加 BPF helper 支持时间戳读取 (LRU-K 需要)

### 8.3 与 CacheBPF 论文对比

| 能力 | CacheBPF | UVM 目标 | UVM 当前 | Gap |
|------|----------|---------|---------|-----|
| 链表 CRUD | 750 行代码 | 400 行 | ~50 行 | -88% |
| 对象属性访问 | 完整 | 基础 | 无 | -100% |
| 遍历效率 | 单次 BPF 调用 | 多次调用 | 不可能 | N/A |

**结论**: 当前实现距离 CacheBPF 论文标准还有**很大差距**,但架构基础良好,可以逐步补充。

### 8.4 对用户的影响

**当前能做**:
- ✅ 简单的 FIFO/LRU 策略
- ✅ 基于单次事件的决策 (不依赖历史)

**当前不能做**:
- ❌ 任何基于访问频率的策略 (LFU, LFU-K)
- ❌ 任何需要扫描链表的策略 (Clock, 2Q)
- ❌ 任何需要多个队列的策略 (ARC, LIRS)
- ❌ 追踪 chunk 跨 VA block 的完整生命周期

**建议**: 优先实现 Phase 1,这将使能力从 3/10 提升到 7/10。

---

**文档版本**: v1.0
**最后更新**: 2025-11-23
**作者**: BPF Trace Analysis
**相关文档**:
- `BPF_HOOKS_ACTUAL_BEHAVIOR_ANALYSIS.md` - Hook 实际行为
- `CHUNK_VA_BLOCK_MAPPING_ANALYSIS.md` - Chunk 重用模式
- `UVM_LIST_HELPERS.md` - CacheBPF API 参考
