# UVM PMM Chunk 生命周期分析报告

**日期**: 2025-11-23
**分析时长**: 5秒生产环境 trace
**捕获事件总数**: 890,884 个事件（丢失 69,517 个事件，因为频率过高）

## 核心结论

本文档分析了 NVIDIA UVM 物理内存管理器 (PMM) 中 GPU 内存 chunk 的生命周期，以理解驱逐行为并指导通过 BPF hooks 实现自定义驱逐策略（如 FIFO）的设计。

**关键发现**: 对于 FIFO 驱逐策略，`populate` 和 `depopulate` hooks 应该 **bypass**（什么都不做），因为这些事件不代表"进入队列时间"，而是已分配 chunk 的引用计数变化。

---

## 1. 分析方法

### 1.1 追踪设置

使用 `bpftrace` 监控以下 UVM 函数：
- `chunk_update_lists_locked` - chunk 激活（从 pinned 变为 unpinned）
- `uvm_pmm_gpu_mark_root_chunk_used` - chunk populate（resident count 增加）
- `uvm_pmm_gpu_mark_root_chunk_unused` - chunk depopulate（resident count 减少）
- `root_chunk_update_eviction_list` - list 更新操作
- `pick_root_chunk_to_evict` - 驱逐候选选择

### 1.2 分析工具

- **bpftrace 脚本**: `/tmp/trace_pmm_chunk.bt`
- **Python 分析器**: `/tmp/analyze_chunk_lifecycle.py`
- **Trace 日志**: GPU 内存压力下 5 秒采样

---

## 2. 总体统计（5秒 trace）

| 指标 | 数量 | 说明 |
|------|------|------|
| 追踪到的唯一 chunk | 15,876 | trace 窗口期间观察到的 chunk |
| 总事件数 | 855,747 | 所有生命周期事件 |
| Chunk 分配（首次见到）| 291,480 | 进入系统的新 chunk |
| Chunk 激活 | 36,911 | 变为可驱逐状态 |
| Chunk populates | 379,317 | 引用计数增加 |
| Chunk depopulates | 36,544 | 引用计数减少 |
| List 更新 | 440,943 | List 重排序操作 |
| 驱逐选择 | 35,910 | 驱逐候选选择次数 |
| Chunk 被驱逐 | 35,099 | 成功驱逐的 chunk |
| 丢失事件 | 69,517 | Ring buffer 溢出（8% 丢失率）|

**关键观察**:
- **Populates (379K) > Evictions (35K)**: 每个 chunk 被 populate **多次**（平均 10.8 次）
- **List updates (440K) > Populates (379K)**: 除了 populate 还有其他操作（激活等）
- **Activations (36K) ≈ Depopulates (36K) ≈ Evictions (35K)**: 这三者数量高度相关

---

## 3. Chunk 首次出现事件分析

当 chunk 第一次出现在 trace 中时，我们看到的是什么事件？

| 首次事件 | 数量 | 百分比 |
|---------|------|--------|
| `evict_selected` | 8,940 | **56.31%** |
| `populate_used` | 6,852 | 43.16% |
| `activate` | 84 | 0.53% |

### 3.1 解读

**56.31% 的 chunk 首次出现就是在驱逐时**，这意味着：
- 这些 chunk 在 trace **开始前**就已经分配了
- 它们在系统中存活，但在 populate 时没有被观察到
- 我们只在驱逐过程选中它们时才"发现"它们

**对 FIFO 的影响**: 使用 `populate` 作为"进入队列时间"是不正确的，因为它遗漏了在 trace 前分配的 chunk，并且代表的是引用变化而非初始分配。

---

## 4. 常见生命周期模式

### 4.1 Top 10 事件序列（每个 chunk 的前 5 个事件）

| 数量 | 模式 | 描述 |
|------|------|------|
| 8,877 (56%) | `evict_selected → depopulate_unused → list_update → populate_used → list_update` | **驱逐-重用循环**: chunk 被驱逐，立即 depopulate，然后重新 populate |
| 3,873 (24%) | `populate_used → list_update → populate_used → list_update → populate_used` | **多次 Populate**: chunk 被不同 VA block 反复引用 |
| 2,693 (17%) | `populate_used → list_update → populate_used → list_update → evict_selected` | **Populate-驱逐**: chunk 被 populate 后不久就被驱逐 |
| 121 | `populate_used → list_update → evict_selected → depopulate_unused → list_update` | **快速驱逐**: populate 后立即驱逐 |

### 4.2 模式分析

**模式 1（驱逐-重用，56%）**: 主导模式
```
evict_selected → depopulate_unused → list_update → populate_used → list_update
     ↑                                                    ↓
     └────────────────── Chunk 被回收 ──────────────────┘
```

这显示：
- 驱逐后立即 `depopulate`（在许多情况下时间戳相同）
- 释放的 chunk 很快被重新 populate 给新的 VA block
- 这是一个 **thrashing 模式** - 驱逐后立即重新分配

**模式 2（多次 Populate，24%）**:
```
populate → populate → populate → ...
```

这显示：
- 单个 chunk 可以被多个 VA block 引用
- `populate` **不是**"首次分配"而是"引用计数++"
- 使用 `populate` 时间作为 FIFO "进入时间"是无效的

---

## 5. Populate → Evict 序列分析

### 5.1 驱逐时间分布

分析了 15,792 个经历了 populate 和 evict 的 chunk：

| 驱逐时间 | 示例 Chunk | Populates | Depopulates | 模式 |
|---------|-----------|-----------|-------------|------|
| **0ms**（最快）| `0xffffcfd7cceb89c8` | 13 | 0 | `evict → populate → ...` |
| 1,999ms | `0xffffcfd7cce6ef58` | 15 | 1 | `populate → ... → evict` |
| 4,448ms（最慢）| `0xffffcfd7ccdd66b8` | 20 | 2 | `populate → ... → evict` |

### 5.2 关键统计

- **立即驱逐（0ms）**: 1 个 chunk（0.01%）
- **多次 populate**: 15,792 个 chunk（**100%**）
- **每个 chunk 平均 populate 次数**: 13-36 次

**关键发现**: **每一个**被驱逐的 chunk 都被 populate 了**多次**。这证明 `populate` 不是一次性的"分配"事件，而是反复发生的"引用增加"事件。

---

## 6. Thrashing 分析

### 6.1 定义

**Thrashing chunks**: 以 `age=0ms` 被驱逐的 chunk（从首次观察到驱逐的时间 = 0）

### 6.2 统计

- **Thrashing chunk 总数**: 11,696 / 15,876（**73.7%**）
- 这些是首次看到就在驱逐的 chunk（`evict_selected` 作为首次事件）

### 6.3 Thrashing 模式示例

```
evict_selected → depopulate_unused → list_update → populate_used → list_update → activate
```

**解读**:
- Chunk 被驱逐
- 立即 depopulate 并移到 unused list
- 立即重新 populate
- 再次激活

这是高压力下的典型内存抖动。

---

## 7. 详细 Chunk 生命周期示例

### 示例 1: 典型 Chunk 生命周期

**Chunk**: `0xffffcfd7ccdebbf8`
**总事件数**: 48
**Populates**: 20, **Depopulates**: 2, **Activates**: 2
**存活时间**: 3,020ms

```
时间    事件                说明
----    ----                ----
10ms    evict_selected      首次看到：正在被驱逐（在 trace 前就存在）
22ms    populate_used       驱逐后 12ms 重新 populate
282ms   populate_used       再次 populate（260ms 后）
282ms   list_update         List 重排序
895ms   evict_selected      第二次驱逐（上次 populate 后 613ms）
895ms   depopulate_unused   立即 depopulate（同一毫秒）
895ms   list_update         List 更新
895ms   populate_used       立即重新 populate（同一毫秒）
895ms   list_update         List 再次更新
895ms   activate            激活（同一毫秒）
896ms   populate_used       再次 populate（1ms 后）
...     ...                 （还有 33 个事件）
```

**关键观察**:
1. **evict → depopulate → populate 在同一毫秒内发生**（895ms）
2. Chunk 在其 3 秒生命周期内被 **populate 了 20 次**
3. 首次事件是驱逐而非 populate（chunk 预先存在）

### 示例 2: 高流转 Chunk

**Chunk**: `0xffffcfd7ccdef438`
**总事件数**: 71
**Populates**: 31, **Depopulates**: 2, **Activates**: 2
**存活时间**: 2,092ms

这个 chunk 在 **2 秒内经历了 31 次 populate**（约 15 次/秒）。

---

## 8. 对 FIFO 驱逐策略的影响

### 8.1 为什么 Populate/Depopulate Hooks 应该 Bypass

**使用 `populate` 作为 FIFO "进入时间"的问题**:

1. **不是一次性事件**: 100% 的被驱逐 chunk 都被 populate 多次
   - 使用最后 populate 时间 → 变成 LRU（最近最少 populate）
   - 使用首次 populate 时间 → 遗漏 trace 前分配的 chunk

2. **Populate ≠ 分配**:
   - Populate 是引用计数增加（VA block 引用 chunk）
   - 真正的分配发生在别处（无法通过 populate 直接追踪）

3. **Depopulate 是驱逐过程的一部分**:
   - 通常在驱逐的同一毫秒发生
   - 不是独立的"使用结束"事件

### 8.2 正确的 FIFO 实现策略

**方案 1: Bypass 所有 hooks（推荐）**
- FIFO = 基于分配顺序的先进先出
- 内核已经维护分配顺序（chunk 按顺序分配）
- 默认 `list_move_tail` 行为已经近似 FIFO
- **操作**: BPF hooks 中什么都不做 → 零开销

**方案 2: 使用 chunk 地址作为时间戳代理**
- 假设：Chunk 地址按顺序分配（地址越低 = 越老）
- 在 `eviction_prepare` hook 中：按地址排序 chunk
- 将低地址移到 head（先驱逐最老的）
- **限制**: 只在分配器是顺序的情况下有效

**方案 3: 追踪真正的分配时间（复杂）**
- 拦截 chunk 分配（目前没有 hook）
- 在 BPF map 中存储分配时间戳
- 在 eviction_prepare 中使用时间戳重排序
- **问题**: 分配函数可能被内联/无法追踪

### 8.3 推荐实现

```c
SEC("struct_ops/uvm_pmm_chunk_populate")
int BPF_PROG(uvm_pmm_chunk_populate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    // BYPASS: 内核已经移到 tail，这对 FIFO 是正确的
    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_depopulate")
int BPF_PROG(uvm_pmm_chunk_depopulate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    // BYPASS: depopulate 是驱逐的一部分，不是独立事件
    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    // BYPASS: 激活不改变分配顺序
    return 0;
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare, uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used, struct list_head *va_block_unused)
{
    // 可选：如果需要可以按 chunk 地址重排序
    // 对于真正的 FIFO，可能不需要，因为内核顺序已经正确
    return 0;
}
```

---

## 9. 性能考虑

### 9.1 BPF Hook 开销

即使是空 hook（return 0），也有开销：

- **每次 populate**: RCU read lock/unlock + 函数调用
- **频率**: 5 秒内 379,317 次 populate = **75,863 次调用/秒**
- **每次 list_update**: BPF hook 调用
- **频率**: 5 秒内 440,943 次更新 = **88,188 次调用/秒**

**BPF 总开销**: 高内存压力下约 16.4 万次 hook 调用/秒

### 9.2 建议

对于 FIFO，通过以下方式**完全移除 BPF hook 调用** populate/depopulate：
1. 不在 struct_ops 中注册这些 hook
2. 或在内核 wrapper 中调用前检查 NULL

**预期收益**: 消除每秒 16.4 万次函数调用开销

---

## 10. 被驱逐 Chunk 的年龄分布

从 5 秒 trace 总结：

```
[0ms]           27,615  (78.7%)  ← 立即驱逐
[512ms-1K)       2,412  ( 6.9%)
[1K-2K)          1,211  ( 3.4%)
[2K-4K)          3,861  (11.0%)
```

**解读**:
- **78.7%** 的被驱逐 chunk 年龄为 0ms（首次看到就在驱逐）
- **21.3%** 在驱逐前存活了 512ms 以上
- 两类群体：**短暂 chunk**（78.7%）和**常驻 chunk**（21.3%）

---

## 11. 结论

1. **Populate 不是"首次分配"**: 每个被驱逐的 chunk 平均被 populate 13-36 次

2. **首次事件通常是驱逐（56%）**: Chunk 在我们通过 populate 观察到它们之前就已存在

3. **Evict → depopulate → populate 原子发生**: 在同一毫秒，表明紧密耦合

4. **对于 FIFO，bypass populate/depopulate hooks**: 它们不代表队列进入/退出

5. **高频事件**: 压力下每秒 16.4 万次 hook 调用 → 性能问题

6. **严重的 Thrashing**: 73.7% 的 chunk 立即被驱逐（age=0ms）

---

## 12. 建议

### 对于 FIFO 实现:
1. ✅ **Bypass populate hook**（不是队列进入时间）
2. ✅ **Bypass depopulate hook**（驱逐的一部分，不是独立事件）
3. ✅ **Bypass activate hook**（不改变分配顺序）
4. ⚠️ **可选的 eviction_prepare hook**（仅当需要按地址重排序时）

### 对于内核优化:
1. 考虑为不需要的策略移除 BPF hook 调用
2. 添加标志以按策略类型禁用 hook
3. 调查 thrashing：78.7% 立即驱逐表明内存严重超额订阅

---

## 附录 A: bpftrace 脚本位置

`/tmp/trace_pmm_chunk.bt` - UVM PMM chunk 生命周期追踪器

## 附录 B: 分析脚本位置

`/tmp/analyze_chunk_lifecycle.py` - Python 脚本用于解析和分析 trace 日志

## 附录 C: 原始 Trace 数据

`/tmp/pmm_trace_5s.log` - 5 秒生产环境 trace（890,884 个事件）

---

## 参考资料

1. UVM_LRU_POLICY.md - 基于 BPF 的驱逐策略的原始设计文档
2. `uvm_pmm_gpu.c` - PMM 实现及 hook 插入点
3. `uvm_bpf_struct_ops.c` - BPF struct_ops 框架和 kfuncs

---

## 13. 深度模式分析（Pattern Analysis）

**分析工具**: `/tmp/analyze_patterns.py` - 高级行为模式聚合分析器

本节对 855,747 个事件和 15,876 个 chunk 进行深度模式挖掘，识别时序模式、事件共现、事件转换序列及行为分类。

---

### 13.1 时序模式分析

#### 13.1.1 事件共现模式（同一毫秒内发生）

| 事件组合 | 次数 | 说明 |
|---------|------|------|
| `list_update + populate_used` | **315,203** | List 更新和 populate 原子发生 |
| `list_update + list_update` | 76,971 | 连续的 list 操作 |
| `populate_used + populate_used` | 42,088 | 多个 VA block 同时引用同一 chunk |
| `activate + depopulate_unused` | 34,765 | 激活和 depopulate 同时发生（矛盾？） |
| `depopulate_unused + evict_selected` | 34,049 | Depopulate 是驱逐的一部分 |
| `evict_selected + list_update` | 33,985 | 驱逐后立即更新 list |

**关键发现**:
1. **315K 次 `list_update + populate_used` 共现**: 这是最频繁的原子操作，表明每次 populate 几乎都伴随 list 更新
2. **34K 次 `activate + depopulate_unused` 共现**: 这看似矛盾（激活但同时 depopulate），可能是复杂的状态转换
3. **34K 次 `depopulate + evict` 共现**: 证实 depopulate 是驱逐流程的组成部分，而非独立事件

#### 13.1.2 事件转换序列（状态机视角）

最频繁的状态转换：

| 转换 | 次数 | 含义 |
|------|------|------|
| `populate_used → list_update` | **357,490** | Populate 后更新 eviction list |
| `list_update → populate_used` | **315,690** | List 更新后进行 populate |
| `evict_selected → depopulate_unused` | 35,075 | 驱逐后立即 depopulate |
| `depopulate_unused → list_update` | 35,035 | Depopulate 后更新 list |
| `list_update → activate` | 35,035 | List 更新触发激活 |
| `activate → populate_used` | 34,916 | 激活后立即 populate |

**状态机模型**:
```
         ┌─────────────────────────────┐
         │                             │
         ▼                             │
    populate_used ─────► list_update ──┘
         │                   │
         │                   ▼
         │               activate ────► populate_used
         │                             (循环)
         ▼
    evict_selected ───► depopulate_unused ───► list_update
         │                                          │
         └──────────────────────────────────────────┘
                     (重新分配循环)
```

---

### 13.2 序列模式分析

#### 13.2.1 Top 3-Event 序列模式

| 序列 | 次数 | 占比 |
|------|------|------|
| `list_update → populate_used → list_update` | **315,690** | 36.9% |
| `populate_used → list_update → populate_used` | **280,657** | 32.8% |
| `evict_selected → depopulate_unused → list_update` | 35,035 | 4.1% |
| `populate_used → list_update → activate` | 35,035 | 4.1% |
| `depopulate_unused → list_update → populate_used` | 35,033 | 4.1% |

**核心循环**:
- **`populate ⇄ list_update` 循环占主导** (69.7%)
- 这证实了 populate 是高频反复事件，而非一次性分配

#### 13.2.2 Top 5-Event 序列模式

| 序列 | 次数 |
|------|------|
| `list_update → populate_used → list_update → populate_used → list_update` | **239,722** |
| `populate_used → list_update → populate_used → list_update → populate_used` | **204,689** |
| `populate_used → list_update → populate_used → list_update → activate` | 35,035 |
| `evict_selected → depopulate_unused → list_update → populate_used → list_update` | 35,033 |

**模式识别**:
1. **连续 populate 模式** (44.4万次): `populate → list_update → populate → ...`
   - 表明 chunk 在短时间内被多个 VA block 反复引用
   - 这是内存压力下的典型"争抢"行为

2. **驱逐-回收-重用模式** (35K次): `evict → depopulate → list_update → populate → list_update`
   - 完整的 thrashing 循环
   - 驱逐的 chunk 立即被重新分配

---

### 13.3 行为模式分类

基于事件序列和时序特征，将所有 chunk 分类为以下行为模式：

| 模式 | 数量 | 占比 | 描述 |
|------|------|------|------|
| **activate_pattern** | **15,876** | **100.00%** | 所有 chunk 都有激活事件（说明所有 chunk 都经历了 pinned→unpinned 转换）|
| **multi_populate** | **15,792** | **99.47%** | 被 populate ≥10 次（几乎所有 chunk 都是多次引用）|
| **thrash_cycle** | **13,963** | **87.95%** | 经历多次驱逐-重用循环（≥2次 evict 和 ≥2次 populate）|
| **stable** | 32 | 0.20% | 长时间存活（>1秒）但事件数少（<10）|
| **thrash_immediate** | 27 | 0.17% | 驱逐后在同一毫秒内重新 populate |
| **evict_no_depop** | 1 | 0.01% | 被驱逐但没有 depopulate 记录 |

#### 13.3.1 详细模式特征

##### Pattern 1: **thrash_immediate** (立即 thrashing, 27 chunks)

**定义**: `evict → depopulate → populate` 在同一毫秒内发生

**统计特征**:
- 事件数: 51-76 (平均 60.4)
- 存活时间: 3,407-4,431ms (平均 3,887ms)
- Top 事件: populate_used (704次), list_update (699次), evict_selected (82次)

**示例 chunk**:
```
0xffffcfd7ccdb3258:
  evict → depopulate → populate → populate → populate → list_update → ...
  (55 events, 3618ms)
```

**含义**: 这些 chunk 被驱逐后**瞬间**重新分配，是最极端的 thrashing 案例

---

##### Pattern 2: **thrash_cycle** (反复 thrashing, 13,963 chunks = 87.95%)

**定义**: 经历 ≥2 次驱逐 和 ≥2 次 populate

**统计特征**:
- 事件数: 32-100 (平均 56.3)
- 存活时间: 2,623-4,454ms (平均 3,552ms)
- Top 事件: list_update (359K), populate_used (326K), evict_selected (33K)

**这是系统的主导模式**: 87.95% 的 chunk 都经历了反复的驱逐-重用循环

**含义**: 系统处于**严重内存压力**下，绝大多数 chunk 无法长期驻留，被反复驱逐和重新分配

---

##### Pattern 3: **multi_populate** (多次引用, 15,792 chunks = 99.47%)

**定义**: 被 populate ≥10 次

**统计特征**:
- 事件数: 26-100 (平均 54.2)
- 存活时间: 665-4,454ms (平均 3,512ms)

**关键洞察**: **99.47% 的 chunk 都被 populate 了 ≥10 次**
- 彻底证明 populate **不是**"首次分配"
- 而是"被 VA block 引用"的高频事件
- **对 FIFO 的影响**: 无法使用 populate 时间作为队列入队时间

---

##### Pattern 4: **stable** (稳定型, 32 chunks = 0.20%)

**定义**: 存活时间 >1秒 且事件数 <10

**统计特征**:
- 事件数: 2-6 (平均 4.3)
- 存活时间: 1,979-2,259ms (平均 2,006ms)
- 主要事件: activate (137次)

**示例**:
```
0xffff8a010edf2f18:
  activate → activate → activate → activate → activate → activate
  (6 events, 1981ms)
```

**含义**: 极少数 chunk (0.2%) 保持稳定，未经历频繁的 populate/evict
- 可能是被 pin 住的 chunk
- 或者是低访问频率的 chunk

---

### 13.4 原子事件分析（0ms 间隔）

**原子事件 chunk 数量**: 15,864 / 15,876 = **99.92%**

几乎**所有 chunk** 都有在同一毫秒内发生的事件序列

#### Top 10 原子事件最多的 chunk:

| Chunk 地址 | 0ms 间隔数 | 总事件数 |
|-----------|----------|---------|
| `0xffffcfd7ccdef848` | **64** | 100 |
| `0xffffcfd7ccddea98` | 64 | 98 |
| `0xffffcfd7cce4e758` | 64 | 98 |
| `0xffffcfd7cce71f78` | 63 | 98 |
| `0xffffcfd7cce88778` | 63 | 94 |

**解读**:
- 这些 chunk 的事件中有 **64%** 是原子发生的（同一毫秒内）
- 说明 PMM 操作高度集中，事件爆发式发生
- 这也是为什么 bpftrace 会丢失 8% 的事件（ring buffer 溢出）

---

### 13.5 模式聚合结论

#### 13.5.1 系统行为总结

1. **系统处于极度内存压力下**:
   - 87.95% 的 chunk 经历 thrash_cycle
   - 驱逐后立即重新分配（35K 次 `evict → depopulate → populate` 序列）

2. **Populate 是高频引用事件，非分配事件**:
   - 99.47% 的 chunk 被 populate ≥10 次
   - 最频繁的模式就是 `populate ⇄ list_update` 循环 (69.7%)

3. **事件高度原子化**:
   - 99.92% 的 chunk 有同一毫秒内的事件
   - 315K 次 `list_update + populate_used` 共现
   - 说明驱逐、depopulate、populate、list_update 是紧密耦合的原子操作序列

4. **100% 的 chunk 都有 activate 事件**:
   - 说明所有 chunk 都经历了从 pinned 到 unpinned 的转换
   - Activate 是进入 eviction list 的必经之路

#### 13.5.2 对 BPF Hook 设计的影响

基于模式分析的设计建议：

| Hook | 是否应该 Bypass? | 理由 |
|------|----------------|------|
| **populate** | ✅ **YES** | 99.47% chunk 被 populate ≥10 次，无法表示"入队时间" |
| **depopulate** | ✅ **YES** | 34K 次与 evict 共现，是驱逐流程的一部分 |
| **activate** | ✅ **YES** | 100% chunk 都有，但不改变分配顺序（只是状态转换）|
| **eviction_prepare** | ⚠️ **可选** | 唯一可以重排序的地方（在驱逐前） |

**推荐策略**:
- **对于 FIFO**: 所有 hooks 都 bypass（return 0）
  - 内核默认的 `list_move_tail` 行为已经近似 FIFO
  - 避免每秒 16.4 万次的 hook 调用开销

- **对于 LRU**: 保留 activate hook
  - Activate 表示 chunk 被访问（从 pinned 变为 unpinned）
  - 在 activate 时 `move_tail` 可以实现真正的 LRU

- **对于自定义策略（如 LFU, S3-FIFO）**:
  - 在 eviction_prepare 中进行全局重排序
  - 或在 activate 中增加访问计数

#### 13.5.3 性能影响重新评估

基于模式频率：

| 事件类型 | 5秒内次数 | 频率 (次/秒) | Hook 开销 |
|---------|---------|------------|----------|
| populate | 379,317 | 75,863 | 高（如果有 hook）|
| list_update | 440,943 | 88,188 | 极高（如果追踪）|
| activate | 36,911 | 7,382 | 中等 |
| evict | 35,099 | 7,020 | 低 |

**总 BPF hook 潜在调用频率**:
- 如果所有 hooks 都启用: **171K 次/秒**
- 如果只启用 activate: **7.4K 次/秒**（可接受）
- 如果只启用 eviction_prepare: **7K 次/秒**（可接受）

**建议**: 对于高频事件（populate, list_update），应完全避免 BPF hook 调用

---

### 13.6 模式可视化

#### 主导生命周期循环 (87.95% 的 chunk):

```
 ┌──────────────────────────────────────────────────────────┐
 │                  Thrash Cycle (87.95%)                   │
 └──────────────────────────────────────────────────────────┘

    [Allocation]
         │
         ▼
    populate_used ◄─────┐
         │              │
         ▼              │
    list_update         │  (多次 populate 循环)
         │              │  99.47% chunk 重复 ≥10 次
         ▼              │
    populate_used ──────┘
         │
         ▼
    activate ──────► list_update
         │
         ▼
   (在 eviction list 中等待)
         │
         ▼
    evict_selected
         │
         ▼
    depopulate_unused  (同一毫秒，34K 次共现)
         │
         ▼
    list_update
         │
         ▼
    populate_used ──────► [重新开始循环]
         │
         └──────────► list_update ──────► ...
```

---

### 13.7 附录：模式分析脚本

**脚本位置**: `/tmp/analyze_patterns.py`

**功能**:
1. 时序模式分析（事件共现和转换）
2. 序列模式分析（3-event 和 5-event 窗口）
3. 行为模式分类（thrash_immediate, thrash_cycle, multi_populate 等）
4. 原子事件分析（0ms 间隔统计）

**运行方法**:
```bash
python3 /tmp/analyze_patterns.py /tmp/pmm_trace_5s.log
```
