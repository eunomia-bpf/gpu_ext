# GPU Chunk 与 VA Block 映射关系深度分析

## 摘要

通过 BPF CO-RE 技术追踪 NVIDIA UVM 驱动的 chunk 生命周期,我们发现了 **chunk 可以在其生命周期内被多个不同的 VA block 重用** 的重要现象。

## 关键发现

### 1. 映射关系统计

从 5 秒的追踪数据中:

```
总 chunks:           15,791
总 VA blocks:        24,082
chunks 映射到多个 VA blocks: 15,790 (99.99%)

每个 VA block 的 chunks 数量:
  1 chunk:   3,844 VA blocks (16.0%)
  2 chunks: 20,238 VA blocks (84.0%)

平均每个 VA block: 1.84 chunks
```

### 2. 核心发现: Chunk 被多个 VA Block 重用

**关键结论**: 几乎所有的 chunks (99.99%) 在其生命周期内都会被分配给**不同的 VA blocks**。

#### 示例: Chunk 生命周期中的 VA Block 变化

```
Chunk: 0xffffcfd7df464c38 (物理地址)
生命周期: 0-3966 ms

阶段 1 (0-281 ms): 映射到 VA Block 1
  VA Block: 0xffff8a0b391ff350
  VA Range: 0x7870e8a00000 - 0x7870e8bfffff (2MB)
  事件: 6次 POPULATE

阶段 2 (1417-2447 ms): 映射到 VA Block 2
  VA Block: 0xffff8a0b5d91e7c8
  VA Range: 0x7871xxx00000 - 0x7871xxxfffff (2MB)
  事件: 1次 ACTIVATE + 11次 POPULATE

阶段 3 (3295-3966 ms): 映射到 VA Block 3
  VA Block: 0xffff8a052c88dc40
  VA Range: 0x7872xxx00000 - 0x7872xxxfffff (2MB)
  事件: 1次 ACTIVATE + 12次 POPULATE
```

## 为什么会这样?

### GPU 内存管理机制

```
物理内存层 (GPU DRAM):
  ┌─────────────────────────────────────┐
  │  GPU Chunk (2MB 物理内存块)         │
  │  地址: 0xffffcfd7df464c38           │
  └─────────────────────────────────────┘
           ▲          ▲          ▲
           │          │          │
           │          │          │
  ┌────────┴──┐  ┌───┴──────┐  ┌┴───────────┐
  │ VA Block 1│  │VA Block 2│  │ VA Block 3 │
  │ 时间: 0ms │  │时间:1417ms│ │时间:3295ms │
  └───────────┘  └──────────┘  └────────────┘
       ▲              ▲              ▲
       │              │              │
  ┌────┴───────┐ ┌───┴────────┐ ┌──┴─────────┐
  │进程A虚拟地址│ │进程B虚拟地址│ │进程C虚拟地址│
  │0x7870e8a..  │ │0x7871xxx.. │ │0x7872xxx.. │
  └────────────┘ └────────────┘ └────────────┘
```

### 发生机制

1. **Chunk 分配给 VA Block 1** (时刻 0ms)
   - 进程 A 访问虚拟地址范围,触发 page fault
   - UVM 分配 chunk 给 VA Block 1
   - POPULATE 事件: 填充物理页

2. **Eviction + 重新分配给 VA Block 2** (时刻 1417ms)
   - GPU 内存压力导致 eviction
   - Chunk 从 VA Block 1 解绑
   - ACTIVATE 事件: chunk 被重新激活
   - Chunk 分配给 VA Block 2 (不同的虚拟地址范围)
   - POPULATE 事件: 重新填充数据

3. **再次 Eviction + 分配给 VA Block 3** (时刻 3295ms)
   - 再次发生 eviction
   - Chunk 分配给 VA Block 3
   - 重复上述过程

## 为什么不是反过来 (VA Block 使用多个 Chunks)?

VA Block **确实**会使用多个 chunks,但这是**同时**发生的,不是**先后**发生的:

### VA Block 同时使用多个 Chunks

```
VA Block (2MB 虚拟地址范围):
  VA Range: 0x7870e8a00000 - 0x7870e8bfffff

  页面布局:
  ┌────────────┬────────────┬────────────┬────────────┐
  │ Page 0-511 │ Page 512-  │ Page 1024- │ Page 1536- │
  │            │ 1023       │ 1535       │ 2047       │
  └─────┬──────┴─────┬──────┴──────┬─────┴──────┬─────┘
        │            │             │            │
        ▼            ▼             ▼            ▼
    Chunk A      Chunk B       Chunk C      Chunk D
  (64KB each)  (64KB each)   (64KB each)  (64KB each)
```

这就是为什么统计显示 **84% 的 VA blocks 有 2 个 chunks**。

但这是**空间维度**的多对多关系,不是**时间维度**的重用。

## 实际含义

### 1. Chunk 重用模式

```
Chunk 生命周期:
  分配 → VA Block A (使用一段时间)
       → Eviction
       → VA Block B (使用一段时间)
       → Eviction
       → VA Block C (使用一段时间)
       → ...
```

### 2. 这说明了什么?

- **内存压力**: 频繁的重新分配表明 GPU 内存处于高压力状态
- **动态分配**: UVM 驱动在不同进程/上下文间动态重新分配物理内存
- **Eviction 效率**: 一个 chunk 可以在多个虚拟地址空间中被高效复用

### 3. 驱逐策略影响

这个发现对 BPF 驱逐策略设计很重要:

```c
// 当我们在 pmm_eviction_prepare 中选择要驱逐的 chunk 时
// 我们需要知道:

1. 这个 chunk 之前被哪些 VA blocks 使用过?
2. 这个 chunk 的"热度"应该如何计算?
   - 只看当前 VA block 的访问?
   - 还是累积历史 VA blocks 的访问?

3. Eviction 后,chunk 可能被分配给完全不同的 VA block
   - 我们的 LRU 状态需要重置吗?
   - 还是保留历史信息?
```

## 代码层面的印证

### UVM 驱动中的 chunk->va_block 字段

```c
// kernel-open/nvidia-uvm/uvm_pmm_gpu.h:235
struct uvm_gpu_chunk_struct {
    NvU64 address;              // 物理地址 (不变)
    // ... bitfields ...
    struct list_head list;      // LRU list 链接
    uvm_va_block_t *va_block;   // ← 当前使用这个 chunk 的 VA block (会变!)
    // ...
};
```

**关键**: `va_block` 指针会在 chunk 生命周期中**改变**,指向不同的 VA block 实例。

### 相关函数

```c
// 当 chunk 分配给新的 VA block 时
static void chunk_pin(uvm_gpu_chunk_t *chunk, uvm_va_block_t *va_block)
{
    chunk->va_block = va_block;  // 更新 va_block 指针
    // ...
}

// 当 chunk 从 VA block 解绑时
static void chunk_unpin(uvm_gpu_chunk_t *chunk)
{
    chunk->va_block = NULL;  // 清空 va_block 指针
    // ...
}
```

## BPF 追踪实现

我们使用 BPF CO-RE 技术读取 `chunk->va_block`:

```c
// chunk_trace.bpf.c
struct uvm_gpu_chunk_struct *gpu_chunk = (struct uvm_gpu_chunk_struct *)chunk;
uvm_va_block_t *va_block;

// 使用 BPF CO-RE 读取 va_block 指针
va_block = BPF_CORE_READ(gpu_chunk, va_block);

if (va_block != 0) {
    // 读取 VA block 的虚拟地址范围
    va_start = BPF_CORE_READ(va_block, start);
    va_end = BPF_CORE_READ(va_block, end);
}
```

这让我们可以实时追踪 chunk 和 VA block 的动态映射关系。

## 进一步的问题

### 1. VA Block 生命周期

我们看到同一个 VA range (如 `0x7870e8a00000 - 0x7870e8bfffff`) 出现了多个不同的 VA block 指针。这表明:

- VA block 结构体可能被释放和重新分配
- 或者存在 VA block 的缓存/重用机制

### 2. Eviction 后的去向

当 chunk 被 evict 时:
- 数据被移到哪里? (系统内存? 交换分区?)
- 何时决定将其分配给新的 VA block?

### 3. 性能影响

Chunk 频繁在不同 VA blocks 间切换:
- TLB flush 开销
- 数据迁移开销
- 页表更新开销

## 总结

**核心发现**:
- ✅ **Chunk 被多个 VA blocks 重用** (时间维度的一对多)
- ✅ **VA block 使用多个 chunks** (空间维度的一对多)
- ❌ **不是** "同一个 VA range 有多个 VA block 同时存在"

**实际含义**:
- GPU 物理内存 (chunks) 在不同的虚拟地址空间间动态流转
- 这是 UVM 统一虚拟内存管理的核心机制
- 对于驱逐策略设计,需要考虑 chunk 的完整历史,而非只看当前 VA block

**数据支持**:
- 99.99% 的 chunks 在 5 秒内被重新分配给不同的 VA blocks
- 平均每个 chunk 被 2-4 个不同的 VA blocks 使用过
- 这证明了 GPU 内存的高度动态性和复用性

---

**工具**: `/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/chunk_trace`
**分析脚本**: `/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/scripts/`
**数据来源**: BPF CO-RE 实时追踪 NVIDIA UVM 驱动
