# NVIDIA UVM 驱动模块架构与工作原理完整分析

**文档版本**: 575.57.08
**分析日期**: 2025-11-14
**基于**: cflow 调用图分析 + 源代码深度剖析

---

## 目录

1. [概述](#1-概述)
2. [核心架构](#2-核心架构)
3. [内存管理层次结构](#3-内存管理层次结构)
4. [GPU 页面错误处理机制](#4-gpu-页面错误处理机制)
5. [性能优化模块](#5-性能优化模块)
6. [策略框架与可扩展性](#6-策略框架与可扩展性)
7. [关键数据流](#7-关键数据流)
8. [与 Linux 内核集成](#8-与-linux-内核集成)
9. [可观测性与调试](#9-可观测性与调试)
10. [总结与设计思想](#10-总结与设计思想)

---

## 1. 概述

### 1.1 什么是 UVM？

**UVM (Unified Virtual Memory)** 是 NVIDIA GPU 驱动中的一个内核模块，实现了 CPU 和 GPU 之间的统一虚拟地址空间。它使得应用程序可以在 CPU 和 GPU 之间透明地共享内存，无需显式的内存拷贝操作。

### 1.2 核心功能

- **统一地址空间**: CPU 和 GPU 使用相同的虚拟地址访问数据
- **按需页面迁移**: 根据访问模式自动在 CPU 和 GPU 内存之间迁移页面
- **GPU 页面错误处理**: 当 GPU 访问不存在的页面时，驱动会自动处理并迁移数据
- **智能预取**: 基于访问模式预测并预取数据
- **抖动检测与抑制**: 检测并处理 CPU-GPU 之间的内存乒乓效应
- **多 GPU 支持**: 支持多个 GPU 和 GPU 间的 P2P 访问

### 1.3 设计目标

1. **透明性**: 对应用程序透明，无需修改代码
2. **性能**: 通过智能策略最小化页面迁移开销
3. **可扩展性**: 模块化设计，易于添加新策略
4. **可观测性**: 提供丰富的性能计数器和调试接口

---

## 2. 核心架构

### 2.1 模块层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户空间应用                                │
│              (CUDA/OpenCL/HIP Applications)                 │
└────────────────────┬────────────────────────────────────────┘
                     │ ioctl / mmap
┌────────────────────┴────────────────────────────────────────┐
│                   UVM Driver (nvidia-uvm.ko)                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │         VA Space Management Layer                    │   │
│  │  • uvm_va_space    • uvm_va_range                   │   │
│  │  • uvm_va_block    • uvm_va_policy                  │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      Fault Handling & Migration Layer               │   │
│  │  • Replayable Faults  • Non-Replayable Faults       │   │
│  │  • Page Migration     • Prefetch Engine             │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Performance Module Layer                     │   │
│  │  • Thrashing Detection  • Access Counters           │   │
│  │  • Prefetch Heuristics  • Event Framework           │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Hardware Abstraction Layer (HAL)          │   │
│  │  • Maxwell  • Pascal  • Volta  • Turing             │   │
│  │  • Ampere   • Ada     • Hopper • Blackwell          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ MMIO / DMA / Interrupts
┌────────────────────┴────────────────────────────────────────┐
│                  GPU Hardware                                │
│  • Fault Buffer  • MMU  • Copy Engine  • Compute Units      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

#### 2.2.1 VA Space (uvm_va_space.h)

**虚拟地址空间** 是 UVM 的顶层数据结构，表示一个进程的统一虚拟地址空间：

```c
uvm_va_space_t {
    // 地址空间的范围树，包含所有 VA ranges
    uvm_range_tree_t range_tree;

    // 注册的 GPU 列表
    uvm_processor_mask_t registered_gpus;

    // 性能模块数据
    uvm_perf_module_data_t perf_modules[UVM_PERF_MODULE_TYPE_COUNT];

    // 与 Linux mm_struct 的关联
    uvm_va_space_mm_t va_space_mm;

    // 读写锁保护整个地址空间
    uvm_rw_semaphore_t lock;
}
```

#### 2.2.2 VA Range (uvm_va_range.h)

**虚拟地址范围** 表示地址空间中的一个连续区域：

```c
类型：
- UVM_VA_RANGE_TYPE_MANAGED: 受 UVM 管理的内存
- UVM_VA_RANGE_TYPE_EXTERNAL: 外部分配（如 RM 分配）
- UVM_VA_RANGE_TYPE_CHANNEL: GPU channel 映射
- UVM_VA_RANGE_TYPE_SKED_REFLECTED: Schedule reflected 映射
- UVM_VA_RANGE_TYPE_SEMAPHORE_POOL: 信号量池
- UVM_VA_RANGE_TYPE_DEVICE_P2P: GPU P2P 内存
```

#### 2.2.3 VA Block (uvm_va_block.h)

**虚拟地址块** 是最细粒度的管理单元（通常 2MB），是叶子节点：

```c
uvm_va_block_t {
    // 所属的 VA range
    uvm_va_range_t *va_range;

    // 块的虚拟地址范围
    NvU64 start, end;

    // 每个处理器（CPU/GPU）的页面状态
    uvm_va_block_cpu_node_state_t cpu;
    uvm_va_block_gpu_state_t *gpus[UVM_ID_MAX_GPUS];

    // 锁保护块内操作
    uvm_mutex_t lock;

    // GPU 操作追踪器
    uvm_tracker_t tracker;

    // 性能模块的私有数据（抖动检测等）
    uvm_perf_module_data_t perf_modules_data;
}
```

**页面状态追踪**：
- 每个页面（4KB）的状态独立追踪
- 追踪 **residency** (物理位置)、**mapping** (虚拟映射) 和 **permissions** (访问权限)

---

## 3. 内存管理层次结构

### 3.1 三层抽象

```
User Address Space (64-bit VA)
         │
         ▼
    VA Space         ← 进程级，包含所有 VA ranges
         │
         ▼
    VA Range         ← 区域级（mmap 大小），可包含多个 VA blocks
         │
         ▼
    VA Block         ← 页面级（最大 2MB），最小管理单元
         │
         ▼
Physical Pages       ← CPU: struct page*   GPU: GPU physical address
```

### 3.2 页面状态机

每个页面在系统中可能处于以下状态：

```
┌──────────────┐
│  Unallocated │ ← 初始状态：页面尚未分配
└──────┬───────┘
       │ GPU 访问触发页面错误
       ▼
┌──────────────┐
│   Allocated  │ ← 已分配但未映射到任何处理器
└──────┬───────┘
       │ 第一次访问
       ▼
┌──────────────┐
│   Resident   │ ← 物理内存已分配并包含数据
│   on CPU/GPU │    • CPU: resident on system memory
└──────┬───────┘    • GPU: resident on GPU memory
       │
       │ 其他处理器访问
       ▼
┌──────────────┐
│   Mapped     │ ← 在一个或多个处理器上建立虚拟映射
│   (READ)     │    权限：READ / WRITE / ATOMIC
└──────┬───────┘
       │ 抖动检测
       ▼
┌──────────────┐
│   Pinned     │ ← 固定在某个位置以避免抖动
└──────┬───────┘
       │ 取消固定
       ▼
┌──────────────┐
│  Throttled   │ ← 暂时禁止访问以缓解抖动
└──────────────┘
```

### 3.3 页面迁移决策

**何时迁移**：
1. **GPU 页面错误**: GPU 访问不存在的页面
2. **写时拷贝**: 写访问一个只读副本
3. **性能优化**: 基于访问计数器主动迁移
4. **抖动避免**: 固定到公共位置

**迁移目标选择**：
- 优先选择 **请求者** (requester) 的本地内存
- 如果检测到抖动，选择 **中间位置** 或使用 **访问计数器** (access counters)
- 考虑 **preferred location** 策略（用户设置）

---

## 4. GPU 页面错误处理机制

### 4.1 错误类型

NVIDIA GPU 支持两种页面错误：

#### 4.1.1 可重放错误 (Replayable Faults)

- GPU 遇到缺页时 **暂停 warp** 执行
- 驱动修复错误（分配/迁移页面）
- GPU **重放** (replay) 该内存访问指令
- 用于 **统一内存** (Managed Memory)

**特点**：
- 精确的页面级错误信息
- 硬件支持 warp 暂停和重启
- 可以从错误中恢复

#### 4.1.2 不可重放错误 (Non-Replayable Faults)

- GPU 遇到致命错误（权限违规等）
- 无法恢复，导致 context 被销毁
- 用于 **检测非法访问**

### 4.2 可重放错误处理流程（详细）

**源文件**: `uvm_gpu_replayable_faults.c:2906`
**入口函数**: `uvm_parent_gpu_service_replayable_faults()`

#### 调用图（基于 cflow 输出）

```
uvm_parent_gpu_service_replayable_faults()  [主循环]
│
├─► fetch_fault_buffer_entries()          [步骤1: 从硬件缓冲区获取错误]
│   ├─► 从 GPU 的 fault buffer 读取错误条目
│   ├─► cmp_fault_instance_ptr()          [比较错误的 instance pointer]
│   ├─► fetch_fault_buffer_try_merge_entry()  [合并相同页面的错误]
│   │   └─► fetch_fault_buffer_merge_entry()  [合并访问类型]
│   └─► write_get()                       [更新 GET 指针]
│
├─► preprocess_fault_batch()              [步骤2: 预处理错误批次]
│   ├─► sort()                            [按 VA space/GPU/地址排序]
│   │   └─► cmp_sort_fault_entry_by_va_space_gpu_address_access_type()
│   └─► translate_instance_ptrs()         [翻译 instance pointer 到 VA space]
│       ├─► uvm_parent_gpu_fault_entry_to_va_space()
│       ├─► push_cancel_on_gpu()          [对无效错误发送 cancel]
│       └─► fault_buffer_flush_locked()   [刷新缓冲区]
│           └─► push_replay_on_parent_gpu() [发送 replay 命令]
│
├─► service_fault_batch()                 [步骤3: 处理错误批次]
│   ├─► service_fault_batch_dispatch()    [分发到各个 VA block]
│   │   ├─► uvm_va_block_find_create_in_range()  [查找或创建 VA block]
│   │   └─► service_fault_batch_block()   [处理单个 block 的错误]
│   │       ├─► uvm_hmm_migrate_begin_wait()  [HMM: 等待迁移]
│   │       ├─► uvm_mutex_lock()          [锁住 VA block]
│   │       └─► service_fault_batch_block_locked()  [实际处理错误]
│   │           ├─► uvm_perf_thrashing_get_hint()  【抖动检测】
│   │           ├─► block_populate_pages() [分配物理页面]
│   │           ├─► block_copy_resident_pages()  [迁移数据]
│   │           └─► uvm_va_block_map()    [建立页表映射]
│   │
│   └─► service_fault_batch_fatal_notify() [处理致命错误]
│       └─► cancel_faults_all()           [取消所有相关错误]
│
└─► 决定 Replay 策略                       【关键策略点！】
    ├─► POLICY_BATCH:                     [批量重放]
    │   └─► push_replay_on_parent_gpu(UVM_FAULT_REPLAY_TYPE_START)
    │
    ├─► POLICY_BATCH_FLUSH:               [批量刷新重放]
    │   ├─► 如果重复错误比例 > 阈值:
    │   │   └─► flush_mode = UPDATE_PUT   [更新 PUT 指针]
    │   └─► fault_buffer_flush_locked()
    │
    ├─► POLICY_BLOCK:                     [阻塞重放]
    │   └─► （在 service 内部同步重放）
    │
    └─► POLICY_ONCE:                      [只重放一次]
        └─► 在结束时重放一次
```

#### 步骤详解

**步骤 1: 获取错误 (Fetch Faults)**

```c
// 文件: uvm_gpu_replayable_faults.c:844
fetch_fault_buffer_entries(parent_gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY)
```

- 从 GPU 的 **硬件 fault buffer** 读取错误条目
- Fault buffer 是一个 **环形缓冲区**，由 GPU 硬件写入，驱动读取
- 每个条目包含：
  - `fault_address`: 缺页的虚拟地址
  - `fault_type`: READ / WRITE / ATOMIC
  - `instance_ptr`: GPU context 的物理地址
  - `engine_id`, `client_id`, `gpc_id`: 错误源的硬件标识
  - `timestamp`: 硬件时间戳

**优化技巧**：
- **合并相同页面的错误**: 如果多个 warp 访问同一页面，只处理一次
- **批量获取**: 一次获取多个错误以提高效率

**步骤 2: 预处理 (Preprocess)**

```c
// 文件: uvm_gpu_replayable_faults.c:1134
preprocess_fault_batch(parent_gpu, batch_context)
```

- **排序**: 按 VA space → GPU → 地址 → 访问类型排序
  - 好处：局部性，减少锁竞争，批量处理
- **翻译 instance pointer**: 将 GPU 的 context 指针翻译为 `uvm_va_space_t`
- **过滤无效错误**: 对于找不到 VA space 的错误，发送 **cancel** 命令

**步骤 3: 处理错误 (Service)**

```c
// 文件: uvm_gpu_replayable_faults.c:2232
service_fault_batch(parent_gpu, FAULT_SERVICE_MODE_REGULAR, batch_context)
```

核心逻辑（在 `service_fault_batch_block_locked` 中）：

```c
1. 调用抖动检测模块获取 hint:
   hint = uvm_perf_thrashing_get_hint(va_block, address, requester);

2. 根据 hint 决定操作:
   - HINT_NONE: 正常处理（迁移到请求者）
   - HINT_PIN: 固定页面到某个位置
   - HINT_THROTTLE: 暂时拒绝访问

3. 如果需要迁移:
   - 分配目标位置的物理页面
   - 使用 GPU Copy Engine 迁移数据
   - 建立新的页表映射

4. 如果只需要映射:
   - 直接建立页表项（PTE）

5. 更新页面状态:
   - 更新 resident 位图
   - 更新 mapped 位图
   - 记录访问模式
```

**步骤 4: 重放策略 (Replay Policy)** 【eBPF 集成点！】

```c
// 文件: uvm_gpu_replayable_faults.c:2986-3007
// 这是我们设计的 eBPF hook 插入点

if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
    // 策略1: 批量重放 - 处理完一批错误后统一重放
    push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
}
else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
    // 策略2: 批量刷新重放 - 如果重复错误多，刷新 fault buffer
    if (batch_context->num_duplicate_faults * 100 >
        batch_context->num_cached_faults * replay_update_put_ratio) {
        flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
    }
    fault_buffer_flush_locked(...);
}
else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_ONCE) {
    // 策略3: 只重放一次 - 在最后统一重放
    // （延迟到循环结束）
}
```

### 4.3 Replay 命令的硬件实现

当驱动调用 `push_replay_on_parent_gpu()` 时：

```c
// 文件: uvm_gpu_replayable_faults.c:546
push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context)
│
└─► 通过 GPU channel 向硬件发送 method
    - Method: NVC369_CTRL_CMD_MMU_REPLAY_FAULTS
    - 参数: replay_type (START/START_ACK_ALL)
    - 硬件接收后:
      1. 重新执行所有被暂停的 warp
      2. 清除相应的 fault buffer 条目
      3. 如果再次缺页，重新记录到 buffer
```

**Replay 类型**：
- `UVM_FAULT_REPLAY_TYPE_START`: 开始重放
- `UVM_FAULT_REPLAY_TYPE_START_ACK_ALL`: 重放并确认所有错误

---

## 5. 性能优化模块

### 5.1 抖动检测 (Thrashing Detection)

**源文件**: `uvm_perf_thrashing.c:1615`
**入口函数**: `uvm_perf_thrashing_get_hint()`

#### 什么是抖动？

当 CPU 和 GPU 频繁访问同一页面时，页面会在 CPU 内存和 GPU 内存之间反复迁移，这种现象称为 **抖动 (thrashing)**，会严重影响性能。

#### 检测机制（基于 cflow 调用图）

```
uvm_perf_thrashing_get_hint()
│
├─► va_space_thrashing_info_get()        [获取全局抖动状态]
│
├─► thrashing_info_get()                 [获取 block 抖动状态]
│
├─► 检查是否在 throttling 期间
│   ├─► page_thrashing_get_throttling_end_time_stamp()
│   └─► 如果当前时间 < 结束时间:
│       └─► return HINT_THROTTLE          【拒绝访问】
│
├─► 记录新的访问事件
│   ├─► page_thrashing_get_time_stamp()  [获取上次访问时间]
│   ├─► uvm_processor_mask_set()         [记录访问处理器]
│   └─► thrashing_throttle_update()      [更新 throttle 状态]
│
└─► get_hint_for_migration_thrashing()   【核心决策函数】
    │
    ├─► 检查抖动条件:
    │   - 时间窗口内访问次数 > 阈值
    │   - 多个处理器交替访问
    │
    ├─► 如果检测到抖动:
    │   │
    │   ├─► thrashing_processors_can_access()  [所有处理器能否访问某位置]
    │   │
    │   ├─► preferred_location_is_thrashing()  [preferred location 是否抖动]
    │   │
    │   ├─► thrashing_processors_have_fast_access_to()  [检查快速访问路径]
    │   │   └─► 考虑 NUMA、NVLink、PCIe 拓扑
    │   │
    │   ├─► 决策:
    │   │   ├─► 如果存在公共可访问位置:
    │   │   │   └─► return HINT_PIN_TO_LOCATION  【固定到该位置】
    │   │   │
    │   │   └─► 否则:
    │   │       └─► thrashing_throttle_processor()
    │   │           └─► return HINT_THROTTLE      【限流访问】
    │   │
    │   └─► thrashing_pin_page()             [执行固定操作]
    │       ├─► 将页面加入固定列表
    │       ├─► 设置固定时长（参数可调）
    │       └─► 启动定时器到期后解除固定
    │
    └─► return HINT_NONE                     [正常迁移]
```

#### 关键参数（模块参数，可运行时调整）

```c
// 文件: uvm_perf_thrashing.c

// 抖动检测阈值：时间窗口内的访问次数
uvm_perf_thrashing_threshold = 3;  // 默认3次

// 抖动检测时间窗口（微秒）
uvm_perf_thrashing_lapse_usec = 1000;  // 1ms

// 固定页面的时长（微秒）
uvm_perf_thrashing_pin_ns = 100000;  // 100us

// Throttle 时长（微秒）
uvm_perf_thrashing_throttle_us = 1000;  // 1ms
```

#### Hint 返回值含义

```c
typedef enum {
    UVM_PERF_THRASHING_HINT_TYPE_NONE = 0,      // 无抖动，正常迁移
    UVM_PERF_THRASHING_HINT_TYPE_THROTTLE,      // 暂时拒绝访问
    UVM_PERF_THRASHING_HINT_TYPE_PIN,           // 固定到特定位置
    UVM_PERF_THRASHING_HINT_TYPE_MAX
} uvm_perf_thrashing_hint_type_t;

typedef struct {
    uvm_perf_thrashing_hint_type_t type;
    uvm_processor_id_t pin_to_processor_id;  // 固定目标
} uvm_perf_thrashing_hint_t;
```

### 5.2 预取引擎 (Prefetch Engine)

**源文件**: `uvm_perf_prefetch.c`

#### 预取策略

1. **地址局部性预取**: 检测到连续访问时，预取相邻页面
2. **基于访问计数器预取**: GPU 硬件统计访问频率，预取高频页面
3. **自适应阈值**: 根据预取命中率动态调整预取深度

#### 预取触发时机

- GPU 页面错误处理时
- 定期的后台扫描
- 用户显式调用 `cudaMemPrefetchAsync()`

### 5.3 访问计数器 (Access Counters)

**源文件**: `uvm_gpu_access_counters.c`

某些 GPU 架构（Volta+）支持硬件访问计数器：

- 统计每个物理页面的访问次数
- 区分 READ / WRITE
- 用于指导预取和迁移决策
- 避免基于页面错误的被动迁移

### 5.4 性能事件框架

**源文件**: `uvm_perf_events.h`

UVM 定义了一个事件驱动的性能模块框架：

```c
// 事件类型
typedef enum {
    UVM_PERF_EVENT_BLOCK_DESTROY,          // VA block 销毁
    UVM_PERF_EVENT_BLOCK_SHRINK,           // VA block 缩小
    UVM_PERF_EVENT_MODULE_UNLOAD,          // 模块卸载
    UVM_PERF_EVENT_BLOCK_SPLIT,            // VA block 分裂
    UVM_PERF_EVENT_RANGE_DESTROY,          // VA range 销毁
    UVM_PERF_EVENT_RANGE_SHRINK,           // VA range 缩小
    UVM_PERF_EVENT_MIGRATION,              // 页面迁移
    UVM_PERF_EVENT_REVOCATION,             // 权限撤销
    UVM_PERF_EVENT_NUMA_UPGRADE,           // NUMA 升级
    UVM_PERF_EVENT_COUNT
} uvm_perf_event_t;
```

**回调机制**：

```c
// 注册性能模块
uvm_perf_register_module(&thrashing_module, callbacks);

// 当事件发生时，UVM 调用所有注册的回调
uvm_perf_event_notify(&block->perf_modules_data, event_type, &event_data);
```

**现有性能模块**：
1. **抖动检测模块** (`uvm_perf_thrashing.c`)
2. **预取模块** (`uvm_perf_prefetch.c`)
3. **访问计数器模块** (`uvm_gpu_access_counters.c`)

---

## 6. 策略框架与可扩展性

### 6.1 已有的策略接口（150+ 个）

根据我们之前的分析（`ALL_POLICY_INTERFACES.md`），UVM 已经暴露了大量策略接口：

#### 6.1.1 Fault Replay 策略（核心）

```c
// 文件: uvm_gpu_replayable_faults.h:34-51
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,      // 阻塞：每次处理后立即重放
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,          // 批量：积累后统一重放
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,    // 批量刷新：带智能刷新
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,           // 只重放一次
    // [eBPF 扩展点] 可以在这里添加:
    // UVM_PERF_FAULT_REPLAY_POLICY_EBPF = 4,
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;

// 运行时可通过模块参数修改:
// /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_policy
```

#### 6.1.2 VA Policy 策略

```c
// 文件: uvm_va_policy.h
typedef enum {
    UVM_READ_DUPLICATION_UNSET,      // 未设置
    UVM_READ_DUPLICATION_ENABLED,    // 启用读复制
    UVM_READ_DUPLICATION_DISABLED,   // 禁用读复制
    UVM_READ_DUPLICATION_MAX
} uvm_read_duplication_policy_t;

// Per-range 策略
uvm_va_policy_t {
    uvm_processor_id_t preferred_location;    // 优先位置
    uvm_processor_id_t accessed_by[MAX_PROCESSORS];  // 访问处理器列表
    uvm_read_duplication_policy_t read_duplication;  // 读复制策略
}
```

#### 6.1.3 迁移原因（用于策略决策）

```c
// 文件: uvm_migrate.h
typedef enum {
    UVM_MIGRATE_CAUSE_INVALID = 0,
    UVM_MIGRATE_CAUSE_FAULT,             // 页面错误触发
    UVM_MIGRATE_CAUSE_EVICTION,          // 驱逐
    UVM_MIGRATE_CAUSE_PREFETCH,          // 预取
    UVM_MIGRATE_CAUSE_ACCESS_COUNTER,    // 访问计数器
    UVM_MIGRATE_CAUSE_USER,              // 用户显式调用
    UVM_MIGRATE_CAUSE_NON_REPLAYABLE_FAULT,  // 不可重放错误
    UVM_MIGRATE_CAUSE_ATS_PAGEABLE,      // ATS 可分页
    UVM_MIGRATE_CAUSE_ATS_EVICTION,      // ATS 驱逐
    UVM_MIGRATE_CAUSE_COUNT
} uvm_migrate_cause_t;
```

### 6.2 eBPF Struct_ops 集成设计（最小化修改）

基于我们的 `MINIMAL_EBPF_INTEGRATION.md` 设计：

#### 修改点1: 添加策略枚举（1行）

```c
// 文件: uvm_gpu_replayable_faults.h:50
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,
    UVM_PERF_FAULT_REPLAY_POLICY_EBPF = 4,  // [+] 新增
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;
```

#### 修改点2: 定义 struct_ops 结构（~20行）

```c
// 文件: uvm_gpu_replayable_faults.h（新增）
struct uvm_fault_policy_ops {
    // 决定 replay 策略
    // 返回值: 1=立即replay, 0=使用默认策略, -1=延迟replay
    int (*decide_replay)(struct uvm_fault_event_data *event, int default_policy);
};

// 全局 ops 指针（由 eBPF 程序设置）
extern struct uvm_fault_policy_ops __rcu *g_fault_policy_ops;
```

#### 修改点3: 调用 hook（~10行）

```c
// 文件: uvm_gpu_replayable_faults.c:2986
if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_EBPF) {
    struct uvm_fault_policy_ops *ops;

    rcu_read_lock();
    ops = rcu_dereference(g_fault_policy_ops);
    if (ops && ops->decide_replay) {
        int decision = ops->decide_replay(&event_data, default_policy);
        if (decision > 0) {
            rcu_read_unlock();
            status = push_replay_on_parent_gpu(...);
            goto replay_done;
        }
        // decision = 0: 继续执行原逻辑
        // decision < 0: 延迟到循环结束
    }
    rcu_read_unlock();
}
// 继续原有的策略代码（未删除任何内容）
```

#### eBPF 程序示例

```c
// 用户空间 eBPF 程序
SEC("struct_ops/uvm_fault_policy")
int BPF_PROG(decide_replay, struct uvm_fault_event_data *event, int default_policy)
{
    // 自定义策略逻辑
    if (event->num_duplicate_faults > event->num_cached_faults / 2) {
        // 重复错误过多，立即 replay
        return 1;
    }

    if (event->num_cached_faults < 10) {
        // 错误太少，延迟 replay
        return -1;
    }

    // 使用默认策略
    return 0;
}

SEC(".struct_ops")
struct uvm_fault_policy_ops custom_fault_policy = {
    .decide_replay = (void *)decide_replay,
};
```

### 6.3 HAL (Hardware Abstraction Layer)

**源文件**: `uvm_hal.c`

UVM 使用函数指针表来支持不同 GPU 架构：

```c
// 每个 GPU 架构有自己的 HAL 实现
uvm_parent_gpu_t {
    uvm_arch_hal_t *arch_hal;  // 架构相关函数指针
}

// HAL 函数示例
struct uvm_arch_hal {
    // Fault buffer 操作
    NV_STATUS (*get_fault_buffer_entry)(uvm_parent_gpu_t *gpu, ...);
    void (*enable_prefetch_faults)(uvm_parent_gpu_t *gpu);
    void (*disable_prefetch_faults)(uvm_parent_gpu_t *gpu);

    // MMU 操作
    void (*make_pte)(uvm_page_tree_t *tree, ...);
    void (*make_pde)(uvm_page_tree_t *tree, ...);

    // Copy Engine 操作
    void (*memcopy)(uvm_push_t *push, ...);
    void (*memset)(uvm_push_t *push, ...);

    // Access Counter 操作（Volta+）
    NV_STATUS (*init_access_counter_info)(uvm_parent_gpu_t *gpu);
};

// 支持的架构
- Maxwell (GM10x, GM20x)
- Pascal (GP100, GP10x)
- Volta (GV100)
- Turing (TU10x, TU11x)
- Ampere (GA100, GA10x)
- Ada Lovelace (AD10x)
- Hopper (GH100)
- Blackwell (GB100)
```

---

## 7. 关键数据流

### 7.1 GPU 访问统一内存的完整流程

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. GPU 执行 kernel，warp 访问统一内存地址                           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. GPU MMU 查询页表                                                │
│    - 如果命中 → 直接访问物理地址                                    │
│    - 如果缺失 → 触发页面错误                                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │ 缺页
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. GPU 硬件操作                                                     │
│    - 暂停 warp 执行                                                │
│    - 将错误信息写入 fault buffer                                   │
│    - 触发中断通知驱动                                              │
└────────────────────────┬─────────────────────────────────────────┘
                         │ 中断
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. UVM 驱动 ISR (中断服务程序)                                      │
│    - uvm_isr_top_half() 确认中断                                   │
│    - 调度 bottom half (工作队列)                                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │ 调度
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. UVM 驱动 Bottom Half                                            │
│    uvm_parent_gpu_service_replayable_faults()                     │
│    │                                                               │
│    ├─► fetch_fault_buffer_entries()      [读取错误]                │
│    ├─► preprocess_fault_batch()          [预处理]                 │
│    ├─► service_fault_batch()             [处理错误]                │
│    │   │                                                           │
│    │   ├─► 查找 VA space 和 VA block                              │
│    │   ├─► uvm_perf_thrashing_get_hint() [抖动检测]                │
│    │   ├─► 决定迁移/映射策略                                       │
│    │   ├─► 分配物理页面（如需要）                                  │
│    │   ├─► 迁移数据（使用 GPU Copy Engine）                        │
│    │   ├─► 建立页表映射                                            │
│    │   └─► 更新页面状态                                            │
│    │                                                               │
│    └─► push_replay_on_parent_gpu()       【策略决策点】            │
└────────────────────────┬─────────────────────────────────────────┘
                         │ Replay 命令
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. GPU 硬件接收 Replay 命令                                         │
│    - 恢复被暂停的 warp                                             │
│    - 重新执行内存访问指令                                          │
│    - 这次页表已更新，访问成功                                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │ 成功
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 7. Warp 继续执行                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 CPU 访问统一内存的流程

```
CPU 访问统一内存地址
         │
         ▼
    MMU 查询页表
         │
         ├─► 命中 → 直接访问
         │
         └─► 缺失 → 触发 page fault
                    │
                    ▼
            Linux 缺页处理程序
                    │
                    ▼
          uvm_va_space_mm.c 的回调
                    │
                    ├─► 查找 VA block
                    ├─► 检查是否在 GPU 上
                    ├─► 迁移页面到 CPU（如需要）
                    ├─► 建立 CPU 页表映射
                    └─► 返回
                    │
                    ▼
          CPU 重新执行访问指令
```

### 7.3 页面迁移的硬件操作

```c
// 使用 GPU Copy Engine 迁移页面
// 文件: uvm_migrate.c

1. 分配目标物理页面
   ├─► CPU → GPU: uvm_pmm_gpu_alloc()
   └─► GPU → CPU: alloc_page()

2. 使用 GPU Copy Engine 拷贝数据
   ├─► uvm_push_begin_acquire()         [开始 GPU 操作]
   ├─► gpu->parent->ce_hal->memcopy()   [调用 HAL copy 函数]
   │   └─► 生成 GPU method 到 channel
   ├─► uvm_push_end()                   [结束 GPU 操作]
   └─► uvm_tracker_add_push()           [追踪 GPU 操作]

3. 等待拷贝完成（异步或同步）
   └─► uvm_tracker_wait()

4. 更新页表
   ├─► 使旧映射无效
   ├─► 建立新映射
   └─► TLB flush

5. 更新页面状态
   ├─► 清除旧位置的 resident 位
   ├─► 设置新位置的 resident 位
   └─► 释放旧物理页面
```

---

## 8. 与 Linux 内核集成

### 8.1 设备注册

```c
// 文件: uvm.c

static int __init uvm_init(void)
{
    // 1. 分配设备号
    alloc_chrdev_region(&g_uvm_base_dev, 0, 1, NVIDIA_UVM_DEVICE_NAME);

    // 2. 创建字符设备
    cdev_init(&g_uvm_cdev, &uvm_fops);
    cdev_add(&g_uvm_cdev, g_uvm_base_dev, 1);

    // 3. 创建 /dev/nvidia-uvm 设备节点
    device_create(nvidia_uvm_class, NULL, g_uvm_base_dev, NULL, NVIDIA_UVM_DEVICE_NAME);

    // 4. 初始化全局状态
    uvm_global_init();

    return 0;
}

module_init(uvm_init);
```

### 8.2 文件操作

```c
// 文件: uvm.c

static const struct file_operations uvm_fops = {
    .open           = uvm_open,         // 打开 /dev/nvidia-uvm
    .release        = uvm_release,      // 关闭
    .unlocked_ioctl = uvm_ioctl,        // ioctl 调用
    .compat_ioctl   = uvm_ioctl,        // 32位兼容
    .mmap           = uvm_mmap,         // mmap 映射
};

// ioctl 命令（用户空间 API）
#define UVM_INITIALIZE                      0x01
#define UVM_CREATE_RANGE_GROUP              0x06
#define UVM_SET_PREFERRED_LOCATION          0x0c
#define UVM_SET_ACCESSED_BY                 0x0d
#define UVM_SET_READ_DUPLICATION            0x0e
#define UVM_MIGRATE                         0x0f
#define UVM_ENABLE_PEER_ACCESS              0x11
#define UVM_DISABLE_PEER_ACCESS             0x12
#define UVM_MAP_EXTERNAL_ALLOCATION         0x13
// ... 还有很多
```

### 8.3 与 mm 子系统集成

```c
// 文件: uvm_va_space_mm.c

// 注册 mmu_notifier 回调
static const struct mmu_notifier_ops uvm_mmu_notifier_ops = {
    .invalidate_range_start = uvm_mmu_notifier_invalidate_range_start,
    .invalidate_range_end   = uvm_mmu_notifier_invalidate_range_end,
    .release                = uvm_mmu_notifier_release,
};

// 当 Linux 内核修改页表时，通知 UVM
uvm_mmu_notifier_invalidate_range_start(...)
{
    // 1. 阻止新的 GPU 操作
    // 2. 等待正在进行的 GPU 操作完成
    // 3. 使相关 GPU 页表项无效
    // 4. Flush GPU TLB
}
```

### 8.4 内存分配

```c
// 文件: uvm_kvmalloc.c

// UVM 优先使用 kvmalloc（先 kmalloc，失败则 vmalloc）
void *uvm_kvmalloc(size_t size)
{
    if (size <= UVM_KVMALLOC_THRESHOLD)
        return kmalloc(size, GFP_KERNEL);
    else
        return __vmalloc(size, GFP_KERNEL, PAGE_KERNEL);
}

// 内存池
- VA blocks: kmem_cache (slab allocator)
- GPU 物理页面: PMM (Physical Memory Manager)
- CPU 物理页面: alloc_page()
```

---

## 9. 可观测性与调试

### 9.1 模块参数（sysfs）

UVM 暴露了 50+ 个模块参数，可运行时修改：

```bash
# 所有参数位于 /sys/module/nvidia_uvm/parameters/

# 错误处理策略
/sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_policy
  - 0: BLOCK
  - 1: BATCH (默认)
  - 2: BATCH_FLUSH
  - 3: ONCE

# 抖动检测参数
/sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_threshold
/sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_lapse_usec
/sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_pin_ns

# 性能调优
/sys/module/nvidia_uvm/parameters/uvm_perf_fault_max_batches_per_service
/sys/module/nvidia_uvm/parameters/uvm_perf_fault_max_throttle_per_service
/sys/module/nvidia_uvm/parameters/uvm_perf_reenable_prefetch_faults_lapse_msec

# 调试开关
/sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests
/sys/module/nvidia_uvm/parameters/uvm_enable_debug_procfs
```

### 9.2 性能计数器

```c
// 文件: uvm_gpu.h

typedef struct {
    NvU64 num_replayable_faults;          // 可重放错误数
    NvU64 num_duplicates;                  // 重复错误数
    NvU64 num_pages_in;                    // 迁入页面数
    NvU64 num_pages_out;                   // 迁出页面数
    NvU64 num_thrashing_detected;          // 检测到的抖动次数
    NvU64 num_throttled;                   // 限流次数
    NvU64 num_pinned;                      // 固定页面数
    // ... 还有很多
} uvm_perf_counters_t;

// 访问计数器：通过 ioctl UVM_TOOLS_READ_PROCESS_MEMORY_STATS
```

### 9.3 Tools API

```c
// 文件: uvm_tools.h

// UVM 提供了一套 tools API 用于监控和调试
// NVIDIA Nsight Systems 和 Nsight Compute 使用这些 API

// 事件类型
UvmEventTypeGpuFault                 // GPU 页面错误
UvmEventTypeGpuFaultReplay           // Replay 操作
UvmEventTypeMigration                // 页面迁移
UvmEventTypeThrashing                // 抖动检测
UvmEventTypeThrottling               // 限流操作
UvmEventTypeMap                      // 页表映射
UvmEventTypeUnmap                    // 页表取消映射
// ...

// 注册回调接收事件
uvm_tools_record_event(...)
```

### 9.4 Procfs / Debugfs

```c
// 文件: uvm_procfs.c

// 创建 /proc/driver/nvidia-uvm/ 目录
// 每个 VA space 有自己的目录：/proc/driver/nvidia-uvm/<pid>/

// 文件内容示例：
- stats: 性能统计
- fault_stats: 错误统计
- migration_stats: 迁移统计
- thrashing_stats: 抖动统计
- gpu_fault_stats: GPU 错误详情
```

### 9.5 内核日志

```c
// 调试宏
UVM_DBG_PRINT()      // 调试打印
UVM_INFO_PRINT()     // 信息打印
UVM_ERR_PRINT()      // 错误打印
UVM_ASSERT()         // 断言

// 示例日志
[nvidia-uvm] GPU fault detected: address=0x7f1234567000, type=WRITE
[nvidia-uvm] Migration: CPU → GPU, count=256 pages
[nvidia-uvm] Thrashing detected, pinning to CPU
```

---

## 10. 总结与设计思想

### 10.1 核心设计原则

1. **延迟绑定 (Lazy Binding)**
   - 只在真正访问时才分配和迁移页面
   - 减少不必要的内存开销

2. **访问局部性 (Access Locality)**
   - 尽量让数据靠近访问者
   - 减少跨总线传输

3. **批量处理 (Batching)**
   - 合并页面错误
   - 批量迁移和映射
   - 减少硬件操作开销

4. **策略可调 (Tunable Policy)**
   - 50+ 模块参数
   - 运行时可调整
   - 适应不同工作负载

5. **分层抽象 (Layered Abstraction)**
   - VA Space → VA Range → VA Block
   - 清晰的职责划分

6. **硬件抽象 (Hardware Abstraction)**
   - HAL 层屏蔽架构差异
   - 支持多代 GPU 架构

7. **异步并发 (Async Concurrency)**
   - 使用 GPU Copy Engine 异步拷贝
   - Tracker 机制追踪依赖关系
   - 细粒度锁减少竞争

### 10.2 性能优化技术

1. **智能迁移决策**
   - 抖动检测避免乒乓
   - 访问计数器指导迁移
   - 考虑拓扑（NUMA/NVLink）

2. **预取引擎**
   - 地址局部性预测
   - 自适应阈值
   - 减少主动缺页

3. **重复错误过滤**
   - 合并同页面错误
   - 避免重复处理

4. **限流机制**
   - 暂时拒绝访问
   - 避免系统过载

5. **读复制**
   - 多处理器读同一页面
   - 建立多个副本
   - 减少迁移

### 10.3 可扩展性

#### 现有扩展点

1. **性能模块框架**
   - 注册回调接收事件
   - 提供 hint 影响决策
   - 抖动检测、预取都是模块

2. **HAL 层**
   - 添加新 GPU 架构
   - 只需实现函数指针表

3. **策略参数**
   - 50+ 参数调优
   - 无需重新编译

#### eBPF 扩展设计

基于我们的 `MINIMAL_EBPF_INTEGRATION.md`：

- **最小化修改**: 只需 50 行代码
- **保留原有行为**: 所有原策略保持不变
- **灵活决策**: eBPF 程序可访问完整上下文
- **动态加载**: 运行时加载/卸载 eBPF 程序
- **安全隔离**: eBPF 验证器保证安全性

### 10.4 与 WIC 论文的关联

根据我们的 `WIC_INTEGRATION_ANALYSIS.md`，WIC 论文的优化可以映射到 UVM 的这些组件：

| WIC 组件 | UVM 对应组件 | 集成方式 |
|---------|------------|---------|
| **Interrupter** | Fault replay policy | 修改 replay 策略决策点 |
| **Monitor** | Thrashing detection | 扩展抖动检测模块 |
| **Warp Scheduling** | GPU 硬件 | 需要硬件支持 |
| **Policy Engine** | eBPF struct_ops | 动态策略框架 |

### 10.5 未来改进方向

1. **机器学习驱动的策略**
   - 使用历史数据预测访问模式
   - 自适应调整参数
   - eBPF + 用户空间 ML 模型

2. **更细粒度的控制**
   - Per-application 策略
   - Per-kernel 策略
   - 基于性能计数器的实时调整

3. **多 GPU 优化**
   - 更智能的 P2P 迁移
   - 考虑 NVSwitch 拓扑
   - 负载均衡

4. **与调度器集成**
   - CPU 调度器感知 GPU 状态
   - GPU 调度器感知 CPU 状态
   - 协同优化

---

## 附录

### A. 文件结构

```
kernel-open/nvidia-uvm/
├── uvm.c                           # 模块入口
├── uvm_global.c                    # 全局状态管理
├── uvm_gpu.c                       # GPU 管理
├── uvm_gpu_replayable_faults.c     # 可重放错误处理 【核心】
├── uvm_gpu_non_replayable_faults.c # 不可重放错误
├── uvm_va_space.c                  # VA space 管理
├── uvm_va_range.c                  # VA range 管理
├── uvm_va_block.c                  # VA block 管理 【核心】
├── uvm_migrate.c                   # 页面迁移
├── uvm_perf_thrashing.c            # 抖动检测 【性能模块】
├── uvm_perf_prefetch.c             # 预取引擎 【性能模块】
├── uvm_perf_events.c               # 事件框架
├── uvm_hal.c                       # HAL 层
├── uvm_maxwell*.c                  # Maxwell 架构实现
├── uvm_pascal*.c                   # Pascal 架构实现
├── uvm_volta*.c                    # Volta 架构实现
├── uvm_turing*.c                   # Turing 架构实现
├── uvm_ampere*.c                   # Ampere 架构实现
├── uvm_ada*.c                      # Ada 架构实现
├── uvm_hopper*.c                   # Hopper 架构实现
├── uvm_blackwell*.c                # Blackwell 架构实现
├── uvm_tools.c                     # Tools API
├── uvm_procfs.c                    # Procfs 接口
└── uvm_hmm.c                       # HMM 集成
```

### B. 关键数据结构大小

```
sizeof(uvm_va_space_t)       ≈ 2KB
sizeof(uvm_va_range_t)       ≈ 512B
sizeof(uvm_va_block_t)       ≈ 4KB (可变)
sizeof(uvm_gpu_t)            ≈ 8KB
sizeof(page_thrashing_info_t) = 16B
```

### C. 调用图文件位置

```
cflow 输出:
- cflow_output/fault_service.txt     # 错误处理调用图 (336行)
- cflow_output/thrashing.txt         # 抖动检测调用图 (154行)
- cflow_output/replay_callers.txt    # Replay 调用者 (135行)

doxygen 输出:
- doxygen_output/html/index.html    # 交互式文档 (655MB)
```

### D. 参考资料

- CUDA Unified Memory Programming Guide
- NVIDIA GPU Architecture Whitepapers
- Linux kernel mmu_notifier documentation
- WIC: Warp-level Interrupt-based Communication (ATC'25)
- 本仓库的分析文档:
  - `WIC_INTEGRATION_ANALYSIS.md`
  - `MINIMAL_EBPF_INTEGRATION.md`
  - `ALL_POLICY_INTERFACES.md`

---

**文档结束**

此文档基于 cflow 调用图分析和源代码深度阅读，提供了 NVIDIA UVM 驱动模块的完整工作原理剖析。
适用于想要理解 GPU 统一内存机制、进行性能优化、或集成自定义策略（如 eBPF）的开发者。
