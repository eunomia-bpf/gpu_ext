# Policy Interface Summary - What You Asked For

你问的："已经有什么 policy 的interface暴露出来了？"

## 简短回答 (Short Answer)

找到了 **150+ 个 policy interfaces**！主要分为以下几类：

1. **Fault Replay Policy** (最关键) - 4种现有策略 + 可以加 EBPF 策略
2. **VA Memory Policy** - preferred_location, accessed_by, read_duplication
3. **Thrashing Policy** - PIN/THROTTLE hints, 8个可调参数
4. **Migration Policy** - 9种 cause types, transfer modes
5. **50+ Module Parameters** - 所有都可以通过 sysfs 运行时调整

## 详细文档 (Detailed Documentation)

我创建了3个文档：

### 1. `MINIMAL_EBPF_INTEGRATION.md` (最小集成方案)
- **已经暴露的接口**：
  - ✅ `uvm_perf_fault_replay_policy` enum (最关键！)
  - ✅ 50+ module parameters
  - ✅ Complete event callback system
  - ✅ Thrashing hint API
  - ✅ Perf module type system

- **最小修改**：
  - 只加 **1个enum值**: `UVM_PERF_FAULT_REPLAY_POLICY_EBPF = 4`
  - 只加 **1个callback**: `int (*decide_replay)(fault_data, default_policy)`
  - **0行删除** - 所有现有代码作为默认行为

### 2. `MINIMAL_PATCH.md` (具体代码修改)
- 精确到行号的代码修改
- 总共约50行新增代码
- 关键决策点: `uvm_gpu_replayable_faults.c:2986`

### 3. `ALL_POLICY_INTERFACES.md` (完整策略接口目录)
- **30个主要类别**，包含：
  - Fault replay policy (8种变体)
  - VA policy (5种类型)
  - Migration policy (9种 causes)
  - Thrashing policy (3种 hints + 8个参数)
  - Access counter policy
  - GPU caching policy
  - Memory barrier policy
  - HAL function pointers (60+)
  - Perf event system (9种事件)
  - 等等...

---

## 核心发现 (Key Findings)

### 1. Fault Replay Policy (WIC论文最相关)

**文件**: `uvm_gpu_replayable_faults.h:34-51`

```c
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,        // 每个block后replay
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,            // 每个batch后replay
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,      // Batch + flush (默认)
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,             // 最后replay一次
    // [+] 可以在这里加: UVM_PERF_FAULT_REPLAY_POLICY_EBPF = 4,
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;
```

**Module Parameter**:
```bash
# 运行时可调！
cat /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_policy
echo 4 > /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_policy  # 设置为EBPF模式
```

**决策点**: Line 2986
```c
// 这里判断用哪个replay policy
if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
    // 策略1: batch后replay
} else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
    // 策略2: batch+flush
    // 这里还有一个policy决策点！
    if (batch_context->num_duplicate_faults * 100 >
        batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
        flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;  // 更新PUT指针
    }
}
// [+] 可以在这里加 EBPF 策略判断
```

### 2. Thrashing Detection Policy (已经完全暴露！)

**文件**: `uvm_perf_thrashing.h:33-74`

```c
typedef enum {
    UVM_PERF_THRASHING_HINT_TYPE_NONE     = 0,  // 无thrashing
    UVM_PERF_THRASHING_HINT_TYPE_PIN      = 1,  // PIN策略：远程映射避免迁移
    UVM_PERF_THRASHING_HINT_TYPE_THROTTLE = 2,  // THROTTLE策略：限流
} uvm_perf_thrashing_hint_type_t;

typedef struct {
    uvm_perf_thrashing_hint_type_t type;
    union {
        struct {
            uvm_processor_id_t residency;      // PIN到哪个处理器
            uvm_processor_mask_t processors;   // 哪些处理器需要映射
        } pin;
        struct {
            NvU64 end_time_stamp;              // 限流到什么时候 (ns)
        } throttle;
    };
} uvm_perf_thrashing_hint_t;
```

**API函数** (已经存在！):
```c
// 查询thrashing hint - 这个函数已经可以用！
uvm_perf_thrashing_hint_t uvm_perf_thrashing_get_hint(
    uvm_va_block_t *va_block,
    uvm_va_block_context_t *va_block_context,
    NvU64 address,
    uvm_processor_id_t requester
);
```

**Module Parameters** (8个可调参数):
```c
uvm_perf_thrashing_enable = 1         // 启用/禁用
uvm_perf_thrashing_threshold = 3      // 检测阈值
uvm_perf_thrashing_pin_threshold = 10 // PIN阈值
uvm_perf_thrashing_lapse_usec = 10000 // 时间窗口(10ms)
uvm_perf_thrashing_nap = 100          // NAP策略
uvm_perf_thrashing_epoch = 3          // Epoch长度
uvm_perf_thrashing_pin = 1            // PIN策略开关
uvm_perf_thrashing_max_resets = 200   // 最大重置次数
```

### 3. VA Policy (用户可见的内存策略)

**文件**: `uvm_va_policy.h:36-82`

```c
struct uvm_va_policy_struct {
    uvm_read_duplication_policy_t read_duplication;  // 读复制策略
    uvm_processor_id_t preferred_location;           // 首选位置
    int preferred_nid;                               // NUMA节点ID
    uvm_processor_mask_t accessed_by;                // 访问处理器掩码
};
```

**用户API** (通过ioctl，已经存在):
```c
// 设置首选位置
NV_STATUS uvm_api_set_preferred_location(params, filp);
// 设置访问处理器
NV_STATUS uvm_api_set_accessed_by(params, filp);
// 启用读复制
NV_STATUS uvm_api_enable_read_duplication(params, filp);
```

### 4. Migration Cause (迁移原因策略)

**文件**: `uvm_va_block_types.h:116-129`

```c
typedef enum {
    UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT,      // GPU可重放fault
    UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT,  // GPU不可重放fault
    UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER,        // 访问计数器触发
    UVM_MAKE_RESIDENT_CAUSE_PREFETCH,              // 预取
    UVM_MAKE_RESIDENT_CAUSE_EVICTION,              // 驱逐
    UVM_MAKE_RESIDENT_CAUSE_API_TOOLS,             // 工具API
    UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE,           // 迁移API
    UVM_MAKE_RESIDENT_CAUSE_API_SET_RANGE_GROUP,   // 范围组API
    UVM_MAKE_RESIDENT_CAUSE_API_HINT,              // 提示API
    UVM_MAKE_RESIDENT_CAUSE_MAX
} uvm_make_resident_cause_t;
```

**用途**: 传递给所有迁移操作，可在perf events中查询

### 5. Perf Event System (回调框架，已完全实现)

**文件**: `uvm_perf_events.h:49-79`

```c
typedef enum {
    UVM_PERF_EVENT_FAULT,       // ⭐ GPU/CPU fault事件
    UVM_PERF_EVENT_MIGRATION,   // ⭐ 页面迁移事件
    UVM_PERF_EVENT_REVOCATION,  // ⭐ 权限撤销事件
    // ... 更多事件类型
} uvm_perf_event_t;
```

**回调注册** (已经存在的API):
```c
// 注册callback
NV_STATUS uvm_perf_register_event_callback(
    uvm_perf_va_space_events_t *va_space_events,
    uvm_perf_event_t event_id,
    uvm_perf_event_callback_t callback  // 函数指针！
);

// 回调函数类型
typedef void (*uvm_perf_event_callback_t)(
    uvm_perf_event_t event_id,
    uvm_perf_event_data_t *event_data  // 包含fault/migration/revocation完整上下文
);
```

### 6. GPU Buffer Flush Mode (缓冲区刷新策略)

**文件**: `uvm_gpu.h:1819-1822`

```c
typedef enum {
    UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT,        // 使用缓存的PUT指针(快)
    UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,        // 更新PUT指针(慢，但少重复)
    UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,   // 等待+更新
} uvm_gpu_buffer_flush_mode_t;
```

**决策点**: Line 2995
```c
// 这是一个policy决策点！
if (batch_context->num_duplicate_faults * 100 >
    batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
    flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
}
```

---

## Module Parameters完整列表 (50+个)

所有这些都可以通过sysfs运行时调整！

### Fault相关 (最关键)
```bash
/sys/module/nvidia_uvm/parameters/
├── uvm_perf_fault_replay_policy          # 0-3 (可加4=EBPF)
├── uvm_perf_fault_batch_count            # 256 (默认)
├── uvm_perf_fault_replay_update_put_ratio # 50 (%)
├── uvm_perf_fault_max_batches_per_service # 20
├── uvm_perf_fault_max_throttle_per_service # 5
└── uvm_perf_fault_coalesce               # 1 (启用)
```

### Thrashing相关
```bash
├── uvm_perf_thrashing_enable             # 1
├── uvm_perf_thrashing_threshold          # 3
├── uvm_perf_thrashing_pin_threshold      # 10
├── uvm_perf_thrashing_lapse_usec         # 10000
├── uvm_perf_thrashing_nap                # 100
├── uvm_perf_thrashing_epoch              # 3
├── uvm_perf_thrashing_pin                # 1
└── uvm_perf_thrashing_max_resets         # 200
```

### Prefetch相关
```bash
├── uvm_perf_prefetch_enable              # 1
├── uvm_perf_prefetch_threshold           # 51 (%)
└── uvm_perf_prefetch_min_faults          # 2
```

### Access Counter相关
```bash
├── uvm_perf_access_counter_migration_enable # 1
├── uvm_perf_access_counter_batch_count   # 64
└── uvm_perf_access_counter_threshold     # 16
```

### Migration相关
```bash
├── uvm_perf_migrate_cpu_preunmap_enable  # 1
└── uvm_perf_migrate_cpu_preunmap_block_order # 2
```

### GPU Cache相关
```bash
├── uvm_exp_gpu_cache_peermem             # 0 (实验性)
└── uvm_exp_gpu_cache_sysmem              # 0 (实验性)
```

### 其他
```bash
├── uvm_ats_mode                          # 0 (ATS支持)
├── uvm_page_table_location               # "auto"
├── uvm_peer_copy                         # "phys"
├── uvm_disable_hmm                       # false
└── uvm_fault_force_sysmem                # 0
```

---

## 如何使用这些Policy接口？

### 方法1: 直接使用现有的Module Parameters (最简单)

```bash
# 查看当前replay policy
cat /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_policy

# 修改batch size
echo 512 > /sys/module/nvidia_uvm/parameters/uvm_perf_fault_batch_count

# 调整thrashing阈值
echo 5 > /sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_threshold
```

### 方法2: 使用现有的Perf Event Callbacks

```c
// 注册一个callback来监控所有GPU faults
void my_fault_callback(uvm_perf_event_t event_id, uvm_perf_event_data_t *event_data) {
    if (event_id == UVM_PERF_EVENT_FAULT) {
        // 访问fault信息
        uvm_va_block_t *va_block = event_data->fault.block;
        uvm_processor_id_t gpu_id = event_data->fault.proc_id;
        uvm_fault_buffer_entry_t *entry = event_data->fault.gpu.buffer_entry;

        // 自定义逻辑...
    }
}

// 注册callback
uvm_perf_register_event_callback(va_space_events,
                                 UVM_PERF_EVENT_FAULT,
                                 my_fault_callback);
```

### 方法3: 使用Thrashing Hint API

```c
// 查询是否需要PIN或THROTTLE
uvm_perf_thrashing_hint_t hint = uvm_perf_thrashing_get_hint(
    va_block, va_block_context, address, requester
);

if (hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
    // PIN策略：映射到hint.pin.residency
    uvm_processor_id_t pin_location = hint.pin.residency;
    uvm_processor_mask_t *processors = &hint.pin.processors;
    // 执行PIN操作...
}
else if (hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
    // THROTTLE策略：等待到hint.throttle.end_time_stamp
    NvU64 wait_until = hint.throttle.end_time_stamp;
    // 执行限流操作...
}
```

### 方法4: 添加eBPF struct_ops (最小修改)

```c
// 1. 添加一个enum值
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,
    UVM_PERF_FAULT_REPLAY_POLICY_EBPF,  // [+] 新增
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;

// 2. 定义struct_ops
struct uvm_fault_policy_ops {
    int (*decide_replay)(uvm_perf_event_data_t *fault_data, int default_policy);
};

// 3. 在决策点调用
if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_EBPF) {
    ops = rcu_dereference(g_fault_policy_ops);
    if (ops && ops->decide_replay) {
        int decision = ops->decide_replay(&event_data, default_policy);
        if (decision > 0) {
            // eBPF说：立即replay
            push_replay_on_parent_gpu(...);
        } else if (decision < 0) {
            // eBPF说：推迟replay
        }
        // decision == 0: 使用默认策略
    }
}
// 原有代码保持不变，作为默认行为
```

---

## 对WIC论文的映射

WIC论文的三个模块可以直接使用现有接口：

### 1. Interrupter (中断器) - 对应fault replay policy
- **现有接口**: `uvm_perf_fault_replay_policy`
- **决策点**: Line 2986 in `uvm_parent_gpu_service_replayable_faults()`
- **最小修改**: 添加 `UVM_PERF_FAULT_REPLAY_POLICY_EBPF = 4`

### 2. Monitor (监控器) - 对应thrashing detection
- **现有接口**: `uvm_perf_thrashing_get_hint()` API (已完全实现！)
- **策略类型**: PIN (远程映射) / THROTTLE (限流)
- **参数**: 8个module parameters可调

### 3. Activator (激活器) - 对应prefetch & access counters
- **现有接口**:
  - Prefetch: `uvm_perf_prefetch_enable`, `uvm_perf_prefetch_threshold`
  - Access counters: `uvm_perf_access_counter_migration_enable`

---

## 总结 (Summary)

### 你问的"已经有什么policy的interface暴露出来了？"

**答案**：非常多！包括：

1. ✅ **Fault replay policy** - 4种策略，可加eBPF (最小修改)
2. ✅ **Thrashing detection** - 完整的PIN/THROTTLE hint API
3. ✅ **VA memory policy** - preferred_location, accessed_by, read_duplication
4. ✅ **Migration policy** - 9种cause types
5. ✅ **50+ module parameters** - 所有运行时可调
6. ✅ **Perf event callbacks** - 9种事件类型，完整的注册API
7. ✅ **HAL function pointers** - 60+个硬件抽象层回调
8. ✅ **GPU buffer flush modes** - 3种策略，带决策逻辑

### 推荐的最小集成方案

**只需要修改2个文件，增加~50行代码**：

1. `uvm_gpu_replayable_faults.h` - 添加1个enum值
2. `uvm_gpu_replayable_faults.c` - 添加struct_ops注册 + 1个判断

**不需要删除任何代码** - 所有现有逻辑作为默认行为保留。

**使用方式**：
```bash
# 加载模块时启用eBPF policy
sudo modprobe nvidia_uvm uvm_perf_fault_replay_policy=4

# 加载eBPF程序
bpftool struct_ops register my_wic_policy.o

# eBPF程序决定什么时候replay faults
```

---

## 三个文档说明

1. **`MINIMAL_EBPF_INTEGRATION.md`** - 回答"已经有什么接口"，介绍最小方案
2. **`MINIMAL_PATCH.md`** - 具体代码修改，精确到行号
3. **`ALL_POLICY_INTERFACES.md`** - 完整目录，30个类别，150+接口

所有文档已经创建在当前目录！
