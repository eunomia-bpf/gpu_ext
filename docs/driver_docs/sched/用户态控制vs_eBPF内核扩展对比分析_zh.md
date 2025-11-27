# 用户态GPU调度控制 vs eBPF内核扩展：深度对比分析

## 目录

1. [两种方案的架构对比](#两种方案的架构对比)
2. [GPreempt用户态方案分析](#gpreempt用户态方案分析)
3. [eBPF内核扩展方案设计](#ebpf内核扩展方案设计)
4. [详细对比分析](#详细对比分析)
5. [混合方案设计](#混合方案设计)
6. [总结与建议](#总结与建议)

---

## 1. 两种方案的架构对比

### 1.1 GPreempt用户态控制方案

```
┌─────────────────────────────────────────────────┐
│              User Space                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Application (CUDA/HIP Program)         │  │
│  │   - 调用GPU API                           │  │
│  │   - 设置任务优先级                        │  │
│  │   - 触发pre-preemption hint              │  │
│  └────────────────┬─────────────────────────┘  │
│                   │ ioctl/syscall             │
│  ┌────────────────▼─────────────────────────┐  │
│  │   GPreempt User Library                   │  │
│  │   - 管理timeslice配置                     │  │
│  │   - 调度preemption kernel                │  │
│  │   - 监控和metrics收集                     │  │
│  └────────────────┬─────────────────────────┘  │
└───────────────────┼──────────────────────────────┘
                    │ ioctl (NV_ESC_RM_*)
┌───────────────────▼──────────────────────────────┐
│              Kernel Space                         │
│  ┌──────────────────────────────────────────┐  │
│  │   GPU Driver (Modified)                   │  │
│  │   - 处理timeslice配置请求                 │  │
│  │   - 执行TSG切换                           │  │
│  │   - 有限的调度决策能力                    │  │
│  └────────────────┬─────────────────────────┘  │
│                   │ MMIO/PCIe                 │
│  ┌────────────────▼─────────────────────────┐  │
│  │   GPU Hardware                            │  │
│  │   - Hardware Scheduler (固定逻辑)        │  │
│  │   - Timeslice rotation (硬件实现)        │  │
│  │   - Context switching                     │  │
│  └──────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘

关键特征：
✓ 调度逻辑在用户态
✓ 依赖硬件timeslice机制
✗ 调度决策延迟高
✗ 缺乏内核事件感知
```

### 1.2 eBPF内核扩展方案（类似sched_ext）

```
┌─────────────────────────────────────────────────┐
│              User Space                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Application (CUDA/HIP Program)         │  │
│  │   - 正常调用GPU API                       │  │
│  │   - 不需要关心调度细节                    │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │   eBPF Loader/Controller                  │  │
│  │   - 加载eBPF调度程序                      │  │
│  │   - 更新调度策略                          │  │
│  │   - 读取metrics (via maps)                │  │
│  └────────────────┬─────────────────────────┘  │
└───────────────────┼──────────────────────────────┘
                    │ bpf() syscall
┌───────────────────▼──────────────────────────────┐
│              Kernel Space                         │
│  ┌──────────────────────────────────────────┐  │
│  │   GPU Driver (eBPF-enabled)               │  │
│  │  ┌────────────────────────────────────┐  │  │
│  │  │ eBPF Hook Points:                  │  │  │
│  │  │ - gpu_task_enqueue()               │  │  │
│  │  │ - gpu_task_dequeue()               │  │  │
│  │  │ - gpu_channel_group_create()       │  │  │
│  │  │ - gpu_runlist_update()             │  │  │
│  │  │ - gpu_context_switch()             │  │  │
│  │  │ - gpu_timeslice_expired()          │  │  │
│  │  └────────┬───────────────────────────┘  │  │
│  │           │ 调用eBPF程序                 │  │
│  │  ┌────────▼───────────────────────────┐  │  │
│  │  │ eBPF Scheduler Programs:           │  │  │
│  │  │ - select_next_task()  ← 核心      │  │  │
│  │  │ - should_preempt()                 │  │  │
│  │  │ - configure_timeslice()            │  │  │
│  │  │ - task_priority()                  │  │  │
│  │  └────────┬───────────────────────────┘  │  │
│  │           │ 访问eBPF Maps               │  │
│  │  ┌────────▼───────────────────────────┐  │  │
│  │  │ eBPF Maps (共享数据):              │  │  │
│  │  │ - task_info_map                    │  │  │
│  │  │ - priority_map                     │  │  │
│  │  │ - metrics_map                      │  │  │
│  │  │ - policy_config_map                │  │  │
│  │  └────────────────────────────────────┘  │  │
│  └────────────────┬─────────────────────────┘  │
│                   │ MMIO/PCIe                 │
│  ┌────────────────▼─────────────────────────┐  │
│  │   GPU Hardware                            │  │
│  │   - 按内核决策执行调度                    │  │
│  └──────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘

关键特征：
✓ 调度逻辑在内核态
✓ 可扩展且安全（eBPF验证）
✓ 零延迟事件响应
✓ 丰富的内核信息访问
```

---

## 2. GPreempt用户态方案分析

### 2.1 工作原理

GPreempt的核心机制：

```c
// 1. 用户态设置timeslice
void configure_preemption() {
    // 打开GPU设备
    int fd = open("/dev/nvidia0", O_RDWR);

    // 为BE任务设置短timeslice
    NVOS54_PARAMETERS params = {
        .hClient = be_client_handle,
        .hObject = be_tsg_handle,
        .timeslice = 200  // 200µs
    };
    ioctl(fd, NV_ESC_RM_CONTROL, &params);

    // 为LC任务设置长timeslice
    params.hClient = lc_client_handle;
    params.hObject = lc_tsg_handle;
    params.timeslice = 5000000;  // 5秒
    ioctl(fd, NV_ESC_RM_CONTROL, &params);
}

// 2. 用户态触发preemption
void submit_lc_task(Task* task) {
    // 开始数据准备
    prepare_data(task);

    // 在后台线程调度preemption
    std::thread([task]() {
        std::this_thread::sleep_for(
            std::chrono::microseconds(
                estimate_prep_time(task) - 100
            )
        );

        // 启动preemption kernel
        launch_preemption_kernel<<<1, 1>>>();
    }).detach();

    // 数据传输
    cudaMemcpy(gpu_data, cpu_data, size, cudaMemcpyHostToDevice);

    // 启动LC kernel
    lc_kernel<<<grid, block>>>(gpu_data);
}
```

### 2.2 优势

1. **不需要修改内核核心代码**
   - 只需要添加配置接口
   - 降低了开发和维护成本

2. **灵活性高**
   - 调度策略可以在用户态快速迭代
   - 不需要重启内核

3. **便于部署**
   - 不涉及内核模块签名
   - 容易在不同GPU上移植

### 2.3 劣势与局限

#### 2.3.1 调度决策延迟高

**问题**：从事件发生到调度决策的路径太长

```
事件: LC任务到达
│
├─ 用户态应用感知: +0µs
│  └─ 调用GPreempt API
│
├─ GPreempt库处理: +10µs
│  ├─ 计算调度时机
│  └─ 准备ioctl参数
│
├─ 系统调用: +5µs
│  ├─ 用户态→内核态切换
│  └─ ioctl处理
│
├─ 驱动处理: +20µs
│  ├─ 查找TSG
│  ├─ 验证权限
│  └─ 配置硬件
│
└─ 硬件执行: +100µs (timeslice轮转)
   └─ 实际抢占发生

总延迟: ~135µs
```

**对比eBPF方案**：
```
事件: LC任务到达
│
├─ 内核感知: +0µs (驱动直接检测)
│
├─ eBPF程序执行: +2µs
│  └─ select_next_task()
│
├─ 驱动执行决策: +5µs
│  └─ 配置硬件
│
└─ 硬件执行: +40µs (直接context switch)

总延迟: ~47µs (快3倍!)
```

#### 2.3.2 缺乏内核事件感知

**问题**：用户态无法感知许多关键内核事件

```c
// 用户态看不到的事件：

// 1. Channel创建/销毁
kchannelConstruct_IMPL() {
    // ❌ 用户态不知道新channel创建了
    // eBPF: ✓ 可以hook这里，立即调整调度策略
}

// 2. Runlist更新
kfifoUpdateUsermodeDoorbell_HAL() {
    // ❌ 用户态不知道任务已提交到GPU
    // eBPF: ✓ 可以hook，获取任务提交时间戳
}

// 3. Page fault
uvmServiceBlockLocked() {
    // ❌ 用户态不知道发生了page fault
    // eBPF: ✓ 可以hook，调整timeslice补偿延迟
}

// 4. GPU错误
gpuServiceNonStallIntr() {
    // ❌ 用户态无法及时响应GPU错误
    // eBPF: ✓ 可以hook，立即切换到备用任务
}
```

**后果**：
- 无法根据实时GPU状态动态调整策略
- 错过优化机会
- 应对异常情况能力弱

#### 2.3.3 信息访问受限

```c
// 用户态无法访问的关键信息：

// 1. 内核数据结构
typedef struct KernelChannelGroup {
    NvU32 timesliceUs;
    NvU32 runlistId;
    // ... 许多内部字段
} KernelChannelGroup;

// ❌ 用户态无法直接读取这些字段
// eBPF: ✓ 可以通过bpf_probe_read访问

// 2. GPU硬件状态
NvU32 gpu_utilization = read_gpu_register(SM_ACTIVE);
// ❌ 用户态无法读取GPU寄存器
// eBPF: ✓ 内核可以读取，eBPF可以访问

// 3. 其他任务的信息
for (each task in gpu) {
    // ❌ 用户态只能看到自己的任务
    // eBPF: ✓ 可以遍历所有任务，全局调度
}
```

#### 2.3.4 安全性问题

**问题**：用户态程序可能是恶意的或有bug

```c
// 恶意用户可能：

// 1. 设置不合理的timeslice
params.timeslice = 0;  // 导致任务无法运行
params.timeslice = UINT32_MAX;  // 独占GPU

// 2. 频繁改变配置，DoS攻击
while (true) {
    ioctl(fd, NV_ESC_RM_CONTROL, &params);
    // 淹没内核
}

// 3. 修改其他任务的配置（如果权限检查不足）
params.hClient = victim_client_handle;
params.timeslice = 0;  // 饿死victim任务
```

**eBPF的优势**：
```c
// eBPF verifier会检查：
// ✓ 无限循环检测
// ✓ 内存访问边界检查
// ✓ 指针安全检查
// ✓ 最大指令数限制

// 恶意程序无法通过verifier
BPF_PROG(configure_timeslice) {
    while (1) { }  // ❌ Verifier拒绝：检测到循环
    return 0;
}
```

#### 2.3.5 无法实现全局优化

**问题**：每个进程只能控制自己的任务

```
进程A (用户态调度器A):
├─ Task A1 (priority=1, timeslice=200µs)
└─ Task A2 (priority=2, timeslice=500µs)

进程B (用户态调度器B):
├─ Task B1 (priority=1, timeslice=100µs)
└─ Task B2 (priority=3, timeslice=1000µs)

问题：
1. 两个调度器无法协调
2. Priority=1在不同进程中含义可能不同
3. 可能导致优先级反转

┌─────────────────────────────────────┐
│   理想调度顺序（全局视角）：         │
│   1. A1 (priority=1, deadline紧)    │
│   2. B1 (priority=1, deadline松)    │
│   3. A2 (priority=2)                │
│   4. B2 (priority=3)                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│   实际调度（硬件timeslice轮转）：    │
│   A1 → B1 → A2 → B2 → A1 → ...     │
│   ❌ 无法体现全局优先级              │
└─────────────────────────────────────┘
```

**eBPF解决方案**：
```c
// 全局调度器可以看到所有任务
BPF_PROG(select_next_task) {
    struct task *best = NULL;
    int best_priority = INT_MAX;

    // 遍历所有进程的所有任务
    bpf_for_each_task(task) {
        int effective_priority = calculate_priority(task);

        if (effective_priority < best_priority) {
            best_priority = effective_priority;
            best = task;
        }
    }

    return best;  // 全局最优
}
```

#### 2.3.6 难以实现复杂调度算法

**问题**：用户态只能通过timeslice间接控制，无法实现精细调度

**例子1：EDF (Earliest Deadline First)**

```c
// 用户态GPreempt方案：
void schedule_edf_userspace() {
    // ❌ 只能近似实现
    for (each task) {
        double slack = task->deadline - now();

        if (slack < 1ms) {
            task->timeslice = INFINITE;  // 最高优先级
        } else if (slack < 10ms) {
            task->timeslice = 1000;
        } else {
            task->timeslice = 200;
        }

        ioctl(fd, NV_ESC_RM_CONTROL, &task);
    }

    // 问题：
    // 1. 无法原子性地更新所有任务
    // 2. Timeslice轮转不等于EDF
    // 3. 硬件仍然round-robin
}

// eBPF方案：
BPF_PROG(select_next_task_edf) {
    struct task *earliest = NULL;
    u64 earliest_deadline = U64_MAX;

    // ✓ 精确实现EDF
    bpf_for_each_task(task) {
        if (task->deadline < earliest_deadline) {
            earliest_deadline = task->deadline;
            earliest = task;
        }
    }

    return earliest;  // 真正的EDF
}
```

**例子2：MLFQ (Multi-Level Feedback Queue)**

```c
// 用户态无法实现真正的MLFQ
// 因为：
// 1. 无法追踪任务的历史行为（CPU burst等）
// 2. 无法原子性地在队列间移动任务
// 3. 硬件不支持多级队列

// eBPF可以完整实现：
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);
    __type(value, struct queue);
    __uint(max_entries, 5);  // 5个优先级队列
} mlfq_queues SEC(".maps");

BPF_PROG(select_next_task_mlfq) {
    // 从高优先级队列开始
    for (int i = 0; i < 5; i++) {
        struct queue *q = bpf_map_lookup_elem(&mlfq_queues, &i);
        if (q && !queue_empty(q)) {
            return queue_dequeue(q);
        }
    }
    return NULL;
}

BPF_PROG(task_timeslice_expired) {
    struct task *task = get_current_task();

    // 降低优先级（feedback）
    if (task->priority_level < 4) {
        task->priority_level++;

        // 移动到低优先级队列
        move_to_queue(task, task->priority_level);
    }
}
```

#### 2.3.7 Race Condition和一致性问题

**问题**：用户态和内核态状态可能不一致

```c
// Timeline:

T0: 用户态读取任务状态
    task->state = RUNNING;

T1: 内核态发生context switch
    task->state = SUSPENDED;  // 用户态不知道！

T2: 用户态基于过时信息做决策
    if (task->state == RUNNING) {
        adjust_timeslice(task, 100);  // ❌ 错误决策
    }

T3: ioctl到内核
    // 此时任务已经不在运行了
    // 但timeslice被错误修改

T4: 任务恢复运行
    // 使用了错误的timeslice
```

**eBPF的优势**：
```c
// 所有决策在内核态，原子性保证
BPF_PROG(context_switch) {
    struct task *prev = get_prev_task();
    struct task *next = get_next_task();

    // ✓ 原子性操作
    prev->state = SUSPENDED;
    next->state = RUNNING;

    // ✓ 基于最新状态做决策
    if (should_boost_priority(next)) {
        next->priority++;
    }

    return next;
}
```

---

## 3. eBPF内核扩展方案设计

### 3.1 架构设计

类似sched_ext，为GPU调度设计eBPF框架：

```c
// 1. 定义eBPF程序类型
enum bpf_gpu_sched_prog_type {
    BPF_GPU_SCHED_SELECT_NEXT_TASK,      // 选择下一个任务
    BPF_GPU_SCHED_TASK_ENQUEUE,          // 任务入队
    BPF_GPU_SCHED_TASK_DEQUEUE,          // 任务出队
    BPF_GPU_SCHED_SHOULD_PREEMPT,        // 是否抢占
    BPF_GPU_SCHED_TIMESLICE_EXPIRED,     // 时间片耗尽
    BPF_GPU_SCHED_CONTEXT_SWITCH,        // 上下文切换
    BPF_GPU_SCHED_TASK_EXIT,             // 任务退出
};

// 2. GPU任务描述符（内核数据结构）
struct gpu_task {
    u64 task_id;
    u32 priority;
    u64 deadline;
    u64 arrival_time;
    u64 exec_time;
    u64 remaining_time;
    u32 tsg_id;
    u32 runlist_id;
    void *kernel_channel_group;  // KernelChannelGroup*
    u64 vruntime;  // 虚拟运行时间（for CFS-like）
    u32 state;     // READY, RUNNING, SUSPENDED
};

// 3. eBPF Helper函数
BPF_HELPER(bpf_gpu_task_set_timeslice,
           struct gpu_task *task, u32 timeslice_us);

BPF_HELPER(bpf_gpu_task_get_info,
           u64 task_id, struct gpu_task *info);

BPF_HELPER(bpf_gpu_for_each_task,
           void *callback, void *ctx);

BPF_HELPER(bpf_gpu_preempt_task,
           struct gpu_task *task);

BPF_HELPER(bpf_gpu_read_hw_counter,
           u32 counter_id, u64 *value);

// 4. 驱动中的Hook点
// drivers/gpu/nvidia/src/kernel/gpu/fifo/kernel_fifo.c

NV_STATUS kfifoRunlistUpdateLocked_IMPL(...) {
    // Hook: task_enqueue
    if (gpu_sched_ops.task_enqueue) {
        struct gpu_task task = {...};
        gpu_sched_ops.task_enqueue(&task);
    }

    // 原有逻辑
    ...
}

// drivers/gpu/nvidia/src/kernel/gpu/fifo/kernel_channel_group.c

void kchangrpTimesliceExpired(...) {
    // Hook: timeslice_expired
    if (gpu_sched_ops.timeslice_expired) {
        struct gpu_task *next =
            gpu_sched_ops.timeslice_expired(current_task);

        if (next != current_task) {
            // 执行抢占
            perform_context_switch(current_task, next);
        }
    }

    // 原有逻辑
    ...
}
```

### 3.2 eBPF调度器实现示例

#### 示例1：优先级调度器

```c
// priority_scheduler.bpf.c

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "gpu_sched.h"

// 优先级队列（简化版）
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);    // task_id
    __type(value, u32);  // priority
    __uint(max_entries, 10000);
} task_priority_map SEC(".maps");

// 统计信息
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);
    __type(value, u64);
    __uint(max_entries, 10);
} stats_map SEC(".maps");

#define STAT_PREEMPTIONS 0
#define STAT_CONTEXT_SWITCHES 1

// 选择下一个任务
SEC("gpu_sched/select_next_task")
struct gpu_task *select_next_task(void *ctx) {
    struct gpu_task *best = NULL;
    u32 highest_priority = 0;

    // 遍历所有就绪任务
    bpf_gpu_for_each_task_in_state(GPU_TASK_READY, task) {
        u32 *priority = bpf_map_lookup_elem(
            &task_priority_map,
            &task->task_id
        );

        if (!priority) {
            priority = &task->priority;  // 使用默认优先级
        }

        if (*priority > highest_priority) {
            highest_priority = *priority;
            best = task;
        }
    }

    return best;
}

// 判断是否应该抢占
SEC("gpu_sched/should_preempt")
bool should_preempt(struct gpu_task *current, struct gpu_task *new) {
    u32 *current_priority = bpf_map_lookup_elem(
        &task_priority_map,
        &current->task_id
    );

    u32 *new_priority = bpf_map_lookup_elem(
        &task_priority_map,
        &new->task_id
    );

    u32 curr_prio = current_priority ? *current_priority : current->priority;
    u32 new_prio = new_priority ? *new_priority : new->priority;

    // 新任务优先级更高，抢占
    if (new_prio > curr_prio) {
        // 更新统计
        u32 key = STAT_PREEMPTIONS;
        u64 *count = bpf_map_lookup_elem(&stats_map, &key);
        if (count) {
            __sync_fetch_and_add(count, 1);
        }

        return true;
    }

    return false;
}

// 任务入队时配置timeslice
SEC("gpu_sched/task_enqueue")
int task_enqueue(struct gpu_task *task) {
    u32 *priority = bpf_map_lookup_elem(
        &task_priority_map,
        &task->task_id
    );

    // 根据优先级设置timeslice
    u32 timeslice;
    if (priority && *priority >= 10) {
        timeslice = 10000000;  // 10秒，实际上不会被中断
    } else if (priority && *priority >= 5) {
        timeslice = 1000;      // 1ms
    } else {
        timeslice = 200;       // 200µs
    }

    bpf_gpu_task_set_timeslice(task, timeslice);

    return 0;
}

// 上下文切换时的记录
SEC("gpu_sched/context_switch")
int context_switch(struct gpu_task *prev, struct gpu_task *next) {
    // 更新统计
    u32 key = STAT_CONTEXT_SWITCHES;
    u64 *count = bpf_map_lookup_elem(&stats_map, &key);
    if (count) {
        __sync_fetch_and_add(count, 1);
    }

    // 更新运行时间
    u64 now = bpf_ktime_get_ns();
    u64 delta = now - prev->last_sched_time;

    prev->exec_time += delta;
    prev->remaining_time -= delta;

    next->last_sched_time = now;

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

#### 示例2：Deadline调度器

```c
// deadline_scheduler.bpf.c

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "gpu_sched.h"

// Deadline信息
struct deadline_info {
    u64 deadline;       // 绝对deadline
    u64 period;         // 周期
    u64 runtime;        // 每周期的运行时间
    u64 remaining;      // 当前周期剩余运行时间
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);    // task_id
    __type(value, struct deadline_info);
    __uint(max_entries, 10000);
} deadline_map SEC(".maps");

// EDF: 选择deadline最早的任务
SEC("gpu_sched/select_next_task")
struct gpu_task *select_next_task(void *ctx) {
    struct gpu_task *earliest = NULL;
    u64 earliest_deadline = U64_MAX;
    u64 now = bpf_ktime_get_ns();

    bpf_gpu_for_each_task_in_state(GPU_TASK_READY, task) {
        struct deadline_info *dl = bpf_map_lookup_elem(
            &deadline_map,
            &task->task_id
        );

        if (!dl) continue;

        // 检查是否超过deadline
        if (dl->deadline < now) {
            // Deadline已过，需要replenish
            dl->deadline += dl->period;
            dl->remaining = dl->runtime;
        }

        // 选择deadline最早的
        if (dl->remaining > 0 && dl->deadline < earliest_deadline) {
            earliest_deadline = dl->deadline;
            earliest = task;
        }
    }

    return earliest;
}

// 时间片耗尽时减少remaining
SEC("gpu_sched/timeslice_expired")
struct gpu_task *timeslice_expired(struct gpu_task *current) {
    struct deadline_info *dl = bpf_map_lookup_elem(
        &deadline_map,
        &current->task_id
    );

    if (dl) {
        // 减少剩余运行时间
        u64 delta = current->last_timeslice_duration;
        if (dl->remaining > delta) {
            dl->remaining -= delta;
        } else {
            dl->remaining = 0;
        }

        // 如果用完了quota，让出CPU
        if (dl->remaining == 0) {
            current->state = GPU_TASK_THROTTLED;
        }
    }

    // 重新选择下一个任务
    return select_next_task(NULL);
}

// 抢占判断：新任务deadline更早
SEC("gpu_sched/should_preempt")
bool should_preempt(struct gpu_task *current, struct gpu_task *new) {
    struct deadline_info *curr_dl = bpf_map_lookup_elem(
        &deadline_map,
        &current->task_id
    );

    struct deadline_info *new_dl = bpf_map_lookup_elem(
        &deadline_map,
        &new->task_id
    );

    if (!curr_dl || !new_dl) return false;

    // EDF: deadline更早的抢占
    return new_dl->deadline < curr_dl->deadline;
}

char LICENSE[] SEC("license") = "GPL";
```

#### 示例3：CFS-like公平调度器

```c
// cfs_scheduler.bpf.c

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "gpu_sched.h"

// Nice值到权重的映射（类似Linux）
static const u32 prio_to_weight[40] = {
    /* -20 */ 88761, 71755, 56483, 46273, 36291,
    /* -15 */ 29154, 23254, 18705, 14949, 11916,
    /* -10 */ 9548, 7620, 6100, 4904, 3906,
    /*  -5 */ 3121, 2501, 1991, 1586, 1277,
    /*   0 */ 1024, 820, 655, 526, 423,
    /*   5 */ 335, 272, 215, 172, 137,
    /*  10 */ 110, 87, 70, 56, 45,
    /*  15 */ 36, 29, 23, 18, 15,
};

#define NICE_0_LOAD 1024

// 任务权重信息
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u64);    // task_id
    __type(value, u32);  // weight
    __uint(max_entries, 10000);
} weight_map SEC(".maps");

// 选择vruntime最小的任务
SEC("gpu_sched/select_next_task")
struct gpu_task *select_next_task(void *ctx) {
    struct gpu_task *leftmost = NULL;
    u64 min_vruntime = U64_MAX;

    bpf_gpu_for_each_task_in_state(GPU_TASK_READY, task) {
        if (task->vruntime < min_vruntime) {
            min_vruntime = task->vruntime;
            leftmost = task;
        }
    }

    return leftmost;
}

// 更新vruntime
static __always_inline void update_vruntime(
    struct gpu_task *task,
    u64 delta_exec
) {
    u32 *weight = bpf_map_lookup_elem(&weight_map, &task->task_id);
    u32 w = weight ? *weight : NICE_0_LOAD;

    // vruntime增长速度与权重成反比
    // delta_vruntime = delta_exec * NICE_0_LOAD / weight
    u64 delta_vruntime = (delta_exec * NICE_0_LOAD) / w;

    task->vruntime += delta_vruntime;
}

// 时间片耗尽
SEC("gpu_sched/timeslice_expired")
struct gpu_task *timeslice_expired(struct gpu_task *current) {
    u64 now = bpf_ktime_get_ns();
    u64 delta = now - current->last_sched_time;

    // 更新vruntime
    update_vruntime(current, delta);

    // 重新选择
    return select_next_task(NULL);
}

// 上下文切换时更新vruntime
SEC("gpu_sched/context_switch")
int context_switch(struct gpu_task *prev, struct gpu_task *next) {
    u64 now = bpf_ktime_get_ns();
    u64 delta = now - prev->last_sched_time;

    // 更新prev的vruntime
    update_vruntime(prev, delta);

    // 重置next的时间戳
    next->last_sched_time = now;

    return 0;
}

// 任务入队时设置初始vruntime
SEC("gpu_sched/task_enqueue")
int task_enqueue(struct gpu_task *task) {
    // 如果是新任务，设置vruntime为当前最小值
    if (task->vruntime == 0) {
        u64 min_vruntime = U64_MAX;

        bpf_gpu_for_each_task(t) {
            if (t->vruntime < min_vruntime) {
                min_vruntime = t->vruntime;
            }
        }

        task->vruntime = min_vruntime;
    }

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

### 3.3 用户态控制接口

```c
// gpu_sched_ctl.c - 用户态工具

#include <stdio.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

int main(int argc, char **argv) {
    struct bpf_object *obj;
    int map_fd, prog_fd;

    // 1. 加载eBPF程序
    obj = bpf_object__open_file("priority_scheduler.bpf.o", NULL);
    if (!obj) {
        fprintf(stderr, "Failed to open BPF object\n");
        return 1;
    }

    if (bpf_object__load(obj)) {
        fprintf(stderr, "Failed to load BPF object\n");
        return 1;
    }

    // 2. 附加eBPF程序到GPU调度钩子
    prog_fd = bpf_program__fd(
        bpf_object__find_program_by_name(
            obj, "select_next_task"
        )
    );

    if (bpf_gpu_sched_attach(prog_fd, BPF_GPU_SCHED_SELECT_NEXT_TASK)) {
        fprintf(stderr, "Failed to attach BPF program\n");
        return 1;
    }

    // 3. 更新优先级（通过map）
    map_fd = bpf_object__find_map_fd_by_name(obj, "task_priority_map");

    u64 task_id = 12345;
    u32 priority = 10;  // 高优先级
    bpf_map_update_elem(map_fd, &task_id, &priority, BPF_ANY);

    // 4. 读取统计信息
    int stats_fd = bpf_object__find_map_fd_by_name(obj, "stats_map");
    u32 key = 0;  // STAT_PREEMPTIONS
    u64 count;
    bpf_map_lookup_elem(stats_fd, &key, &count);
    printf("Preemptions: %llu\n", count);

    return 0;
}
```

---

## 4. 详细对比分析

### 4.1 性能对比

| 指标 | GPreempt (用户态) | eBPF (内核态) | 差异 |
|-----|------------------|--------------|------|
| **调度决策延迟** | 100-150µs | 2-10µs | **eBPF快10-75倍** |
| **上下文切换延迟** | 40µs | 40µs | 相同（硬件决定） |
| **总抢占延迟** | 140-190µs | 42-50µs | **eBPF快3-4倍** |
| **吞吐量影响** | 中等 | 极小 | eBPF更优 |
| **CPU开销** | 高（用户态轮询） | 低（事件驱动） | **eBPF节省CPU** |

**延迟breakdown详细分析**：

```
GPreempt用户态方案：
┌─────────────────────────────────────────┐
│ LC任务到达                               │
├─────────────────────────────────────────┤
│ 用户态感知                    +0µs      │
│ GPreempt库处理                +10µs     │
│ 准备ioctl                     +5µs      │
│ syscall进入内核               +5µs      │
│ 驱动查找TSG                   +10µs     │
│ 驱动配置timeslice             +5µs      │
│ 等待timeslice轮转             +100µs    │ ← 最大瓶颈
│ 上下文切换                    +40µs     │
├─────────────────────────────────────────┤
│ 总计                          ~175µs    │
└─────────────────────────────────────────┘

eBPF内核方案：
┌─────────────────────────────────────────┐
│ LC任务到达                               │
├─────────────────────────────────────────┤
│ 驱动检测到任务入队             +0µs     │
│ 调用eBPF hook                  +1µs     │
│ eBPF程序执行select_next_task   +2µs     │
│ eBPF程序返回决策               +1µs     │
│ 驱动执行抢占                   +5µs     │
│ 上下文切换                     +40µs    │
├─────────────────────────────────────────┤
│ 总计                           ~49µs    │
└─────────────────────────────────────────┘

性能差异来源：
1. ❌ GPreempt: 用户态→内核态切换 (~15µs)
2. ❌ GPreempt: 等待timeslice轮转 (~100µs)
3. ✓ eBPF: 直接在内核态决策，立即抢占
```

### 4.2 功能对比

| 功能 | GPreempt | eBPF | 说明 |
|-----|----------|------|------|
| **基础抢占** | ✓ | ✓ | 两者都支持 |
| **优先级调度** | ✓ (受限) | ✓ | eBPF可实现真正的优先级 |
| **EDF调度** | ✗ | ✓ | GPreempt无法实现 |
| **CFS调度** | ✗ | ✓ | GPreempt无法实现 |
| **全局调度** | ✗ | ✓ | GPreempt每个进程独立 |
| **动态策略切换** | ✓ | ✓ | 两者都支持 |
| **内核事件响应** | ✗ | ✓ | GPreempt看不到内核事件 |
| **访问GPU状态** | ✗ | ✓ | GPreempt无法访问硬件状态 |
| **自定义调度算法** | ✗ | ✓ | eBPF完全可编程 |
| **原子性保证** | ✗ | ✓ | GPreempt有race condition |

### 4.3 安全性对比

| 安全特性 | GPreempt | eBPF | 说明 |
|---------|----------|------|------|
| **权限检查** | 依赖驱动 | Verifier + CAP_BPF | eBPF更严格 |
| **资源隔离** | ✗ | ✓ | eBPF有map限制 |
| **DoS防护** | ✗ | ✓ | eBPF有指令数限制 |
| **内存安全** | N/A | ✓ | eBPF verifier保证 |
| **死锁检测** | ✗ | ✓ | eBPF禁止无限循环 |

**安全性详细分析**：

```c
// GPreempt安全问题：

// 1. DoS攻击
while (true) {
    ioctl(fd, NV_ESC_RM_CONTROL, &params);  // ❌ 可以淹没内核
}

// 2. 权限提升
params.hClient = root_client;  // ❌ 如果驱动检查不严格
params.timeslice = INFINITE;

// 3. 资源耗尽
params.timeslice = 0;  // ❌ 让其他任务无法运行

// eBPF防护：

// 1. Verifier拒绝无限循环
BPF_PROG(malicious) {
    while (1) { }  // ❌ Verifier: 检测到循环，拒绝加载
}

// 2. 指令数限制
BPF_PROG(too_complex) {
    // ❌ 如果超过1M指令，Verifier拒绝
    for (int i = 0; i < 1000000; i++) {
        // ...
    }
}

// 3. 内存访问边界检查
BPF_PROG(out_of_bounds) {
    char buf[100];
    char *ptr = buf + 200;  // ❌ Verifier: 越界访问
    *ptr = 'A';
}

// 4. Map大小限制
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1000000000);  // ❌ 太大，加载失败
} huge_map SEC(".maps");
```

### 4.4 可维护性对比

| 方面 | GPreempt | eBPF | 说明 |
|-----|----------|------|------|
| **开发难度** | 中等 | 较高 | eBPF需要理解verifier |
| **调试难度** | 低 | 中等 | eBPF有bpftool辅助 |
| **部署难度** | 低 | 低 | 两者都较容易 |
| **热更新** | ✓ | ✓ | 两者都支持 |
| **代码重用** | ✗ | ✓ | eBPF有CO-RE |
| **文档/工具** | 少 | 丰富 | eBPF生态成熟 |

### 4.5 适用场景对比

#### GPreempt适合的场景：

```
✓ 1. 快速原型验证
   - 不需要修改内核核心代码
   - 可以快速迭代调度策略

✓ 2. 简单的优先级调度
   - 只有2-3个优先级
   - 不需要复杂调度算法

✓ 3. 单用户环境
   - 所有GPU任务属于同一个进程
   - 不需要全局调度

✓ 4. 延迟要求不严格
   - 可以接受100-200µs的抢占延迟
   - 吞吐量优先于延迟
```

#### eBPF适合的场景：

```
✓ 1. 严格延迟要求
   - 需要<50µs的抢占延迟
   - 实时系统

✓ 2. 复杂调度算法
   - EDF, MLFQ, CFS等
   - 需要全局优化

✓ 3. 多用户环境
   - 多个进程共享GPU
   - 需要公平性保证

✓ 4. 需要内核信息
   - 访问GPU硬件状态
   - 响应内核事件

✓ 5. 安全关键场景
   - 多租户云环境
   - 需要隔离和DoS防护
```

---

## 5. 混合方案设计

最优方案：**eBPF为主，用户态辅助**

```
┌─────────────────────────────────────────────────┐
│              User Space                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Policy Configuration                    │  │
│  │   - 定义调度策略（Python/YAML）           │  │
│  │   - 设置SLO目标                           │  │
│  │   - 监控和可视化                          │  │
│  └────────────────┬─────────────────────────┘  │
│                   │ 编译为eBPF                │
│  ┌────────────────▼─────────────────────────┐  │
│  │   eBPF Compiler & Loader                  │  │
│  │   - 编译策略为eBPF字节码                  │  │
│  │   - 验证和加载                            │  │
│  │   - 热更新支持                            │  │
│  └────────────────┬─────────────────────────┘  │
└───────────────────┼──────────────────────────────┘
                    │ bpf() syscall
┌───────────────────▼──────────────────────────────┐
│              Kernel Space                         │
│  ┌──────────────────────────────────────────┐  │
│  │   Core Scheduler (eBPF)                   │  │
│  │   - 快速路径：所有关键调度决策            │  │
│  │   - 事件驱动：hook所有内核事件            │  │
│  │   - 低延迟：<10µs决策时间                 │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                           │
│  ┌────────────────▼─────────────────────────┐  │
│  │   Fallback to User (Optional)             │  │
│  │   - 慢路径：复杂决策（ML预测等）          │  │
│  │   - 通过ringbuf通知用户态                 │  │
│  │   - 用户态计算后写入map                   │  │
│  └──────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### 5.1 混合方案实现

```python
# policy.py - 高层策略定义

from gpu_sched import Policy, Task, SLO

# 定义调度策略
class AdaptiveScheduler(Policy):
    def __init__(self):
        self.slo_target = SLO(p99=10_000)  # 10ms P99延迟

    # 编译为eBPF
    def compile(self):
        return """
        // 快速路径：eBPF实现
        SEC("gpu_sched/select_next_task")
        struct gpu_task *select_next_task(void *ctx) {
            // 简单启发式：优先级 + deadline
            return select_by_priority_and_deadline();
        }

        SEC("gpu_sched/should_preempt")
        bool should_preempt(struct gpu_task *curr,
                           struct gpu_task *new) {
            // 快速判断
            if (new->priority > curr->priority + 2) {
                return true;
            }

            // 复杂判断交给用户态
            if (need_complex_decision(curr, new)) {
                send_to_userspace(curr, new);
                return false;  // 暂时不抢占
            }

            return false;
        }
        """

    # 慢路径：用户态处理
    def handle_complex_decision(self, curr: Task, new: Task):
        # 使用机器学习预测
        pred_latency = self.ml_model.predict(new)

        if pred_latency > self.slo_target.p99:
            # 需要抢占
            return self.preempt(curr, new)
        else:
            # 继续运行当前任务
            return curr

# 部署策略
scheduler = AdaptiveScheduler()
scheduler.deploy(gpu_id=0)
```

### 5.2 混合方案的优势

```
┌──────────────────────────────────────────────┐
│              混合方案优势                     │
├──────────────────────────────────────────────┤
│ 1. 性能优势                                  │
│    ✓ 快速路径在内核（<10µs）                 │
│    ✓ 只在必要时才调用用户态                  │
│    ✓ 避免频繁context switch                  │
│                                              │
│ 2. 灵活性优势                                │
│    ✓ 复杂逻辑在用户态（ML、优化算法）        │
│    ✓ 策略热更新                              │
│    ✓ 易于调试                                │
│                                              │
│ 3. 安全性优势                                │
│    ✓ eBPF verifier保证内核安全               │
│    ✓ 用户态逻辑隔离                          │
│    ✓ 资源限制                                │
│                                              │
│ 4. 可维护性优势                              │
│    ✓ 高层策略用Python等高级语言              │
│    ✓ 关键路径用eBPF保证性能                  │
│    ✓ 清晰的责任分离                          │
└──────────────────────────────────────────────┘
```

---

## 6. 总结与建议

### 6.1 GPreempt的不足

基于代码分析和论文阅读，GPreempt用户态方案的主要不足：

| 类别 | 具体问题 | 严重程度 |
|-----|---------|---------|
| **性能** | 调度决策延迟高（100-150µs） | ⭐⭐⭐ 严重 |
| **功能** | 无法实现EDF、CFS等调度算法 | ⭐⭐⭐ 严重 |
| **功能** | 缺乏全局调度能力 | ⭐⭐⭐ 严重 |
| **可靠性** | Race condition和一致性问题 | ⭐⭐ 中等 |
| **安全性** | 缺乏DoS防护 | ⭐⭐ 中等 |
| **可观测性** | 无法访问内核事件 | ⭐⭐ 中等 |
| **扩展性** | 依赖特定硬件timeslice机制 | ⭐ 轻微 |

### 6.2 eBPF方案的优势

```
eBPF内核扩展方案能做到GPreempt做不到的：

1. ✓ 超低延迟调度（<10µs决策）
   - 直接在内核态响应事件
   - 无需用户态↔内核态切换

2. ✓ 精确的调度算法
   - 真正的EDF（earliest deadline first）
   - 真正的CFS（完全公平调度）
   - 任意自定义算法

3. ✓ 全局优化
   - 跨进程调度
   - 全局优先级管理
   - 系统级资源分配

4. ✓ 丰富的内核信息
   - GPU硬件状态
   - 内核事件（page fault等）
   - 其他任务信息

5. ✓ 强安全保证
   - Verifier静态检查
   - 资源限制
   - DoS防护

6. ✓ 原子性保证
   - 无race condition
   - 一致性保证
   - 事务性操作
```

### 6.3 建议的演进路径

```
阶段1: GPreempt (现状)
└─ 快速验证想法
└─ 简单场景够用

阶段2: 添加eBPF钩子
└─ 在驱动中添加hook点
└─ 实现基础eBPF支持
└─ 向后兼容GPreempt

阶段3: 混合方案
└─ 快速路径用eBPF
└─ 复杂决策用用户态
└─ 最佳性能+灵活性

阶段4: 纯eBPF (理想)
└─ 所有调度逻辑在eBPF
└─ 用户态只做配置和监控
└─ 最高性能和安全性
```

### 6.4 具体实施建议

**短期（1-3个月）**：
```
1. 在开源GPU驱动中添加eBPF hook点
   - kfifoRunlistUpdate
   - kchannelConstruct/Destruct
   - timeslice_expired

2. 实现基础的eBPF helper函数
   - bpf_gpu_task_set_timeslice
   - bpf_gpu_task_get_info
   - bpf_gpu_for_each_task

3. 提供示例eBPF调度器
   - 优先级调度器
   - Round-robin调度器
```

**中期（3-6个月）**：
```
1. 实现高级调度算法
   - EDF调度器
   - CFS-like公平调度器
   - MLFQ调度器

2. 添加性能优化
   - JIT编译eBPF
   - 优化helper函数
   - 减少hook开销

3. 完善工具链
   - libbpf集成
   - bpftool支持
   - 调试工具
```

**长期（6-12个月）**：
```
1. 生产级特性
   - 完整的错误处理
   - 监控和告警
   - 自动调优

2. 高级功能
   - 机器学习集成
   - 自适应调度
   - 多GPU协同

3. 社区生态
   - 文档和教程
   - 示例调度器库
   - 性能测试套件
```

---

## 附录：参考实现

### A. Linux sched_ext架构

```c
// Linux 6.6+ sched_ext示例

// 1. 定义调度类
struct sched_ext_ops {
    // 选择下一个任务
    struct task_struct *(*select_task_rq)(struct task_struct *p,
                                          int prev_cpu,
                                          int wake_flags);

    // 任务入队
    void (*enqueue_task)(struct rq *rq, struct task_struct *p,
                        int flags);

    // 任务出队
    void (*dequeue_task)(struct rq *rq, struct task_struct *p,
                        int flags);

    // 时间片耗尽
    void (*task_tick)(struct rq *rq, struct task_struct *p);
};

// 2. GPU调度可以借鉴类似架构
struct gpu_sched_ext_ops {
    struct gpu_task *(*select_next_task)(struct gpu_device *gpu);
    void (*task_enqueue)(struct gpu_task *task);
    void (*task_dequeue)(struct gpu_task *task);
    bool (*should_preempt)(struct gpu_task *curr,
                          struct gpu_task *new);
    void (*timeslice_expired)(struct gpu_task *task);
    void (*context_switch)(struct gpu_task *prev,
                          struct gpu_task *next);
};
```

### B. 完整的混合方案示例

见文档后续章节...
