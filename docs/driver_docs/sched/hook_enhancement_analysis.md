# GPU 调度 Hook 点增强分析

## 当前 Hook 点

### 已实现的 Hook

| Hook 名称 | 触发时机 | 源码位置 |
|----------|---------|---------|
| `nv_gpu_sched_task_init` | TSG 创建 | `kernel_channel_group.c` |
| `nv_gpu_sched_schedule` | TSG 调度/启用 | `kernel_channel_group_api.c` |
| `nv_gpu_sched_token_request` | 用户请求 work submit token | `kernel_channel.c` |
| `nv_gpu_sched_task_destroy` | TSG 销毁 | `kernel_channel_group.c` |

### token_request hook 说明 (原 work_submit)

**重要**: 已从 `work_submit` 改名为 `token_request`，因为：

1. **不是每次 kernel launch 都触发** - 只在用户态通过 ioctl 查询 work submit token 时调用
2. **触发的 ioctl**: `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` (0xc36f0108)
3. **用途**: 获取一个 token 用于追踪工作完成状态，通常在同步前调用

```c
// 触发调用链:
用户态: ioctl(fd, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN)
           ↓
内核态: kchannelCtrlCmdGpfifoGetWorkSubmitToken_IMPL()
           ↓
        kfifoGenerateWorkSubmitToken()  // 生成 token
           ↓
        kchannelNotifyWorkSubmitToken() // 通知 + hook
           ↓
        nv_gpu_sched_token_request(&ctx)  // eBPF hook
```

## 为什么无法追踪每次 Kernel Launch

GPU 实际工作提交是 **bypass 内核** 的：

```
用户态 kernel launch 流程：
  用户态 libcuda.so
       │
       ├─ 写 pushbuffer (用户态直接写 GPU 映射内存，不经过内核！)
       │
       └─ 写 doorbell 寄存器 (用户态直接 MMIO，不经过内核！)
           │
           └─ GPU 硬件收到通知，从 pushbuffer 读取命令执行
```

## 可以增强的 Hook 点

### 1. kfifoChannelGroupSetTimeslice - 时间片修改 hook

**位置**: `kernel_fifo.c:1666`

**触发时机**: 当用户态调用 `NVA06C_CTRL_CMD_SET_TIMESLICE` 修改 TSG 时间片时

**可以做什么**:
- 追踪时间片修改请求
- 在 eBPF 中拦截/修改时间片值
- 实现基于策略的时间片限制

```c
// 建议的 hook 结构
struct nv_gpu_set_timeslice_ctx {
    u64 tsg_id;
    u64 old_timeslice_us;
    u64 new_timeslice_us;
    u32 pid;
    // output
    u64 adjusted_timeslice_us;  // eBPF 可以调整这个值
};
```

### 2. kchangrpapiCtrlCmdSetInterleaveLevel - 交错级别修改 hook

**位置**: `kernel_channel_group_api.c`

**触发时机**: 当修改 TSG 的 interleave level 时

**可以做什么**:
- 限制某些进程只能使用 LOW interleave
- 追踪优先级变化

### 3. kfifoUpdateUsermodeDoorbell - 内核态 doorbell 触发 hook

**位置**: `kernel_fifo_tu102.c:45` (Turing), `kernel_fifo_ga100.c:153` (Ampere)

**触发时机**: 当内核态代码触发 doorbell 时（不是用户态直接写）

**注意**: 这只捕获内核态触发的 doorbell，用户态直接 MMIO 写 doorbell 是无法捕获的

### 4. Preemption hook

**位置**: `kernel_fifo_ctrl.c:752` `deviceCtrlCmdFifoDisableChannels_IMPL`

**触发时机**: 当 channel 被 preempt/disable 时

**可以做什么**:
- 追踪 preemption 事件
- 实现自定义 preemption 策略

## 架构限制 - 为什么无法追踪每次 kernel launch

```
┌─────────────────────────────────────────────────────────────┐
│                        用户态                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CUDA Runtime (libcuda.so)                           │   │
│  │                                                      │   │
│  │  cudaLaunchKernel() {                               │   │
│  │      // 1. 写 pushbuffer - 直接写 GPU 映射内存       │   │
│  │      *pushbuf_ptr = gpu_commands;                   │   │
│  │                                                      │   │
│  │      // 2. 写 doorbell - 直接 MMIO，不经过内核！     │   │
│  │      *doorbell_ptr = work_submit_token;             │   │
│  │  }                                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           │ (用户态直接访问，bypass 内核)    │
│                           ▼                                  │
├─────────────────────────────────────────────────────────────┤
│                     GPU 硬件                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Doorbell 寄存器 ──► FIFO 引擎 ──► 执行 Pushbuffer   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**关键点**:
- 用户态 `cudaLaunchKernel` 直接写 GPU 映射的内存
- doorbell 是通过 usermode MMIO 直接触发
- **内核完全不参与单次 kernel launch 过程**
- 内核只参与 context 创建/销毁、TSG 调度、资源分配等

## 可行的增强方案

### 方案 1: 增加更多控制 ioctl hook

在以下控制命令处添加 hook：

| 控制命令 | 功能 | Hook 价值 |
|---------|------|----------|
| `NVA06C_CTRL_CMD_SET_TIMESLICE` | 设置时间片 | 高 - 可以限制/调整时间片 |
| `NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL` | 设置交错级别 | 高 - 可以控制优先级 |
| `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE` | 调度控制 | 已有 |
| `NV906F_CTRL_CMD_RESET_CHANNEL` | 重置 channel | 中 - 追踪错误恢复 |

### 方案 2: 使用 UVM (nvidia-uvm.ko) hook

nvidia-uvm.ko 是用 Kbuild 编译的，函数可追踪。UVM 处理：
- GPU 页面迁移
- 内存映射
- 缺页处理

这些操作每次访问 managed memory 都会触发。

### 方案 3: 硬件性能计数器 (不需要修改驱动)

通过 NVIDIA 的 CUPTI 或 perf 接口获取：
- SM 利用率
- 内存带宽
- 实际 kernel 执行统计

## 建议的新 Hook

```c
// 1. 时间片修改 hook
void nv_gpu_sched_set_timeslice(struct nv_gpu_set_timeslice_ctx *ctx);

// 2. 交错级别修改 hook
void nv_gpu_sched_set_interleave(struct nv_gpu_set_interleave_ctx *ctx);

// 3. Channel preemption hook
void nv_gpu_sched_preempt(struct nv_gpu_preempt_ctx *ctx);

// 4. Runlist 更新 hook (每次 runlist 变化时)
void nv_gpu_sched_runlist_update(struct nv_gpu_runlist_ctx *ctx);
```

## cudaDeviceSynchronize 同步机制

CUDA 同步可以通过以下两种方式实现：

### 方式 1: poll() 系统调用 (阻塞等待)

```c
// kernel-open/nvidia/nv.c:2296
nvidia_poll(struct file *file, poll_table *wait)
{
    // 等待 event_data_head 或 dataless_event_pending
    poll_wait(file, &nvlfp->waitqueue, wait);

    if ((nvlfp->event_data_head != NULL) || nvlfp->dataless_event_pending)
    {
        mask = (POLLPRI | POLLIN);  // 有事件发生
    }
}
```

流程:
```
cudaDeviceSynchronize()
   ↓
poll(/dev/nvidia0, POLLIN)  ← syscall，可被追踪
   ↓
内核: nvidia_poll() 等待 waitqueue
   ↓
GPU 完成 → 中断 → wake_up(&nvlfp->waitqueue)
   ↓
poll() 返回，用户态继续
```

### 方式 2: Notifier 内存轮询 (无 syscall)

驱动在 channel 创建时设置 **notifier memory**：
- 这是 GPU 可写、用户态可读的共享内存
- GPU 完成工作后会写入 notifier
- 用户态可以直接轮询这个内存，**不需要 syscall**

```
cudaDeviceSynchronize()
   ↓
while (*notifier_ptr != expected_value) {
    // 用户态直接读 GPU 映射的 notifier 内存
    // 完全不经过内核！
}
```

### 操作可追踪性总结

| 操作 | 是否有 syscall | 能否被追踪 |
|-----|--------------|-----------|
| `GET_WORK_SUBMIT_TOKEN` ioctl | ✓ 有 | ✓ 能 (token_request hook) |
| `poll()` 等待完成 | ✓ 有 | ✓ 能 (可以加 poll hook) |
| Notifier 轮询 | ✗ 无 | ✗ 不能 (用户态直接读内存) |
| Kernel launch | ✗ 无 | ✗ 不能 (用户态直接写 pushbuffer + doorbell) |

## 结论

1. **无法追踪每次 kernel launch** - 这是架构限制，用户态直接 bypass 内核
2. **可以追踪控制面操作** - TSG 创建/销毁、调度、时间片修改等
3. **token_request** (原 work_submit) 追踪的是 ioctl 请求 token，不是实际工作提交
4. **同步操作** - poll() 方式可追踪，notifier 轮询方式不可追踪
5. **建议增加时间片和交错级别 hook** - 这些是运行时可控的调度参数
6. **如果需要追踪 kernel launch** - 考虑使用 CUPTI 或修改 CUDA 运行时
