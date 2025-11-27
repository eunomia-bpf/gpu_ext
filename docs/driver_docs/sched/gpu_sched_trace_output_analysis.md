# GPU Scheduler Trace 输出分析

基于 eBPF/kprobe 的 NVIDIA GPU 调度事件追踪工具输出解释。

## 示例输出

```
23:00:23.509353 [CPU01] TASK_INIT    PID=637320 python3          TSG=2    engine=UNKNOWN  timeslice=1024 us interleave=UNKNOWN runlist=13
23:00:24.523615 [CPU01] TASK_INIT    PID=637320 python3          TSG=0    engine=UNKNOWN  timeslice=1024 us interleave=UNKNOWN runlist=14
23:00:24.561798 [CPU01] TASK_INIT    PID=637320 python3          TSG=1    engine=COPY     timeslice=1024 us interleave=UNKNOWN runlist=0
23:00:24.591625 [CPU01] SCHEDULE     PID=637320 python3          TSG=1    channels=8 timeslice=1024 us interleave=LOW allowed=yes
23:00:24.592495 [CPU01] TASK_INIT    PID=637320 python3          TSG=3    engine=UNKNOWN  timeslice=1024 us interleave=UNKNOWN runlist=13
23:00:24.599449 [CPU01] SCHEDULE     PID=637320 python3          TSG=3    channels=4 timeslice=1024 us interleave=LOW allowed=yes
23:00:24.600243 [CPU01] TASK_INIT    PID=637320 python3          TSG=1    engine=UNKNOWN  timeslice=1024 us interleave=UNKNOWN runlist=14
23:00:24.607274 [CPU01] SCHEDULE     PID=637320 python3          TSG=1    channels=4 timeslice=1024 us interleave=LOW allowed=yes
23:00:25.697801 [CPU01] TASK_DESTROY PID=637320 cuda-EvtHandlr   TSG=0
23:00:25.701934 [CPU01] TASK_DESTROY PID=637320 cuda-EvtHandlr   TSG=2
23:00:25.727580 [CPU01] TASK_DESTROY PID=637320 cuda-EvtHandlr   TSG=1
23:00:25.731589 [CPU01] TASK_DESTROY PID=637320 cuda-EvtHandlr   TSG=3
23:00:25.735378 [CPU01] TASK_DESTROY PID=637320 cuda-EvtHandlr   TSG=1
```

## 字段说明

| 字段 | 说明 |
|------|------|
| 时间戳 | `HH:MM:SS.微秒` 格式的事件发生时间 |
| CPU | 事件发生的 CPU 核心编号 |
| 事件类型 | `TASK_INIT`, `SCHEDULE`, `WORK_SUBMIT`, `TASK_DESTROY` |
| PID | 触发事件的进程 ID (tgid) |
| 进程名 | 触发事件的进程/线程名称 (comm, 最多16字符) |
| TSG | Task/Channel Group ID，GPU 调度的基本单位 |

## 事件类型详解

### 1. TASK_INIT - TSG 创建

当用户态程序（如 CUDA 应用）初始化 GPU context 时触发。

```
TASK_INIT PID=637320 python3 TSG=1 engine=COPY timeslice=1024 us interleave=UNKNOWN runlist=0
```

- **engine**: 引擎类型
  - `GRAPHICS` (0): 图形引擎
  - `COPY` (1): DMA 复制引擎
  - `NVDEC` (2): 视频解码引擎
  - `NVENC` (3): 视频编码引擎
  - `UNKNOWN`: 未知/通用计算引擎 (runlist 13/14 通常是计算引擎)
- **timeslice**: 时间片，单位微秒（默认 1024us）
- **interleave**: 交错级别（LOW/MEDIUM/HIGH）
- **runlist**: 运行队列 ID，不同引擎有不同的 runlist
  - runlist 0: COPY 引擎
  - runlist 13/14: 计算/图形引擎

### 2. SCHEDULE - TSG 调度

当 TSG 被调度/启用时触发，这是准入控制点。

```
SCHEDULE PID=637320 python3 TSG=1 channels=8 timeslice=1024 us interleave=LOW allowed=yes
```

- **channels**: TSG 中的 channel 数量
- **allowed**: 是否允许调度（eBPF 可以在此拒绝调度）

### 3. WORK_SUBMIT - 工作提交（需要 -v 参数）

当用户态查询 work submit token 时触发。

```
WORK_SUBMIT PID=637320 python3 TSG=1 channel=2 token=1073741826 gpu=0
```

- **channel**: Channel ID
- **token**: Work submit token，用于追踪工作完成
- **gpu**: GPU 实例 ID

**重要限制**: 这个 hook **不是每次 kernel launch 都会触发**！
- GPU kernel launch 是直接写 pushbuffer 给硬件，不经过内核
- work_submit hook 只在用户态主动通过 ioctl 查询 token 时触发
- 适合追踪同步点，不适合追踪每次 kernel 提交

### 4. TASK_DESTROY - TSG 销毁

当 TSG 被销毁时触发，通常在 CUDA context 销毁时。

```
TASK_DESTROY PID=637320 cuda-EvtHandlr TSG=1
```

注意：进程名变成了 `cuda-EvtHandlr`，这是 CUDA 运行时的事件处理线程负责清理。

## 典型 CUDA 应用事件序列

```
时间线:
  |
  | [应用启动]
  |   TASK_INIT (TSG=2, runlist=13)  - 计算引擎 TSG
  |   TASK_INIT (TSG=0, runlist=14)  - 计算引擎 TSG
  |   TASK_INIT (TSG=1, engine=COPY) - DMA 复制引擎 TSG
  |
  | [开始工作]
  |   SCHEDULE (TSG=1, channels=8)   - 调度 COPY 引擎
  |   TASK_INIT (TSG=3, runlist=13)  - 额外计算 TSG
  |   SCHEDULE (TSG=3, channels=4)   - 调度计算引擎
  |   SCHEDULE (TSG=1, channels=4)   - 调度计算引擎
  |
  | [工作执行中 - 无内核事件，GPU 直接执行 pushbuffer]
  |
  | [应用退出]
  |   TASK_DESTROY (TSG=0)
  |   TASK_DESTROY (TSG=2)
  |   TASK_DESTROY (TSG=1)
  |   TASK_DESTROY (TSG=3)
  |   TASK_DESTROY (TSG=1)
  v
```

## 关键观察

1. **多个 TSG**: 一个 CUDA 应用会创建多个 TSG，分别对应不同的引擎（计算、复制等）

2. **Runlist 分离**:
   - runlist 0: COPY 引擎
   - runlist 13/14: 计算引擎
   - 不同 runlist 可以并行执行

3. **进程名变化**: 创建时是主进程 (python3)，销毁时是 CUDA 事件处理线程 (cuda-EvtHandlr)

4. **调度时机**: SCHEDULE 事件在 TASK_INIT 之后，表示 TSG 准备好被 GPU 执行

## Hook 点源码位置

| Hook | 源文件 | 函数 |
|------|--------|------|
| task_init | kernel_channel_group.c | kchangrpConstruct_IMPL |
| schedule | kernel_channel_group_api.c | kchangrpapiCtrlCmdGpFifoSchedule_IMPL |
| work_submit | kernel_channel.c | kchannelNotifyWorkSubmitToken_IMPL |
| task_destroy | kernel_channel_group.c | kchangrpDestruct_IMPL |

## 统计信息解读

```
=== Statistics ===
task_init:    5      # 创建了 5 个 TSG
schedule:     3      # 3 次调度事件
work_submit:  36     # 36 次 token 查询（不等于 kernel launch 次数）
task_destroy: 5      # 销毁了 5 个 TSG
dropped:      0      # ring buffer 未丢失事件
```
