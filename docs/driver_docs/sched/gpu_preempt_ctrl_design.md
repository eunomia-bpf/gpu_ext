# GPU Preempt Control Tool Design Document

基于 eBPF tracepoint 的 GPU TSG 抢占控制工具设计文档。

## 概述

`gpu_preempt_ctrl` 是一个用户态工具，通过 eBPF tracepoint 监听 NVIDIA GPU TSG（Time Slice Group）生命周期事件，捕获 `hClient` 和 `hTsg` 句柄信息。

## 当前状态

### 工作的功能

1. **TSG 事件监控** - 通过 tracepoint 捕获 TSG 创建、调度、销毁事件
2. **句柄捕获** - 成功获取 hClient 和 hTsg 句柄
3. **进程追踪** - 按 PID 跟踪 TSG 归属

### 限制：跨进程 ioctl 不工作

测试结果显示，从外部进程发送 preempt ioctl 返回 `-22 (EINVAL)`：

```
=== Testing PREEMPT ===
  PREEMPT hClient=0xc1d00100 hTsg=0x5c000046 result=-22 (0xffffffea) duration=6 us
```

**原因**：NVIDIA RM（Resource Manager）的安全模型要求 ioctl 控制命令必须在**创建资源的同一进程上下文**中发起。RM 会检查 `clientOSInfo`（进程 fd 信息）来验证调用者身份。

这与 GPreempt 论文面临的问题相同，GPreempt 的解决方案是修改驱动绕过安全检查（见 `GPreempt.patch` 中 `Nv04ControlWithSecInfo` -> `Nv04Control`）。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Space                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              gpu_preempt_ctrl                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │ │
│  │  │ Ring Buffer  │  │ TSG Tracker  │  │ ioctl Commands   │   │ │
│  │  │   Reader     │  │   (in-mem)   │  │ [NOT WORKING]    │   │ │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │ │
│  └─────────┼─────────────────┼───────────────────┼─────────────┘ │
│            │                 │                   │               │
│            │ Works!          │ Works!            × EINVAL        │
└────────────┼─────────────────┼───────────────────┼───────────────┘
             │                 │                   │
═════════════╪═════════════════╪═══════════════════╪═══════════════
             ▼                 ▼                   ▼ Kernel Space
┌─────────────────────────────────────────────────────────────────┐
│              gpu_preempt_ctrl.bpf.c (eBPF)                      │
│  tracepoint/nvidia/nvidia_gpu_tsg_create   ───────────┐         │
│  tracepoint/nvidia/nvidia_gpu_tsg_schedule ───────────┤ Works!  │
│  tracepoint/nvidia/nvidia_gpu_tsg_destroy  ───────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NVIDIA Driver (nvidia.ko)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  NV_ESC_RM_CONTROL ioctl                                  │   │
│  │    → Nv04ControlWithSecInfo()                            │   │
│  │       → rmclientValidate() ← 检查 clientOSInfo 失败!     │   │
│  │                               返回 NV_ERR_INVALID_CLIENT │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 测试验证

### 1. 监控功能正常工作

```bash
$ sudo ./gpu_preempt_ctrl -v
Opened NVIDIA device (fd=3)
GPU Preempt Control started.
---
16:58:51.175171 [CPU01] TSG_CREATE   PID=486829 test_preempt_ct  hClient=0xc1d000cb hTsg=0x5c000013 tsg=1 engine=COPY timeslice=1024 runlist=0
16:58:51.202662 [CPU01] TSG_SCHEDULE PID=486829 test_preempt_ct  hClient=0xc1d000cb hTsg=0x5c000013 tsg=1 channels=8 timeslice=1024
16:58:51.204162 [CPU01] TSG_CREATE   PID=486829 test_preempt_ct  hClient=0xc1d000cb hTsg=0x5c000038 tsg=3 engine=UNKNOWN timeslice=1024 runlist=13
...

=== Statistics ===
tsg_create:   9
tsg_schedule: 3
tsg_destroy:  1
```

### 2. Preempt ioctl 失败

```bash
$ sudo ./test_preempt_ioctl ./test_preempt_ctrl
...
=== Testing PREEMPT ===
  PREEMPT hClient=0xc1d00100 hTsg=0x5c000046 result=-22 (EINVAL)

=== Testing SET_TIMESLICE ===
  SET_TIMESLICE result=-22

=== Testing SET_INTERLEAVE_LEVEL ===
  SET_INTERLEAVE result=-22
```

## 可行的解决方案

### 方案 1: 使用信号注入（推荐）

在目标 CUDA 进程中注入代码来执行 preempt ioctl：

```c
// 目标进程需要链接的库 libgpu_preempt.so
// 通过 LD_PRELOAD 或 ptrace 注入

void __attribute__((constructor)) init_preempt_handler(void) {
    // 注册信号处理器
    signal(SIGUSR1, handle_preempt_signal);
}

void handle_preempt_signal(int sig) {
    // 从共享内存读取要 preempt 的 hTsg
    uint32_t hClient = shm->hClient;
    uint32_t hTsg = shm->hTsg;

    // 在目标进程上下文中执行 ioctl - 这会成功！
    do_preempt(nvidia_fd, hClient, hTsg, 100);
}
```

控制流程：
```
gpu_preempt_ctrl                    Target CUDA Process
     │                                      │
     │ 1. Capture hClient/hTsg              │
     │    via tracepoint                    │
     │                                      │
     │ 2. Write to shared memory            │
     │──────────────────────────────────────>│
     │                                      │
     │ 3. Send SIGUSR1                      │
     │──────────────────────────────────────>│
     │                                      │ 4. Signal handler
     │                                      │    reads shm
     │                                      │    calls preempt ioctl
     │                                      │    (SUCCESS!)
```

### 方案 2: 修改驱动绕过安全检查

类似 GPreempt patch，修改 `escape.c`：

```c
// 在 NV_ESC_RM_CONTROL 处理中
-   Nv04ControlWithSecInfo(pApi, secInfo);
+   Nv04Control(pApi);  // 绕过 clientOSInfo 检查
```

**缺点**：安全风险，任意进程可控制任意 GPU 资源。

### 方案 3: 添加特权 ioctl

添加新的 ioctl `NV_ESC_RM_PREEMPT_TSG`，仅 root 可用：

```c
case NV_ESC_RM_PREEMPT_TSG:
{
    // 仅允许 root
    if (!capable(CAP_SYS_ADMIN))
        return -EPERM;

    // 直接查找 TSG 并 preempt，不检查 client ownership
    status = kchangrpPreemptTsg(pApi->hClient, pApi->hObject);
    break;
}
```

### 方案 4: 目标进程主动集成

修改 CUDA 应用程序，主动检查控制命令：

```c
// 在 CUDA 应用中
void gpu_work_loop() {
    while (running) {
        // 检查是否需要 preempt
        if (check_preempt_request()) {
            do_self_preempt();
        }

        launch_kernel<<<...>>>();
        cudaDeviceSynchronize();
    }
}
```

## 文件结构

```
src/
├── gpu_preempt_ctrl.bpf.c      # eBPF 程序 (tracepoint handlers) [工作]
├── gpu_preempt_ctrl.c          # 用户态程序 (监控功能工作，ioctl 不工作)
├── gpu_preempt_ctrl_event.h    # 共享数据结构定义

docs/driver_docs/sched/test/
├── test_preempt_ctrl.cu        # CUDA 测试程序
├── test_preempt_ioctl.c        # ioctl 测试程序（验证失败）
└── test_gpu_preempt.sh         # 测试脚本
```

## 当前工具用途

虽然 preempt ioctl 不能跨进程工作，当前工具仍有价值：

1. **监控和观测** - 实时查看 GPU TSG 活动
2. **调试辅助** - 跟踪 CUDA 应用的 GPU 资源使用
3. **研究基础** - 为后续实现提供句柄信息
4. **集成准备** - 配合信号注入方案使用

## 使用方法

### 编译

```bash
cd src/
make gpu_preempt_ctrl
```

### 监控 GPU TSG 活动

```bash
# 启动监控（verbose 模式）
sudo ./gpu_preempt_ctrl -v

# 在另一个终端运行 CUDA 程序
./any_cuda_program

# 观察输出
17:05:43.123 [CPU01] TSG_CREATE   PID=12345 my_cuda_app  hClient=0x... hTsg=0x...
17:05:43.456 [CPU01] TSG_SCHEDULE PID=12345 my_cuda_app  channels=8 timeslice=1024
...
```

### 交互命令（监控功能）

```
gpu> list
=== Tracked TSGs (3) ===
hClient    hTsg       tsg_id     engine   runlist  timeslice  level    process
--------------------------------------------------------------------------------
0xc1d0008b 0x5c000013 1          COPY     0        1024       LOW      my_app (pid=12345)

gpu> help
gpu> quit
```

## 下一步计划

1. **实现信号注入方案** - 最少侵入性的方法
2. **或者修改驱动添加特权 ioctl** - 如果需要系统级控制
3. **完善监控功能** - 添加统计、导出、告警等

## 参考

- [GPreempt.patch](./GPreempt.patch) - GPreempt 论文的驱动修改
- [test_preempt_cuda.c](./test_preempt_cuda.c) - 原始的 preempt 测试程序
- NVIDIA open-gpu-kernel-modules escape.c - RM ioctl 处理
