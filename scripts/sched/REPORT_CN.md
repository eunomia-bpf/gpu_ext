# CPU 调度器与 IRQ 对 GPU 工作负载的影响：定量分析研究

GPU 工作负载在 AI/ML 应用中日益关键，但其性能可能受到 CPU 端调度决策和中断处理的显著影响。受 Meta 关于 AI 训练优化的 sched_ext 研究启发，本研究定量分析了 CPU 调度器上下文切换和硬件/软件中断对 GPU kernel launch 性能的影响。

我们开发了基于 eBPF 的追踪工具 (`cuda_sched_trace`)，以纳秒级精度捕获 CUDA kernel launch、CPU 调度器事件和 IRQ 事件。使用 Qwen3 0.6B LLM 推理作为基准工作负载，我们在六种场景下进行了系统性实验：干净基线、CPU 密集型干扰 (stress-ng)、网络 I/O 干扰 (iperf3)、磁盘 I/O 干扰 (fio)、组合重度负载，以及优化配置。

**主要发现**：
- 在干净环境中，调度器影响很小（运行时间的 1.2%）
- IRQ 开销对本地推理可忽略不计（0.0276%）
- CPU 密集型 noisy neighbor 导致上下文切换增加 524 倍，性能下降 8.8%
- 组合重度负载（CPU+Network+Disk）导致 **20.5% 性能下降** — 最严重的情况
- 使用 `taskset` 进行 CPU 绑核可减少 **96.3%** 的上下文切换，但无法完全消除干扰

这些结果表明，虽然在干净环境中调度器优化价值有限，但在类似生产环境的 noisy neighbor 条件下变得至关重要。

---

## 1. 引言

### 1.1 背景

现代 GPU 计算工作负载，特别是大型语言模型（LLM）推理和 AI 训练，需要 CPU 和 GPU 执行之间的紧密协调。CPU 负责：
- 准备输入数据和 kernel 参数
- 通过 CUDA API 调用启动 GPU kernel
- 管理内存传输和同步

对这一 CPU 端工作流程的任何中断都可能延迟 GPU kernel 提交，潜在地导致 GPU 空闲时间和吞吐量降低。

### 1.2 动机

Meta 关于 AI 训练 sched_ext 的研究（LPC 2025）指出：
- "IRQ 抢占我们的重要任务"是常见的生产环境问题
- 网络中断（NET_RX/NET_TX）和块设备中断显著影响训练吞吐量
- 自定义调度策略可将 AI 工作负载性能提升 5-20%

然而，实际影响因工作负载特性而显著不同。本研究旨在：
1. 量化 CPU 调度对 GPU 工作负载的真实影响
2. 区分调度器问题和应用固有行为
3. 评估常见优化技术的有效性
4. 为生产部署提供可操作的建议

### 1.3 研究目标

**主要目标**：确定 CPU 调度器优化对 GPU 工作负载是否值得投入，以及在什么条件下能提供有意义的收益。

**具体目标**：
1. 测量 CPU 调度对 GPU kernel launch 的基线影响
2. 表征 IRQ 干扰模式及其性能成本
3. 量化各种 noisy neighbor 场景的影响
4. 评估 CPU 绑核和优先级优化的有效性

---

## 2. 方法论

### 2.1 追踪工具设计

我们开发了 `cuda_sched_trace`，一个基于 eBPF 的追踪工具，使用以下组件：

#### 2.1.1 CUDA API 追踪 (uprobes)

```c
// 附加到 CUDA Driver API
SEC("uprobe/cuLaunchKernel")
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    // 捕获: 时间戳, pid, tid, grid/block 维度, shared memory, stream
    // 将进程标记为 GPU 进程以便调度器追踪
}

// 附加到 CUDA Runtime API
SEC("uprobe/cudaLaunchKernel")
int trace_cudaLaunchKernel(struct pt_regs *ctx) { ... }

SEC("uprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_enter(struct pt_regs *ctx) { ... }

SEC("uretprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_exit(struct pt_regs *ctx) { ... }
```

#### 2.1.2 调度器事件追踪 (tracepoints)

```c
SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next) {
    // 仅当 prev 或 next 是 GPU 进程时追踪
    // 记录: 时间戳, prev/next pid, off-cpu/on-cpu 持续时间
}
```

#### 2.1.3 IRQ 追踪 (tracepoints)

```c
SEC("tp_btf/irq_handler_entry")
int BPF_PROG(irq_handler_entry, int irq, struct irqaction *action) {
    // 追踪硬中断入口，记录 IRQ 编号和处理器名称
}

SEC("tp_btf/irq_handler_exit")
int BPF_PROG(irq_handler_exit, int irq, struct irqaction *action) {
    // 计算 IRQ 持续时间
}

SEC("tp_btf/softirq_entry")
int BPF_PROG(softirq_entry, unsigned int vec_nr) {
    // 追踪软中断: TIMER, NET_RX, NET_TX, BLOCK, SCHED, RCU 等
}

SEC("tp_btf/softirq_exit")
int BPF_PROG(softirq_exit, unsigned int vec_nr) {
    // 计算软中断持续时间
}
```

#### 2.1.4 数据收集架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户空间                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ GPU 应用    │    │ cuda_sched  │    │ 分析脚本            │  │
│  │ (qwen3.cu)  │    │ _trace      │    │ (Python)            │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘  │
│         │                  │                       │             │
│         │ CUDA 调用        │ perf_event           │ CSV 解析    │
│         ▼                  ▼                       ▼             │
├─────────────────────────────────────────────────────────────────┤
│                         内核空间                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ uprobes     │    │ tracepoints │    │ BPF Ring Buffer     │  │
│  │ (CUDA API)  │    │ (sched,irq) │    │ (事件队列)          │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 测试程序

**基准测试**: Qwen3 0.6B LLM 推理 (qwen3.cu)

| 属性 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B-FP32 |
| 任务 | 单轮问答 |
| 输入 | "What is eBPF?" |
| 输出 | 约 30-50 tokens |
| Kernel 模式 | 批量提交（每个 token 约 950 次 launch） |
| GPU 内存 | 约 3 GB |

**为什么选择这个基准?**
- 代表现代 LLM 推理工作负载
- 混合计算密集型和内存密集型 kernel
- 清晰的批量提交模式（每个 token 多个 transformer 层）
- 可测量的吞吐量指标（tok/s）

### 2.3 测试环境

| 组件 | 规格 |
|------|------|
| CPU | 24 核心 |
| GPU | 支持 CUDA 的 NVIDIA GPU |
| 内存 | 足够支持模型 + 系统 |
| 操作系统 | Linux 6.15.11-061511-generic |
| 内核 | 启用 BTF 以支持 CO-RE eBPF |
| CUDA | Driver API + Runtime API |

### 2.4 干扰工具

| 工具 | 用途 | 配置 |
|------|------|------|
| stress-ng | CPU 负载 | `--cpu 0 --cpu-method fft`（所有核心） |
| iperf3 | 网络 I/O | 服务端 + 客户端，10 并行流，60 秒 |
| fio | 磁盘 I/O | `randwrite, bs=4k, iodepth=32, 4 jobs` |

### 2.5 测试场景

| 场景 | 描述 | 干扰 |
|------|------|------|
| Baseline | 干净环境 | 无 |
| Noisy CPU | CPU 密集型 | stress-ng 在所有核心 |
| Noisy Network | 网络 I/O | iperf3 本地环回 |
| Noisy Disk | 磁盘 I/O | fio 随机写入 |
| Heavy Load | 组合 | CPU + Network + Disk 同时运行 |
| Optimized | CPU 绑核 | stress-ng + taskset -c 0-3 + nice -n -10 |

### 2.6 分析方法

#### 2.6.1 Launch Pair 分析

比较连续的 kernel launch 以识别调度器影响：

```
Launch_i → [间隔] → Launch_i+1

Group A: 间隔内无上下文切换的 launch（正常流程）
Group B: 间隔内有上下文切换的 launch（被抢占）

抢占惩罚 = median(Group B 间隔) - median(Group A 间隔)
```

#### 2.6.2 归一化指标

为了考虑可变的工作负载长度，所有指标都进行归一化：

```
Sched/1K = (上下文切换总数 / Kernel Launch 总数) × 1000
IRQ/1K = (IRQ 总数 / Kernel Launch 总数) × 1000
```

#### 2.6.3 性能影响计算

```
性能下降 % = (Baseline tok/s - 场景 tok/s) / Baseline tok/s × 100
```

### 2.7 数据收集命令

```bash
# 启动追踪
sudo ./cuda_sched_trace > trace.csv 2> trace.log &
TRACE_PID=$!

# 运行基准测试
cd qwen3.cu
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1

# 停止追踪
sudo kill -SIGINT $TRACE_PID

# 分析结果
python3 analyze_scheduler_impact.py
```

---

## 3. 研究问题与结果

### RQ1: CPU 调度器在干净环境中是否显著影响 GPU 性能？

#### 3.1.1 实验设计

- **条件**: 干净系统，无人工干扰
- **指标**: 上下文切换频率、抢占惩罚、总运行时间影响
- **分析**: Launch pair 对比（有/无上下文切换）

#### 3.1.2 结果

**基础统计**:

| 指标 | 值 |
|------|-----|
| 总运行时间 | 79.5 秒 |
| Kernel Launch 数 | 51,464 |
| 上下文切换 | 592 (7.44 Hz) |
| OFF-CPU 时间 | 7.88 ms (0.01%) |

**Launch Pair 分析**:

| 分组 | 数量 | 百分比 | P50 间隔 | P90 间隔 | P99 间隔 |
|------|------|--------|----------|----------|----------|
| 无上下文切换 | 51,401 | 99.9% | 2 µs | 4 µs | 4 µs |
| 有上下文切换 | 62 | 0.1% | 15.3 ms | 15.5 ms | 5.0 s |

**抢占惩罚**: 15.3 ms（中位数）

**尾延迟归因**:

| 百分位 | 总异常值 | 有上下文切换 | 归因 |
|--------|----------|--------------|------|
| P95+ | 2,580 | 62 (2.4%) | 97.6% 应用问题 |
| P99+ | 515 | 62 (12.0%) | 88.0% 应用问题 |

**总调度器影响**:
```
影响 = 受影响的 pairs × 惩罚 = 62 × 15ms = 0.93 秒
百分比 = 0.93 / 79.5 = 1.2%
```

#### 3.1.3 发现

**在干净环境中，CPU 调度器影响很小（1.2%）**。绝大多数（99.9%）的 kernel launch pair 不受上下文切换影响。尾延迟主要由应用行为（token 生成边界）引起，而非调度器抢占。

---

### RQ2: IRQ 中断对 GPU 性能的影响有多大？

#### 3.2.1 实验设计

- **条件**: 启用 IRQ 追踪的干净系统
- **指标**: IRQ 频率、持续时间、类型分布
- **分析**: IRQ 时间占总运行时间的百分比

#### 3.2.2 结果

**IRQ 概览**:

| 指标 | 值 |
|------|-----|
| 总运行时间 | 4.99 秒 |
| Kernel Launch 数 | 125,236 |
| 软中断 | 653 事件 |
| 硬中断 | 0 事件 |

**软中断类型分布**:

| 类型 | 次数 | 总时间 | 平均时间 | 最大时间 | 百分比 |
|------|------|--------|----------|----------|--------|
| TIMER | 317 | 0.77 ms | 2.4 µs | 30.1 µs | 49% |
| RCU | 291 | 0.40 ms | 1.4 µs | 17.2 µs | 45% |
| NET_RX | 30 | 0.13 ms | 4.5 µs | 14.0 µs | 4.6% |
| SCHED | 15 | 0.07 ms | 4.9 µs | 18.9 µs | 2.3% |

**总 IRQ 影响**:
```
总 IRQ 时间: 1.38 ms
运行时间占比: 0.0276%
```

#### 3.2.3 理论影响 vs 实际影响

**理论关注点**:
1. **直接时间成本**: IRQ handler 执行时间
2. **Cache 污染**: IRQ 将 GPU 进程数据从 L1/L2/L3 驱逐
3. **流水线停顿**: 中断时 CPU 流水线刷新
4. **延迟累积**: 关键路径上的延迟

**为什么 qwen3 的实际影响较低**:

1. **批量提交模式**: 950 次 launch 在 <100µs 窗口内
   - IRQ 很少在如此短的窗口内发生
   - 大多数 IRQ 发生在 burst 之间（CPU 计算期间）

2. **TIMER 占主导（49%）**:
   - Cache footprint 小
   - 仅访问内核定时器结构
   - Cache 污染最小

3. **无网络 I/O**:
   - NET_RX 仅 30 事件（4.6%）
   - 分布式训练会看到更高的 NET_RX

4. **无数据加载**:
   - 零硬中断
   - 无 NVMe/SSD 块设备中断

#### 3.2.4 发现

**IRQ 影响对本地 LLM 推理可忽略不计（0.0276%）**。然而，这一发现是特定于工作负载的。具有网络通信或即时数据加载的分布式训练会经历显著更高的 IRQ 影响（估计 5-20%）。

---

### RQ3: Noisy Neighbor 如何影响 GPU 性能？

#### 3.3.1 实验设计

使用相同的 GPU 工作负载测试六种场景：

| 场景 | 干扰 | 目的 |
|------|------|------|
| Baseline | 无 | 参考点 |
| Noisy CPU | stress-ng（所有核心） | CPU 竞争 |
| Noisy Network | iperf3（10 流） | 网络 IRQ |
| Noisy Disk | fio（4 jobs，randwrite） | 块 IRQ |
| Heavy Load | 三者组合 | 生产环境模拟 |
| Optimized | CPU stress + taskset + nice | 缓解测试 |

#### 3.3.2 结果

**归一化指标（每 1000 个 kernel launch）**:

| 场景 | Launch 数 | Sched/1K | Soft IRQ/1K | Hard IRQ/1K | IRQ 时间 (ms) |
|------|----------|----------|-------------|-------------|---------------|
| Baseline | 56,882 | 22.8 | 5.8 | 0.0 | 0.62 |
| Noisy CPU | 61,184 | **11,932.8** | 6.4 | 0.0 | 0.33 |
| Noisy Network | 154,394 | 6.0 | 2.7 | 0.0 | 0.92 |
| Noisy Disk | 126,670 | 29.3 | 3.9 | **0.1** | 1.03 |
| **Heavy Load** | **99,424** | **6,044.6** | 2.4 | 0.0 | 0.37 |
| Optimized | 108,984 | 445.2 | 2.8 | 0.0 | 0.71 |

**性能影响**:

| 场景 | tok/s | 运行时间 (s) | 性能下降 | 上下文切换增加 |
|------|-------|-------------|----------|----------------|
| Baseline | 54.77 | 3.00 | - | 1x |
| Noisy CPU | 49.93 | 4.15 | **8.8%** | **524x** |
| Noisy Network | 53.23 | 7.22 | 2.8% | 0.26x |
| Noisy Disk | 54.95 | 5.60 | -0.3% | 1.3x |
| **Heavy Load** | **43.56** | **6.97** | **20.5%** | **265x** |
| Optimized | 53.75 | 5.10 | 1.9% | 19.5x |

#### 3.3.3 按场景详细分析

**Noisy CPU (stress-ng)**:
- 上下文切换增加 **524 倍**（22.8 → 11,932.8 每 1K launch）
- 性能下降 **8.8%**
- 机制: CFS 调度器在 GPU 进程和 stress-ng 工作进程之间分时

**Noisy Network (iperf3)**:
- 上下文切换实际**减少**（网络负载减少 CPU 竞争）
- 软中断略有增加
- 性能仅下降 **2.8%**
- 发现: 网络 I/O 主要影响 IRQ，而非调度

**Noisy Disk (fio)**:
- 首次出现**硬中断**（BLOCK 中断）
- 上下文切换保持较低
- 性能几乎不变（-0.3%）
- 发现: 磁盘 I/O 对 GPU 工作负载影响最小

**Heavy Load (CPU + Network + Disk)**:
- 性能下降 **20.5%** — 最严重的情况
- 上下文切换: 6,044.6/1K（265 倍增加）
- 有趣的是，仅为 Noisy CPU 上下文切换的 **50.7%**
- **关键洞察**: 干扰相互竞争，但组合效应仍然最严重

**Heavy Load 软中断分解**:

| 类型 | 次数 | 总时间 | 平均时间 |
|------|------|--------|----------|
| RCU | 213 | 217.4 µs | 1.0 µs |
| TIMER | 17 | 122.9 µs | 7.2 µs |
| SCHED | 5 | 33.3 µs | 6.7 µs |

#### 3.3.4 发现

**Noisy neighbor 显著影响 GPU 性能，组合干扰导致 20.5% 性能下降**。不同干扰类型有不同的特征：
- CPU 竞争 → 上下文切换大幅增加
- 网络 I/O → IRQ 开销
- 磁盘 I/O → 块中断但性能影响最小
- 组合 → 由于累积效应，总体影响最严重

---

### RQ4: CPU 绑核能否有效缓解调度器影响？

#### 3.4.1 实验设计

- **基线**: Noisy CPU 场景（stress-ng 在所有核心）
- **优化**: 相同的 stress-ng + GPU 进程使用：
  - `taskset -c 0-3`（绑定到核心 0-3）
  - `nice -n -10`（更高优先级）

#### 3.4.2 结果

**对比**:

| 指标 | Noisy CPU | Optimized | 改善 |
|------|-----------|-----------|------|
| Sched/1K | 11,932.8 | 445.2 | **96.3% 减少** |
| tok/s | 49.93 | 53.75 | **7.6% 提升** |
| vs. Baseline | 8.8% 更慢 | 1.9% 更慢 | 显著恢复 |

**剩余差距分析**:
- Optimized 仍有 445.2 sched/1K，而 Baseline 为 22.8
- 这仍是 Baseline 的 **19.5 倍**
- CPU 绑核减少了竞争但无法完全消除

#### 3.4.3 为什么无法完全消除

1. **共享核心竞争**: stress-ng 工作进程仍可能被调度到核心 0-3
2. **内核任务**: 系统守护进程和内核线程无法被排除
3. **IRQ 亲和性**: 中断仍可能针对绑定的核心

#### 3.4.4 更好隔离的建议

```bash
# 1. 使用 isolcpus 内核参数（启动时）
isolcpus=4-7 nohz_full=4-7

# 2. 将 GPU 进程绑定到隔离的核心
taskset -c 4-7 ./gpu_app

# 3. 将 IRQ 绑定到远离 GPU 的核心
echo 0-3 > /proc/irq/*/smp_affinity_list

# 4. 使用 cgroups 进行 CPU 隔离
cgcreate -g cpu:gpu_workload
cgset -r cpuset.cpus=4-7 gpu_workload
cgexec -g cpu:gpu_workload ./gpu_app
```

#### 3.4.5 发现

**CPU 绑核非常有效，减少 96.3% 的上下文切换并恢复 7.6% 的性能**。然而，在重负载条件下无法完全恢复到基线性能。完全隔离需要内核级配置（isolcpus）和 IRQ 亲和性管理。

---

## 4. 讨论

### 4.1 关键洞察

1. **环境很重要**: 调度器影响从 1.2%（干净）到 20.5%（重负载）不等

2. **工作负载模式至关重要**: 批量提交模式（如 qwen3）对中断更有弹性，因为 IRQ 通常发生在 burst 之间

3. **不同干扰有不同特征**:
   | 干扰 | 主要影响 | 次要影响 |
   |------|----------|----------|
   | CPU | 上下文切换 | 无 |
   | 网络 | IRQ 开销 | 轻微调度 |
   | 磁盘 | 硬中断 | 最小 |
   | 组合 | 以上全部 | 最严重 |

4. **优化有效性**:
   - CPU 绑核: 非常有效（96% 减少）
   - 优先级调整: 有帮助但有限
   - 完全隔离: 需要内核配置

### 4.2 与 Meta 研究的对比

| 方面 | Meta（AI 训练） | 本研究（LLM 推理） |
|------|----------------|-------------------|
| 主要问题 | 网络 IRQ (NET_RX) | CPU 调度 |
| IRQ 影响 | 5-20% | 0.03%（本地推理） |
| 优化方案 | sched_ext layer | taskset + nice |
| 工作负载 | 分布式训练 | 单节点推理 |

**关键区别**: Meta 的分布式训练有持续的网络通信（all-reduce），使 NET_RX 成为主要瓶颈。本地推理网络 I/O 最少。

### 4.3 局限性

1. **eBPF 开销**: 追踪本身增加 1-5% 开销
2. **仅 CUDA**: 不支持其他 GPU API（OpenCL、HIP）
3. **无 GPU 端数据**: 无法测量实际 kernel 执行时间
4. **有限的 IRQ 归因**: 无法识别哪个进程导致了 IRQ
5. **单 GPU**: 未测试多 GPU 场景

### 4.4 实践建议

**生产部署**:

| 环境 | 建议 | 预期收益 |
|------|------|----------|
| 专用服务器 | 无需优化 | - |
| 共享服务器（轻负载） | taskset + nice | 5-10% 提升 |
| 共享服务器（重负载） | isolcpus + IRQ 亲和性 | 15-20% 提升 |
| Kubernetes | CPU limits + nodeSelector | 视情况而定 |

**决策树**:
```
GPU 工作负载对延迟敏感吗？
├── 否 → 无需优化
└── 是 → 服务器是共享的吗？
    ├── 否 → 仅监控，需要时优化
    └── 是 → 共存负载有多重？
        ├── 轻 → taskset + nice
        └── 重 → isolcpus + 专用核心
```

---

## 5. 结论

本研究为 CPU 调度器和 IRQ 对 GPU 工作负载的影响提供了定量证据：

1. **干净环境显示影响最小**（1.2% 调度器，0.03% IRQ），表明调度器优化并非普遍必要。

2. **Noisy neighbor 条件导致显著性能下降**（组合 CPU+Network+Disk 负载高达 20.5%），证明了共享环境中资源隔离的重要性。

3. **不同干扰类型需要不同的缓解策略**:
   - CPU 竞争 → CPU 绑核（96.3% 减少）
   - 网络 IRQ → IRQ 亲和性 + 中断合并
   - 磁盘 IRQ → I/O 调度器调优

4. **追踪工具成功识别和量化**调度问题，使数据驱动的优化决策成为可能，而非猜测。

**建议**: 在投资调度器优化之前，使用提供的追踪工具分析您的特定工作负载。影响因工作负载特性和部署环境而显著不同。

---

## 6. 附录

### A. 工具安装和使用

```bash
# 构建追踪工具
cd tools
make cuda_sched_trace

# 运行追踪
sudo ./cuda_sched_trace > trace.csv 2> trace.log &

# 在另一个终端运行 GPU 工作负载
./your_gpu_app

# 停止追踪
sudo pkill cuda_sched_trace

# 分析结果
cd ../scripts/sched
python3 analyze_scheduler_impact.py /path/to/trace.csv
```

### B. CSV 输出格式

| 字段 | 描述 |
|------|------|
| timestamp_ns | 相对时间戳（纳秒） |
| event_type | cuLaunchKernel, cudaLaunchKernel, syncEnter, syncExit, schedSwitch, hardirqEntry, hardirqExit, softirqEntry, softirqExit |
| pid, tid | 进程/线程 ID |
| comm | 进程名 |
| cpu | CPU 核心编号 |
| grid_x/y/z | CUDA grid 维度（launch 事件） |
| block_x/y/z | CUDA block 维度（launch 事件） |
| shared_mem | Shared memory 大小（launch 事件） |
| stream | CUDA stream 指针（launch 事件） |
| last_offcpu_ns | 上次 OFF-CPU 时间戳（schedSwitch） |
| last_oncpu_ns | 上次 ON-CPU 时间戳（schedSwitch） |
| irq_num | IRQ 编号（硬中断）或向量（软中断） |
| irq_name | 处理器名称或类型（TIMER, NET_RX 等） |
| duration_ns | IRQ 持续时间（仅 exit 事件） |

### C. Noisy Neighbor 测试脚本

```bash
#!/bin/bash
# test_noisy_neighbor.sh - 完整测试套件

# 场景：
# 1. Baseline（干净）
# 2. Noisy CPU（stress-ng）
# 3. Noisy Network（iperf3）
# 4. Noisy Disk（fio）
# 5. Heavy Load（三者组合）
# 6. Optimized（taskset + nice）

# 完整实现见 scripts/sched/test_noisy_neighbor.sh
```

---

## 参考文献

1. Meta Platforms, Inc. "Accelerating AI Training with sched_ext." Linux Plumbers Conference 2025. https://lpc.events/event/19/contributions/2039/

2. NVIDIA Corporation. "CUDA Driver API Reference." https://docs.nvidia.com/cuda/cuda-driver-api/

3. Linux Kernel Documentation. "BPF Documentation." https://www.kernel.org/doc/html/latest/bpf/

4. stress-ng. "A tool to load and stress a computer system." https://github.com/ColinIanKing/stress-ng

5. iperf3. "A TCP, UDP, and SCTP network bandwidth measurement tool." https://github.com/esnet/iperf

6. fio. "Flexible I/O Tester." https://github.com/axboe/fio

---

**工件**:
- 追踪工具: `tools/cuda_sched_trace`
- 测试脚本: `scripts/sched/test_noisy_neighbor.sh`
- 分析脚本: `scripts/sched/analyze_scheduler_impact.py`
- 本报告: `scripts/sched/REPORT_CN.md`
- 英文版: `scripts/sched/REPORT.md`
- 快速参考: `scripts/sched/README.md`
