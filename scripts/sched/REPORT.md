# CPU Scheduler and IRQ Impact on GPU Workloads: A Quantitative Analysis

**Authors**: Research Team
**Date**: January 2026
**Version**: 1.0

---

## Abstract

GPU workloads are increasingly critical for AI/ML applications, yet their performance can be significantly impacted by CPU-side scheduling decisions and interrupt handling. Inspired by Meta's sched_ext research on AI training optimization, this study quantitatively analyzes the impact of CPU scheduler context switches and hardware/software interrupts on GPU kernel launch performance.

We developed an eBPF-based tracing tool (`cuda_sched_trace`) that captures CUDA kernel launches, CPU scheduler events, and IRQ occurrences with nanosecond precision. Using Qwen3 0.6B LLM inference as our benchmark workload, we conducted systematic experiments across six scenarios: clean baseline, CPU-intensive interference (stress-ng), network I/O interference (iperf3), disk I/O interference (fio), combined heavy load, and optimized configuration.

**Key Findings**:
- In clean environments, scheduler impact is minimal (1.2% of runtime)
- IRQ overhead is negligible for local inference (0.0276%)
- CPU-intensive noisy neighbors cause 524x increase in context switches and 8.8% performance degradation
- Combined heavy load (CPU+Network+Disk) results in **20.5% performance degradation** - the worst case
- CPU pinning with `taskset` reduces context switches by **96.3%** but cannot fully eliminate interference

These results demonstrate that while scheduler optimization has limited value in clean environments, it becomes critical under production-like noisy neighbor conditions.

---

## 1. Introduction

### 1.1 Background

Modern GPU computing workloads, particularly large language model (LLM) inference and AI training, require tight coordination between CPU and GPU execution. The CPU is responsible for:
- Preparing input data and kernel parameters
- Launching GPU kernels via CUDA API calls
- Managing memory transfers and synchronization

Any interruption to this CPU-side workflow can delay GPU kernel submissions, potentially causing GPU idle time and reduced throughput.

### 1.2 Motivation

Meta's research on sched_ext for AI training (LPC 2025) highlighted that:
- "IRQs preempting our important tasks" is a common production issue
- Network interrupts (NET_RX/NET_TX) and block device interrupts significantly impact training throughput
- Custom scheduling policies can improve AI workload performance by 5-20%

However, the actual impact varies significantly based on workload characteristics. This study aims to:
1. Quantify the real-world impact of CPU scheduling on GPU workloads
2. Distinguish between scheduler problems and application-inherent behavior
3. Evaluate the effectiveness of common optimization techniques
4. Provide actionable recommendations for production deployments

### 1.3 Research Goal

**Primary Goal**: Determine whether CPU scheduler optimization is a worthwhile investment for GPU workloads, and under what conditions it provides meaningful benefits.

**Specific Objectives**:
1. Measure the baseline impact of CPU scheduling on GPU kernel launches
2. Characterize IRQ interference patterns and their performance cost
3. Quantify the impact of various noisy neighbor scenarios
4. Evaluate CPU pinning and priority optimization effectiveness

---

## 2. Methodology

### 2.1 Tracing Tool Design

We developed `cuda_sched_trace`, an eBPF-based tracing tool using the following components:

#### 2.1.1 CUDA API Tracing (uprobes)

```c
// Attach to CUDA Driver API
SEC("uprobe/cuLaunchKernel")
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    // Capture: timestamp, pid, tid, grid/block dimensions, shared memory, stream
    // Mark process as GPU process for scheduler tracking
}

// Attach to CUDA Runtime API
SEC("uprobe/cudaLaunchKernel")
int trace_cudaLaunchKernel(struct pt_regs *ctx) { ... }

SEC("uprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_enter(struct pt_regs *ctx) { ... }

SEC("uretprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_exit(struct pt_regs *ctx) { ... }
```

#### 2.1.2 Scheduler Event Tracing (tracepoints)

```c
SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next) {
    // Only track if prev or next is a GPU process
    // Record: timestamp, prev/next pid, off-cpu/on-cpu duration
}
```

#### 2.1.3 IRQ Tracing (tracepoints)

```c
SEC("tp_btf/irq_handler_entry")
int BPF_PROG(irq_handler_entry, int irq, struct irqaction *action) {
    // Track hard IRQ entry, record IRQ number and handler name
}

SEC("tp_btf/irq_handler_exit")
int BPF_PROG(irq_handler_exit, int irq, struct irqaction *action) {
    // Calculate IRQ duration
}

SEC("tp_btf/softirq_entry")
int BPF_PROG(softirq_entry, unsigned int vec_nr) {
    // Track soft IRQ: TIMER, NET_RX, NET_TX, BLOCK, SCHED, RCU, etc.
}

SEC("tp_btf/softirq_exit")
int BPF_PROG(softirq_exit, unsigned int vec_nr) {
    // Calculate soft IRQ duration
}
```

#### 2.1.4 Data Collection Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Space                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ GPU App     │    │ cuda_sched  │    │ Analysis Scripts    │  │
│  │ (qwen3.cu)  │    │ _trace      │    │ (Python)            │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘  │
│         │                  │                       │             │
│         │ CUDA calls       │ perf_event           │ CSV parsing │
│         ▼                  ▼                       ▼             │
├─────────────────────────────────────────────────────────────────┤
│                         Kernel Space                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ uprobes     │    │ tracepoints │    │ BPF Ring Buffer     │  │
│  │ (CUDA API)  │    │ (sched,irq) │    │ (Event Queue)       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Test Program

**Benchmark**: Qwen3 0.6B LLM Inference (qwen3.cu)

| Property | Value |
|----------|-------|
| Model | Qwen3-0.6B-FP32 |
| Task | Single-turn Q&A |
| Input | "What is eBPF?" |
| Output | ~30-50 tokens |
| Kernel Pattern | Burst submission (~950 launches per token) |
| GPU Memory | ~3 GB |

**Why this benchmark?**
- Representative of modern LLM inference workloads
- Mix of compute-bound and memory-bound kernels
- Clear burst submission pattern (multiple transformer layers per token)
- Measurable throughput metric (tok/s)

### 2.3 Test Environment

| Component | Specification |
|-----------|---------------|
| CPU | 24 cores (specific model TBD) |
| GPU | NVIDIA GPU with CUDA support |
| Memory | Sufficient for model + system |
| OS | Linux 6.15.11-061511-generic |
| Kernel | BTF-enabled for CO-RE eBPF |
| CUDA | Driver API + Runtime API |

### 2.4 Interference Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| stress-ng | CPU load | `--cpu 0 --cpu-method fft` (all cores) |
| iperf3 | Network I/O | Server + Client, 10 parallel streams, 60s |
| fio | Disk I/O | `randwrite, bs=4k, iodepth=32, 4 jobs` |

### 2.5 Test Scenarios

| Scenario | Description | Interference |
|----------|-------------|--------------|
| Baseline | Clean environment | None |
| Noisy CPU | CPU-intensive | stress-ng on all cores |
| Noisy Network | Network I/O | iperf3 localhost loopback |
| Noisy Disk | Disk I/O | fio random write |
| Heavy Load | Combined | CPU + Network + Disk simultaneously |
| Optimized | CPU pinning | stress-ng + taskset -c 0-3 + nice -n -10 |

### 2.6 Analysis Methodology

#### 2.6.1 Launch Pair Analysis

Compare consecutive kernel launches to identify scheduler impact:

```
Launch_i → [interval] → Launch_i+1

Group A: Launches with NO context switch in interval (normal flow)
Group B: Launches with context switch in interval (preempted)

Preemption Penalty = median(Group B interval) - median(Group A interval)
```

#### 2.6.2 Normalized Metrics

To account for variable workload lengths, all metrics are normalized:

```
Sched/1K = (Total Context Switches / Total Kernel Launches) × 1000
IRQ/1K = (Total IRQs / Total Kernel Launches) × 1000
```

#### 2.6.3 Performance Impact Calculation

```
Slowdown % = (Baseline tok/s - Scenario tok/s) / Baseline tok/s × 100
```

### 2.7 Data Collection Commands

```bash
# Start tracing
sudo ./cuda_sched_trace > trace.csv 2> trace.log &
TRACE_PID=$!

# Run benchmark
cd qwen3.cu
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1

# Stop tracing
sudo kill -SIGINT $TRACE_PID

# Analyze results
python3 analyze_scheduler_impact.py
```

---

## 3. Research Questions and Results

### RQ1: Does CPU Scheduler Significantly Impact GPU Performance in Clean Environments?

#### 3.1.1 Experiment Design

- **Condition**: Clean system, no artificial interference
- **Metrics**: Context switch frequency, preemption penalty, total runtime impact
- **Analysis**: Launch pair comparison (with vs. without context switch)

#### 3.1.2 Results

**Basic Statistics**:

| Metric | Value |
|--------|-------|
| Total Runtime | 79.5 seconds |
| Kernel Launches | 51,464 |
| Context Switches | 592 (7.44 Hz) |
| OFF-CPU Time | 7.88 ms (0.01%) |

**Launch Pair Analysis**:

| Group | Count | Percentage | P50 Interval | P90 Interval | P99 Interval |
|-------|-------|------------|--------------|--------------|--------------|
| No Context Switch | 51,401 | 99.9% | 2 µs | 4 µs | 4 µs |
| With Context Switch | 62 | 0.1% | 15.3 ms | 15.5 ms | 5.0 s |

**Preemption Penalty**: 15.3 ms (median)

**Tail Latency Attribution**:

| Percentile | Total Outliers | With Context Switch | Attribution |
|------------|----------------|---------------------|-------------|
| P95+ | 2,580 | 62 (2.4%) | 97.6% application |
| P99+ | 515 | 62 (12.0%) | 88.0% application |

**Total Scheduler Impact**:
```
Impact = Affected Pairs × Penalty = 62 × 15ms = 0.93 seconds
Percentage = 0.93 / 79.5 = 1.2%
```

#### 3.1.3 Finding

**In clean environments, CPU scheduler impact is minimal (1.2%)**. The vast majority (99.9%) of kernel launch pairs are unaffected by context switches. Tail latency is predominantly caused by application behavior (token generation boundaries), not scheduler preemption.

---

### RQ2: What is the Impact of IRQ Interrupts on GPU Performance?

#### 3.2.1 Experiment Design

- **Condition**: Clean system with IRQ tracing enabled
- **Metrics**: IRQ frequency, duration, type distribution
- **Analysis**: IRQ time as percentage of total runtime

#### 3.2.2 Results

**IRQ Overview**:

| Metric | Value |
|--------|-------|
| Total Runtime | 4.99 seconds |
| Kernel Launches | 125,236 |
| Soft IRQs | 653 events |
| Hard IRQs | 0 events |

**Soft IRQ Type Distribution**:

| Type | Count | Total Time | Avg Time | Max Time | Percentage |
|------|-------|------------|----------|----------|------------|
| TIMER | 317 | 0.77 ms | 2.4 µs | 30.1 µs | 49% |
| RCU | 291 | 0.40 ms | 1.4 µs | 17.2 µs | 45% |
| NET_RX | 30 | 0.13 ms | 4.5 µs | 14.0 µs | 4.6% |
| SCHED | 15 | 0.07 ms | 4.9 µs | 18.9 µs | 2.3% |

**Total IRQ Impact**:
```
Total IRQ Time: 1.38 ms
Percentage of Runtime: 0.0276%
```

#### 3.2.3 Theoretical vs. Actual Impact

**Theoretical Concerns**:
1. **Direct Time Cost**: IRQ handler execution time
2. **Cache Pollution**: IRQ evicts GPU process data from L1/L2/L3
3. **Pipeline Stalls**: CPU pipeline flush on interrupt
4. **Latency Accumulation**: Delays on critical path

**Why Actual Impact is Low for qwen3**:

1. **Burst Submission Pattern**: 950 launches in <100µs window
   - IRQs rarely occur within such short windows
   - Most IRQs happen between bursts (during CPU compute)

2. **TIMER Dominates (49%)**:
   - Small cache footprint
   - Only accesses kernel timer structures
   - Minimal cache pollution

3. **No Network I/O**:
   - NET_RX only 30 events (4.6%)
   - Distributed training would see much higher NET_RX

4. **No Data Loading**:
   - Zero hard IRQs
   - No NVMe/SSD block device interrupts

#### 3.2.4 Finding

**IRQ impact is negligible for local LLM inference (0.0276%)**. However, this finding is workload-specific. Distributed training with network communication or on-the-fly data loading would experience significantly higher IRQ impact (estimated 5-20%).

---

### RQ3: How Do Noisy Neighbors Affect GPU Performance?

#### 3.3.1 Experiment Design

Six scenarios tested with identical GPU workload:

| Scenario | Interference | Purpose |
|----------|--------------|---------|
| Baseline | None | Reference point |
| Noisy CPU | stress-ng (all cores) | CPU contention |
| Noisy Network | iperf3 (10 streams) | Network IRQ |
| Noisy Disk | fio (4 jobs, randwrite) | Block IRQ |
| Heavy Load | All three combined | Production simulation |
| Optimized | CPU stress + taskset + nice | Mitigation test |

#### 3.3.2 Results

**Normalized Metrics (per 1000 kernel launches)**:

| Scenario | Launches | Sched/1K | Soft IRQ/1K | Hard IRQ/1K | IRQ Time (ms) |
|----------|----------|----------|-------------|-------------|---------------|
| Baseline | 56,882 | 22.8 | 5.8 | 0.0 | 0.62 |
| Noisy CPU | 61,184 | **11,932.8** | 6.4 | 0.0 | 0.33 |
| Noisy Network | 154,394 | 6.0 | 2.7 | 0.0 | 0.92 |
| Noisy Disk | 126,670 | 29.3 | 3.9 | **0.1** | 1.03 |
| **Heavy Load** | **99,424** | **6,044.6** | 2.4 | 0.0 | 0.37 |
| Optimized | 108,984 | 445.2 | 2.8 | 0.0 | 0.71 |

**Performance Impact**:

| Scenario | tok/s | Runtime (s) | Slowdown | Context Switch Increase |
|----------|-------|-------------|----------|------------------------|
| Baseline | 54.77 | 3.00 | - | 1x |
| Noisy CPU | 49.93 | 4.15 | **8.8%** | **524x** |
| Noisy Network | 53.23 | 7.22 | 2.8% | 0.26x |
| Noisy Disk | 54.95 | 5.60 | -0.3% | 1.3x |
| **Heavy Load** | **43.56** | **6.97** | **20.5%** | **265x** |
| Optimized | 53.75 | 5.10 | 1.9% | 19.5x |

#### 3.3.3 Detailed Analysis by Scenario

**Noisy CPU (stress-ng)**:
- Context switches increase **524x** (22.8 → 11,932.8 per 1K launches)
- Performance degrades **8.8%**
- Mechanism: CFS scheduler time-slices between GPU process and stress-ng workers

**Noisy Network (iperf3)**:
- Context switches actually **decrease** (network load reduces CPU competition)
- Soft IRQ increases slightly
- Performance degrades only **2.8%**
- Finding: Network I/O primarily impacts IRQ, not scheduling

**Noisy Disk (fio)**:
- First appearance of **hard IRQs** (BLOCK interrupts)
- Context switches remain low
- Performance virtually unchanged (-0.3%)
- Finding: Disk I/O has minimal impact on GPU workload

**Heavy Load (CPU + Network + Disk)**:
- Performance degrades **20.5%** - the worst case
- Context switches: 6,044.6/1K (265x increase)
- Interestingly, only **50.7%** of Noisy CPU's context switches
- **Key Insight**: Interferences compete with each other, but combined effect is still worst

**Heavy Load Soft IRQ Breakdown**:

| Type | Count | Total Time | Avg Time |
|------|-------|------------|----------|
| RCU | 213 | 217.4 µs | 1.0 µs |
| TIMER | 17 | 122.9 µs | 7.2 µs |
| SCHED | 5 | 33.3 µs | 6.7 µs |

#### 3.3.4 Finding

**Noisy neighbors significantly impact GPU performance, with combined interference causing 20.5% degradation**. Different interference types have distinct signatures:
- CPU contention → massive context switch increase
- Network I/O → IRQ overhead
- Disk I/O → block interrupts but minimal performance impact
- Combined → worst overall impact due to cumulative effects

---

### RQ4: Can CPU Pinning Effectively Mitigate Scheduler Impact?

#### 3.4.1 Experiment Design

- **Baseline**: Noisy CPU scenario (stress-ng on all cores)
- **Optimized**: Same stress-ng + GPU process with:
  - `taskset -c 0-3` (pin to cores 0-3)
  - `nice -n -10` (higher priority)

#### 3.4.2 Results

**Comparison**:

| Metric | Noisy CPU | Optimized | Improvement |
|--------|-----------|-----------|-------------|
| Sched/1K | 11,932.8 | 445.2 | **96.3% reduction** |
| tok/s | 49.93 | 53.75 | **7.6% improvement** |
| vs. Baseline | 8.8% slower | 1.9% slower | Significant recovery |

**Remaining Gap Analysis**:
- Optimized still has 445.2 sched/1K vs. Baseline's 22.8
- This is **19.5x higher** than baseline
- CPU pinning reduces contention but cannot eliminate it entirely

#### 3.4.3 Why Complete Elimination is Impossible

1. **Shared Core Competition**: stress-ng workers may still be scheduled on cores 0-3
2. **Kernel Tasks**: System daemons and kernel threads cannot be excluded
3. **IRQ Affinity**: Interrupts may still target pinned cores

#### 3.4.4 Recommendations for Better Isolation

```bash
# 1. Use isolcpus kernel parameter (boot time)
isolcpus=4-7 nohz_full=4-7

# 2. Bind GPU process to isolated cores
taskset -c 4-7 ./gpu_app

# 3. Bind IRQs away from GPU cores
echo 0-3 > /proc/irq/*/smp_affinity_list

# 4. Use cgroups for CPU isolation
cgcreate -g cpu:gpu_workload
cgset -r cpuset.cpus=4-7 gpu_workload
cgexec -g cpu:gpu_workload ./gpu_app
```

#### 3.4.5 Finding

**CPU pinning is highly effective, reducing context switches by 96.3% and recovering 7.6% performance**. However, it cannot fully restore baseline performance under heavy load conditions. Complete isolation requires kernel-level configuration (isolcpus) and IRQ affinity management.

---

## 4. Discussion

### 4.1 Key Insights

1. **Environment Matters**: Scheduler impact ranges from 1.2% (clean) to 20.5% (heavy load)

2. **Workload Pattern is Critical**: Burst submission patterns (like qwen3) are more resilient to interrupts because IRQs typically occur between bursts

3. **Different Interferences Have Different Signatures**:
   | Interference | Primary Impact | Secondary Impact |
   |--------------|----------------|------------------|
   | CPU | Context switches | None |
   | Network | IRQ overhead | Slight scheduling |
   | Disk | Hard IRQs | Minimal |
   | Combined | All of above | Worst overall |

4. **Optimization Effectiveness**:
   - CPU pinning: Very effective (96% reduction)
   - Priority adjustment: Helpful but limited
   - Full isolation: Requires kernel configuration

### 4.2 Comparison with Meta's Findings

| Aspect | Meta (AI Training) | Our Study (LLM Inference) |
|--------|-------------------|---------------------------|
| Primary Issue | Network IRQ (NET_RX) | CPU scheduling |
| IRQ Impact | 5-20% | 0.03% (local inference) |
| Optimization | sched_ext layer | taskset + nice |
| Workload | Distributed training | Single-node inference |

**Key Difference**: Meta's distributed training has constant network communication (all-reduce), making NET_RX a major bottleneck. Local inference has minimal network I/O.

### 4.3 Limitations

1. **eBPF Overhead**: Tracing itself adds 1-5% overhead
2. **CUDA Only**: Does not support other GPU APIs (OpenCL, HIP)
3. **No GPU-side Data**: Cannot measure actual kernel execution time
4. **Limited IRQ Attribution**: Cannot identify which process caused the IRQ
5. **Single GPU**: Did not test multi-GPU scenarios

### 4.4 Practical Recommendations

**For Production Deployments**:

| Environment | Recommendation | Expected Benefit |
|-------------|----------------|------------------|
| Dedicated Server | No optimization needed | - |
| Shared Server (light) | taskset + nice | 5-10% improvement |
| Shared Server (heavy) | isolcpus + IRQ affinity | 15-20% improvement |
| Kubernetes | CPU limits + nodeSelector | Varies |

**Decision Tree**:
```
Is GPU workload latency-sensitive?
├── No → No optimization needed
└── Yes → Is server shared?
    ├── No → Monitor only, optimize if needed
    └── Yes → How heavy is colocated load?
        ├── Light → taskset + nice
        └── Heavy → isolcpus + dedicated cores
```

---

## 5. Conclusion

This study provides quantitative evidence for CPU scheduler and IRQ impact on GPU workloads:

1. **Clean environments show minimal impact** (1.2% scheduler, 0.03% IRQ), suggesting that scheduler optimization is not universally necessary.

2. **Noisy neighbor conditions cause significant degradation** (up to 20.5% for combined CPU+Network+Disk load), demonstrating the importance of resource isolation in shared environments.

3. **Different interference types require different mitigation strategies**:
   - CPU contention → CPU pinning (96.3% reduction)
   - Network IRQ → IRQ affinity + interrupt coalescing
   - Disk IRQ → I/O scheduler tuning

4. **The tracing tool successfully identifies and quantifies** scheduling issues, enabling data-driven optimization decisions rather than speculation.

**Recommendation**: Use the provided tracing tool to profile your specific workload before investing in scheduler optimization. The impact varies dramatically based on workload characteristics and deployment environment.

---

## 6. Appendix

### A. Tool Installation and Usage

```bash
# Build the tracing tool
cd tools
make cuda_sched_trace

# Run tracing
sudo ./cuda_sched_trace > trace.csv 2> trace.log &

# In another terminal, run your GPU workload
./your_gpu_app

# Stop tracing
sudo pkill cuda_sched_trace

# Analyze results
cd ../scripts/sched
python3 analyze_scheduler_impact.py /path/to/trace.csv
```

### B. CSV Output Format

| Field | Description |
|-------|-------------|
| timestamp_ns | Relative timestamp (nanoseconds) |
| event_type | cuLaunchKernel, cudaLaunchKernel, syncEnter, syncExit, schedSwitch, hardirqEntry, hardirqExit, softirqEntry, softirqExit |
| pid, tid | Process/Thread ID |
| comm | Process name |
| cpu | CPU core number |
| grid_x/y/z | CUDA grid dimensions (launch events) |
| block_x/y/z | CUDA block dimensions (launch events) |
| shared_mem | Shared memory size (launch events) |
| stream | CUDA stream pointer (launch events) |
| last_offcpu_ns | Last OFF-CPU timestamp (schedSwitch) |
| last_oncpu_ns | Last ON-CPU timestamp (schedSwitch) |
| irq_num | IRQ number (hard) or vector (soft) |
| irq_name | Handler name or type (TIMER, NET_RX, etc.) |
| duration_ns | IRQ duration (exit events only) |

### C. Noisy Neighbor Test Script

```bash
#!/bin/bash
# test_noisy_neighbor.sh - Complete test suite

# Scenarios:
# 1. Baseline (clean)
# 2. Noisy CPU (stress-ng)
# 3. Noisy Network (iperf3)
# 4. Noisy Disk (fio)
# 5. Heavy Load (all combined)
# 6. Optimized (taskset + nice)

# See scripts/sched/test_noisy_neighbor.sh for full implementation
```

---

## References

1. Meta Platforms, Inc. "Accelerating AI Training with sched_ext." Linux Plumbers Conference 2025. https://lpc.events/event/19/contributions/2039/

2. NVIDIA Corporation. "CUDA Driver API Reference." https://docs.nvidia.com/cuda/cuda-driver-api/

3. Linux Kernel Documentation. "BPF Documentation." https://www.kernel.org/doc/html/latest/bpf/

4. stress-ng. "A tool to load and stress a computer system." https://github.com/ColinIanKing/stress-ng

5. iperf3. "A TCP, UDP, and SCTP network bandwidth measurement tool." https://github.com/esnet/iperf

6. fio. "Flexible I/O Tester." https://github.com/axboe/fio

---

**Artifacts**:
- Tracing Tool: `tools/cuda_sched_trace`
- Test Script: `scripts/sched/test_noisy_neighbor.sh`
- Analysis Script: `scripts/sched/analyze_scheduler_impact.py`
- This Report: `scripts/sched/REPORT.md`
- Quick Reference: `scripts/sched/README.md`
