# CPU Scheduler Overhead Analysis: llama-server with Noisy Neighbors

**Date**: January 2026
**Workload**: llama-server (gpt-oss-20b model, 65536 context)
**Benchmark**: vLLM bench serve (ShareGPT dataset, 100 prompts, 1 QPS)

---

## Executive Summary

This study quantifies the impact of CPU scheduling on llama-server performance under various interference scenarios. We measured both actual performance (without tracing) and detailed scheduler metrics (with cuda_sched_trace).

**Key Findings**:
- In clean environments, scheduler impact is minimal (3.6 context switches per 1K kernel launches)
- CPU-intensive noisy neighbors cause **36,414x increase** in context switches and **10% performance degradation**
- Combined heavy load (CPU + Network + Disk) results in **30.1% performance degradation**
- Network and disk I/O interference have negligible impact (<1% slowdown)
- The tracing tool adds ~0.1% overhead in clean environments

---

## 1. Test Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5090 (32GB) |
| CPU | 24 cores |
| OS | Linux 6.15.11-061511-generic |
| Model | gpt-oss-20b-mxfp4.gguf (20B parameters) |
| Context | 65536 tokens |
| Server | llama-server with `--gpt-oss-20b-default` |

---

## 2. Test Methodology

### 2.1 Test Matrix

| Scenario | Description | Interference Tool |
|----------|-------------|-------------------|
| Baseline | Clean environment | None |
| CPU Stress | CPU-intensive load | `stress-ng --cpu 0 --cpu-method fft` |
| Network Stress | Network I/O | `iperf3` loopback (10 streams) |
| Disk Stress | Disk I/O | `fio randwrite bs=4k iodepth=32` |
| Heavy Load | Combined interference | CPU + Network + Disk |
| CPU Pinned | Optimized configuration | `taskset -c 0-3 nice -n -10` + stress-ng |

Each scenario was tested in two modes:
1. **Without Tracing**: Measure actual performance
2. **With Tracing**: Collect scheduler metrics via `cuda_sched_trace`

### 2.2 Benchmark Configuration

```bash
uv run vllm bench serve \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name sharegpt \
  --num-prompts 100 \
  --base-url http://127.0.0.1:8013 \
  --max-concurrency=1 \
  --request-rate 1
```

---

## 3. Results

### 3.1 Performance Results (Without Tracing)

| Scenario | tok/s | TPOT Mean | TPOT P99 | TTFT Mean | TTFT P99 | Slowdown |
|----------|-------|-----------|----------|-----------|----------|----------|
| **Baseline** | 209.82 | 2.86ms | 3.25ms | 64.40ms | 100.53ms | - |
| CPU Stress | 188.93 | 3.92ms | 6.54ms | 85.09ms | 152.72ms | **10.0%** |
| Network Stress | 208.54 | 3.02ms | 3.89ms | 59.24ms | 96.33ms | 0.6% |
| Disk Stress | 209.74 | 2.87ms | 3.24ms | 64.07ms | 101.88ms | 0.0% |
| **Heavy Load** | 146.69 | 5.31ms | 9.09ms | 100.58ms | 262.93ms | **30.1%** |
| CPU Pinned | 181.62 | 3.96ms | 6.50ms | 87.09ms | 151.07ms | 13.4% |

### 3.2 Scheduler Metrics (With Tracing)

| Scenario | Kernel Launches | Sched/1K | Soft IRQ/1K | Hard IRQ/1K | IRQ Time |
|----------|-----------------|----------|-------------|-------------|----------|
| **Baseline** | 394,664 | 3.6 | 77.0 | 0.1 | 211.14ms |
| CPU Stress | 352,957 | **132,585.0** | 78.0 | 0.0 | 102.95ms |
| Network Stress | 394,664 | 15.2 | 71.7 | 0.9 | 166.14ms |
| Disk Stress | 394,664 | 3.8 | 73.3 | 1.5 | 199.19ms |
| **Heavy Load** | 304,503 | **81,006.7** | 85.2 | 84.6 | 133.34ms |
| CPU Pinned | 358,444 | 132,748.9 | 71.5 | 0.2 | 83.02ms |

**Sched/1K** = Context switches per 1000 kernel launches

### 3.3 Tracer Overhead

| Scenario | Without Trace | With Trace | Overhead |
|----------|---------------|------------|----------|
| Baseline | 209.82 tok/s | 209.67 tok/s | **0.1%** |
| CPU Stress | 188.93 tok/s | 163.96 tok/s | 13.2% |
| Heavy Load | 146.69 tok/s | 135.82 tok/s | 7.4% |

---

## 4. Analysis

### 4.1 RQ1: Does CPU Scheduler Significantly Impact Performance in Clean Environments?

**Finding: Minimal impact (3.6 context switches per 1K launches)**

In clean environments, llama-server experiences very few context switches relative to kernel launches. The scheduler overhead is negligible, with:
- 3.6 context switches per 1000 kernel launches
- 211ms total IRQ time over the benchmark duration
- 0.1% tracer overhead

### 4.2 RQ2: What is the Impact of Different Interference Types?

| Interference | Slowdown | Context Switch Increase | Primary Mechanism |
|--------------|----------|------------------------|-------------------|
| CPU Stress | 10.0% | 36,414x | CFS time-slicing |
| Network Stress | 0.6% | 4.2x | Soft IRQ (NET_RX) |
| Disk Stress | 0.0% | 1.1x | Hard IRQ (BLOCK) |
| Heavy Load | 30.1% | 22,248x | Combined effects |

**Key Observations**:
1. **CPU contention is the dominant factor** - 36,414x increase in context switches
2. **Network I/O has minimal impact** on GPU workloads (only soft IRQ overhead)
3. **Disk I/O has negligible impact** - hard IRQs don't affect llama-server significantly
4. **Combined load is worse than sum of parts** - interactions amplify degradation

### 4.3 RQ3: Heavy Load Soft IRQ Breakdown

| IRQ Type | Count | Total Time | Avg Time |
|----------|-------|------------|----------|
| NET_RX | 16,215 | 95.8ms | 5.9μs |
| RCU | 3,513 | 4.4ms | 1.3μs |
| TIMER | 2,950 | 4.4ms | 1.5μs |
| BLOCK | 2,942 | 3.3ms | 1.1μs |
| SCHED | 326 | 1.9ms | 5.8μs |

Under heavy load, NET_RX dominates soft IRQ time, but the total IRQ overhead (133ms) is much less than the performance impact suggests. The primary degradation comes from context switches, not IRQ handling.

### 4.4 RQ4: CPU Pinning Effectiveness

**Finding: CPU pinning did NOT improve performance in this test**

| Metric | CPU Stress | CPU Pinned | Change |
|--------|------------|------------|--------|
| tok/s | 188.93 | 181.62 | -3.9% |
| Sched/1K | 132,585 | 132,749 | +0.1% |

**Why pinning failed**:
- `stress-ng --cpu 0` runs on ALL cores, including the pinned cores (0-3)
- The llama-server was constrained to fewer cores while still competing with stress-ng
- Proper isolation requires `isolcpus` kernel parameter or cgroup CPU isolation

**Correct approach**:
```bash
# Option 1: Pin stress-ng to different cores
stress-ng --cpu 4 --taskset 4-23 --cpu-method fft &
taskset -c 0-3 nice -n -10 ./llama-server ...

# Option 2: Use isolcpus (kernel boot parameter)
isolcpus=0-3 nohz_full=0-3

# Option 3: Use cgroups for CPU isolation
cgcreate -g cpuset:gpu_workload
cgset -r cpuset.cpus=0-3 gpu_workload
cgexec -g cpuset:gpu_workload ./llama-server
```

---

## 5. Comparison with qwen3.cu Results

| Metric | qwen3.cu (REPORT.md) | llama-server (this study) |
|--------|---------------------|---------------------------|
| Baseline Sched/1K | 22.8 | 3.6 |
| CPU Stress Slowdown | 8.8% | 10.0% |
| Heavy Load Slowdown | 20.5% | 30.1% |
| Network Slowdown | 2.8% | 0.6% |
| Disk Slowdown | -0.3% | 0.0% |

**Observations**:
- llama-server has lower baseline context switches (more efficient scheduling)
- CPU stress impact is similar (~10%)
- Heavy load impact is more severe for llama-server (30% vs 20%)
- Network/Disk interference remains minimal for both workloads

---

## 6. Conclusions

1. **Clean environments show minimal scheduler impact** - optimization is not necessary for dedicated servers

2. **CPU-intensive noisy neighbors cause significant degradation** (10-30%) - isolation is critical in shared environments

3. **Network and disk I/O interference are negligible** for local LLM inference workloads

4. **CPU pinning requires proper configuration** - simply using `taskset` without isolating the interference source is ineffective

5. **The tracing tool has minimal overhead** (0.1%) in clean environments, making it suitable for production profiling

---

## 7. Recommendations

| Environment | Recommendation | Expected Benefit |
|-------------|----------------|------------------|
| Dedicated Server | No optimization needed | - |
| Shared Server (light CPU) | Monitor context switches | Early warning |
| Shared Server (heavy CPU) | `isolcpus` + `taskset` | 10-30% improvement |
| Kubernetes | CPU limits + node affinity | Isolation |

---

## 8. Files

| File | Description |
|------|-------------|
| `test_cpu_sched_overhead.sh` | Test orchestration script |
| `analyze_sched_overhead.py` | Analysis and reporting script |
| `*_notrace_bench.log` | Performance results (no tracing) |
| `*_trace_bench.log` | Performance results (with tracing) |
| `*_trace.csv` | Raw scheduler trace data |
| `*_trace.log` | Trace summary statistics |

---

## Appendix: Raw Commands

**Start llama-server:**
```bash
/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server \
  --gpt-oss-20b-default -c 65536
```

**Run benchmark:**
```bash
cd ~/workspace/gpu/schedcp/workloads/llama.cpp
uv run vllm bench serve \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name sharegpt \
  --num-prompts 100 \
  --dataset-path /path/to/ShareGPT.json \
  --base-url http://127.0.0.1:8013 \
  --max-concurrency=1 \
  --request-rate 1
```

**Run tracing:**
```bash
sudo ./cuda_sched_trace > trace.csv 2> trace.log &
```
