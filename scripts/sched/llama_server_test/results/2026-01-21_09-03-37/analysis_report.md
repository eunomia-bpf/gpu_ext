# CPU Scheduler Overhead Analysis Report

> **Note:** This report is part of the research on understanding how CPU scheduling overhead affects GPU-accelerated LLM inference workloads. The experiments were conducted using `cuda_sched_trace` tool to capture context switches, IRQs, and other scheduling events during llama-server inference.

---

## Overview

This document presents a comprehensive analysis of CPU scheduler overhead impact on LLM inference performance. The goal is to answer the following research questions:

- **RQ1:** How much does the CPU scheduler impact GPU inference in a clean environment?
- **RQ2:** Is I/O-bound interference different from CPU-bound interference?
- **RQ3:** What is the performance impact of "noisy neighbor" workloads?
- **RQ4:** Can CPU pinning mitigate scheduler overhead?

### Test Environment

- **Hardware:** Server with NVIDIA GPU
- **OS:** Linux 6.15.11
- **Date:** 2026-01-21

---

**Generated:** 2026-01-22 12:35:42
**Data Directory:** `results/2026-01-21_09-03-37`

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experiment Configuration](#experiment-configuration)
3. [Performance Results](#performance-results)
4. [Scheduler Metrics](#scheduler-metrics)
5. [Tracer Overhead Analysis](#tracer-overhead-analysis)
6. [Detailed Per-Run Results](#detailed-per-run-results)
7. [Key Findings](#key-findings)
8. [Conclusions](#conclusions)

## Executive Summary

This report analyzes the impact of CPU scheduling overhead on LLM inference performance using llama-server with the **Qwen/Qwen3-30B-A3B-FP8** model.

### Key Metrics Summary

| Metric | Value |
|--------|-------|
| Baseline Throughput | **218.7 tok/s** |
| Max Performance Degradation | **26.5%** (Heavy Load) |
| Baseline Context Switches | **0.1** per 1K launches |
| Max Context Switch Increase | **21990x** (CPU Stress) |

## Experiment Configuration

### Test Scenarios

| Scenario | Description |
|----------|-------------|
| **Baseline** | Clean system, no additional workloads |
| **CPU Stress** | `stress-ng --cpu` running on all cores |
| **Network Stress** | `iperf3` generating network traffic |
| **Disk Stress** | `stress-ng --hdd` performing disk I/O |
| **Heavy Load** | Combined CPU + Network + Disk stress |
| **CPU Pinned** | llama-server pinned to specific CPU cores under stress |

### Benchmark Configuration

- **Model:** Qwen/Qwen3-30B-A3B-FP8
- **Dataset:** ShareGPT
- **Prompts per run:** 200
- **Runs per scenario:** 3 (for averaging)
- **Warmup:** 1 run before measurement

## Performance Results

### Table 1: Throughput and Latency (Without Tracing)

| Scenario | Runs | Throughput (tok/s) | TPOT Mean (ms) | TPOT P99 (ms) | TTFT Mean (ms) | TTFT P99 (ms) | Slowdown |
|----------|------|-------------------|----------------|---------------|----------------|---------------|----------|
| Baseline | 3 | 218.7 ± 0.1 | 4.10 ± 0.07 | 23.04 | 53.7 ± 5.9 | 156.17 | - |
| CPU Stress | 3 | 177.1 ± 3.6 | 4.24 ± 0.13 | 6.69 | 72.2 ± 13.6 | 148.77 | 19.0% |
| Network Stress | 3 | 218.2 ± 0.3 | 3.55 ± 0.16 | 4.11 | 50.9 ± 3.6 | 95.34 | 0.2% |
| Disk Stress | 3 | 218.8 ± 0.1 | 3.48 ± 0.17 | 4.05 | 51.1 ± 5.6 | 95.90 | -0.0% |
| Heavy Load | 2 | 160.8 ± 6.9 | 5.24 | 10.35 | 72.0 ± 22.8 | 167.44 | 26.5% |
| CPU Pinned | 3 | 176.8 ± 2.3 | 4.52 ± 0.20 | 6.62 | 55.2 ± 21.6 | 129.28 | 19.2% |

### Table 2: Inter-Token Latency (ITL) Metrics

| Scenario | ITL Mean (ms) | ITL Median (ms) | ITL P99 (ms) |
|----------|---------------|-----------------|--------------|
| Baseline | 3.90 | 3.61 | 23.33 |
| CPU Stress | 3.86 | 3.31 | 8.35 |
| Network Stress | 3.58 | 3.62 | 4.15 |
| Disk Stress | 3.52 | 3.58 | 3.91 |
| Heavy Load | 5.16 | 4.89 | 12.02 |
| CPU Pinned | 4.19 | 3.80 | 8.61 |

## Scheduler Metrics

### Table 3: Context Switches and IRQs (With Tracing)

| Scenario | CUDA Launches | Sched Switches | Sched/1K | Soft IRQs | SoftIRQ/1K | Hard IRQs | HardIRQ/1K |
|----------|---------------|----------------|----------|-----------|------------|-----------|------------|
| Baseline | 70,906,427 | 8,314 | 0.1 | 219,242 | 3.1 | 196 | 0.0 |
| CPU Stress | 61,133,304 | 157,624,800 | 2578.4 | 165,375 | 2.7 | 1,256 | 0.0 |
| Network Stress | 70,906,427 | 41,527 | 0.6 | 208,566 | 2.9 | 1,530 | 0.0 |
| Disk Stress | 70,906,427 | 9,417 | 0.1 | 182,497 | 2.6 | 48,936 | 0.7 |
| Heavy Load | 60,272,085 | 174,511,329 | 2895.4 | 155,803 | 2.6 | 184,033 | 3.1 |
| CPU Pinned | 48,807,901 | 155,176,940 | 3179.3 | 123,993 | 2.5 | 199 | 0.0 |

### Context Switch Analysis

| Scenario | Sched/1K | Increase vs Baseline |
|----------|----------|---------------------|
| Baseline | 0.1 | - |
| CPU Stress | 2578.4 | 21990x |
| Network Stress | 0.6 | 5x |
| Disk Stress | 0.1 | 1x |
| Heavy Load | 2895.4 | 24694x |
| CPU Pinned | 3179.3 | 27115x |

## Tracer Overhead Analysis

### Table 4: Performance Impact of Tracing

| Scenario | No Trace (tok/s) | With Trace (tok/s) | Overhead |
|----------|------------------|--------------------| ---------|
| Baseline | 218.75 | 218.69 | 0.0% |
| CPU Stress | 177.13 | 162.10 | 8.5% |
| Network Stress | 218.25 | 218.21 | 0.0% |
| Disk Stress | 218.77 | 218.76 | 0.0% |
| Heavy Load | 160.84 | 142.42 | 11.5% |
| CPU Pinned | 176.80 | 151.46 | 14.3% |

## Detailed Per-Run Results

### Baseline (No Trace)

| Run | Throughput (tok/s) | TPOT Mean (ms) | TTFT Mean (ms) | Duration (s) |
|-----|-------------------|----------------|----------------|--------------|
| Run 1 | 218.63 | 4.02 | 61.90 | 205.9 |
| Run 2 | 218.78 | 4.10 | 50.95 | 205.8 |
| Run 3 | 218.83 | 4.19 | 48.37 | 205.7 |

### CPU Stress (No Trace)

| Run | Throughput (tok/s) | TPOT Mean (ms) | TTFT Mean (ms) | Duration (s) |
|-----|-------------------|----------------|----------------|--------------|
| Run 1 | 178.81 | 4.07 | 88.32 | 200.8 |
| Run 2 | 180.48 | 4.28 | 73.40 | 203.3 |
| Run 3 | 172.11 | 4.37 | 54.99 | 203.9 |

### Network Stress (No Trace)

| Run | Throughput (tok/s) | TPOT Mean (ms) | TTFT Mean (ms) | Duration (s) |
|-----|-------------------|----------------|----------------|--------------|
| Run 1 | 218.05 | 3.32 | 55.57 | 206.4 |
| Run 2 | 218.01 | 3.67 | 50.16 | 206.5 |
| Run 3 | 218.69 | 3.66 | 46.88 | 205.8 |

### Disk Stress (No Trace)

| Run | Throughput (tok/s) | TPOT Mean (ms) | TTFT Mean (ms) | Duration (s) |
|-----|-------------------|----------------|----------------|--------------|
| Run 1 | 218.66 | 3.24 | 58.87 | 205.9 |
| Run 2 | 218.77 | 3.60 | 48.69 | 205.8 |
| Run 3 | 218.89 | 3.60 | 45.71 | 205.6 |

### Heavy Load (No Trace)

| Run | Throughput (tok/s) | TPOT Mean (ms) | TTFT Mean (ms) | Duration (s) |
|-----|-------------------|----------------|----------------|--------------|
| Run 1 | 153.98 | 5.23 | 94.84 | 202.3 |
| Run 2 | 167.70 | 5.25 | 49.18 | 209.8 |

### CPU Pinned (No Trace)

| Run | Throughput (tok/s) | TPOT Mean (ms) | TTFT Mean (ms) | Duration (s) |
|-----|-------------------|----------------|----------------|--------------|
| Run 1 | 176.63 | 4.24 | 85.71 | 203.3 |
| Run 2 | 179.65 | 4.67 | 40.36 | 200.8 |
| Run 3 | 174.13 | 4.65 | 39.54 | 203.0 |

## Key Findings

### RQ1: CPU Scheduler Impact in Clean Environment

- **Context Switches:** 0.1 per 1K kernel launches
- **Soft IRQs:** 3.1 per 1K kernel launches
- **Conclusion:** Minimal scheduler interference in clean environment

### RQ2: I/O vs CPU-bound Interference

| Stress Type | Performance Impact |
|-------------|-------------------|
| Network I/O | 0.2% slowdown |
| Disk I/O | -0.0% slowdown |
| CPU Stress | 19.0% slowdown |

- **Conclusion:** I/O-bound workloads have minimal impact; CPU-bound workloads cause significant degradation

### RQ3: Noisy Neighbor Impact

| Scenario | Slowdown | Context Switch Increase |
|----------|----------|------------------------|
| CPU Stress | 19.0% | 21990x |
| Network Stress | 0.2% | 5x |
| Disk Stress | -0.0% | 1x |
| Heavy Load | 26.5% | 24694x |

- **Worst Case (Heavy Load):** 26.5% performance degradation
- **Root Cause:** 2895x increase in context switches

### RQ4: CPU Pinning Effectiveness

| Metric | CPU Stress | CPU Pinned | Change |
|--------|------------|------------|--------|
| Throughput (tok/s) | 177.1 | 176.8 | -0.2% |
| Sched/1K | 2578.4 | 3179.3 | +23.3% |

- **Observation:** CPU pinning did NOT reduce context switches in this configuration
- **Possible Cause:** Pinning to already-contested cores may increase contention

## Conclusions

### Summary of Findings

1. **Clean Environment Performance:** LLM inference operates with minimal scheduler overhead (~0.1 context switches per 1K kernel launches)

2. **I/O Workload Isolation:** Network and disk I/O stress have negligible impact on GPU inference performance, suggesting good I/O subsystem isolation

3. **CPU Contention is Critical:** CPU-intensive workloads cause significant performance degradation (19-27%) due to massive increases in context switching (2500-3000x)

4. **CPU Pinning Considerations:** Simple CPU pinning may not be effective if the pinned cores are already under contention; careful core selection is needed

5. **Tracer Overhead:** The cuda_sched_trace tool adds 0-14% overhead depending on scheduler activity levels

---

## Methodology Notes

### Data Collection

- Each scenario was run **3 times** (except Heavy Load which has 2 valid runs due to `heavy_load_notrace_run3` being incomplete)
- Results are averaged with standard deviation reported where applicable
- Two modes per scenario:
  - **No Trace:** Pure performance measurement without tracing overhead
  - **With Trace:** Performance with `cuda_sched_trace` enabled for scheduler metrics

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **tok/s** | Output token throughput (tokens generated per second) |
| **TPOT** | Time Per Output Token - latency for each generated token |
| **TTFT** | Time To First Token - initial response latency |
| **ITL** | Inter-Token Latency - time between consecutive tokens |
| **Sched/1K** | Context switches per 1000 CUDA kernel launches |
| **SoftIRQ/1K** | Soft interrupts per 1000 CUDA kernel launches |

### Limitations

1. **Single GPU Configuration:** Results may vary with multi-GPU setups
2. **CPU Pinning Strategy:** Only tested one pinning configuration; alternative strategies may yield different results
3. **Workload Specificity:** ShareGPT dataset may not represent all inference workloads
4. **Incomplete Run:** `heavy_load_notrace_run3` did not complete, so Heavy Load no-trace results are based on 2 runs

---

## Files in This Directory

| File Pattern | Description |
|--------------|-------------|
| `*_notrace_run*.log` | vLLM benchmark output without tracing |
| `*_trace_run*.log` | vLLM benchmark output with tracing enabled |
| `*_trace.log` | cuda_sched_trace summary statistics |
| `*_trace.csv` | Detailed trace events (large files, git-ignored) |
| `analysis_report.md` | This report |

---

## Recommendations

Based on these findings, we recommend:

1. **Monitor CPU Utilization:** High CPU usage from co-located workloads significantly impacts LLM inference
2. **Isolate GPU Workloads:** Consider using cgroups or containers to isolate GPU inference from CPU-intensive tasks
3. **Careful CPU Pinning:** If using CPU pinning, ensure pinned cores are not under contention
4. **I/O Workloads Are Safe:** Network and disk I/O workloads have minimal impact on GPU inference

---

*Report generated by `analyze_sched_overhead.py`*
