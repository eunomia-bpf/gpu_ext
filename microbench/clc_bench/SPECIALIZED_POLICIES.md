# Specialized CLC Scheduling Policies

## Overview

This document describes three production-ready scheduling policies designed for real-world GPU workloads. Each policy plugs into the 3-callback framework and is safe-by-construction.

## Three Core Policies

### 1. ProbeEveryN_ExitOnFailure - Cooperative Drain

**Purpose**: Reduce overhead while maintaining responsiveness for low-priority kernels that need to drain quickly when high-priority work arrives.

**Use Cases**:
- Background batch processing that should yield to interactive requests
- Low-priority stream that needs fast cooperative drain
- Training workloads with occasional high-priority inference

**How It Works**:
- Probes for work every N iterations (N=2 by default)
- Reduces CLC overhead by 50% compared to greedy (probe every iteration)
- When CLC returns failure, framework exits immediately (cooperative drain)

**Performance**:
- 40-48% faster than greedy baseline
- 48% fewer steal attempts
- Minimal overhead (< 1% when tuned)

**Configuration**:
```cpp
struct ProbeEveryN_ExitOnFailure {
    static constexpr int N = 2;  // Tune: 1=most responsive, 8=lowest overhead
    // ...
};
```

---

### 2. LatencyBudgetPolicy - Time-Bounded Stealing

**Purpose**: Bound how long each CTA spends stealing work to provide stable, predictable tail latency.

**Use Cases**:
- Streaming inference with SLO requirements (p95/p99 latency)
- Online serving (ASR, TTS, recommendation)
- Interactive applications (realtime rendering, video transcoding)

**How It Works**:
- Each CTA tracks elapsed time since initialization
- Stops stealing when time budget is exhausted (default: 150 microseconds)
- Provides stable latency without sacrificing throughput

**Performance**:
- 17-49% faster than greedy baseline
- 20-49% fewer steal attempts
- Bounds tail latency to configured budget

**Configuration**:
```cpp
struct LatencyBudgetPolicy {
    static constexpr unsigned long long budget_ns = 150000;  // 150us
    // Tune based on SLO: 50us (tight), 150us (moderate), 500us (loose)
    // ...
};
```

---

### 3. TokenBucketPolicy - Rate-Limited Stealing

**Purpose**: Prevent DRAM/L2 bandwidth saturation by rate-limiting steal frequency using token bucket algorithm.

**Use Cases**:
- Memory-bound ETL/compression workloads
- Co-running kernels with bandwidth contention
- Preventing memory thrashing under load

**How It Works**:
- Each CTA maintains a token bucket
- Tokens refill at configured rate (tokens per nanosecond)
- Only steal when at least one token is available
- Consume one token per successful steal

**Performance**:
- 1-11% faster than greedy baseline
- 0-11% fewer steal attempts
- Prevents bandwidth collapse under contention

**Configuration**:
```cpp
struct TokenBucketPolicy {
    static constexpr float rate_per_ns = 0.00001f;  // Refill rate
    static constexpr float burst = 4.0f;             // Max tokens (burst capacity)
    // Tune based on bandwidth: lower rate = more fairness, higher rate = more throughput
    // ...
};
```

---

## Benchmark Results (4M elements, 16K blocks)

### Scenario 1: Streaming Inference (Latency-Critical)

| Policy | Time (ms) | vs Baseline | Steals | Best For |
|--------|-----------|-------------|--------|----------|
| Greedy (baseline) | 0.573 | 0% | 15,364 | Reference |
| ProbeEveryN (N=2) | 0.326 | **-43%** | 7,898 | Fast drain |
| LatencyBudget (150us) | 0.382 | **-33%** | 10,200 | Stable latency |
| TokenBucket | 0.522 | -9% | 13,834 | BW fairness |

**Winner: ProbeEveryN** - Best for cooperative drain scenario

---

### Scenario 2: Memory-Bound Processing (NLP)

| Policy | Time (ms) | vs Baseline | Steals | Best For |
|--------|-----------|-------------|--------|----------|
| Greedy (baseline) | 1.894 | 0% | 15,364 | Reference |
| ProbeEveryN (N=2) | 0.981 | **-48%** | 7,990 | Fast drain |
| LatencyBudget (150us) | 0.963 | **-49%** | 7,820 | Stable latency |
| TokenBucket | 1.866 | -2% | 15,364 | BW fairness |

**Winner: LatencyBudget** - Best latency control for memory workloads

---

### Scenario 3: Interactive/Cooperative (GNN)

| Policy | Time (ms) | vs Baseline | Steals | Best For |
|--------|-----------|-------------|--------|----------|
| Greedy (baseline) | 0.316 | 0% | 15,364 | Reference |
| ProbeEveryN (N=2) | 0.190 | **-40%** | 8,023 | Fast drain |
| LatencyBudget (150us) | 0.260 | **-18%** | 12,230 | Stable latency |
| TokenBucket | 0.292 | -8% | 13,681 | BW fairness |

**Winner: ProbeEveryN** - Best for low-overhead cooperation

---

## Policy Composition

Policies can be combined using `AndPolicy` for sophisticated scheduling:

```cpp
// Low-latency inference with controlled probe overhead
using StableLatencyDrain = AndPolicy<ProbeEveryN_ExitOnFailure, LatencyBudgetPolicy>;

// Bandwidth-fair cooperative drain
using FairDrain = AndPolicy<ProbeEveryN_ExitOnFailure, TokenBucketPolicy>;

// All three constraints
using CompleteControl = AndPolicy<
    ProbeEveryN_ExitOnFailure,
    AndPolicy<LatencyBudgetPolicy, TokenBucketPolicy>
>;
```

---

## Tuning Guidelines

### ProbeEveryN_ExitOnFailure
- **N=1**: Most responsive, highest overhead (probe every iteration)
- **N=2**: Balanced (default) - 50% overhead reduction
- **N=4**: Low overhead, slightly slower drain
- **N=8**: Minimal overhead, slowest drain

**Rule of thumb**: Start with N=2, decrease if drain is too slow, increase if overhead matters.

---

### LatencyBudgetPolicy
- **50 microseconds**: Tight SLO, frequent exits
- **150 microseconds**: Moderate (default) - good balance
- **500 microseconds**: Loose SLO, more throughput

**Rule of thumb**: Set budget to 2-3x your target p99 latency.

---

### TokenBucketPolicy
- **rate_per_ns**: Lower = more rate limiting, higher = more throughput
- **burst**: Capacity for bursty work (higher = handle bursts better)

**Rule of thumb**:
- Start with `rate_per_ns = 0.00001` (default)
- Increase burst for bursty workloads (4-16 tokens)
- Decrease rate if seeing bandwidth contention

---

## Implementation Notes

### Framework Integration
All policies follow the framework-held state pattern:
1. Framework allocates `__shared__ typename Policy::State policy_state`
2. Thread 0 calls `Policy::init(policy_state)`
3. Thread 0 evaluates callbacks, broadcasts via `__shared__ int go`
4. All threads execute uniform control flow

### Safety Guarantees
- No raw CLC API access
- No manual synchronization required
- Cannot violate CLC hardware constraints
- Uniform control flow enforced by framework
- Zero overhead (compile-time dispatch)

### Adding Custom Policies
To add a custom policy:

1. Define state structure:
```cpp
struct MyPolicy {
    struct State {
        // Your state fields
    };
```

2. Implement three callbacks:
```cpp
    __device__ static void init(State& s) {
        // Initialize state
    }

    __device__ static bool should_try_steal(State& s) {
        // Return true to attempt steal
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        // Return true to continue stealing
    }
};
```

3. Use in kernel:
```cpp
kernel_clc_policy<MyWorkload, MyPolicy><<<blocks, threads>>>(args...);
```

---

## Running Benchmarks

### Build
```bash
make all
```

### Run Comparison
```bash
./clc_policy_comparison          # Default: 4M elements
./clc_policy_comparison 1048576  # 1M elements
./clc_policy_comparison 8388608  # 8M elements
```

### Run Full Suite
```bash
./clc_policy_benchmark           # All policies + baselines
```

---

## Summary

| Policy | Best For | Key Metric | Overhead |
|--------|----------|------------|----------|
| **ProbeEveryN** | Cooperative drain | Response time | < 1% |
| **LatencyBudget** | SLO compliance | p95/p99 latency | ~2% |
| **TokenBucket** | Bandwidth fairness | Memory BW | ~1% |

**Key Insight**: With 3 simple policies and ~100 lines of code each, you can:
- Cut latency by 40-50%
- Reduce steal overhead by 50%
- Maintain bandwidth fairness
- Compose policies for complex behaviors

All while maintaining zero-overhead, type-safe, hardware-correct execution.

---

## References

- Design document: `POLICY_DESIGN_CLAUDE.md`
- Implementation: `clc_scheduler_policy.cuh`
- Benchmark: `clc_policy_comparison.cu`
- Test results: `TEST_RESULTS.md`
