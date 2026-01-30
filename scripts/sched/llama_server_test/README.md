# CPU Scheduler Overhead Test for llama-server

Test CPU scheduler overhead on llama-server with various noisy neighbors.

## Files

- `test_cpu_sched_overhead.sh` - Main test orchestration script
- `analyze_sched_overhead.py` - Analysis script for results

## Test Matrix

### Scenarios (6 total)
1. **Baseline** - llama-server running alone
2. **CPU Stress** - stress-ng on all cores
3. **Network Stress** - iperf3 loopback
4. **Disk Stress** - fio random write
5. **Heavy Load** - CPU + Network + Disk combined
6. **CPU Pinned** - taskset + nice with CPU stress

### Modes (2 per scenario)
- **No Trace** - Measure actual performance without tracing overhead
- **With Trace** - Measure scheduler metrics with cuda_sched_trace

Total: 12 test runs

## Usage

```bash
# Run all tests
./test_cpu_sched_overhead.sh

# Results are saved to /tmp/llama_sched_test/

# Run analysis separately
python3 analyze_sched_overhead.py /tmp/llama_sched_test/
```

## Output

### Table 1: Performance (No Trace)
- tok/s, TPOT, TTFT, slowdown %

### Table 2: Scheduler Metrics (With Trace)
- Kernel launches, Sched/1K, IRQ/1K, IRQ time

### Table 3: Tracer Overhead
- Performance difference between trace/no-trace modes

## Requirements

- llama-server built at `/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/`
- cuda_sched_trace built at `/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/tools/`
- ShareGPT dataset at `/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/`
- Tools: stress-ng, iperf3, fio, uv (for vllm bench)
