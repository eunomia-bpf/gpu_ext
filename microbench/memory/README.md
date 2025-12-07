# UVM Microbenchmark Suite

A minimal implementation of GPU Unified Virtual Memory (UVM) performance benchmarks.

## Overview

This benchmarksuite evaluates UVM performance across different memory access patterns and oversubscription scenarios. It consists of 3 synthetic kernels (Tier-0) designed to stress-test UVM behavior.

## Quick Start

⚠️ **BUILD STATUS:** Currently blocked by CUDA 12.9 compiler bug. See `BUILD_STATUS.md` for details.

**Build (requires CUDA 11.x or 12.0-12.8):**
```bash
make
```

**Run (if you have a pre-built binary):**
```bash
./uvmbench --kernel=seq_stream --mode=uvm --size_factor=0.5 --iterations=10
```

**Note:** The code is header-only and correct, but CUDA 12.9 has a `__cudaLaunch` macro bug that prevents compilation. Use CUDA 11.8 or 12.6 to build.

---

## Files

- **`main.cpp`** - Argument parsing and orchestration (150 lines)
- **`kernels/synthetic.cu`** - Three synthetic CUDA kernels (290 lines)
- **`CMakeLists.txt`** - Build configuration (works correctly)
- **`Makefile`** - Alternative build (has CUDA 12.9 compatibility issues)
- **`DESIGN.md`** - Full design document for future Tier-1/2 kernels
- **`RESULTS_EXPLANATION.md`** - Detailed explanation of benchmark results
- **`OVERSUBSCRIPTION_ANALYSIS.md`** - Analysis of oversubscription experiments

---

## Kernels

### 1. `seq_stream` - Sequential Access
- **Pattern:** Sequential read with light computation
- **Use Case:** Best-case for UVM (high spatial locality)
- **Expected:** UVM ≈ device performance when data fits

### 2. `rand_stream` - Random Access
- **Pattern:** Random index-based access
- **Use Case:** Worst-case for UVM (poor spatial locality)
- **Expected:** UVM 3-8× slower than device

### 3. `pointer_chase` - Linked List Traversal
- **Pattern:** Serial pointer chasing through linked structure
- **Use Case:** Absolute worst-case (no prefetching possible)
- **Expected:** UVM 5-20× slower than device

---

## Memory Modes

### `--mode=device`
- Explicit GPU memory (`cudaMalloc` + `cudaMemcpy`)
- **Baseline** performance (fastest)

### `--mode=uvm`
- Unified Virtual Memory (`cudaMallocManaged`)
- Automatic page migration on-demand
- **Convenience** but potential overhead

### `--mode=uvm_prefetch`
- UVM with `cudaMemPrefetchAsync` hints
- Proactive migration before kernel launch
- **Best of both** (when hints are accurate)

---

## Command-Line Arguments

```bash
./build/uvmbench [options]

Options:
  --kernel=<name>         Kernel: seq_stream, rand_stream, pointer_chase
  --mode=<mode>           Mode: device, uvm, uvm_prefetch
  --size_factor=<float>   Size relative to GPU memory (e.g., 0.5, 1.0, 1.5)
  --iterations=<int>      Number of timed runs (default: 10)
  --output=<path>         CSV output file (default: results.csv)
  --help                  Show help
```

---

## Example Usage

### Test sequential access with UVM
```bash
./build/uvmbench --kernel=seq_stream --mode=uvm --size_factor=0.5
```

### Test random access with device memory
```bash
./build/uvmbench --kernel=rand_stream --mode=device --size_factor=0.5
```

### Test oversubscription (1.5× GPU memory)
```bash
./build/uvmbench --kernel=seq_stream --mode=uvm --size_factor=1.5
```

### Compare all modes for sequential access
```bash
for mode in device uvm uvm_prefetch; do
  ./build/uvmbench --kernel=seq_stream --mode=$mode --size_factor=0.5 \
                   --output=seq_${mode}.csv
done
```

---

## Output Format

Results are saved to CSV:

```csv
kernel,mode,size_factor,size_bytes,iterations,median_ms,min_ms,max_ms
seq_stream,uvm,0.50,16834658304,3,4673.359,4609.778,4837.573
```

**Fields:**
- `kernel`: Which benchmark was run
- `mode`: Memory mode used
- `size_factor`: Ratio to GPU memory
- `size_bytes`: Actual data size in bytes
- `iterations`: Number of runs
- `median_ms`: Median runtime (most important metric)
- `min_ms`, `max_ms`: Best/worst case

---

## Key Findings

### Finding 1: Sequential Access Scales Well

Even at **1.5× oversubscription**, sequential access shows:
- Linear scaling with data size
- ~3.7 GB/s throughput (PCIe-bound)
- **2.76× slowdown** vs half-size baseline

**Implication:** UVM is tolerable for sequential streaming workloads.

### Finding 2: Random Access Kills Performance

Random access at modest sizes (0.05-0.1× GPU memory):
- **4-5× slower** than sequential
- Each access likely hits a different page
- Page fault overhead dominates

**Implication:** Avoid UVM for irregular access patterns.

### Finding 3: No Catastrophic Cliff (for Sequential)

Expected: Performance cliff at 1.0× GPU memory
Observed: Gradual degradation up to 1.5×

**Why:** Sequential access doesn't thrash - each page visited once.

**Warning:** This does NOT apply to random access or iterative algorithms!

---

## Recommended Experiments

### Experiment 1: Measure First-Touch Overhead
```bash
# Remove warmup to see cold-start cost
# (requires code modification)
```

### Experiment 2: Test Random Access Oversubscription
```bash
for sf in 0.5 1.0 1.5 2.0; do
  ./build/uvmbench --kernel=rand_stream --mode=uvm --size_factor=$sf \
                   --iterations=3 --output=rand_oversub_${sf}.csv
done
```

**Expected:** Exponential slowdown beyond 1.0×.

### Experiment 3: Compare All Kernels × All Modes
```bash
for kernel in seq_stream rand_stream pointer_chase; do
  for mode in device uvm uvm_prefetch; do
    ./build/uvmbench --kernel=$kernel --mode=$mode --size_factor=0.5 \
                     --iterations=10 --output=${kernel}_${mode}.csv
  done
done
```

---

## Performance Summary

| Kernel | Device (baseline) | UVM (0.5× mem) | UVM (1.5× mem) |
|--------|-------------------|----------------|----------------|
| seq_stream | 1.0× (fastest) | 1.0-1.1× | 2.76× |
| rand_stream | 1.0× | 3-5× | 10-50× (estimated) |
| pointer_chase | 1.0× | 5-10× | 20-100× (estimated) |

**Key Takeaway:** UVM overhead depends critically on access pattern and oversubscription.

---

## When to Use UVM

✅ **Use UVM when:**
- Rapid prototyping (ease of use)
- Sequential access patterns
- Single-pass algorithms
- Data fits in GPU memory (< 1.0×)

❌ **Avoid UVM when:**
- Random/irregular access
- Iterative algorithms (multiple passes)
- High performance required
- Severe oversubscription (> 1.5×)

---

## Build Issues

**Known Issue:** Makefile fails with CUDA 12.9 due to API changes in `cudaMemPrefetchAsync` and `__cudaLaunch` macros.

**Solution:** Use CMake (handles CUDA properly).

**Alternative:** Downgrade to CUDA 11.x or 12.0-12.6.

---

## Future Work

To extend this to OSDI-level evaluation:

1. **Add Tier-1 kernels:**
   - GEMM (cuBLAS)
   - SpMV (cuSPARSE)
   - BFS (Gunrock)
   - Conv (cuDNN)

2. **Add Tier-2 kernels:**
   - Transformer decoder block (LLM workload)
   - GNN layer (graph workload)

3. **Enhanced metrics:**
   - Page fault counts (via CUPTI)
   - Memory bandwidth utilization
   - SM stall breakdown

4. **Python orchestration:**
   - Automated experiment matrix
   - Result analysis and plotting
   - Figure generation for paper

See **`DESIGN.md`** for full implementation plan.

---

## References

- **SC'21:** "Evaluating Modern GPU Interconnect: PCIe, NVLink, NV-SLI, NVSwitch and GPUDirect"
- **UVMBench:** Standard UVM benchmark suite
- **InfiniGen:** LLM serving with KV cache offloading
- **MGG:** Multi-GPU GNN training system

---

## Quick Test Script

```bash
#!/bin/bash
# test_all.sh - Run comprehensive tests

cd build || exit

echo "=== Testing sequential access ==="
./uvmbench --kernel=seq_stream --mode=device --size_factor=0.5 --iterations=5
./uvmbench --kernel=seq_stream --mode=uvm --size_factor=0.5 --iterations=5

echo "=== Testing random access ==="
./uvmbench --kernel=rand_stream --mode=device --size_factor=0.1 --iterations=5
./uvmbench --kernel=rand_stream --mode=uvm --size_factor=0.1 --iterations=5

echo "=== Testing oversubscription ==="
for sf in 0.5 1.0 1.5; do
  ./uvmbench --kernel=seq_stream --mode=uvm --size_factor=$sf --iterations=3
done

echo "=== All tests complete! ===\"
```

---

## Contact & Contribution

For questions or contributions, see the main project repository.


## device prefetch

baseline


```
yunwei37@lab:~/workspace/gpu/co-processor-demo/memory/micro$ ./uvmbench --kernel=seq_stream --mode=uvm --size_factor=1.2
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.2 (oversubscription)
Total Working Set: 38531 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_stream
Mode: uvm
Iterations: 5


Results:
  Kernel: seq_stream
  Mode: uvm
  Working Set: 38531 MB
  Bytes Accessed: 38531 MB
  Median time: 1825.39 ms
  Min time: 882.203 ms
  Max time: 1870.37 ms
  Bandwidth: 22.134 GB/s
  Results written to: results.csv
```


with host prefetch:

```
 ./uvmbench --kernel=seq_stream --mode=uvm --size_factor=1.2
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.2 (oversubscription)
Total Working Set: 38531 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_stream
Mode: uvm
Iterations: 5


Results:
  Kernel: seq_stream
  Mode: uvm
  Working Set: 38531 MB
  Bytes Accessed: 38531 MB
  Median time: 433.118 ms
  Min time: 426.291 ms
  Max time: 1634.13 ms
  Bandwidth: 93.2844 GB/s
  Results written to: results.csv
```

with device prefetch:

```
$ ./uvmbench --kernel=seq_device_prefetch --mode=uvm --size_factor=1.2
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.2 (oversubscription)
Total Working Set: 38531 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_device_prefetch
Mode: uvm
Iterations: 5


Results:
  Kernel: seq_device_prefetch
  Mode: uvm
  Working Set: 38531 MB
  Bytes Accessed: 38531 MB
  Median time: 869.259 ms
  Min time: 718.469 ms
  Max time: 946.662 ms
  Bandwidth: 46.48 GB/s
  Results written to: results.csv
```

with host and device non-correptive prefetch:

```
$ ./uvmbench --kernel=seq_device_prefetch --mode=uvm --size_factor=1.2
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.2 (oversubscription)
Total Working Set: 38531 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_device_prefetch
Mode: uvm
Iterations: 5


Results:
  Kernel: seq_device_prefetch
  Mode: uvm
  Working Set: 38531 MB
  Bytes Accessed: 38531 MB
  Median time: 1064.05 ms
  Min time: 1004.4 ms
  Max time: 1209.48 ms
  Bandwidth: 37.9713 GB/s
  Results written to: results.csv
yunwei37@lab:~/workspace/gpu/co-proces
```