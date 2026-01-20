# GPU Performance Bottleneck Benchmark

A formal benchmark suite for quantifying four major GPU performance bottlenecks:

1. **Memory Coalescing** - Impact of non-contiguous memory access patterns
2. **Thread Divergence** - Cost of warp divergence due to conditional branches
3. **Atomic Contention** - Serialization overhead from concurrent atomic operations
4. **Arithmetic Intensity (Roofline)** - Memory-bound vs compute-bound performance

## Quick Start

```bash
make
./bench
python3 analyze.py  # Optional: generate visualization
```

## Benchmark Design

Each test uses a single formal parameter to isolate and quantify one performance factor.

### 1. Memory Coalescing

**Parameter:** `stride` (access pattern)

| Pattern | Description | Expected Effect |
|---------|-------------|-----------------|
| stride=1 | Contiguous access `data[tid]` | Optimal bandwidth |
| stride=32 | Strided access `data[tid*32]` | ~5x slowdown (no coalescing within warp) |
| random | Random access | Worst case |

### 2. Thread Divergence

**Parameter:** `div_factor` (number of execution paths within a warp)

| Factor | Description | Expected Effect |
|--------|-------------|-----------------|
| div=1 | All threads same path | Baseline (1x) |
| div=32 | Each thread different path | ~32x slowdown |

### 3. Atomic Contention

**Parameter:** `contention_factor` (threads per counter)

| Factor | Description | Expected Effect |
|--------|-------------|-----------------|
| 1 | No contention | Baseline (1x) |
| 32 | Warp-level contention | ~10x slowdown |
| ALL | Global contention | ~200x+ slowdown |

### 4. Roofline (Arithmetic Intensity)

**Parameter:** `flops_per_elem` (compute operations per memory access)

| AI (FLOP/Byte) | Description | Expected Behavior |
|----------------|-------------|-------------------|
| 0.25 | Low compute | Memory-bound |
| 16-64 | Balanced | Transition region |
| 256+ | High compute | Compute-bound |

## Output

### Console Output

```
=== Memory Coalescing Benchmark ===
pattern      time(ms) bandwidth(GB/s)   efficiency   slowdown
-------------------------------------------------------------
stride=1        0.004          1999.0       111.5%      1.00x
stride=32       0.023           372.4        20.8%      5.37x
random          0.023           371.4        20.7%      5.38x

=== Thread Divergence Benchmark ===
div_factor   time(ms)     slowdown
-------------------------------------
1               0.096        1.00x
32              2.767       28.73x

=== Atomic Contention Benchmark ===
contention     time(ms) throughput(M/s)     slowdown
----------------------------------------------------
1                 0.183       573252.00         1.0x
ALL              43.192         2427.69       236.1x

=== Roofline (Arithmetic Intensity) Benchmark ===
FLOPs/elem       AI     time(ms)       GFLOPS   slowdown        bound
------------------------------------------------------------------------
1              0.25        0.004        512.2       1.00x          MEM
1024         256.00        0.027      81033.7       6.47x      COMPUTE
```

### Generated Files

- `results.csv` - Raw benchmark data
- `results.png` - Visualization (5 charts including full Roofline model)

## Sample Results (RTX 5090)

| Bottleneck | Worst Case | Slowdown |
|------------|------------|----------|
| Memory Coalescing | random access | 5.4x |
| Thread Divergence | div=32 | 28.7x |
| Atomic Contention | ALL threads | 236x |
| Roofline | AI=256 | 6.5x (compute-bound) |

## Requirements

- CUDA Toolkit (nvcc)
- Python 3 with pandas, matplotlib, numpy (for visualization)

## License

Part of the GPU Extension Policy project.
