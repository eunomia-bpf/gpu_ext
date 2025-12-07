一套设计，把 microbench 分成三层：

* Tier‑0：极简 synthetic kernel（保留，但只是用来解释机制）
* Tier‑1：典型 “real kernel”（GEMM、Stencil、SpMV、BFS、Conv 等）
* Tier‑2：领域特定 kernel（Transformer layer、GNN layer、一层 CNN）

下面全部按"你准备写 OSDI measurement/system paper"的标准来设计。

---

## 一、Workload 设计：三层结构 (Tier-0/1/2)

### 1.1 Tier‑0：Synthetic Kernels（机制解释工具）

这一层你可以保留 2–3 个最基础的访问模式就够了，用来解释 UVM 行为本身：

* `seq_stream`：顺序读一大段 managed 数组（带计算），看 cold fault + steady‑state 吞吐。
* `rand_stream`：完全随机读写 managed 数组，看最坏场景下每页只用几个字节的情况。
* `pointer_chase`：典型 TLB + pointer-chasing 场景。

这层不再是 main evaluation，而是在 Section "UVM Behavior Characterization" 里给 real kernel 提供解读 basis——比如说明为什么 SpMV 这么烂，而 GEMM 相对没那么惨。

#### T0-RQ3 当前实现：Oversubscription Characterization with Page-Level Probing

**实验目标**

在统一的页级访问粒度（4096B stride）下，对比三种访问模式在 oversubscription 时的行为：
- Sequential stream: 顺序页扫描
- Random stream: 页级随机访问
- Pointer chase: 固定长度依赖链

**Kernel 设计**

统一的 Chunk-Based 抽象

所有 kernel 使用相同的线程配置和数据分块模型：

```
固定参数：
- blockSize = 256
- numBlocks = 256
- total_threads = 65536
- chunks_per_thread = 1
- total_chunks = 65536

计算 chunk size：
chunk_elems = (N / total_chunks)，向上对齐到页边界
nodes_per_chunk = (total_nodes / total_chunks)，向上对齐到页边界
```

每个线程负责处理固定数量的 chunk（当前为 1），总 WSS 由 `total_working_set` 参数控制，与线程数解耦。

#### Kernel 1: Sequential Stream (`seq_stream`)

**语义**：每个线程顺序扫描自己的 tile，模拟 GEMM/stencil 类型的访问模式。

**实现**：
```cpp
__global__ void seq_chunk_kernel(const float* input, float* output,
                                 size_t N,
                                 size_t chunk_elems,
                                 int chunks_per_thread,
                                 size_t stride_elems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int c = 0; c < chunks_per_thread; ++c) {
        size_t chunk_id = tid * chunks_per_thread + c;
        size_t chunk_start = chunk_id * chunk_elems;
        size_t chunk_end = min(chunk_start + chunk_elems, N);

        // Sequential access with stride
        for (size_t i = chunk_start; i < chunk_end; i += stride_elems) {
            float val = input[i];
            val = val * 1.5f + 2.0f;  // Light computation
            output[i] = val;
        }
    }
}
```

**内存布局**：
- Input array: 50% of total_working_set
- Output array: 50% of total_working_set
- Stride: 4096B (page-level probing)

**Bytes Accessed 计算**：
```cpp
size_t num_accesses = (N + stride_elems - 1) / stride_elems;
if (stride_bytes >= 4096) {
    // Page-level: count UVM migration bytes
    size_t num_pages = num_accesses;
    bytes_accessed = num_pages * 4096 * 2;  // input + output
}
```

#### Kernel 2: Random Stream (`rand_stream`)

**语义**：每个线程对自己 chunk 内的 pages 做无重复随机 permutation。

**实现**：
```cpp
__global__ void rand_chunk_kernel(const float* input, float* output,
                                  size_t N,
                                  size_t chunk_elems,
                                  int chunks_per_thread,
                                  size_t stride_elems,
                                  unsigned int base_seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = base_seed ^ tid;
    size_t elems_per_page = 4096 / sizeof(float);

    for (int c = 0; c < chunks_per_thread; ++c) {
        size_t chunk_id = tid * chunks_per_thread + c;
        size_t chunk_start = chunk_id * chunk_elems;
        size_t pages_in_chunk = (chunk_size + elems_per_page - 1) / elems_per_page;

        // Multiplicative congruential permutation (visit each page exactly once)
        seed = lcg_random(seed);
        size_t step = (seed | 1u);  // Ensure odd (coprime with power of 2)
        size_t offset = lcg_random(seed ^ 0xDEADBEEF) % pages_in_chunk;

        for (size_t p = 0; p < pages_in_chunk; ++p) {
            size_t random_page = (offset + p * step) % pages_in_chunk;
            size_t page_start = chunk_start + (random_page * elems_per_page);

            // Access one element per page
            float val = input[page_start];
            val = val * 1.5f + 2.0f;
            output[page_start] = val;
        }
    }
}
```

**关键特性**：
- 使用 `(offset + p * step) % pages_in_chunk` 实现无重复 permutation
- `step` 为奇数，与 2^k 互质，保证覆盖所有 pages
- 每个 page 恰好被访问 1 次，只是顺序被打散

**Bytes Accessed 计算**：
```cpp
size_t total_pages = (N * sizeof(float) + 4095) / 4096;
if (stride_bytes >= 4096) {
    bytes_accessed = total_pages * 4096 * 2;  // input + output
}
```

#### Kernel 3: Pointer Chase (`pointer_chase`)

**语义**：每个线程在自己的 chunk 内追踪固定长度的依赖链，测量 latency micro。

**数据结构**：
```cpp
struct Node {
    unsigned int next;  // Index of next node
    float data;
    float padding[1];   // Align to 16 bytes
};
```

**初始化（GPU-based）**：
```cpp
__global__ void init_chunks_kernel(Node* nodes, size_t nodes_per_chunk, int total_chunks) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t total_nodes = total_chunks * nodes_per_chunk;

    for (size_t i = tid; i < total_nodes; i += stride) {
        size_t chunk_id = i / nodes_per_chunk;

        // Random next pointer WITHIN the same chunk
        unsigned int r = lcg_random((unsigned int)i);
        size_t next_offset = r % nodes_per_chunk;
        size_t next_idx = chunk_id * nodes_per_chunk + next_offset;

        nodes[i].next = (unsigned int)next_idx;
        nodes[i].data = 1.0f;
    }
}
```

**Chase kernel**：
```cpp
__global__ void pointer_chunk_kernel(const Node* nodes, float* output,
                                     size_t nodes_per_chunk,
                                     int chunks_per_thread,
                                     int chase_steps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int c = 0; c < chunks_per_thread; ++c) {
        size_t chunk_id = tid * chunks_per_thread + c;
        size_t chunk_start = chunk_id * nodes_per_chunk;

        unsigned int cur = (unsigned int)chunk_start;
        float sum = 0.0f;

        // Chase pointers (dependent loads)
        #pragma unroll 4
        for (int s = 0; s < chase_steps; ++s) {
            sum += nodes[cur].data;
            cur = nodes[cur].next;
        }

        output[chunk_id] = sum;
    }
}
```

**关键参数**：
- `chase_steps = 8`：每个线程追踪的固定步数
- `nodes_per_chunk`：向上对齐到页边界（4096 / sizeof(Node)）

**内存布局**：
- Nodes array: 90% of total_working_set
- Output array: 10% of total_working_set

**Bytes Accessed 计算**：
```cpp
// Fixed logical work: total_chunks * chase_steps accesses
size_t logical_accesses = total_chunks * chunks_per_thread * chase_steps;
size_t logical_bytes = logical_accesses * sizeof(Node);

// Estimate pages touched
size_t pages_touched = (logical_bytes + 4095) / 4096;
size_t total_pages = (n_alloc * sizeof(Node) + 4095) / 4096;
pages_touched = min(pages_touched, total_pages);

bytes_accessed = pages_touched * 4096;
```

**注意**：Pointer chase 是 latency micro（固定工作量），不是 throughput micro（扫完整 WSS）。

---

### 1.2 Tier‑1：Real Kernel Family（和 SC'21 / UVMBench 一样的级别）

这一层是重点，目标是用一批**覆盖不同 *memory behavior* 的真实 kernel**，但每个 kernel 本身足够小、单一：

#### 1.2.1 Dense Linear Algebra：GEMM（高重用、高算密度）

* 对应 SC’21 的 cuBLAS sgemm  和 UVMBench 里的 GEMM/SGEMM
* 工作负载：

  * 单精度 GEMM：`C = A×B`，A,B,C 都是 `cudaMallocManaged`；
  * 用 cuBLAS 或 CUTLASS 实现（不用自己写 naive GEMM，否则 reviewer 直接问“为什么不用 cuBLAS？”）。
* 参数控制：

  * 矩阵大小 N×N，从 fit‑in 到 oversub（确保 A+B+C 合起来是 {0.25×,0.5×,1×,1.25×,1.5×,2×} GPU mem）；
  * 访问模式固定是 dense + cache friendly。
* 意义：

  * 这是 **“best‑case real kernel”**：算密度高、数据重用多，看 UVM 在极友好场景下的 overhead 下限。

#### 1.2.2 Stencil / 2D/3D Convolution：高空间局部性、有限重用

* 对应 UVMBench 里的 2DCONV/3DCONV、SC’21 的 Gauss‑Seidel / HPGMG‑FV。

* 工作负载：

  1. **2D 5-point stencil**（或 9‑point）：

     * 大数组 `A[N][N]` 和 `B[N][N]` 是 managed；
     * kernel：对每个 internal cell 做固定模板更新；
     * 时间步 t 可以做 1–10 步（增加 compute reuse）。
  2. **3D 7‑point stencil**（可选，增加内存压力）。

* 参数控制：

  * N 调到数组大小 {0.25×–2×} GPU mem；
  * 时间步数 t ∈ {1, 5}，看算密度变化对 UVM 影响。

* 意义：

  * 代表 HPC / PDE 类型应用：有空间局部性，但重用窗口远不及 GEMM；
  * 对 UVM 来说是中等难度 case。

#### 1.2.3 Sparse Linear Algebra：SpMV / SpMM（典型 irregular）

* 对应 UVMBench / SC 系列大量用的 SpMV、以及许多 graph/GNN 底层算子。
* 工作负载：

  * 用 cuSPARSE 的 CSR SpMV：`y = A x`，A 是 sparse matrix；
  * 稀疏矩阵来源：

    * SuiteSparse 的标准测试矩阵（如 webbase‑1M、kron_g500 等）；
    * 或 OGBN 的 graph adjacency 转成 CSR（和 MGG 中图数据接轨）。
  * A, x, y 全部 managed。
* 参数控制：

  * 控制 matrix size / nnz，使得 `sizeof(A) + sizeof(x) + sizeof(y)` 在 {0.25×–2×} GPU mem；
  * 矩阵类型区分：

    * “结构规整”（banded、block‑diag）；
    * “结构随机”。
* 意义：

  * 这是典型 **“TLB / page‑fault hell”**；
  * 跟 MGG 的 irregular graph pattern完全一脉相承，用来解释为什么 GNN + UVM 会被打爆。

#### 1.2.4 Graph Traversal：BFS / PageRank（一阶 graph kernel）

* 对应 UVMBench 的 BFS、SC/Graph500 测试中最常见的 kernel。
* 工作负载：

  * 选一个简化 BFS / PageRank kernel（可以直接基于 Gunrock 的实现，或者用 UVMBench 改的版本）；
  * 图数据用：

    * scale‑free graph（Twitter, com‑orkut 等）；
    * mesh‑like graph（road network）；
  * 顶点属性 / frontier 等结构全部用 UVM。
* 参数控制：

  * 图规模调到顶点数 / 边数使得 graph + aux arrays 占 {0.25×–2×} GPU mem；
  * 多次 BFS/PR 迭代，稳定阶段采样。
* 意义：

  * 这就是 MGG 前面的“单 GPU graph kernel”版本；
  * 显示 UVM 在 neighbor‑exploration 上表现如何，对比 SpMV。

#### 1.2.5 CNN 层 / Conv+BN+ReLU（DL 真实 kernel）

* 对应 PipeSwitch/TGS 那种 DL 模型的基本 building block。
* 工作负载：

  * 模仿 ResNet‑50 的某个 conv block（如 3×3 conv + BN + ReLU）：

    * 输入 feature map: `N×C_in×H×W`；
    * filter: `C_out×C_in×K×K`；
    * 输出: `N×C_out×H'×W'`；
  * 用 cuDNN 或 PyTorch C++ frontend 跑 forward，只是把所有 tensor 改为 `cudaMallocManaged` 分配。
* 参数控制：

  * 调 batch size N、feature map 分辨率，使整个 layer 的 working set 覆盖 {0.25×–2×} GPU mem；
  * 也可以固定 N、H、W，增大 C_in / C_out。
* 意义：

  * 对应真实 DL 推理 kernel，空间局部性较好，但中间 activation 大、权重 reuse 高；
  * 跟 PipeSwitch / TGS 里的场景有直接对齐。

---

### 1.3 Tier‑2：领域特定 kernel（LLM / GNN）

这一层用**简化但真实的“层”**，直接对上 InfiniGen / MGG 里的 evaluation 场景：

#### 1.3.1 Transformer decoder block（LLM‑style KV heavy kernel）

* 对应 InfiniGen 的单层 KV cache 使用模式。
* 工作负载：

  * 实现一个简化的 decoder block：

    * multi‑head attention（Q,K,V projection + softmax + matmul）；
    * MLP（两层全连接）；
  * KV cache 以 `[layers][heads][seq_len][head_dim]` 存放，全部用 `cudaMallocManaged`。
* 参数控制：

  * 固定 hidden size/head 数，扫 seq_len，让 KV cache 占 {0.5×–4×} GPU mem；
  * 可选：只把 KV cache 放 UVM，weights 用普通 device mem，复现实际场景。
* 意义：

  * 直接对应 InfiniGen 的 baseline：UVM 做 KV offloading。
  * 把 micro‑bench 跟完整 LLM 系统实验连起来。

#### 1.3.2 GNN layer（GCN message passing）

* 对应 MGG 的单层 GCN 聚合逻辑。
* 工作负载：

  * 给定 CSR graph + node feature matrix `H`，实现：

    * `H' = σ(Ā H W)`（典型的 GCN 一层）；
  * graph 结构、feature matrix 统统用 UVM；
  * 单 GPU 版本即可（multi‑GPU 留给系统部分）。
* 参数控制：

  * graph 用 ogbn-products / ogbn-papers100M 的子图（sampling 不要太 aggressive）；
  * 扫 feature 维度 d，使得 `|H| + |H'| ≈ {0.25×–2×} GPU mem`。
* 意义：

  * Irregular neighbor aggregation + medium compute intensity；
  * 直接给你解释 MGG 里 MGG‑UVM 那条惨不忍睹的曲线的 micro‑bench 版本。

---

## 二、Research Questions (RQ)

### 2.1 概述

这套 benchmark 最终要支撑的是类似这样的一组问题：

* **RQ1：UVM vs Device Memory 性能对比** - 在真实 kernel 上，UVM 相对显式 GPU 内存管理的性能损失有多大？
* **RQ2：访存模式影响** - UVM 在不同访存模式和计算强度的 kernel 上表现是否一致？
* **RQ3：Oversubscription 行为** - 在 realistic oversubscription（1.0×–2.0× 显存）下，哪些 kernel 还能"勉强可用"，哪些直接被 thrash 掉？
* **RQ4：Prefetch 效果** - 简单 prefetch（如 cudaMemPrefetchAsync）在 real kernel 上的收益/副作用有多大？

Tier‑0 synthetic 只用来给这几类 real kernel 找"解释工具"；**所有结论必须在 Tier‑1/2 上复现**。

---

### 2.2 RQ1: UVM vs Device Memory Performance Comparison

**目标**：量化 UVM 在 fits-in-memory 场景下的基础开销。

**实验设计**：

参数配置：
```python
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']  # Tier-0
MODES   = ['device', 'uvm']
SIZE_FACTORS = [0.25, 0.5, 0.75]  # All fits in GPU memory
STRIDE_BYTES = [4, 4096]  # Element-level vs Page-level
ITERATIONS = 5
```

对比维度：
- **Mode**: device (cudaMalloc + memcpy) vs uvm (cudaMallocManaged, no prefetch)
- **Access Pattern**: sequential, random, pointer-chase
- **Stride**: 4B (element) vs 4096B (page)

输出指标：
```csv
kernel,mode,size_factor,stride_bytes,median_ms,min_ms,max_ms,bytes_accessed,bw_GBps
```

**可视化**：

图 RQ1-1: Slowdown vs Access Pattern
- X 轴: kernel × stride
- Y 轴: Slowdown (UVM / Device)
- 每组两个 bar: stride=4B, stride=4096B
- Size factor 固定在 0.5x

图 RQ1-2: Throughput Comparison
- X 轴: Size Factor
- Y 轴: Bandwidth (GB/s)
- 曲线: device vs uvm，每个 kernel 单独一张子图
- 3×2 布局

**输出文件**：
- `rq1_results.csv`
- `rq1_mode_comparison.{pdf,png}`

---

### 2.3 RQ2: Access Pattern Impact

**目标**：理解不同访存模式对 UVM 性能的影响。

**实验设计**：

参数配置：
```python
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODES   = ['uvm']  # 只关注 UVM
SIZE_FACTORS = [0.5]  # 固定 fits-in
STRIDE_BYTES = [4, 16, 64, 256, 1024, 4096]  # 扫描不同访问粒度
ITERATIONS = 5
```

对比维度：
- **Stride**: 从 element-level (4B) 到 page-level (4096B)
- **Pattern**: seq vs random vs pointer-chase

输出指标：
```csv
kernel,stride_bytes,median_ms,bw_GBps,pages_touched,page_faults
```

**可视化**：

图 RQ2-1: Bandwidth vs Stride
- X 轴: Stride (bytes)，对数刻度
- Y 轴: Bandwidth (GB/s)
- 曲线: 三种 kernel
- 垂直线标记: 4096B (page boundary)

图 RQ2-2: Pattern Characterization
- 热力图: kernel × stride，颜色表示 normalized bandwidth

**输出文件**：
- `rq2_results.csv`
- `rq2_access_pattern.{pdf,png}`

---

### 2.4 RQ3: Oversubscription Behavior (当前 T0-RQ3 实现)

**目标**：表征 UVM 在 oversubscription 下的性能退化行为。

**实验设计**：

参数配置：
```python
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODES   = ['device', 'uvm']
SIZE_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]  # 覆盖 fits-in 到 oversub
BASELINE_SF = 0.5  # Baseline for normalization
STRIDE_BYTES = 4096  # 统一使用 page-level，公平对比
ITERATIONS = 3
```

**Size Factor 定义**：
- `total_working_set = size_factor × GPU_memory`
- 0.5x, 0.75x: fits-in memory (baseline)
- 1.0x: exactly at capacity
- 1.25x, 1.5x: oversubscription

**Mode 说明**：
- `device`: cudaMalloc + explicit memcpy (只跑 ≤0.8x，避免 OOM)
- `uvm`: cudaMallocManaged，无 prefetch

执行流程：
```python
for kernel in KERNELS:
    for mode in MODES:
        for size_factor in SIZE_FACTORS:
            # Skip device mode for oversubscription
            if mode == 'device' and size_factor > 0.8:
                continue

            run_benchmark(
                kernel=kernel,
                mode=mode,
                size_factor=size_factor,
                stride_bytes=4096,
                iterations=3
            )
```

每个配置运行流程：
1. Warmup: 2 iterations
2. Timed: 3 iterations
3. 统计: median, min, max runtime

输出指标：
```csv
kernel,mode,size_factor,stride_bytes,median_ms,min_ms,max_ms,bytes_accessed,bw_GBps
```

其中：
- `median_ms`: 中位数运行时间
- `bytes_accessed`: 基于 stride 和访问模式的逻辑字节数
- `bw_GBps`: `bytes_accessed / (median_ms / 1000)`

**可视化**：

图 RQ3-1: Runtime vs Size Factor (per kernel)

布局：3 行 × 2 列（每个 kernel 一行）

左列 - Runtime (Log Scale)：
- X 轴: Size Factor (× GPU Memory)
- Y 轴: Median Runtime (ms)，对数刻度
- 曲线: device vs uvm
- 垂直线: 1.0x 处标记 "GPU capacity"

右列 - Normalized Throughput：
- X 轴: Size Factor (× GPU Memory)
- Y 轴: Normalized Throughput (vs Device @ 0.5x)
- 计算: `norm_bw = current_bw / baseline_bw`
- 水平线: 1.0 处标记 baseline
- 垂直线: 1.0x 处标记 "GPU capacity"

图 RQ3-2: Summary Statistics Table

```
=================================================================
T0-RQ3 SUMMARY: Oversubscription with Page-Level Probing
=================================================================
Configuration: stride_bytes=4096 (page-level), iterations=3

Sequential Stream:
-----------------------------------------------------------------
  UVM at 1.0x: XXX.XXXms (XX.XXx vs 0.5x)
  UVM at 1.25x: XXXX.XXXms (XXX.XXx vs 0.5x)
  UVM at 1.5x: XXXX.XXXms (XXX.XXx vs 0.5x)

Random Stream:
-----------------------------------------------------------------
  UVM at 1.0x: XXX.XXXms (XX.XXx vs 0.5x)
  ...

Pointer Chase:
-----------------------------------------------------------------
  UVM at 1.0x: X.XXXms (XX.XXx vs 0.5x)
  ...
```

Slowdown 计算：
```python
uvm_base = kdf[(kdf['mode'] == 'uvm') &
               (kdf['size_factor'] == BASELINE_SF)]
uvm_at_t = kdf[(kdf['mode'] == 'uvm') &
               (kdf['size_factor'] == threshold)]

slowdown = uvm_at_t['median_ms'] / uvm_base['median_ms']
```

**输出文件**：
- `t0rq3_results.csv`: 所有配置的原始数据
- `t0rq3_oversub_page_stride.{pdf,png}`: 可视化图表

---

### 2.5 RQ4: Prefetch Effectiveness

**目标**：评估 cudaMemPrefetchAsync 对不同访问模式的影响。

**实验设计**：

参数配置：
```python
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODES   = ['uvm', 'uvm_prefetch']
SIZE_FACTORS = [0.5, 1.0, 1.25]
STRIDE_BYTES = 4096
ITERATIONS = 5
```

对比维度：
- **Prefetch**: 无 prefetch vs prefetch all to GPU
- **Size Factor**: fits-in (0.5x) vs at-capacity (1.0x) vs oversub (1.25x)

输出指标：
```csv
kernel,mode,size_factor,median_ms,prefetch_overhead_ms,speedup
```

**可视化**：

图 RQ4-1: Prefetch Speedup
- X 轴: Size Factor
- Y 轴: Speedup (uvm_prefetch / uvm)
- 曲线: 三种 kernel
- 水平线: speedup=1.0

图 RQ4-2: Overhead Breakdown
- 堆叠柱状图: prefetch overhead + kernel runtime

**输出文件**：
- `rq4_results.csv`
- `rq4_prefetch_effect.{pdf,png}`

---

### 2.6 RQ4 (扩展): UVM Prefetch + Fault Batching Heuristics Under Oversubscription

**目标**：在 oversub 场景下，调整 driver 的 prefetch 和 fault batching heuristics，能否在不同访问模式上带来可观改善？与用户层 `cudaMemPrefetchAsync` 相比，谁更有效、更稳定？

**实验设计**：

A. Driver-level prefetch & fault batching 配置：

不做全 3×3 穷举，选 4 个代表性组合：

```
C1 = P0 + B2 (prefetch 关，batch 默认)
  - uvm_perf_prefetch_enable=0
  - uvm_perf_fault_batch_count=256

C2 = P1 + B2 (driver 默认)
  - uvm_perf_prefetch_enable=1
  - uvm_perf_prefetch_threshold=51
  - uvm_perf_prefetch_min_faults=1
  - uvm_perf_fault_batch_count=256

C3 = P2 + B3 (aggressive prefetch + large batch)
  - uvm_perf_prefetch_enable=1
  - uvm_perf_prefetch_threshold=20
  - uvm_perf_prefetch_min_faults=1
  - uvm_perf_fault_batch_count=512

C4 = P1 + B1 (保守 prefetch + small batch，偏向低 latency)
  - uvm_perf_prefetch_enable=1
  - uvm_perf_prefetch_threshold=51
  - uvm_perf_prefetch_min_faults=1
  - uvm_perf_fault_batch_count=128
```

B. User-level `cudaMemPrefetchAsync`：

- 在 launch 前，对主要数组 `cudaMemPrefetchAsync(ptr, size, dev)`
- 对 oversub 版本的两种策略：
  - Sliding window prefetch（分片预取）
  - Hotspot prefetch（只预取热点数据，如 KV cache / feature hotset）

C. Size factor 范围：

重点关心 oversub 的"转折点"：
```python
SIZE_FACTORS = [0.5, 1.0, 1.25]  # 主要点
# 对 Transformer/GNN 可加 1.5x, 2.0x
```

**Tier-0 实验**：

```python
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODE = 'uvm'
SIZE_FACTORS = [0.5, 1.0, 1.25]
CONFIGS = ['C1', 'C2', 'C3', 'C4']
STRIDE_BYTES = 4096
ITERATIONS = 3-5

# 额外：mode=uvm_prefetch 在 C2 下跑，对比用户态 vs 内核态 prefetch
```

**Tier-1/2 实验**：

对每个 kernel（GEMM/stencil/SpMV/Conv/Transformer/GNN）：
```python
SIZE_FACTORS = [0.5, 1.25]  # 两点
CONFIGS = ['C1', 'C2', 'C3']  # 三种配置
ITERATIONS = 5
```

**输出指标**：

```csv
kernel,mode,size_factor,config,median_ms,bw_GBps,page_fault_count,prefetch_fault_count,migrated_bytes
```

从 `/proc/driver/nvidia-uvm/stats` 采样：
- `page_fault_count`
- `prefetch_fault_count` / migrated bytes
- fault service time（如果暴露）

**可视化**：

图 RQ4-A: T0 per-kernel throughput vs size_factor
- X 轴: size_factor (0.5/1.0/1.25)
- Y 轴: normalized throughput
- 曲线: C1–C4 + `uvm_prefetch`
- 布局: 3 个子图（seq/rand/pointer）

图 RQ4-B: Tier-1/2 bar chart – per real kernel best/worst config
- X 轴: kernel（GEMM, stencil, SpMV, Conv, Transformer, GNN）
- Y 轴: slowdown vs device@0.5×
- 每个 kernel 三根 bar: C1、C2、C3（在 1.25× 下）

图 RQ4-C: (可选) Fault stats 对比
- 堆叠柱状图: fault rate / migrated bytes
- 证明 prefetch 早了/晚了导致的 fault 链形态改变

**输出文件**：
- `rq4_extended_results.csv`
- `rq4_heuristics_sweep.{pdf,png}`

---

### 2.7 RQ5: UVM Thrashing Detection + Pin Strategy Effectiveness

**目标**：对 pointer-chase / CPU+GPU 共同访问的数据结构，UVM 内建的 thrash detector + pin heuristics 在多大程度上能缓解"来回迁移"的灾难？调得激进时会不会伤及正常 kernel？

**实验设计**：

A. Thrashing 相关参数配置：

给 3–4 个代表性 policy：

```
T0 – Thrashing OFF
  - uvm_perf_thrashing_enable=0

T1 – Driver Default
  - 保持默认值 (threshold=3, pin=300ms, lapse=500us)

T2 – Conservative detection, light pinning
  - uvm_perf_thrashing_threshold=5
  - uvm_perf_thrashing_pin_threshold=10
  - uvm_perf_thrashing_pin=300
  - uvm_perf_thrashing_lapse_usec=1000

T3 – Aggressive detection + long pin
  - uvm_perf_thrashing_threshold=3
  - uvm_perf_thrashing_pin_threshold=3
  - uvm_perf_thrashing_pin=1000
  - uvm_perf_thrashing_lapse_usec=1000
```

**Tier-0 实验**：

1. `pointer_chase` (GPU-only)：

```python
SIZE_FACTORS = [0.5, 1.0, 1.25]
POLICIES = ['T0', 'T1', 'T2', 'T3']
# 看 oversub 时从 10µs → 秒级的退化能否被 T2/T3 拉回来
```

2. CPU+GPU shared synthetic：

设计一个 managed array W：
```python
size_factor = [1.0, 1.25]
loop K 次，每轮：
  - GPU kernel 顺序/随机扫 W 一遍
  - CPU 在 host 上对 W 的前一半做 random walk
记录每轮 GPU/CPU 端的时间
```

在 T0/T1/T2/T3 下跑，对比：
- GPU runtime 是否因为 thrashing policy 改善
- CPU 端是否被 pin 策略饿死（latency 变长）

**Tier-1/2 实验**：

SpMV/BFS/GNN：选 "CPU 部分参与" 的版本
- CPU 做 advance/compaction、GPU 做 neighbor gather
- 或在每次 GPU kernel 间隔插一个 CPU 访问 pass

```python
SIZE_FACTORS = [1.25]  # oversub 点
POLICIES = ['T0', 'T1', 'T3']
```

**输出指标**：

```csv
kernel,policy,size_factor,median_ms,thrashing_events,pinned_pages,pin_duration_avg,migration_count
```

从 UVM stats 获取：
- #thrashing events
- #pinned pages
- pin duration 平均值
- eviction/migration 次数

**可视化**：

图 RQ5-1: pointer-chase latency vs size_factor vs policy
- X 轴: size_factor
- Y 轴: runtime (log scale)
- 曲线: T0/T1/T2/T3
- 看 1.25× 点是否对 T3 有明显改善

图 RQ5-2: CPU+GPU synthetic heatmap
- X 轴: policy（T0–T3）
- Y 轴: iteration index
- 颜色: GPU 时间或合计时间
- 一眼看出哪种策略最稳定而不 thrash

图 RQ5-3: real kernels bar chart
- X 轴: kernel（SpMV/BFS/GNN）
- Y 轴: throughput (normalized to device-only)
- Bar: T0/T1/T3 在 1.25× 下

**输出文件**：
- `rq5_results.csv`
- `rq5_thrashing_policies.{pdf,png}`

---

### 2.8 RQ6: Access Counter Migration + Page Table Placement Impact

**目标**：在 H100 上，access-counter-based migrations + GPU-resident page tables 在大规模 oversubscription 下能否真正改善性能？还是只是二阶优化，跟 10×–100× 退化相比只是噪声？

**实验设计**：

A. Access counter 和 page table 参数配置：

定义几组策略：

```
A0 – Access counter OFF + page table in sys
  - uvm_perf_access_counter_migration_enable=0
  - uvm_page_table_location=sys

A1 – Default
  - migration_enable=-1 (auto)
  - uvm_perf_access_counter_threshold=256
  - uvm_page_table_location=auto

A2 – Aggressive migration, sys page table
  - uvm_perf_access_counter_migration_enable=1
  - uvm_perf_access_counter_threshold=64
  - uvm_page_table_location=sys

A3 – Aggressive + page table in VRAM
  - uvm_perf_access_counter_migration_enable=1
  - uvm_perf_access_counter_threshold=64
  - uvm_page_table_location=vid
```

**Tier-0 实验**：

`rand_stream` / `pointer_chase`：
```python
SIZE_FACTORS = [0.75, 1.0, 1.25, 1.5]
CONFIGS = ['A0', 'A2', 'A3']
STRIDE_BYTES = 4096
ITERATIONS = 3-5
```

`seq_stream` 作为控制组：
```python
SIZE_FACTORS = [0.75, 1.25]  # 两点
CONFIGS = ['A0', 'A3']
# 看 dense sequential 场景下是否几乎无差
```

**Tier-1/2 实验**：

SpMV/BFS：
```python
SIZE_FACTORS = [1.0, 1.25]  # oversub 转折点
CONFIGS = ['A0', 'A2', 'A3']
```

Transformer/GNN：
```python
# KV cache 1.5×、feature matrix 1.5× (明显 oversub)
SIZE_FACTORS = [1.5]
CONFIGS = ['A0', 'A2', 'A3']
```

注意：page table 放 `vid` 会消耗少量显存，size_factor 估算时预留 margin（如 0.95× 当 1.0×）

**输出指标**：

```csv
kernel,config,size_factor,median_ms,bw_GBps,access_counter_migrations,migrated_bytes,pcie_traffic,tlb_misses
```

从 UVM stats 获取：
- `access_counter_migrations` 数量
- migrated bytes

从 Nsight Systems / nvprof 获取：
- PCIe traffic
- TLB miss / page table walk（如果能拿到）

**可视化**：

图 RQ6-1: T0 rand/pointer throughput vs size_factor
- X 轴: size_factor
- Y 轴: normalized throughput
- 曲线: A0 / A2 / A3
- 重点看 1.25× / 1.5× 时是否 A2/A3 明显优于 A0

图 RQ6-2: real kernels per-kernel bar chart
- X 轴: kernel（SpMV, BFS, Transformer, GNN）
- Y 轴: throughput normalized to device fit-in baseline
- 三根 bar: A0/A2/A3（在某个 oversub 点 1.25× or 1.5×）

图 RQ6-3: (可选) access-counter effectiveness scatter
- X 轴: #migrations (A2/A3)
- Y 轴: throughput improvement vs A0
- 如果点云贴近 x 轴，说明 access-counter migration 作用有限

**输出文件**：
- `rq6_results.csv`
- `rq6_access_counter_placement.{pdf,png}`

**预期结论**：

如果最后发现：
- 不管 rand_stream 还是 SpMV/BFS/Transformer/GNN，A2/A3 相对 A0 的改进只有 ±10–20%
- 而 oversub 本身带来的 regression 是 10×–100×
- page table 在 `vid` vs `sys` 差别 <10%

则可给出尖锐结论：**现有 access-counter / page-table placement 这些高级机制只是在原本就很糟糕的 oversub 行为上做小修小补，无法从根本上改变性能形状**。这为后续机制（page-subdividing、device-driven migration、multi-GPU 协调）提供了强动机。

---

## 三、实验矩阵：real kernel 为主，synthetic 为辅

### 3.1 维度 1：内存模式（还是那四个）

对**每个 Tier‑1 / Tier‑2 kernel**（至少）跑四种模式：

1. **Device‑only baseline（显式显存 + memcpy）**

   * 所有数据 `cudaMalloc`；
   * host side 用 pinned mem + `cudaMemcpyAsync` 预拷贝；
   * kernel 本身只碰 device mem；
   * optional：对 oversub 实验用 “滑动窗口 + double buffering” 版本做 best‑effort out‑of‑core baseline。

2. **UVM（默认，无 prefetch）**

   * 全部 `cudaMallocManaged`；
   * 不用 `cudaMemPrefetchAsync` / `cudaMemAdvise`；
   * 完全交给 UVM driver。

3. **UVM + prefetch to GPU**

   * kernel 前调用 `cudaMemPrefetchAsync(ptr, size, device)` 对所有主要数组 prefetch；
   * 对 LLM/GNN 这类带 KV/feature big tensor 的，考虑只对热数据 prefetch。

4. **UVM + oversubscription**

   * working set > GPU mem；
   * 模拟 “KV cache / graph / activation 放不下” 的真实情况。

这四个是 **固定组合**，贯穿所有 kernel，这样 reviewer 会觉得你是在做系统性对比，而不是 cherry-pick。

---

### 3.2 维度 2：working set / GPU mem 比例

对每个 kernel 至少扫这些点：

* `S / GPU_mem ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`
* 对 LLM / KV cache，可以再加 `3.0, 4.0`，模拟特别长 context。

不要每个 kernel 都把所有点跑齐，但至少：

* dense GEMM / stencil：跑到 1.5× 或 2×；
* SpMV / BFS / GNN：重点看 1.0×–1.5× 这一段，说明 “刚开始 oversub 的时候已经很惨”；
* Transformer：跑全程，说明 LLM KV 场景下 UVM 有多不可用。

---

### 3.3 维度 3：并发度 / 算密度

这里不要搞到 combinatorial explosion，只挑关键两个维度：

* **并发度**：

  * 给每个 kernel 选两种 launch config：

    * “低 occupancy”（gridDim ≈ #SM, blockDim 小）；
    * “高 occupancy”（gridDim > 4×#SM, blockDim = 256/512）；
  * 用来验证：简单多发几个 warps 并不能掩盖 UVM latency（援引 SC’21 的结论）。

* **算密度（compute intensity）**（只对 stencil / GNN / Transformer 做）：

  * stencil：调整 time steps t = 1 vs t = 10；
  * GNN：增加 feature dim / 增加 nonlinearity；
  * Transformer：加入/去掉 MLP 部分。
    目的：看看 compute heavy kernel 是否对 UVM 更“宽容”。

---

## 四、实现建议：怎么把 real kernel 塞进同一个 microbench 框架

### 4.1 框架结构

你可以设计成一个统一的 `uvmbench` 可执行程序，形态类似 UVMBench，但更偏 “mechanism measurement”：

* 命令行参数：

  * `--kernel={gemm,stencil2d,spmv,bfs,conv,transformer,gnn,...}`
  * `--mode={device,uvm,uvm_prefetch,uvm_oversub}`
  * `--size_factor=0.25..4.0`（按 GPU mem 比例）
  * `--occupancy={low,high}`
* 内部每个 kernel 是一个 C++/CUDA 函数：

  * `run_gemm(const Config&, Result*)`
  * `run_stencil2d(...)`
  * …
* 每个函数负责：

  1. 按 `size_factor` 推导实际问题规模（N, nnz, H×W, seq_len 等）；
  2. 按 `mode` 选择 `cudaMalloc` 还是 `cudaMallocManaged` + prefetch；
  3. 调用 cuBLAS/cuDNN/cuSPARSE 或自己的 kernel；
  4. 用 `cudaEvent` 计时，跑多次取 median；
  5. 收集 profiler/driver 统计（后面说）。

### 4.2 具体 kernel 实现建议

* GEMM：

  * 用 cuBLAS `cublasSgemm`，调用前后插 event 计时；
  * 确保 A,B,C 都是 managed 或 device。
* Stencil：

  * 写自己的 2D/3D kernel（教材级别那个），用 shared mem 做 basic 优化，保证不是完全 naive；
  * 迭代 t 次。
* SpMV：

  * cuSPARSE `cusparseSpMV`，CSR + managed arrays；
  * 对 nnz 很大的矩阵（百万以上）才有意义。
* BFS / PageRank：

  * 可以直接拿 Gunrock 的 BFS kernel 抽出 main loop，简化成单 GPU 版本；
  * 或者用 UVMBench 改好的 BFS 变种。
* Conv layer：

  * cuDNN `cudnnConvolutionForward` + BN + ReLU；
  * 或者直接在 PyTorch C++ API 里 instantiate 一层 conv 模型，跑 forward；
  * 反正内存模式对他们都是透明的。
* Transformer：

  * 可以复用 HuggingFace/DeepSpeed 中某个 decoder layer 的 CUDA kernel，或者简单用 PyTorch Eager 的一层，前提是你能控制 memory 分配方式（managed vs device）。
* GNN layer：

  * 用 DGL/PyG 的 GCNConv/GATConv 内核，自己写 wrapper 控制 memory；
  * 或者照 MGG 的 pseudo‑code 实現一层 gather‑aggregate kernel。

这部分的策略很简单：**凡是有成熟库的，就用库**，不要自己造一个 naive 版本然后被 reviewer 问“为啥你不跑 cuBLAS/cuDNN”。

---

## 五、测量与图：重写版（以 real kernel 为主）

### 5.1 指标

和之前说的类似，但强调要在每个 real kernel 上都收：

* Runtime：每次 kernel 或每个 iteration 的 wall‑clock（取 median 或 P95）；
* Effective throughput：

  * 对 GEMM：GFLOPS；
  * 对 stencil/SpMV/BFS：GB/s 或 GEdges/s；
  * 对 conv：images/s；
  * 对 Transformer/GNN：tokens/s 或 nodes/s；
* Slowdown vs device‑only baseline；
* UVM‑specific：

  * migrated bytes；
  * #page faults；
  * batch 处理时间（如果你 patch 了 driver）。
* GPU counters：

  * 内存带宽利用率；
  * SM stall breakdown。

### 5.2 推荐图（这次全部以 real kernel 为主）

#### 图 A：不同 kernel 的整体性能损失

* X 轴：kernel 类型（GEMM, stencil2d, SpMV, BFS, Conv, Transformer, GNN）
* Y 轴：slowdown (UVM vs device‑only) 在 `S = 1.0× GPU mem` 时的值；
* 每个 kernel 画两个 bar：

  * UVM no prefetch；
  * UVM + prefetch。

**解释方向：**

* 可以直接写一句非常“硬”的话：

  * “On average, UVM slows down real‑world kernels by 1.3×–8× even when data fits in GPU memory; sparse and graph kernels suffer the most.”
* 把 SC’21 “管理开销 > memcpy”的结论搬出来佐证。

#### 图 B：Oversubscription 扫描（per kernel）

选代表性的 3–4 个 kernel（比如 GEMM, stencil2d, SpMV, Transformer），画 4 张类似的图：

* X 轴：`S / GPU_mem`；
* Y 轴：throughput（normalize 到 device‑only fit‑in mem 情况 = 1.0）；
* 曲线：device sliding‑window baseline（如果有）、UVM、UVM+prefetch。

**你能讲的故事：**

* GEMM 在 1.25× 之前还能苟一苟，2× 基本折半甚至更差；
* SpMV / BFS 在 1.1× 左右就开始疯狂 thrashing；
* Transformer/LLM 在 >1× 时完全 PCIe‑bound，延迟线性/超线性炸裂（对应 InfiniGen 里 UVM baseline 的行为）。

#### 图 C：算密度对 UVM 的影响（stencil / GNN / Transformer）

* 例如对 stencil2d，画：

  * X 轴：time steps t；
  * Y 轴：slowdown (UVM vs device‑only)，分别在 `S=1.0×` 和 `1.25×` 两种规模下。
* 类似地，对 GNN/Transformer 改变 feature dim / 是否包含 MLP。

**解释：**

* 如果算密度加大，对 GEMM/Conv 来说 UVM 的额外 latency 可以部分隐藏；
* 对极度 memory bound 的 SpMV/BFS 基本帮不上忙；
* 这直接支持后面系统设计里“某些 kernel 适合作为 out‑of‑core candidate，某些完全不适合”。

#### 图 D：UVM 行为解剖（从 real kernel 映射回 synthetic）

这一步是把 Tier‑0 synthetic 拿回来发挥作用：

* 先给例如 SpMV/BFS/Transformer 分析他们的访问 pattern：

  * 通过 driver instrumentation 或 trace 统计「每次 kernel 实际访问的 unique pages / total accesses」；
  * 推出一个“等价 stride / randomness”。
* 然后把这些 pattern 映射回你之前的 `seq_stream` / `rand_stream` synthetic 实验中对应的点，展示：

  * “the behavior of real kernels falls between synthetic cases X and Y”；
* 用来证明你对 UVM 行为的解释不是拍脑袋，而是有系统 mapping 的。

Reviewer 看这个会觉得你不是瞎凑 microbench，而是**用 synthetic 作为分析工具，真正服务 real workload**。

---

## 六、怎么在论文里“讲”这套设计（而不是像 benchmark paper 那样罗列）

你最后的 narrative，应该不是“我们设计了一套很酷的 benchmark”，而是类似：

1. **先定义 scope：**

   * “We focus on four representative application domains where UVM is widely used or frequently proposed as a fallback: dense HPC (GEMM), stencil/HPC simulations, sparse/graph analytics, and DL/LLM/GNN workloads.”
   * 引 SC’21、UVMBench、PipeSwitch、MGG、InfiniGen，证明这四类不是瞎挑的。

2. **然后说选择的 kernel 是 “realistic building blocks”：**

   * “Our benchmark suite consists of seven real‑world kernels that are directly reused or abstracted from widely‑deployed libraries and systems (cuBLAS/cuDNN/cuSPARSE, Gunrock, DGL, HuggingFace), plus a small set of synthetic kernels used only for low‑level analysis.”

3. **接着强调你不是搞 workload zoo，而是为了回答前面那 4–5 个 RQ：**

   * 每个 kernel 类别对应某个 RQ 的极端点（best‑case / worst‑case）。

4. **最后用图 A–D 把故事串起来：**

   * “UVM’s best‑case overhead even on dense GEMM is X–Y×…”
   * “Irregular kernels such as SpMV and BFS suffer Z× slowdown even when their working set fits in GPU memory…”
   * “Naive prefetch helps only in dense/stencil workloads; in LLM‑style KV access, UVM becomes completely PCIe‑bound beyond 1.25× oversubscription…”

你要真按这个路数来写，**microbench 部分就已经是一个非常 solid 的 measurement section**，接下来你可以自然引出任何你想做的机制（比如 device‑driven migration、sub‑page UVM、multi‑GPU coordinated eviction 等等）。

---

如果你接下来想，我可以直接把这一套 Tier‑1/Tier‑2 kernel 列成一个“实验 checklist”（比如：行 = kernel，列 = mode × size_factor × occupancy，标出必须跑的组合、optional 的组合），你丢给学生去实现的时候就不会乱。
