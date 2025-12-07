# UVM Microbenchmark 设计文档

## 定位与 Scope

### 这套 Benchmark 能说明什么

1. **UVM 策略在不同访存模式下的行为差异**
   - page fault 数量、read/write 比例
   - migration 次数、数据量
   - HtoD / DtoH 带宽利用
   - end-to-end 吞吐 slowdown

2. **策略是否利用了 temporal/spatial locality**
   - 当同一数据被重复访问时，策略能否避免无谓的 eviction/thrash
   - 在不同 oversub factor 下的性能退化曲线

3. **不同访存 pattern 的对比**
   - LLM-style 权重复用 vs 顺序流式 vs 随机访问 vs stencil 迭代

### 这套 Benchmark 不能说明什么

1. ❌ 不能证明"在真实 LLM 服务上的 end-to-end 改善有多大" — 需要 vLLM/llama.cpp 等真实框架
2. ❌ 不能证明"策略在所有模型、所有 batch size 下都有效" — 只对应特定访问模式
3. ❌ 不能替代多租户/多优先级调度实验 — 需要真实多租户场景

### 使用建议

| 用途 | Benchmark 类型 |
|------|---------------|
| **主要评估** (throughput/p99) | 真实应用: vLLM, llama.cpp, PyTorch GNN, Faiss |
| **机制解释** (page fault 行为) | 本 microbenchmark suite |
| **overhead 测量** | seq_stream, rand_stream |

---

# GEMM Kernel 设计

## 目标

设计 **LLM-style 权重复用** 的 synthetic benchmark，用于评估 UVM 策略在"大静态权重 + token 重复访问 + oversubscribe"场景下的效果。

## 真实模型参数对照

| 模型 | dim | hidden | num_layers | 每层权重 | 总权重 |
|------|-----|--------|------------|---------|--------|
| Qwen-0.6B | 1024 | 2816 | 28 | ~11MB | ~300MB |
| Qwen-1.8B | 2048 | 5504 | 24 | ~45MB | ~1.1GB |
| LLaMA-7B | 4096 | 11008 | 32 | ~180MB | ~5.8GB |
| LLaMA-13B | 5120 | 13824 | 40 | ~283MB | ~11.3GB |
| LLaMA-70B | 8192 | 28672 | 80 | ~940MB | ~75GB |

### 推荐的 Benchmark 配置

基于真实模型，选择以下配置作为默认参数：

```cpp
// 配置 1: 类似 LLaMA-7B (推荐)
const int dim = 4096;
const int hidden = 11008;
const int NUM_LAYERS = 32;
// 每层 ~180MB，总权重 ~5.8GB
// size_factor=1.0 时约 32/5.8 ≈ 5.5x 权重，模拟 ~180 层

// 配置 2: 类似 Qwen-1.8B (较小)
const int dim = 2048;
const int hidden = 5504;
const int NUM_LAYERS = 24;
// 每层 ~45MB，总权重 ~1.1GB
```

## 现有 Benchmark 设计分析

### PolyBench/UVM benchmark 的方式

```cpp
// UVM_benchmarks: 512 x 512，约 3MB
// UVM_benchmarks_oversub: 12800 x 12800 (512*25)，约 1.9GB x 3 ≈ 5.6GB
#define NI 512 * 25
#define NJ 512 * 25
#define NK 512 * 25

// 单个 GEMM，访问全部数据
gemm_kernel<<<grid, block>>>(A_gpu, B_gpu, C_gpu);
```

**核心思路**：
- 直接放大单个矩阵维度
- **一个 kernel launch 访问全部数据**
- 简单直接，肯定会触发 page fault

### Hotspot 的方式

```cpp
// 迭代算法：同一数据被反复访问
for (t = 0; t < total_iterations; t += num_iterations) {
    calculate_temp<<<dimGrid, dimBlock>>>(
        MatrixPower, MatrixTemp[src], MatrixTemp[dst], ...);
}
```

**核心思路**：
- 固定大小的数据
- **通过迭代反复访问同一数据**
- 测试 UVM 对重复访问的处理

## 两种设计方案

### 方案 A：PolyBench 方式（单个大 GEMM）

```cpp
// 直接用一个巨大的 GEMM
const size_t N = calculate_dim_for_size(total_working_set);
cudaMallocManaged(&A, N * N * sizeof(float));
cudaMallocManaged(&B, N * N * sizeof(float));
cudaMallocManaged(&C, N * N * sizeof(float));

gemm_kernel<<<grid, block>>>(N, N, N, alpha, beta, A, B, C);
```

| 优点 | 缺点 |
|------|------|
| 简单直接 | 单个 kernel 永远不会访问 32GB |
| 肯定触发 page fault | 不像真实 DL 负载 |
| 与 PolyBench 一致 | 计算量巨大（O(N³)）|

### 方案 B：多 batch 独立（当前设计）

```cpp
// 当前设计：682 个独立 batch
for (int b = 0; b < num_batches; b++) {
    gemm_kernel(A[b], B[b], C[b]);  // 每个 48MB，互不相关
}
```

| 优点 | 缺点 |
|------|------|
| 接近真实 batch 处理 | 每个 kernel 工作集只有 48MB |
| 顺序执行，无 sync | GPU 可以缓存所有数据 |
| | **不会触发 eviction** |

**问题分析**：虽然总分配 32GB，但每个 kernel 只访问 48MB，GPU 有足够缓存，不会产生真正的内存压力。

## 真实推理场景分析

### 参考：Qwen3-0.6B 推理 (runcu.cu)

```cpp
// 每个 token 的 forward pass
for (int l = 0; l < n_layers; l++) {
    // Attention
    matmul(q, x, Wq[l], dim, att_dim);    // 1024 x 2048
    matmul(k, x, Wk[l], dim, kv_dim);     // 1024 x 1024
    matmul(v, x, Wv[l], dim, kv_dim);     // 1024 x 1024
    matmul(out, attn, Wo[l], att_dim, dim); // 2048 x 1024

    // FFN
    matmul(h1, x, W1[l], dim, hidden);    // 1024 x 3072
    matmul(h2, x, W3[l], dim, hidden);    // 1024 x 3072
    matmul(x, h, W2[l], hidden, dim);     // 3072 x 1024
}
// 无显式 sync，CUDA stream 自动保证顺序
```

### 关键特征

| 特征 | 说明 |
|------|------|
| 矩阵大小 | 中等固定大小 (1024x3072 等) |
| 执行方式 | 串行，layer by layer，无 sync |
| 权重复用 | 同一组 W 被每个 token 反复访问 |
| 访问模式 | GEMM 混合模式（A 顺序，B 跳跃，C 顺序）|

## 当前设计问题

### 问题 1：独立 batch 没有内存压力

```cpp
// 当前设计：682 个独立 batch
for (int b = 0; b < num_batches; b++) {
    gemm_kernel(A[b], B[b], C[b]);  // 每个 48MB，互不相关
}
```

- 每个 kernel 工作集只有 48MB
- GPU 32GB 可以缓存所有 batch
- 不会触发 eviction，没有真正的内存压力

### 问题 2：不符合权重复用模式

真实推理中，权重被多个 token 反复访问：
```
Token 1: W[0] → W[1] → ... → W[27]
Token 2: W[0] → W[1] → ... → W[27]  // 同一组 W
Token 3: W[0] → W[1] → ... → W[27]
```

## 建议设计

### 模拟权重复用的神经网络

```cpp
void run_gemm(...) {
    // 1. 固定层大小（模拟真实网络）
    const int dim = 1024;
    const int hidden = 3072;
    const int num_layers = 28;

    // 2. 分配权重（共享，只分配一次）
    // size_factor 控制总权重大小
    std::vector<DATA_TYPE*> W(num_layers);
    size_t weight_per_layer = dim * hidden * sizeof(DATA_TYPE);  // ~12MB
    for (int l = 0; l < num_layers; l++) {
        cudaMallocManaged(&W[l], weight_per_layer);
        // 初始化...
    }

    // 3. 分配 activation（每个 token 复用）
    DATA_TYPE *x, *h, *out;
    cudaMallocManaged(&x, dim * sizeof(DATA_TYPE));
    cudaMallocManaged(&h, hidden * sizeof(DATA_TYPE));
    cudaMallocManaged(&out, dim * sizeof(DATA_TYPE));

    // 4. 计算需要多少 token 来达到目标工作量
    size_t total_weights = num_layers * weight_per_layer;
    int num_tokens = total_working_set / total_weights;
    if (num_tokens < 1) num_tokens = 1;

    // 5. 模拟推理：每个 token 遍历所有层
    auto launch = [&]() {
        for (int t = 0; t < num_tokens; t++) {
            for (int l = 0; l < num_layers; l++) {
                // GEMV: W[l] @ x → out
                gemm_kernel<<<grid, block>>>(dim, hidden, 1,
                                             1.0f, 0.0f,
                                             x, W[l], out);
                // 不 sync，继续下一层
            }
        }
    };

    time_kernel(launch, warmup, iterations, runtimes, result);
}
```

### 内存压力来源

| size_factor | 总权重 | GPU 内存 | 效果 |
|-------------|--------|----------|------|
| < 1.0 | < 32GB | 32GB | 权重全部驻留，热缓存 |
| = 1.0 | = 32GB | 32GB | 边界情况 |
| > 1.0 | > 32GB | 32GB | **触发 eviction**，测试 UVM 策略 |

当 `size_factor > 1` 时：
- 权重无法全部放入 GPU
- 每个 token 需要重新加载被 evict 的权重
- 反复的 page fault 测试 UVM 预取/eviction 策略

### 访问模式

```
Token 0: W[0] → W[1] → ... → W[27]
              ↑ 顺序访问，可预测
Token 1: W[0] → W[1] → ... → W[27]
              ↑ 重复访问相同权重
Token 2: W[0] → W[1] → ... → W[27]
              ...
```

- **顺序性**：layer 按顺序访问
- **重复性**：权重被多个 token 反复访问
- **可预测性**：访问模式确定，利于测试预取策略

## 对比其他 kernel

| Kernel | 访问模式 | 工作集 | 适合测试 |
|--------|---------|--------|---------|
| seq_stream | 纯顺序 | 连续大块 | 顺序预取 |
| rand_stream | 随机 | 分散访问 | 按需调页 |
| gemm (新设计) | 顺序+重复 | 权重复用 | **神经网络场景** |
| hotspot/jacobi | 5点 stencil | 邻居访问 | 迭代求解 |
| bfs | 图随机 | 不规则 | 图算法 |

## 实现注意事项

1. **不要 sync**：kernel 之间不要显式 sync，让 CUDA stream 自动排队
2. **权重在 CPU 初始化**：UVM 模式下在 CPU 初始化，确保页面初始在 CPU
3. **activation 可以小**：每个 token 的 activation 很小，不是内存压力来源
4. **权重是内存压力来源**：控制 num_layers 和 layer_size 来达到目标 size_factor

## 结论与建议

### 核心问题

内存压力的来源：
1. **PolyBench 方式**：单个 kernel 访问的数据量 > GPU 内存
2. **Hotspot 方式**：同一数据被迭代反复访问
3. **当前多 batch 方式**：❌ 每个 kernel 只访问 48MB，不会触发 eviction

### 建议

**对于 UVM benchmark 测试，采用 PolyBench 方式更简单有效**：

```cpp
// 根据 size_factor 计算矩阵维度
// size_factor=1.0 → N ≈ 3200 (单矩阵 ~10GB，总 ~30GB)
size_t target_per_matrix = total_working_set / 3;
int N = (int)sqrt(target_per_matrix / sizeof(DATA_TYPE));

cudaMallocManaged(&A, N * N * sizeof(DATA_TYPE));
cudaMallocManaged(&B, N * N * sizeof(DATA_TYPE));
cudaMallocManaged(&C, N * N * sizeof(DATA_TYPE));

// 单个大 GEMM
gemm_kernel<<<grid, block>>>(N, N, N, alpha, beta, A, B, C);
```

这样：
- size_factor < 1.0：数据全部驻留 GPU，测试热缓存性能
- size_factor > 1.0：数据超过 GPU 内存，触发 page fault 和 eviction

### 不同场景的选择

| 测试目标 | 推荐方案 |
|---------|---------|
| UVM 基本功能 | PolyBench 方式（单个大 GEMM）|
| UVM 策略（预取/eviction）| Hotspot 方式（迭代访问）|
| 真实 DL 负载模拟 | 权重复用设计 |

## 方案 C：简化的权重复用设计

核心思路：**一个大 buffer + 偏移访问**

```cpp
void run_gemm_weight_reuse(size_t total_working_set, ...) {
    const int dim = 1024;
    const int hidden = 3072;
    size_t layer_size = dim * hidden * sizeof(DATA_TYPE);  // ~12MB
    int num_layers = total_working_set / layer_size;
    if (num_layers < 1) num_layers = 1;

    // 1. 一个大 buffer 存所有权重
    DATA_TYPE* weights;
    cudaMallocManaged(&weights, (size_t)num_layers * layer_size);

    // CPU 初始化（确保页面在 CPU）
    for (size_t i = 0; i < num_layers * dim * hidden; i++) {
        weights[i] = (DATA_TYPE)(i % 1000) / 1000.0f;
    }

    // 2. 小的 activation buffer
    DATA_TYPE *x, *out;
    cudaMallocManaged(&x, dim * sizeof(DATA_TYPE));
    cudaMallocManaged(&out, hidden * sizeof(DATA_TYPE));

    // 3. 多个 token 遍历所有层
    int num_tokens = 10;  // 或根据需要调整

    auto launch = [&]() {
        for (int t = 0; t < num_tokens; t++) {
            for (int l = 0; l < num_layers; l++) {
                DATA_TYPE* W = weights + (size_t)l * dim * hidden;
                gemm_kernel<<<grid, block>>>(dim, hidden, 1,
                                             1.0f, 0.0f, x, W, out);
                // 不 sync
            }
        }
    };

    time_kernel(launch, warmup, iterations, runtimes, result);
}
```

**访问模式**：
```
Token 0: W[0:12MB] → W[12MB:24MB] → ... → W[31GB:32GB]
         ↑ 顺序访问大 buffer 的不同部分
Token 1: W[0:12MB] → W[12MB:24MB] → ... → W[31GB:32GB]
         ↑ 重复访问，测试 eviction 后的重新加载
```

**与其他方案对比**：

| 方案 | 单次 kernel 访问 | 内存压力来源 | 复杂度 |
|------|----------------|--------------|--------|
| PolyBench | 全部数据 | 数据量 | 简单 |
| 多 batch | 48MB（独立）| 无 | 简单 |
| 权重复用 | 12MB（共享buffer偏移）| 数据量 + 重复访问 | 简单 |

---

# 其他 Kernel 设计分析

## 当前实现状态

| Kernel | 当前实现 | 真实算法特点 | 状态 |
|--------|----------|--------------|------|
| **Hotspot** | 60 次迭代，每次访问全部数据 | 热传导模拟，迭代求解 | ✅ 正确 |
| **Jacobi2D** | 20 次迭代，5-point stencil | PDE 求解器，迭代收敛 | ✅ 正确 |
| **Kmeans** | 100 次迭代，聚类收敛 | ML 聚类，迭代优化 | ✅ 正确 |
| **Conv2D** | 单次 3x3 卷积 | CNN 多层卷积 | ⚠️ 可改进 |
| **BFS** | 单源 BFS | 图分析（PageRank 等）| ⚠️ 可改进 |
| **GEMM** | 权重复用设计 | 神经网络推理 | ✅ 已修改 |

## Hotspot / Jacobi2D - 已正确 ✅

这两个 kernel 天然符合"迭代访问同一数据"的模式：

```cpp
// Hotspot: 60 次迭代
for (int t = 0; t < 60; t++) {
    calculate_temp<<<grid, block>>>(power, temp[src], temp[dst], ...);
    swap(src, dst);
}

// Jacobi2D: 20 次迭代
for (int t = 0; t < 20; t++) {
    jacobi_kernel1<<<grid, block>>>(A, B);  // B = stencil(A)
    jacobi_kernel2<<<grid, block>>>(A, B);  // A = B
}
```

**特点**：
- 固定大小数据，通过迭代产生内存压力
- 每次迭代访问全部数据
- 符合真实科学计算场景

## Kmeans - 已正确 ✅

```cpp
// 100 次聚类迭代
for (int rep = 0; rep < 100; rep++) {
    get_dst<<<n, k>>>(dst, x, y, mu_x, mu_y);  // 计算距离
    regroup<<<n, 1>>>(group, dst, k);           // 重新分组
    recenter<<<1, k>>>(mu_x, mu_y, ...);        // 更新中心
}
```

**特点**：
- 迭代优化算法
- 每次迭代访问所有数据点
- 符合真实 ML 训练场景

## Conv2D - 需要改进 ⚠️

### 当前问题

```cpp
// 当前：单次卷积
convolution2D_kernel<<<grid, block>>>(A, B);
// 只执行一次，数据访问一遍就结束
```

### 真实 CNN 场景

VGG-16 有 13 个卷积层，ResNet-50 有 53 个卷积层：

```cpp
// 真实 CNN: 多层卷积
for (int l = 0; l < num_layers; l++) {
    conv_kernel<<<grid, block>>>(input, filters[l], output);
    // 可能有 ReLU, pooling 等
}
```

### 建议修改（类似 GEMM 的权重复用设计）

```cpp
void run_conv2d(size_t total_working_set, ...) {
    // 固定特征图大小
    const int H = 224, W = 224, C = 64;  // 特征图
    const int K = 3;  // 3x3 卷积核

    // 根据 working_set 计算层数
    size_t filter_size = K * K * C * C * sizeof(float);  // ~144KB per layer
    int num_layers = total_working_set / filter_size;

    // 一个大 buffer 存所有 filter
    float* filters;
    cudaMallocManaged(&filters, num_layers * filter_size);

    // 特征图（小，复用）
    float *input, *output;
    cudaMallocManaged(&input, H * W * C * sizeof(float));
    cudaMallocManaged(&output, H * W * C * sizeof(float));

    // 多层顺序卷积
    auto launch = [&]() {
        for (int l = 0; l < num_layers; l++) {
            float* F = filters + l * K * K * C * C;
            conv_kernel<<<grid, block>>>(input, F, output);
            std::swap(input, output);  // 输出变输入
        }
    };
}
```

## PageRank - 已实现 ✅

### 实现文件

`kernels/pagerank.cuh` - 基于 GPreempt/EMOGI 的 PageRank 实现

### 设计原则

与 GEMM/Hotspot/Jacobi 相同：**固定图大小 + 迭代次数控制总工作量**

```cpp
// 固定图大小：1M 节点，avgDegree=16，约 150MB
const uint64_t NUM_NODES = 1000000;
const int AVG_DEGREE = 16;

// 根据 working_set 计算迭代次数
int num_iterations = total_working_set / single_iteration_bytes;

// PageRank 迭代
for (int iter = 0; iter < num_iterations; iter++) {
    pr_kernel_coalesce<<<...>>>(label, delta, residual, ...);
    pr_update<<<...>>>(label, delta, residual, value, ...);
}
```

### 核心 Kernel（来自 EMOGI）

```cpp
// Warp-coalesced PageRank kernel
__global__ void pr_kernel_coalesce(bool *label, float *delta, float *residual,
                                   uint64_t vertex_count, uint64_t *vertexList,
                                   uint64_t *edgeList) {
    uint64_t warpIdx = tid >> PR_WARP_SHIFT;
    uint64_t laneIdx = tid & (PR_WARP_SIZE - 1);

    if (warpIdx < vertex_count && label[warpIdx]) {
        // Warp 协作访问邻居
        for (uint64_t i = shift_start + laneIdx; i < end; i += PR_WARP_SIZE) {
            atomicAdd(&residual[edgeList[i]], delta[warpIdx]);
        }
    }
}
```

### 访问模式

- **随机访问**：通过邻接表间接访问
- **时间局部性**：多次迭代访问同一图结构
- **Warp 协作**：利用 warp coalescing 优化内存访问

### 与原 BFS 对比

| 特性 | BFS (保留) | PageRank (新增) |
|------|-----------|----------------|
| 遍历方式 | 单源一次 | 迭代多次 |
| 时间局部性 | 无 | 有 |
| 控制变量 | 图大小 | 迭代次数 |
| 适用测试 | 纯随机访问 | 随机+迭代 |

## 总结

| Kernel | 内存压力模式 | 访问特点 | 状态 |
|--------|-------------|---------|------|
| Hotspot | 迭代 × 网格 | 顺序 stencil | ✅ 已改 |
| Jacobi2D | 迭代 × 网格 | 顺序 stencil | ✅ 已改 |
| GEMM | 多token × 权重 | 顺序层访问 | ✅ 已改 |
| PageRank | 迭代 × 边数 | 随机图访问 | ✅ 新增 |
| BFS | 一次 × 图大小 | 纯随机访问 | 保留原始 |
| Kmeans | 迭代 × 数据点 | 顺序扫描 + 随机聚类 | ✅ 正确 |
| Conv2D | 多层 × filter | 顺序卷积 | ⚠️ 可改进 |

**核心原则**：内存压力 = 数据量 × 重复访问次数

---

# 数据局部性与真实 Workload 分析

## 32GB 工作集的局部性问题

从 kernel 访问模式角度分析，不同 kernel 对大工作集的适应性不同：

| Kernel | 单次访问大小 | 访问模式 | 32GB 局部性 |
|--------|-------------|---------|-------------|
| Hotspot/Jacobi | 整个网格 | 顺序 stencil | ⚠️ 网格太大 |
| GEMM | 12MB/层 | 顺序跳跃 | ⚠️ stride 太大 |
| BFS | 随机边 | 完全随机 | ❌ 很差 |
| Kmeans | 全部点 | 顺序扫描 | ✅ 较好 |

## Hotspot/Jacobi - Stencil 类

### 当前实现

```cpp
// 根据 working_set 计算网格大小
// 32GB → sqrt(32GB / 3 / 4) ≈ 53000 x 53000
int grid_size = (int)sqrt(array_bytes / sizeof(float));

// 固定 60 次迭代
int total_iterations = 60;
```

### 问题分析

- 53000 x 53000 网格 **不符合真实 workload**
- 真实 CFD/天气模拟通常是 1000~8000 的网格
- 单次 kernel 访问 32GB，page fault 会很密集
- 空间局部性好（stencil），但数据量太大

### 真实 Workload 参考

| 应用 | 典型网格大小 | 迭代次数 |
|------|-------------|---------|
| 热传导模拟 | 1024 x 1024 | 1000+ |
| CFD (2D) | 4096 x 4096 | 10000+ |
| 图像处理 | 1920 x 1080 | 1-10 |

### 建议修改

```cpp
void run_hotspot(size_t total_working_set, ...) {
    // 固定合理的网格大小
    const int GRID_SIZE = 4096;  // 4096 x 4096 ≈ 200MB (3 arrays)
    size_t single_iteration_bytes = 3 * GRID_SIZE * GRID_SIZE * sizeof(float);

    // 根据 working_set 计算迭代次数
    int total_iterations = total_working_set / single_iteration_bytes;
    if (total_iterations < 10) total_iterations = 10;

    // 这样：
    // - 单次 kernel 访问 ~200MB，局部性好
    // - 通过迭代次数控制总工作量
    // - size_factor=1.0 → ~160 次迭代 (32GB / 200MB)
}
```

**访问模式**：
```
Iteration 0: 访问全部 200MB（stencil 局部性好）
Iteration 1: 再次访问全部 200MB（数据可能已在 GPU）
...
Iteration 159: 第 160 次访问同一 200MB
```

- 时间局部性 ✅：同一数据被反复访问
- 空间局部性 ✅：stencil 相邻访问
- 符合真实 workload ✅

## GEMM - 神经网络类

### 当前实现

```cpp
const int dim = 1024;
const int hidden = 3072;
size_t layer_size = dim * hidden * sizeof(float);  // ~12MB

int num_layers = total_working_set / layer_size;
// 32GB → 2700 层
```

### 问题分析

- 2700 层 **不符合真实网络**
- 真实 LLM：28-96 层
- 每层 12MB 太小，stride 跳跃大
- 预取器难以预测 12MB 间隔的访问

### 真实 Workload 参考

| 模型 | 层数 | 每层权重 | 总权重 |
|------|------|---------|--------|
| Qwen-0.6B | 28 | ~20MB | ~600MB |
| Llama-7B | 32 | ~200MB | ~7GB |
| Llama-70B | 80 | ~800MB | ~70GB |

### 建议修改

```cpp
void run_gemm(size_t total_working_set, ...) {
    // 固定合理的层数
    const int NUM_LAYERS = 32;  // 类似 Llama-7B

    // 根据 working_set 计算每层大小
    size_t layer_size = total_working_set / NUM_LAYERS;

    // 反算 dim 和 hidden
    // layer_size = dim * hidden * sizeof(float)
    // 假设 hidden = 4 * dim (常见比例)
    int dim = (int)sqrt(layer_size / sizeof(float) / 4);
    int hidden = dim * 4;

    // 这样：
    // - size_factor=0.5 → 每层 ~500MB，dim≈11000
    // - size_factor=1.0 → 每层 ~1GB，dim≈16000
    // - 每层足够大，连续访问
}
```

或者用更真实的参数：

```cpp
void run_gemm(size_t total_working_set, ...) {
    // 真实的层大小（类似 Llama-7B 的一层）
    const int dim = 4096;
    const int hidden = 11008;
    size_t layer_size = dim * hidden * sizeof(float);  // ~180MB

    // 根据 working_set 计算层数
    int num_layers = total_working_set / layer_size;
    if (num_layers < 1) num_layers = 1;

    // 多个 token 遍历所有层
    int num_tokens = 10;

    // 这样：
    // - size_factor=0.5 → ~90 层
    // - 每层 180MB，局部性好
    // - 符合真实 LLM 推理
}
```

## BFS - 图算法类

### 当前实现

```cpp
// 一个大图
int numNodes = total_working_set / (avgDegree * sizeof(int) + 4 * sizeof(int));
// 32GB → ~1亿节点

// 单源 BFS
d_distance[0] = 0;
while (changed) {
    simpleBfs<<<...>>>(level, ...);
}
```

### 问题分析

- 1 亿节点的单个大图 **不符合真实 workload**
- 随机访问邻接表，局部性极差
- 单源 BFS 只遍历一次

### 真实 Workload 参考

| 场景 | 图大小 | 处理方式 |
|------|--------|---------|
| GNN mini-batch | 1000-10000 节点/子图 | 多子图并行 |
| 推荐系统 | 10000 节点/用户子图 | 多用户并行 |
| 社交网络分析 | 百万节点 | 迭代算法 (PageRank) |

### 建议修改：多小图并行

```cpp
void run_bfs(size_t total_working_set, ...) {
    // 固定单个图的大小（GNN mini-batch 风格）
    const int NODES_PER_GRAPH = 10000;
    const int AVG_DEGREE = 16;
    size_t single_graph_bytes = NODES_PER_GRAPH * (AVG_DEGREE + 4) * sizeof(int);
    // ~800KB per graph

    // 根据 working_set 计算图的数量
    int num_graphs = total_working_set / single_graph_bytes;
    if (num_graphs < 1) num_graphs = 1;

    // 分配所有图的邻接表（连续存储）
    // Graph 0: [0, 800KB)
    // Graph 1: [800KB, 1.6MB)
    // ...

    auto launch = [&]() {
        for (int g = 0; g < num_graphs; g++) {
            // 每个图独立做 BFS 或 PageRank
            int* adj = all_adjacency + g * edges_per_graph;
            bfs_kernel<<<blocks, threads>>>(adj, ...);
        }
    };
}
```

**优点**：
- 每个小图有较好的局部性
- 多图并行利用 GPU 并行性
- 符合 GNN 的 mini-batch 训练模式

### 或者：PageRank 迭代

```cpp
void run_bfs(size_t total_working_set, ...) {
    // 中等大小的图
    int numNodes = 1000000;  // 100 万节点
    int avgDegree = 16;
    // 图大小 ~300MB

    // 多次 PageRank 迭代
    int num_iterations = total_working_set / graph_bytes;

    auto launch = [&]() {
        for (int iter = 0; iter < num_iterations; iter++) {
            pagerank_kernel<<<...>>>(adj, rank_in, rank_out);
            std::swap(rank_in, rank_out);
        }
    };
}
```

**优点**：
- 迭代访问同一图，有时间局部性
- 符合真实图分析 workload

## Conv2D - 类似 GEMM

### 建议：固定特征图大小，调整层数

```cpp
void run_conv2d(size_t total_working_set, ...) {
    // 固定特征图大小（类似 ResNet 中间层）
    const int H = 56, W = 56, C = 256;
    const int K = 3;
    size_t feature_size = H * W * C * sizeof(float);     // ~3MB
    size_t filter_size = K * K * C * C * sizeof(float);  // ~2.3MB

    // 根据 working_set 计算层数
    int num_layers = total_working_set / filter_size;

    // 每层卷积，filter 连续存储
}
```

## 总结：真实 Workload 的设计原则

| 原则 | 说明 |
|------|------|
| **固定合理的单次访问大小** | 不要让单次 kernel 访问 32GB |
| **通过迭代/层数控制总量** | 用迭代次数或批次数量调整 |
| **保持空间局部性** | 连续访问，避免大 stride |
| **保持时间局部性** | 同一数据被多次访问 |

### 建议的参数设计

| Kernel | 单次访问 | 控制变量 | size_factor=1.0 时 |
|--------|---------|---------|-------------------|
| Hotspot | ~200MB (4K grid) | 迭代次数 | 160 次迭代 |
| Jacobi2D | ~130MB (4K grid) | 迭代次数 | 250 次迭代 |
| GEMM | ~180MB/层 | 层数 | 180 层 |
| PageRank | ~150MB (1M nodes) | 迭代次数 | 220 次迭代 |
| Conv2D | ~2.3MB/层 | 层数 | 14000 层 |
| Kmeans | 全部点 | 迭代次数 | 当前已正确 |

---

# 实验设计建议

## 实验 1: Oversub Factor → 性能退化曲线

**目标**: 对比不同 UVM 策略在不同 oversubscription 程度下的表现

```
x 轴: size_factor (0.5×, 1×, 1.5×, 2×, 4×)
y 轴:
  - end-to-end 吞吐 (tokens/s 或 slowdown)
  - page faults / token
  - migrated GB / token

对比:
  - baseline: 默认 UVM (LRU + 内建 prefetch)
  - your_policy: 你的策略
  - upper_bound: 手动 cudaMemPrefetchAsync
```

**预期结果**:
- size_factor ≈ 1: baseline 还凑合，你的策略略好
- size_factor > 2: baseline 出现 thrashing，你的策略显著更好

## 实验 2: Temporal Locality 利用率

**目标**: 验证策略是否真正利用了重复访问的局部性

```cpp
// 固定 total_working_set 和 size_factor
// 改变 num_tokens (权重被重复访问的次数)

Case 1: num_tokens = 1   // 只扫一遍，无 temporal reuse
Case 2: num_tokens = 16  // 每层被访问 16 次，有明显 locality
```

**预期结果**:
- baseline 在 Case 2 可能仍然 thrash（刚用完就被 evict）
- 你的策略在 Case 2 应该明显优于 Case 1（利用了 reuse）

## 实验 3: 不同访存 Pattern 对比

**目标**: 说明策略不是只在一种极端模式下生效

| Pattern | 代表 Kernel | 策略预期效果 |
|---------|------------|-------------|
| 顺序流式 | seq_stream | 硬件 prefetch 足够，策略提升小 |
| 完全随机 | rand_stream | 无局部性，策略提升有限 |
| Stencil 迭代 | Hotspot/Jacobi | 空间+时间局部性，策略有效 |
| 权重复用 | GEMM | 时间局部性强，策略显著有效 |
| 图随机迭代 | BFS/PageRank | 随机但有时间局部性，策略有效 |

## 指标收集

每个实验应收集:

| 指标 | 来源 | 说明 |
|------|------|------|
| runtime_ms | cudaEvent | 端到端时间 |
| page_faults | nvidia-smi / nsight | 总 page fault 数 |
| bytes_migrated | UVM counters | HtoD + DtoH 数据量 |
| bandwidth_util | 计算 | bytes_migrated / runtime |
| slowdown | 计算 | runtime / baseline_fit_in_gpu |

---

# Benchmark Suite 总览

## Kernel 分类

| 类别 | Kernel | 访问模式 | 代表应用 |
|------|--------|---------|---------|
| **Synthetic** | seq_stream | 顺序流式 | 内存带宽测试 |
| | rand_stream | 完全随机 | 随机访问测试 |
| | pointer_chase | 链表追踪 | 延迟测试 |
| **Scientific** | Hotspot | 5-point stencil + 迭代 | CFD, 热传导 |
| | Jacobi2D | 5-point stencil + 迭代 | PDE 求解器 |
| **ML/DL** | GEMM | 权重复用 (LLaMA-style) | LLM 推理 |
| | Conv2D | 多层卷积 | CNN 推理 |
| | Kmeans | 全量扫描迭代 | 聚类 |
| **Graph** | PageRank | 图随机 + 迭代 | 社交网络分析, 推荐 |
| | BFS | 图随机 (单次) | 图遍历基准 |

## 参数设计原则

1. **单次访问大小**: 100MB ~ 500MB（符合真实 workload）
2. **总工作量控制**: 通过迭代次数/层数/图数量调整
3. **Oversub factor**: 0.5× ~ 4× 覆盖常见场景
4. **对齐真实模型**: 参数选择参考实际 LLM/CNN/GNN 规模
